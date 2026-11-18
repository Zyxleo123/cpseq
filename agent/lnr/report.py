#!/usr/bin/env python3
"""Report the LNR experiment: real ring closure, no response model.

Reads the rule rows, the LLM rows, and the frontier, all JSONL written by
lnr/run.py and lnr/frontier.py, so every number is redrawable without
re-running a torsion solve. Arms are compared paired over targets at matched
evaluation budget; the frontier ceiling is context only.

    python -m lnr.report --dir <lnr_outdir>

The quantity scored is the retention of the best window that actually CLOSED
within the evaluation budget, 0 if none did. It is a real coordinate fraction,
not a modelled utility. Two summaries are reported per arm:

  closure rate  -- fraction of targets where the arm found ANY closing window.
                   A controller that never closes has solved nothing.
  retention     -- mean best-closing retention. A controller is only credited
                   interface it kept on a ring that actually closes.

And the one comparison the whole project is about, here on real geometry:
mask-authoring vs tool-selection vs the strong deterministic rule, paired.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb, erf, sqrt
from pathlib import Path
from statistics import mean


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def wilcoxon(deltas: list[float]):
    nz = [d for d in deltas if abs(d) > 1e-9]
    n = len(nz)
    if n == 0:
        return None, "all_tied"
    if n < 6:
        k = min(sum(1 for d in nz if d < 0), sum(1 for d in nz if d > 0))
        return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), f"sign_n{n}"
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w = min(sum(r for r, d in zip(ranks, nz) if d > 0),
            sum(r for r, d in zip(ranks, nz) if d < 0))
    mu = n * (n + 1) / 4.0
    sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu + 0.5) / sd if sd else 0.0
    return min(1.0, 2.0 * 0.5 * (1.0 + erf(-abs(z) / sqrt(2.0)))), f"wilcoxon_n{n}"


def mcnemar(a: dict, b: dict, parents: list[str], closed_field: str = "any_closed"):
    ao = sum(1 for p in parents if a[p].get(closed_field) and not b[p].get(closed_field))
    bo = sum(1 for p in parents if b[p].get(closed_field) and not a[p].get(closed_field))
    n = ao + bo
    p = None
    if n:
        k = min(ao, bo)
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return ao, bo, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--clash-file", default="clashes.jsonl",
                    help="clash map filename inside --dir (default clashes.jsonl, "
                    "the CA gate; use clashes_bb.jsonl for the backbone-triplet gate).")
    ap.add_argument("--clash-field", default="clash_free",
                    help="which boolean in the clash map marks a clean closure: "
                    "'clash_free' (CA, 3.0 A) or 'clash_free_bb' (backbone, 2.5 A).")
    ap.add_argument("--metric",
                    choices=["retention", "binding", "clean", "clean-binding"],
                    default="retention",
                    help="which quantity the PAIRED tests and arm ordering use: "
                    "'retention' = CA-contact fraction of the best GEOMETRICALLY "
                    "closing window (always available); "
                    "'clean' = CA-contact fraction of the best CLASH-FREE closing "
                    "window -- the physically meaningful metric (needs clashes.jsonl); "
                    "'binding' = native interface ENERGY the best geometric window "
                    "keeps (Rosetta alanine scan, needs rosetta_interface.jsonl); "
                    "'clean-binding' = native interface ENERGY the best CLASH-FREE "
                    "window keeps -- physical validity AND real binding, the "
                    "strongest combined metric (needs both clashes.jsonl + Rosetta).")
    args = ap.parse_args()
    d = Path(args.dir)

    rows = []
    for f in sorted(d.glob("rules*.jsonl")):
        rows += load(f)
    for f in sorted(d.glob("llm_*.jsonl")):
        rows += load(f)
    frontier = {r["target_id"]: r for r in load(d / "frontier.jsonl")}

    # clean-closure re-grade (if the clash map exists): each campaign's best
    # retention over the windows it tried that close WITHOUT clashes. Added as
    # best_retention_clean / any_clean on every row, so the report can show
    # both the geometric metric and the stricter physical one.
    clash = {}
    for c in load(d / args.clash_file):
        clash[(c["target_id"], c["a"], c["b"])] = c.get(args.clash_field, False)

    # per-residue REAL binding share (Rosetta alanine scan), for energy-weighted
    # retention: a held window's binding quality = fraction of the native
    # interface ENERGY its residues carry, vs the CA-contact fraction.
    bind = {}
    for b in load(d / "rosetta_interface.jsonl"):
        if "per_residue_ddG_ala" not in b:   # failed target (timeout/crash)
            continue
        pr = {int(k): (v or 0.0) for k, v in b["per_residue_ddG_ala"].items()}
        tot = sum(v for v in pr.values() if v > 0) or 1.0
        bind[b["target_id"]] = (pr, tot, b.get("native_dG"))

    def binding_retention(tid, a, b):
        if tid not in bind:
            return None
        pr, tot, _ = bind[tid]
        kept = sum(pr[k] for k in range(a, b + 1) if k in pr and pr[k] > 0)
        return kept / tot
    for r in rows:
        if clash:
            clean = [(h["retention"], h["a"], h["b"]) for h in r.get("history", [])
                     if h.get("closes")
                     and clash.get((r["target_id"], h["a"], h["b"]), False)]
            if clean:
                ret_c, ca, cb = max(clean, key=lambda x: x[0])
                r["best_retention_clean"] = ret_c
                r["best_clean_window"] = [ca, cb]
                r["any_clean"] = True
            else:
                r["best_retention_clean"] = 0.0
                r["best_clean_window"] = None
                r["any_clean"] = False
        else:
            r["best_retention_clean"] = None
            r["best_clean_window"] = None
            r["any_clean"] = None
        # binding-weighted retention of the best closing window the arm found.
        # None means only "this target was never Rosetta-scored", so it drops
        # out of binding comparisons. A target that WAS scored but on which the
        # arm closed nothing keeps 0.0 -- it kept no binding because it never
        # closed, which is a real score, not missing data.
        if r["target_id"] in bind:
            if r.get("best_window"):
                a, b = r["best_window"]
                r["binding_retention"] = binding_retention(r["target_id"], a, b)
            else:
                r["binding_retention"] = 0.0
        else:
            r["binding_retention"] = None
        # clean-binding: binding energy retained by the best CLASH-FREE window
        # (the physically-valid ring the arm committed to -- the same window
        # best_retention_clean picks), not the geometric one. The report's plain
        # bindRet grades the geometric window and so UNDERSELLS an arm that wins
        # by finding clean rings. None if the target was never Rosetta-scored;
        # 0.0 if scored but the arm found no clash-free closer (kept no binding).
        if r["target_id"] in bind and clash:
            if r.get("best_clean_window"):
                a, b = r["best_clean_window"]
                r["clean_binding_retention"] = binding_retention(r["target_id"], a, b)
            else:
                r["clean_binding_retention"] = 0.0
        else:
            r["clean_binding_retention"] = None

    # arm key: rules keep their name; LLM arms carry the model
    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        name = r["arm"]
        if name.startswith("llm-"):
            name = f"{r.get('model_id', '?')}|{name}"
        arms[name][r["target_id"]] = r

    print("=" * 96)
    print("LNR -- real ring closure, no response model")
    print("=" * 96)
    has_clean = any(v.get("any_clean") is not None
                    for a in arms for v in arms[a].values())
    has_bind = any(v.get("binding_retention") is not None
                   for a in arms for v in arms[a].values())
    has_cleanbind = any(v.get("clean_binding_retention") is not None
                        for a in arms for v in arms[a].values())
    cl_hdr = f" {'clean%':>6s} {'cleanRet':>8s}" if has_clean else ""
    b_hdr = f" {'bindRet':>7s}" if has_bind else ""
    cb_hdr = f" {'cleanBind':>9s}" if has_cleanbind else ""
    print(f"{'arm':30s} {'n':>3s} {'closes':>7s} {'retention':>9s}{cl_hdr}{b_hdr}{cb_hdr} "
          f"{'rel':>4s} {'evals':>5s} {'calls':>6s} {'cost$':>7s}")
    def _order_val(v):
        if args.metric == "clean-binding" and has_cleanbind:
            return v.get("clean_binding_retention") or 0.0
        if args.metric == "binding" and has_bind:
            return v.get("binding_retention") or 0.0
        if has_clean and v.get("best_retention_clean") is not None:
            return v["best_retention_clean"]
        return v["best_retention"]
    order = sorted(arms, key=lambda a: -mean(_order_val(v) for v in arms[a].values()))
    for a in order:
        v = list(arms[a].values())
        n = len(v)
        close_rate = sum(x["any_closed"] for x in v) / n
        ret = mean(x["best_retention"] for x in v)
        clean_cell = ""
        if has_clean:
            crate = sum(1 for x in v if x.get("any_clean")) / n
            cret = mean(x.get("best_retention_clean") or 0.0 for x in v)
            clean_cell = f" {crate:6.2f} {cret:8.3f}"
        bind_cell = ""
        if has_bind:
            bvals = [x.get("binding_retention") for x in v
                     if x.get("binding_retention") is not None]
            bind_cell = f" {(mean(bvals) if bvals else 0.0):7.3f}"
        cb_cell = ""
        if has_cleanbind:
            cbvals = [x.get("clean_binding_retention") for x in v
                      if x.get("clean_binding_retention") is not None]
            cb_cell = f" {(mean(cbvals) if cbvals else 0.0):9.3f}"
        # mean release of the best closing window, over closers
        rels = [x["best_window"][1] - x["best_window"][0]
                for x in v if x["best_window"]]
        rel = mean(x["length"] - (w + 1) for x, w in
                   zip([x for x in v if x["best_window"]], rels)) if rels else 0.0
        print(f"{a:30s} {n:3d} {close_rate:7.2f} {ret:9.3f}{clean_cell}{bind_cell}{cb_cell} {rel:4.1f} "
              f"{sum(x['n_eval'] for x in v)/n:5.1f} "
              f"{sum(x.get('llm_calls', 0) for x in v):6d} "
              f"{sum(x.get('cost_usd', 0.0) for x in v):7.3f}")

    if frontier:
        ceils = [f["ceiling"]["retention"] for f in frontier.values()
                 if f["ceiling"]]
        n_ceiled = sum(1 for f in frontier.values() if f["ceiling"])
        print(f"{'frontier (ceiling)':30s} {len(frontier):3d} "
              f"{n_ceiled/len(frontier):7.2f} "
              f"{mean(ceils) if ceils else 0:9.3f}   <- best real retention attainable")

    # ---- the decisive paired comparisons --------------------------------
    metric_field = {"binding": "binding_retention",
                    "clean": "best_retention_clean",
                    "clean-binding": "clean_binding_retention",
                    "retention": "best_retention"}[args.metric]
    # closure for McNemar follows the metric: clean closure (a clash-free ring
    # was found) for the clash-gated metrics, else geometric closure.
    closed_field = ("any_clean" if args.metric in ("clean", "clean-binding")
                    else "any_closed")
    metric_name = args.metric
    print()
    print("=" * 96)
    print(f"PAIRED, matched budget, shared targets  --  metric: {metric_name}"
          + ("  (energy-weighted, Rosetta alanine scan)" if args.metric == "binding"
             else "  (CA-contact fraction)"))
    print("=" * 96)
    if args.metric == "binding" and not has_bind:
        print("  WARNING: --metric binding but no rosetta_interface.jsonl found -- "
              "every binding comparison will be empty. Run LNR_MODE=rosif first.")
    if args.metric == "clean" and not has_clean:
        print("  WARNING: --metric clean but no clashes.jsonl found -- "
              "every clean comparison will be empty. Run LNR_MODE=clash first.")
    if args.metric == "clean-binding" and not (has_clean and has_bind):
        print("  WARNING: --metric clean-binding needs BOTH clashes.jsonl and "
              "rosetta_interface.jsonl -- comparisons will be empty without them.")

    def pair(an: str, bn: str, label: str):
        if an not in arms or bn not in arms:
            print(f"  {label}: missing ({an} or {bn})")
            return
        shared = sorted(set(arms[an]) & set(arms[bn]))
        # paired only over targets BOTH arms carry a value for under this metric
        # (binding_retention is None on targets Rosetta never scored).
        comparable = [p for p in shared
                      if arms[an][p].get(metric_field) is not None
                      and arms[bn][p].get(metric_field) is not None]
        if not comparable:
            why = "no shared targets" if not shared \
                else f"no shared targets with {metric_name} scores"
            print(f"  {label}: {why}")
            return
        dret = [arms[an][p][metric_field] - arms[bn][p][metric_field]
                for p in comparable]
        pv, how = wilcoxon(dret)
        wins = sum(1 for x in dret if x > 1e-9)
        losses = sum(1 for x in dret if x < -1e-9)
        ao, bo, pmc = mcnemar(arms[an], arms[bn], comparable, closed_field)
        verdict = "tie"
        if pv is not None and pv < 0.05:
            verdict = "A better" if mean(dret) > 0 else "B better"
        pvs = "   n/a" if pv is None else f"{pv:.4f}"
        pmcs = "   n/a" if pmc is None else f"{pmc:.4f}"
        print(f"  {label}")
        print(f"    {metric_name:>9s}: n={len(comparable)} W/L={wins}/{losses} "
              f"mean_d={mean(dret):+.3f} p={pvs} [{how}] -> {verdict}")
        print(f"    closure:   A_only={ao} B_only={bo} p(mcnemar)={pmcs}")

    # ---- proxy validation: does CA-contact count track real binding energy?
    brows = [b for b in load(d / "rosetta_interface.jsonl")
             if "per_residue_ddG_ala" in b and "per_res_contacts" in b]
    if brows:
        def _sp(x, y):
            n = len(x)
            if n < 3:
                return None
            def rk(v):
                o = sorted(range(n), key=lambda i: v[i]); r = [0.0] * n; i = 0
                while i < n:
                    j = i
                    while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                        j += 1
                    for t in range(i, j + 1):
                        r[o[t]] = (i + j) / 2 + 1
                    i = j + 1
                return r
            rx, ry = rk(x), rk(y); mx = sum(rx) / n; my = sum(ry) / n
            num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            dx = sum((a - mx) ** 2 for a in rx) ** .5
            dy = sum((b - my) ** 2 for b in ry) ** .5
            return num / (dx * dy) if dx and dy else None
        rhos, conc = [], []
        for b in brows:
            pr = {int(k): v for k, v in b["per_residue_ddG_ala"].items()}
            con = b["per_res_contacts"]
            idx = [k for k in sorted(pr) if pr[k] is not None and k < len(con)]
            r = _sp([con[k] for k in idx], [pr[k] for k in idx])
            if r is not None:
                rhos.append(r)
            vals = sorted((pr[k] for k in idx if pr[k] > 0), reverse=True)
            conc.append(sum(vals[:2]) / (sum(vals) or 1))
        rhos.sort()
        print("=" * 96)
        print("PROXY VALIDATION -- per-residue CA contacts vs REAL Rosetta "
              f"binding (alanine scan, n={len(brows)})")
        print("=" * 96)
        print(f"  Spearman(contacts, binding ddG): mean={sum(rhos)/len(rhos):+.2f} "
              f"median={rhos[len(rhos)//2]:+.2f}  -> contacts are a WEAK-MODERATE "
              f"predictor of binding energy")
        print(f"  binding concentration: top-2 residues carry "
              f"{sum(conc)/len(conc):.0%} of interface energy on average "
              f"-> binding is hotspot-driven, which equal-weighted contact "
              f"retention only partly captures")
        print(f"  native dG: {min(b['native_dG'] for b in brows):.0f} .. "
              f"{max(b['native_dG'] for b in brows):.0f} REU (binding-quality reference)")
        print(f"  => report binding-weighted retention (bindRet), not contact "
              f"retention, as the binding-quality metric")
        print()

    # the strong bar is placement_search if present, else feasible
    bar = "placement_search" if "placement_search" in arms else "feasible"
    models = sorted({a.split("|")[0] for a in arms if "|" in a})
    has_bo = "bo" in arms
    for m in models:
        mask, tool = f"{m}|llm-mask", f"{m}|llm-tool"
        pair(mask, tool, f"[{m}] mask-authoring (A) vs tool-selection (B)")
        pair(tool, bar, f"[{m}] tool-selection (A) vs rule:{bar} (B) [strong bar]")
        pair(mask, bar, f"[{m}] mask-authoring (A) vs rule:{bar} (B) [strong bar]")
        # the published-method bar: same action space, budget and feedback as
        # the LLM, differing only in the controller (a GP, not a language model)
        if has_bo:
            pair(tool, "bo", f"[{m}] tool-selection (A) vs rule:bo (B) [BO, published bar]")
            pair(mask, "bo", f"[{m}] mask-authoring (A) vs rule:bo (B) [BO, published bar]")
        # the floor: every LLM arm must clear random, or it has shown nothing
        pair(tool, "random", f"[{m}] tool-selection (A) vs rule:random (B) [FLOOR]")
        pair(mask, "random", f"[{m}] mask-authoring (A) vs rule:random (B) [FLOOR]")
        # the information-matched arms (only when present in the data): the LLM
        # shown the SAME analytic closure-feasibility bound the geometry-aware
        # bar reads. Two questions per scaffold: (1) what did the information
        # buy over the plain arm, and (2) does the info-matched LLM close the
        # gap to the strong deterministic bar it was previously denied.
        for info_arm, plain, kind in ((f"{m}|llm-info-mask", mask, "mask"),
                                      (f"{m}|llm-info-tool", tool, "tool")):
            if info_arm not in arms:
                continue
            pair(info_arm, plain,
                 f"[{m}] info-{kind} (A) vs {kind} (B) [value of the analytic bound]")
            pair(info_arm, bar,
                 f"[{m}] info-{kind} (A) vs rule:{bar} (B) [info-matched vs strong bar]")
            if has_bo:
                pair(info_arm, "bo",
                     f"[{m}] info-{kind} (A) vs rule:bo (B) [info-matched vs BO]")
            pair(info_arm, "random",
                 f"[{m}] info-{kind} (A) vs rule:random (B) [FLOOR]")
    pair(bar, "random", f"rule:{bar} (A) vs rule:random (B) [strong bar vs FLOOR]")
    if has_bo:
        pair(bar, "bo", f"rule:{bar} (A) vs rule:bo (B) [hand-coded strategy vs BO]")
        pair("bo", "random", f"rule:bo (A) vs rule:random (B) [BO vs FLOOR]")
    pair(bar, "contact_core",
         f"rule:{bar} (A) vs rule:contact_core (B) [geometry vs naive]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
