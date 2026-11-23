"""Does the SCOPE choice change the answer? An exact, search-free comparison.

The open question this settles the factual half of: the in-loop fitness averages
the per-residue likelihood over a position set, and there are three defensible
sets -- `all` (the whole peptide, which is what was graded), `window` (the held
window, which is what the LNR objective actually asks for) and `iface` (the
contacting residues). Picking one is a judgement about what the objective IS.
That judgement is not made here. What is made here is the measurement that tells
you whether the judgement MATTERS.

Two things make this exact rather than a search. First, `pll_bg` is additive
over positions, so the best `<= k` substitution set under any scope is read off
the cached PSSM table -- `PLLScorer.optimum` already does it. Second, scoring
someone else's optimum under your scope is another table lookup. So the whole
3x3 question is answered with no model calls and no search bias.

That matters: the obvious cheap version of this analysis -- correlate the three
scope columns already recorded in `seqmut_iter_pll_scores.jsonl` -- is biased,
because every sequence in that file was proposed by a search optimising `all`.
It never visited the `window` landscape, so it would understate the divergence.

The number to read is the OFF-DIAGONAL of the regret matrix:

    regret[X][Y] = fraction of scope Y's available gain that is NOT captured
                   by deploying scope X's optimum

regret[X][X] is 0 by construction. If regret[all][window] is near 0 the scope
question is low-stakes and `all` can stay. If it is large, the project has been
optimising the wrong objective and the rows cannot settle it.

Held fixed throughout: `mutable` = the interface positions, the operator's real
action space. Scope and action space are DIFFERENT axes -- scope says which
positions the fitness averages over, `mutable` says which ones may change -- and
varying both at once would confound them. Grading inside the action space is
also the third trap in docs/seqmut_notes.md.

Run (CPU, needs the cached PSSM tables and the committed windows):
  python agent/lnr/pll_scope.py --manifest ... --pdb-dir ... \
      --pssm-dir .../pssm --seed-windows .../seqmut_iter_pll_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

SCOPES = ("all", "window", "iface")


def _positions_for(scope, t, window, PI):
    """(positions, interface_only) for a scope, matching PLLScorer.optimum."""
    if scope == "window":
        return PI._window_positions(t, window), False
    if scope == "iface":
        return None, True
    return None, False


def compare_target(sc_by_scope, t, window, mutable, k_mut, PI):
    """The 3x3 cross-scope table for one target, or None if unscorable.

    `sc_by_scope` is a PLLScorer per scope, sharing one PSSM cache.
    """
    opt, pssm = {}, None
    for s in SCOPES:
        o = sc_by_scope[s].optimum(t, k_mut, window=window, mutable=mutable)
        if o is None:
            return None
        opt[s] = o
    try:
        pssm = sc_by_scope["all"].pssm(t)
    except Exception:
        return None

    # Apply a scope's optimal subs to the native sequence.
    def mutate(subs):
        seq = list(t.seq)
        for i, a in subs:
            seq[int(i)] = a
        return "".join(seq)

    seq_of = {s: mutate(opt[s]["subs"]) for s in SCOPES}

    row = {"target_id": t.target_id, "length": int(t.length),
           "n_mutable": len(set(mutable)) if mutable is not None else None,
           "window": window, "subs": {s: opt[s]["subs"] for s in SCOPES},
           "subs_key": {s: opt[s]["subs_key"] for s in SCOPES},
           "gain": {}, "regret": {x: {} for x in SCOPES},
           "degenerate_scopes": []}

    for y in SCOPES:
        pos, iface_only = _positions_for(y, t, window, PI)
        kw = dict(metric=sc_by_scope[y].metric, positions=pos,
                  interface_only=iface_only)
        wt_y = pssm.score(t.seq, **kw)
        best_y = pssm.score(seq_of[y], **kw)
        gain_y = best_y - wt_y          # higher pll is better
        row["gain"][y] = round(gain_y, 6)
        for x in SCOPES:
            got = pssm.score(seq_of[x], **kw)
            # Fraction of y's reachable gain forgone by deploying x's optimum.
            r = ((best_y - got) / gain_y) if gain_y > 1e-9 else 0.0
            row["regret"][x][y] = round(r, 6)
        if gain_y <= 1e-9:
            # No reachable gain under this scope: a regret of 0 here means
            # "nothing to win", not "the scopes agree". Flagged so the fleet
            # mean cannot be read as agreement.
            row["degenerate_scopes"].append(y)

    row["all_scopes_agree"] = len({row["subs_key"][s] for s in SCOPES}) == 1
    # `iface` collapses onto `all` whenever every residue contacts the receptor,
    # and `window` collapses onto `all` whenever the window spans the peptide.
    # Those targets cannot discriminate and are counted separately.
    pos_sets = {}
    for s in SCOPES:
        pos, iface_only = _positions_for(s, t, window, PI)
        pos_sets[s] = tuple(sorted(pssm._positions(iface_only, pos)))
    row["position_sets_identical"] = len(set(pos_sets.values())) == 1
    row["n_positions"] = {s: len(pos_sets[s]) for s in SCOPES}
    return row


def summarize(rows):
    """Fleet means of the off-diagonal, plus the discriminating subset."""
    disc = [r for r in rows if not r["position_sets_identical"]]
    out = {"n": len(rows), "n_discriminating": len(disc),
           "n_agree_on_optimum": sum(1 for r in rows if r["all_scopes_agree"]),
           "matrix": {}, "matrix_discriminating": {}}
    for x in SCOPES:
        out["matrix"][x] = {}
        out["matrix_discriminating"][x] = {}
        for y in SCOPES:
            v = [r["regret"][x][y] for r in rows
                 if y not in r["degenerate_scopes"]]
            vd = [r["regret"][x][y] for r in disc
                  if y not in r["degenerate_scopes"]]
            out["matrix"][x][y] = round(st.mean(v), 4) if v else None
            out["matrix_discriminating"][x][y] = (round(st.mean(vd), 4)
                                                  if vd else None)
    if disc:
        out["agree_on_optimum_discriminating"] = sum(
            1 for r in disc if r["all_scopes_agree"])
    return out


def _print_matrix(m, title):
    print(f"\n  {title}")
    print(f"  {'deploy\\true':>14s} " + " ".join(f"{y:>9s}" for y in SCOPES))
    for x in SCOPES:
        cells = []
        for y in SCOPES:
            v = m[x][y]
            cells.append("     n/a " if v is None else f"{v:>9.3f}")
        print(f"  {x:>14s} " + " ".join(cells))


def report(manifest, pdb_dir, pssm_dir, seed_windows=None, targets=(),
           metric="pll_bg", k_mut=2, out_json=None, model=None, device="cpu"):
    from lnr import env as E
    from lnr import pll_inloop as PI
    from lnr import seqmut_iter as SI
    import lnr.bindscore as bs

    ts = E.load_manifest(str(manifest), str(pdb_dir))
    ts.sort(key=lambda x: x.target_id)
    want = {s for s in targets if s}
    if want:
        ts = [x for x in ts if x.target_id in want]
    if not ts:
        print(f"GATE: no targets from {manifest} / {pdb_dir}")
        return 1

    windows = {}
    if seed_windows and Path(seed_windows).exists():
        windows = SI._load_windows(Path(seed_windows))
        print(f"reusing {len(windows)} committed windows from {seed_windows}")
    else:
        print("NOTE: no committed windows -- scope='window' will fall back to "
              "the whole peptide and cannot discriminate. Pass --seed-windows.")

    sc_by_scope = {}
    for s in SCOPES:
        sc_by_scope[s] = PI.PLLScorer(metric=metric, scope=s,
                                      model_name=(model or bs.PRIMARY_MODEL),
                                      device=device, pssm_dir=pssm_dir)
    print(f"[pll-scope] {len(ts)} target(s)  metric={metric}  k_mut={k_mut}  "
          f"scopes={','.join(SCOPES)}")

    rows, skipped = [], []
    for t in ts:
        win = windows.get(t.target_id)
        mutable = SI._iface_positions(t)
        try:
            r = compare_target(sc_by_scope, t, win, mutable, k_mut, PI)
        except Exception as exc:                      # unscorable target
            skipped.append((t.target_id, f"{type(exc).__name__}: {exc}"))
            continue
        if r is None:
            skipped.append((t.target_id, "no usable pose"))
            continue
        rows.append(r)

    if not rows:
        print("GATE: no scorable target. Is --pssm-dir the cached table dir?")
        return 1

    s = summarize(rows)
    print(f"\nscorable {len(rows)}  skipped {len(skipped)}")
    print(f"scopes select the SAME optimum on {s['n_agree_on_optimum']}/{s['n']}")
    print(f"targets where the three position sets DIFFER (can discriminate): "
          f"{s['n_discriminating']}/{s['n']}")
    if s.get("agree_on_optimum_discriminating") is not None:
        print(f"  of those, same optimum anyway: "
              f"{s['agree_on_optimum_discriminating']}/{s['n_discriminating']}")

    print("\n  normalised cross-scope regret: fraction of the TRUE scope's "
          "reachable\n  gain forgone by deploying the DEPLOYED scope's optimum "
          "(0 = no cost)")
    _print_matrix(s["matrix"], "all scorable targets")
    if s["n_discriminating"]:
        _print_matrix(s["matrix_discriminating"],
                      "only targets whose position sets differ")

    if skipped:
        print(f"\n  skipped: " + ", ".join(f"{a} ({b})" for a, b in skipped[:8])
              + (" ..." if len(skipped) > 8 else ""))

    # The verdict is deliberately about stakes, not about which scope to pick.
    off = [s["matrix_discriminating"][x][y] for x in SCOPES for y in SCOPES
           if x != y and s["matrix_discriminating"][x][y] is not None]
    worst = max(off) if off else 0.0
    print(f"\n  worst off-diagonal (discriminating subset): {worst:.3f}")
    if not s["n_discriminating"]:
        print("  VERDICT: UNDECIDABLE HERE -- no target has three distinct "
              "position sets, so this data cannot separate the scopes.")
    elif worst < 0.05:
        print("  VERDICT: LOW STAKES -- deploying any scope's optimum captures "
              ">95% of every other scope's reachable gain, so the scope choice "
              "does not change which sequence you ship.")
    elif worst < 0.25:
        print("  VERDICT: MATTERS MODERATELY -- a scope mismatch forgoes a "
              "noticeable slice of the true objective's gain.")
    else:
        print("  VERDICT: MATTERS -- the scopes are materially different "
              "objectives. Results graded under `all` do NOT transfer, and the "
              "scope must be decided before any further claim rests on it.")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(
            {"summary": s, "k_mut": k_mut, "metric": metric,
             "worst_off_diagonal_discriminating": worst,
             "skipped": [{"target_id": a, "why": b} for a, b in skipped],
             "targets": rows}, indent=2))
        print(f"\nwrote {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--pssm-dir", default="")
    ap.add_argument("--seed-windows", default="")
    ap.add_argument("--targets", default="")
    ap.add_argument("--metric", default="pll_bg")
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k-mut", type=int, default=2)
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    return report(a.manifest, a.pdb_dir, a.pssm_dir or None,
                  seed_windows=(a.seed_windows or None),
                  targets=[s.strip() for s in a.targets.split(",") if s.strip()],
                  metric=a.metric, k_mut=a.k_mut,
                  out_json=(a.out_json or None),
                  model=(a.model or None), device=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
