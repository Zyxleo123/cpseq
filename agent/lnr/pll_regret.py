"""REGRET analysis for the pose-conditioned-likelihood in-loop fitness.

Reads only saved artefacts (`seqmut_iter_pll_rows.jsonl`), so every number and
table here is redrawable with no recomputation and no model -- the LNR
convention. It exists because `--evaluator pll` is the first in-loop fitness in
this project whose OPTIMUM IS KNOWN, which makes a question askable that was not
askable before: not "which arm scored higher" but "how much of the reachable
improvement did each arm actually capture, and what predicts failure".

THE CONTROL THIS MODULE IS BUILT AROUND
---------------------------------------
Raw regret is in nats/residue and is NOT comparable across targets. `pll_bg` is a
MEAN over peptide positions, so a fixed-k edit is diluted by 1/L: on the 79-target
run, available gain falls with length at rho -0.681. Any raw regret-vs-length
correlation is therefore fighting a size confound -- the same trap that produced
this project's "interface dG measures size, not affinity" result. So the headline
quantity is NORMALISED regret, `regret / available_gain`: the fraction of the
reachable improvement the search failed to capture. Both are reported, next to the
confound itself, and every correlation gets a permutation null rather than a
parametric p -- here a plain shuffle, because one row per target really is
independent (unlike the within-series pairs that needed a restricted permutation,
[[pooled-pairs-need-a-permutation-null]]).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LENGTH_BINS = ((0, 8), (9, 11), (12, 14), (15, 19), (20, 10**6))
DISCRIMINATING_REGRET = 0.05

# Peptides that are PURIFICATION ARTIFACTS rather than binders. Optimising a
# pose-conditioned likelihood over an expression tag measures the tag: a
# low-complexity repeat has no binding signal to recover, so its regret is not
# evidence about an operator. Excluded from headline claims via
# --exclude-artifacts, and reported as a separate line rather than dropped
# silently, so the n always reconciles.
#
# Audited across all 85 selftest-indexable targets (the native sequences
# printed by job 119637): 5a29 is the ONLY one. It is in the 29-target
# discriminating subset, so it carries ~3.4% of the graded weight -- small, but
# it is also L=18, which puts it in the longest bin where the length effect is
# measured.
ARTIFACT_TARGETS = {
    "5a29": "His7 tag + thrombin cleavage site LVPRGS (HHHHHHHSSGLVPRGSHA)",
}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _ranks(v):
    """Average ranks, so ties (and regret has many exact 0.0 ties) do not get an
    arbitrary order that would inflate or deflate rho."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def pearson(a, b):
    if len(a) < 2:
        return 0.0
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def perm_p(a, b, n=20000, seed=0):
    """Two-sided permutation p for |rho|. Targets are independent rows, so the
    null is a plain shuffle of one side."""
    if len(a) < 3:
        return float("nan")
    obs = abs(spearman(a, b))
    rng = random.Random(seed)
    bb = list(b)
    hits = 0
    for _ in range(int(n)):
        rng.shuffle(bb)
        if abs(spearman(a, bb)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (int(n) + 1)


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------
def load(path, arm=None, exclude=None):
    """Graded rows from a seqmut-iter pll rows file: those carrying BOTH a regret
    and the optimum it was measured against.

    `exclude` drops target_ids outright (see ARTIFACT_TARGETS). It is applied
    here rather than in each table so that every statistic, correlation and
    paired comparison downstream sees the same n.
    """
    drop = set(exclude or ())
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = r.get("iter") or {}
        if r.get("mut_arm") == "nomut" or it.get("regret") is None:
            continue
        if not it.get("optimum"):
            continue
        if arm and r.get("mut_arm") != arm:
            continue
        if r.get("target_id") in drop:
            continue
        o = it["optimum"]
        gain = abs(float(o["fitness"]) - float(o["wt_fitness"]))
        out.append({
            "target_id": r.get("target_id"),
            "arm": r.get("mut_arm"),
            "length": int(r.get("length") or 0),
            "n_mutable": o.get("n_mutable") or int(r.get("length") or 0),
            "regret": float(it["regret"]),
            # fraction of the reachable improvement that was NOT captured
            "norm_regret": (float(it["regret"]) / gain) if gain > 1e-9 else 0.0,
            "gain": gain,
            "solved": 1.0 if float(it["regret"]) <= 0.0 else 0.0,
            # Regret is `best - optimum` and CANNOT be negative against a bar
            # the arm is confined to. A negative value means the arm reached
            # something the optimum excluded -- measured on 1t5z and 2hwl, where
            # the LLM edited an OFF-INTERFACE position while the optimum was
            # constrained to the interface to match the EA's action space
            # (`_point_mutate` draws from `_iface_positions`, `parse_many`
            # accepts any position). Flagged rather than averaged, because a
            # negative mean regret silently reported as an improvement is how
            # this went unnoticed for six targets.
            "regret_negative": float(it["regret"]) < -1e-9,
            "free_fitness": o.get("free_fitness"),
            "regret_vs_free": (
                None if o.get("free_fitness") is None
                or it.get("best_recovered") is None
                else float(it["best_recovered"]) - float(o["free_fitness"])),
            "n_evals": it.get("n_evals"),
            "n_gen_run": it.get("n_gen_run"),
            "evals_to_optimum": it.get("evals_to_optimum"),
            "action_space_binds": bool(o.get("free_minus_constrained")),
            # The MECHANISM columns. A paired regret difference says an arm did
            # better; these say whether an operator did it. `dup_frac` is
            # duplicates as a share of proposal SLOTS -- slots are
            # llm_calls * m_off, because proposals arrive m_off per CALL and the
            # re-ask fires most generations, so dividing by generations
            # overstates the denominator and understates the waste (the
            # "34-45%" in the notes was really 57.6%).
            "llm_calls": it.get("llm_calls") or 0,
            "n_dup_proposed": it.get("n_dup_proposed") or 0,
            "llm_cost_usd": float(it.get("llm_cost_usd") or 0.0),
            "fallback_gens": it.get("fallback_gens") or 0,
            "effort": it.get("effort"),
            "effort_sent": it.get("effort_sent"),
        })
    return out


def arms(path, exclude=None):
    return sorted({r["arm"] for r in load(path, exclude=exclude)})


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------
def by_length(rows, bins=LENGTH_BINS):
    out = []
    for lo, hi in bins:
        g = [r for r in rows if lo <= r["length"] <= hi]
        if not g:
            continue
        out.append({
            "bin": f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+",
            "n": len(g),
            "n_solved": int(sum(r["solved"] for r in g)),
            "mean_regret": st.mean(r["regret"] for r in g),
            "median_regret": st.median(r["regret"] for r in g),
            "mean_norm_regret": st.mean(r["norm_regret"] for r in g),
            "median_norm_regret": st.median(r["norm_regret"] for r in g),
            "mean_gain": st.mean(r["gain"] for r in g),
        })
    return out


def correlations(rows, n_perm=20000, seed=0):
    """Every correlation WITH its size control beside it, never alone."""
    L = [r["length"] for r in rows]
    pairs = [
        ("length vs regret (raw, nats)", [r["regret"] for r in rows]),
        ("length vs regret / available gain", [r["norm_regret"] for r in rows]),
        ("length vs available gain  (THE CONFOUND)", [r["gain"] for r in rows]),
        ("length vs solved (0/1)", [r["solved"] for r in rows]),
        ("n_mutable vs regret (raw)", None),
    ]
    out = []
    for name, y in pairs:
        x = L
        if y is None:                      # the n_mutable row uses a different x
            x = [r["n_mutable"] for r in rows]
            y = [r["regret"] for r in rows]
        out.append({"name": name, "rho": spearman(x, y),
                    "perm_p": perm_p(x, y, n=n_perm, seed=seed), "n": len(rows)})
    return out


def discriminating(rows, thresh=DISCRIMINATING_REGRET):
    """Targets where an arm left enough regret that a better operator could show
    a difference -- the subset worth spending API budget on."""
    hi = [r for r in rows if r["regret"] > thresh]
    return [r["target_id"] for r in sorted(hi, key=lambda r: -r["regret"])]


def budget_check(rows, max_evals=None):
    """Was the search limited by its EVALUATION budget, or by its generations?
    Answering this wrong turns an under-run into a claimed operator ceiling."""
    ev = [r["n_evals"] for r in rows if r["n_evals"] is not None]
    if not ev:
        return {}
    at_cap = (sum(1 for e in ev if max_evals and e >= max_evals)
              if max_evals else None)
    return {"mean_evals": st.mean(ev), "median_evals": st.median(ev),
            "max_evals_used": max(ev), "min_evals_used": min(ev),
            "n": len(ev), "cap": max_evals, "n_at_cap": at_cap}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(path, out_json=None, thresh=DISCRIMINATING_REGRET, n_perm=20000,
           max_evals=None, arm=None, exclude=None):
    rows = load(path, arm=arm, exclude=exclude)
    if not rows:
        print(f"GATE: no graded rows in {path} (need iter.regret + iter.optimum)")
        return 1
    summary = {"source": str(path), "n": len(rows),
               "arms": sorted({r["arm"] for r in rows}),
               "excluded": sorted(exclude or ())}
    print(f"PLL REGRET -- {path}")
    print(f"rows: {len(rows)}   arms: {', '.join(summary['arms'])}")
    # Printed, never silent: an excluded target changes every n below it, and a
    # reader comparing two reports needs to see which set each one is over.
    for t in summary["excluded"]:
        why = ARTIFACT_TARGETS.get(t, "excluded by request")
        print(f"  EXCLUDED {t}: {why}")

    for a in summary["arms"]:
        g = [r for r in rows if r["arm"] == a]
        ns = int(sum(r["solved"] for r in g))
        e2o = [r["evals_to_optimum"] for r in g if r["evals_to_optimum"]]
        print(f"\n=== {a} (n={len(g)}) ===")
        print(f"  solved {ns}/{len(g)} = {ns/len(g):.0%}   "
              f"mean regret {st.mean(r['regret'] for r in g):.4f}   "
              f"mean normalised regret "
              f"{st.mean(r['norm_regret'] for r in g):.3f}")
        if e2o:
            print(f"  evaluations to the optimum, where reached: "
                  f"mean {st.mean(e2o):.0f}  median {st.median(e2o):.0f}")
        bc = budget_check(g, max_evals)
        if bc:
            cap = (f"; {bc['n_at_cap']}/{bc['n']} reached the {bc['cap']} cap"
                   if bc.get("n_at_cap") is not None else "")
            print(f"  evaluations used: mean {bc['mean_evals']:.0f} "
                  f"(min {bc['min_evals_used']}, max {bc['max_evals_used']}){cap}")
            if bc.get("n_at_cap") == 0:
                print("  -> NOT budget-limited: no target reached the cap, so a "
                      "solve rate below 100% here is a GENERATION or operator "
                      "limit, and must not be reported as an operator ceiling "
                      "until the generations are raised.")
        nb = [r["target_id"] for r in g if r["action_space_binds"]]
        if nb:
            print(f"  action space binds on {len(nb)}/{len(g)}: "
                  f"{', '.join(nb)} -- regret there is a bound, not a failure")

        print(f"\n  by peptide length  (normRegret = fraction of the AVAILABLE "
              f"gain not captured)")
        print(f"  {'bin':>7s} {'n':>3s} {'solved':>9s} {'regret':>8s} "
              f"{'normRegret':>11s} {'medNorm':>8s} {'availGain':>10s}")
        for b in by_length(g):
            print(f"  {b['bin']:>7s} {b['n']:3d} "
                  f"{f'{b['n_solved']}/{b['n']}':>9s} {b['mean_regret']:8.4f} "
                  f"{b['mean_norm_regret']:11.3f} {b['median_norm_regret']:8.3f} "
                  f"{b['mean_gain']:10.4f}")

        print(f"\n  correlations, each with a {n_perm}-permutation null "
              f"(the size confound is listed, never omitted):")
        print(f"  {'':42s} {'rho':>7s} {'perm p':>9s}")
        cors = correlations(g, n_perm=n_perm)
        for c in cors:
            print(f"  {c['name']:42s} {c['rho']:+7.3f} {c['perm_p']:9.4f}")
        if len(cors) >= 3:
            raw, norm, conf = cors[0]["rho"], cors[1]["rho"], cors[2]["rho"]
            if abs(norm) > abs(raw) and conf < -0.2:
                print("  -> available gain SHRINKS with length, which SUPPRESSES "
                      "the raw correlation; the normalised one is the real "
                      "effect size, not the raw one.")

        disc = discriminating(g, thresh)
        print(f"\n  discriminating subset (regret > {thresh}): "
              f"{len(disc)}/{len(g)} = {len(disc)/len(g):.0%} of targets")
        if disc:
            print("  " + ",".join(disc))
        summary[a] = {
            "n": len(g), "n_solved": ns, "solved_frac": ns / len(g),
            "mean_regret": st.mean(r["regret"] for r in g),
            "mean_norm_regret": st.mean(r["norm_regret"] for r in g),
            "by_length": by_length(g), "correlations": cors,
            "budget": bc, "action_space_binds": nb,
            "discriminating": disc, "discriminating_thresh": thresh,
            "worst": sorted(g, key=lambda r: -r["regret"])[:12],
        }

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {out_json}")
    return 0


def compare(path_a, path_b, label_a="A", label_b="B", arm=None, exclude=None,
            m_off=12, max_evals=0):
    """PAIRED comparison of two runs over their SHARED targets.

    Written for one question: is a solve rate below 100% a BUDGET limit, a
    GENERATION limit, or the operator's ceiling? Those are not distinguishable
    from one run, and calling an under-run an operator ceiling is the easiest
    wrong conclusion available here. Pairing per target is what makes the
    comparison readable at n=79 -- an unpaired mean difference would be swamped
    by the between-target spread in available gain.

    Reports a sign test over the targets whose regret MOVED, since the ties are
    uninformative: a run where 74 of 79 targets are bit-identical and 5 improved
    has not "improved by 0.012 nats on average", it has moved 5 targets.
    """
    A = {r["target_id"]: r for r in load(path_a, arm=arm, exclude=exclude)}
    B = {r["target_id"]: r for r in load(path_b, arm=arm, exclude=exclude)}
    shared = sorted(set(A) & set(B))
    if not shared:
        print(f"GATE: no shared targets between {path_a} and {path_b}")
        return None
    for lab, d in ((label_a, A), (label_b, B)):
        neg = [k for k in shared if d[k].get("regret_negative")]
        if neg:
            print(f"\n  !! {lab}: NEGATIVE regret on {len(neg)} target(s) "
                  f"({', '.join(neg)}).\n"
                  f"     That arm reached a genome the optimum EXCLUDED, so this "
                  f"bar is not valid for it:\n"
                  f"     the optimum is constrained to the interface to match "
                  f"the EA's action space, while\n"
                  f"     the LLM arm may edit any position. Use regret_vs_free "
                  f"for a cross-arm claim.")
    better = [k for k in shared if B[k]["regret"] < A[k]["regret"] - 1e-12]
    worse = [k for k in shared if B[k]["regret"] > A[k]["regret"] + 1e-12]
    moved = len(better) + len(worse)
    # one-sided sign test on the targets that moved (ties carry no information)
    p = float("nan")
    if moved:
        p = sum(math.comb(moved, k) for k in range(len(better), moved + 1)) \
            / (2 ** moved)
    out = {"n_shared": len(shared), "n_better": len(better),
           "n_worse": len(worse), "n_unchanged": len(shared) - moved,
           "sign_p": p, "better": better, "worse": worse}
    print(f"\nPAIRED: {label_b} vs {label_a} over {len(shared)} shared target(s)")
    for lab, d in ((label_a, A), (label_b, B)):
        g = [d[k] for k in shared]
        ns = int(sum(r["solved"] for r in g))
        ev = [r["n_evals"] for r in g if r["n_evals"] is not None]
        slots = sum(r["llm_calls"] for r in g) * m_off
        dup = sum(r["n_dup_proposed"] for r in g)
        out[lab] = {"solved": ns, "solved_frac": ns / len(g),
                    "mean_regret": st.mean(r["regret"] for r in g),
                    "mean_norm_regret": st.mean(r["norm_regret"] for r in g),
                    "mean_evals": st.mean(ev) if ev else None,
                    "mean_calls": st.mean(r["llm_calls"] for r in g),
                    "dup_slots": slots, "n_dup": dup,
                    "dup_frac": (dup / slots) if slots else None,
                    "total_cost_usd": round(sum(r["llm_cost_usd"] for r in g), 4),
                    "n_at_eval_cap": sum(1 for r in g
                                         if max_evals and r["n_evals"]
                                         and r["n_evals"] >= max_evals)}
        print(f"  {lab:24s} solved {ns}/{len(g)} = {ns/len(g):4.0%}   "
              f"mean regret {out[lab]['mean_regret']:.4f}   "
              f"mean normalised {out[lab]['mean_norm_regret']:.3f}   "
              f"mean evals {out[lab]['mean_evals'] or 0:.0f}")
        if slots:
            print(f"  {'':24s} duplicates {dup}/{slots} slots = "
                  f"{100 * dup / slots:.1f}%   mean calls "
                  f"{out[lab]['mean_calls']:.0f}   "
                  f"${out[lab]['total_cost_usd']:.2f}")
        if out[lab]["n_at_eval_cap"]:
            print(f"  {'':24s} !! {out[lab]['n_at_eval_cap']}/{len(g)} target(s) "
                  f"REACHED the {max_evals}-eval cap -- for those the budget "
                  f"bound, not the operator")
    print(f"  moved: {len(better)} better, {len(worse)} worse, "
          f"{out['n_unchanged']} bit-identical"
          + (f"   sign test p({label_b} better than {label_a}) = {p:.4f}"
             if moved else ""))
    if moved and len(worse) > len(better):
        # the one-sided test was asked in the losing direction; say so rather
        # than leave a p near 1.0 to be read as "no difference"
        rev = sum(math.comb(moved, k) for k in range(len(worse), moved + 1)) \
            / (2 ** moved)
        print(f"  (that p is one-sided in the stated direction and {label_b} is "
              f"the WORSE run here; p({label_a} better than {label_b}) = "
              f"{rev:.4f})")
    if better:
        print(f"  better on: {', '.join(better)}")
    if out["n_unchanged"] / len(shared) > 0.8 and len(better) < 0.2 * len(shared):
        print("  -> the second run is NOT finding materially more: most targets "
              "are bit-identical. If its evaluation cap was also never reached, "
              "this is the OPERATOR's ceiling, not a budget or generation limit.")
    return out


def compare_arms(path, arm_a="iter-ea", arm_b="iter-llm", exclude=None,
                 label_a=None, label_b=None):
    """PAIRED comparison of two ARMS inside ONE rows file.

    `compare()` pairs two FILES on the same arm, which answers "did more
    generations help?". This answers the different question the LLM arm was run
    for: at a matched budget and on the same targets, does the LLM operator
    capture more of the reachable gain than the random EA?

    Pairing is essential rather than cosmetic here. Between-target spread in
    available gain is large (rho(length, gain) = -0.688), so an unpaired
    difference of means at n~26 is swamped by which targets happen to be in the
    set -- both arms see the SAME targets and the same seed pool, so the
    per-target difference removes that variance entirely.

    Ties are excluded from the sign test, not counted as agreement: on an
    additive objective both arms frequently find the identical optimum, and
    calling that "the operator made no difference" confuses a saturated target
    with a contested one. The count of ties is reported separately because it is
    the actual finding when it is large.
    """
    label_a = label_a or arm_a
    label_b = label_b or arm_b
    A = {r["target_id"]: r for r in load(path, arm=arm_a, exclude=exclude)}
    B = {r["target_id"]: r for r in load(path, arm=arm_b, exclude=exclude)}
    shared = sorted(set(A) & set(B))
    if not shared:
        print(f"GATE: no targets carry BOTH {arm_a} and {arm_b} in {path}")
        return None
    better = [k for k in shared if B[k]["regret"] < A[k]["regret"] - 1e-12]
    worse = [k for k in shared if B[k]["regret"] > A[k]["regret"] + 1e-12]
    moved = len(better) + len(worse)
    p = float("nan")
    if moved:
        p = sum(math.comb(moved, k) for k in range(len(better), moved + 1)) \
            / (2 ** moved)
    diffs = [B[k]["regret"] - A[k]["regret"] for k in shared]
    ndiffs = [B[k]["norm_regret"] - A[k]["norm_regret"] for k in shared]
    out = {"source": str(path), "arm_a": arm_a, "arm_b": arm_b,
           "n_shared": len(shared), "n_better": len(better),
           "n_worse": len(worse), "n_tied": len(shared) - moved,
           "sign_p": p, "better": better, "worse": worse,
           "excluded": sorted(exclude or ()),
           "mean_regret_diff": round(st.mean(diffs), 6),
           "mean_norm_regret_diff": round(st.mean(ndiffs), 6)}
    print(f"\nPAIRED ARMS: {label_b} vs {label_a} over {len(shared)} target(s) "
          f"in one file")
    for t in sorted(exclude or ()):
        print(f"  EXCLUDED {t}: {ARTIFACT_TARGETS.get(t, 'excluded by request')}")
    for lab, d in ((label_a, A), (label_b, B)):
        g = [d[k] for k in shared]
        ns = int(sum(r["solved"] for r in g))
        ev = [r["n_evals"] for r in g if r["n_evals"] is not None]
        out[lab] = {"solved": ns, "solved_frac": ns / len(g),
                    "mean_regret": st.mean(r["regret"] for r in g),
                    "mean_norm_regret": st.mean(r["norm_regret"] for r in g),
                    "mean_evals": st.mean(ev) if ev else None}
        print(f"  {lab:24s} solved {ns}/{len(g)} = {ns/len(g):4.0%}   "
              f"mean regret {out[lab]['mean_regret']:.4f}   "
              f"mean normalised {out[lab]['mean_norm_regret']:.3f}   "
              f"mean evals {out[lab]['mean_evals'] or 0:.0f}")
    print(f"  paired mean difference: {out['mean_regret_diff']:+.4f} raw, "
          f"{out['mean_norm_regret_diff']:+.4f} normalised "
          f"(negative = {label_b} captures more)")
    print(f"  moved: {len(better)} better, {len(worse)} worse, "
          f"{out['n_tied']} tied"
          + (f"   sign test p({label_b} better) = {p:.4f}" if moved else ""))
    if moved and len(worse) > len(better):
        rev = sum(math.comb(moved, k) for k in range(len(worse), moved + 1)) \
            / (2 ** moved)
        print(f"  (one-sided in the stated direction, and {label_b} is the WORSE "
              f"arm here; p({label_a} better) = {rev:.4f})")
    if better:
        print(f"  {label_b} better on: {', '.join(better)}")
    if worse:
        print(f"  {label_b} worse on : {', '.join(worse)}")
    # The interpretation that is easiest to get wrong, stated explicitly.
    if moved and 0.05 < p < 0.95:
        print(f"  -> NOT separated at n={len(shared)}: the operators are within "
              f"noise of each other on this set. That is a statement about "
              f"POWER, not a demonstration that they are equivalent.")
    elif not moved:
        print("  -> every target is a tie: both operators found the identical "
              "subs everywhere. On an additive objective this is the objective "
              "being saturated, not the operators being equal in general.")
    return out


def novelty(traj_path, exclude=None) -> dict:
    """PROPOSAL NOVELTY per arm, read from the trajectory.

    READ `dup_frac` PER ARM, NEVER ACROSS ARMS. The arms are not measured at
    the same point and the asymmetry is structural:

      * `llm_offspring` DROPS proposals that repeat a known genome before it
        returns, so an LLM arm's returned children are novel by construction --
        `dup_frac` is ~0 whatever the model proposed. It was ~0.7% before the
        complete-history fix and ~0.6% after, which says nothing about the fix.
        An LLM arm pays for its duplicates in RE-ASKS (extra calls, extra
        dollars) and in generations that return fewer than `m_off` children;
        that cost lives in `iter.n_dup_proposed` and `iter.llm_calls`.
      * `ea_offspring` gives up after 8 colliding tries and returns the
        duplicate anyway, so its waste DOES land here.

    The quantity that IS comparable is `novel_per_gen`: novel children actually
    delivered per generation. It is what decides how much of an evaluation
    budget an operator can spend, and it is measured identically for every arm.

    Seed generations are skipped: the seed pool is shared by construction, so
    counting it would dilute every arm by the same amount and flatter whichever
    arm ran fewest generations.
    """
    drop = set(exclude or ())
    agg: dict = {}
    for line in Path(traj_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("phase") == "seed" or d.get("n_proposed") is None:
            continue
        if d.get("target_id") in drop:
            continue
        a = agg.setdefault(d.get("arm"), {"proposed": 0, "novel": 0,
                                          "gens": 0, "targets": set()})
        a["proposed"] += int(d["n_proposed"])
        a["novel"] += int(d.get("n_new") or 0)
        a["gens"] += 1
        a["targets"].add(d.get("target_id"))
    out = {}
    for arm, a in sorted(agg.items()):
        p = a["proposed"]
        out[arm] = {"proposed": p, "novel": a["novel"], "gens": a["gens"],
                    "dup_frac": ((p - a["novel"]) / p) if p else None,
                    # the cross-arm number: novel children delivered per
                    # generation, measured identically for every operator
                    "novel_per_gen": (a["novel"] / a["gens"]) if a["gens"] else None,
                    "n_targets": len(a["targets"])}
    return out


def print_novelty(traj_path, exclude=None):
    nov = novelty(traj_path, exclude=exclude)
    if not nov:
        print(f"\nNOVELTY: no usable generations in {traj_path}")
        return nov
    print(f"\nPROPOSAL NOVELTY (from the trajectory)")
    print(f"  {'arm':14s} {'targets':>7s} {'gens':>6s} {'proposed':>9s} "
          f"{'novel':>8s} {'dup%':>7s} {'novel/gen':>10s}")
    for arm, v in nov.items():
        print(f"  {arm:14s} {v['n_targets']:7d} {v['gens']:6d} "
              f"{v['proposed']:9d} {v['novel']:8d} "
              f"{100 * v['dup_frac']:6.1f}% {v['novel_per_gen']:10.1f}")
    print("  novel/gen IS comparable across arms -- it is what decides how much "
          "of an\n  evaluation budget an operator can spend.")
    print("  dup% IS NOT comparable across arms: llm_offspring drops repeats "
          "BEFORE\n  returning, so an LLM arm reads ~0% whatever it proposed "
          "(its duplicate cost\n  is re-asks, in iter.n_dup_proposed and "
          "llm_calls), while ea_offspring\n  returns the duplicate and is "
          "charged for it here.")
    return nov


def regret_at_budget(traj_path, budget, arm=None, exclude=None) -> dict:
    """Best regret each target had reached by its FIRST `budget` evaluations.

    The analysis that separates "the operator proposes better" from "the
    operator spends more of the budget" -- two things the headline regret cannot
    tell apart, and which came apart badly in job 120668/120763: the fixed LLM
    arm reaches the 900-eval cap where the same arm previously stopped near 516,
    so part of its lower regret is simply more evaluations. Reading both runs at
    the SAME eval count removes that, for free, from trajectories already on
    disk.

    Each trajectory row carries the cumulative `evals` and the regret of the
    best genome found so far, so the value at a budget is the last row with
    `evals <= budget`. Targets that never reached the budget are reported
    separately rather than silently given their final value -- crediting a
    target with a regret it only achieved later is exactly the error this
    function exists to prevent.
    """
    drop = set(exclude or ())
    per: dict = {}
    for line in Path(traj_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if arm and d.get("arm") != arm:
            continue
        if d.get("target_id") in drop:
            continue
        ev, rg = d.get("evals"), d.get("regret")
        if ev is None or rg is None:
            continue
        k = (d.get("target_id"), d.get("arm"))
        cur = per.setdefault(k, {"regret": None, "evals": 0, "max_evals": 0})
        cur["max_evals"] = max(cur["max_evals"], int(ev))
        if int(ev) <= budget and (cur["regret"] is None
                                 or float(rg) < cur["regret"]):
            cur["regret"] = float(rg)
            cur["evals"] = int(ev)
    reached = {k: v for k, v in per.items()
               if v["regret"] is not None and v["max_evals"] >= budget}
    short = {k: v for k, v in per.items() if k not in reached}
    return {"budget": budget, "reached": reached, "short_of_budget": short}


def compare_at_budget(traj_a, traj_b, budget, arm="iter-llm",
                      label_a="A", label_b="B", exclude=None):
    """Paired regret of one arm in two runs, at a MATCHED evaluation budget."""
    A = regret_at_budget(traj_a, budget, arm=arm, exclude=exclude)
    B = regret_at_budget(traj_b, budget, arm=arm, exclude=exclude)
    a = {t: v["regret"] for (t, _), v in A["reached"].items()}
    b = {t: v["regret"] for (t, _), v in B["reached"].items()}
    shared = sorted(set(a) & set(b))
    print(f"\nMATCHED-BUDGET regret for {arm} at the first {budget} "
          f"evaluation(s), over {len(shared)} shared target(s)")
    if A["short_of_budget"] or B["short_of_budget"]:
        print(f"  (excluded for never reaching {budget} evals: "
              f"{len(A['short_of_budget'])} in {label_a}, "
              f"{len(B['short_of_budget'])} in {label_b})")
    if not shared:
        print("  GATE: no shared targets reached this budget in both runs.")
        return None
    better = [t for t in shared if b[t] < a[t] - 1e-12]
    worse = [t for t in shared if b[t] > a[t] + 1e-12]
    moved = len(better) + len(worse)
    pv = (sum(math.comb(moved, k) for k in range(len(better), moved + 1))
          / (2 ** moved)) if moved else float("nan")
    print(f"  {label_a:24s} mean regret {st.mean(a[t] for t in shared):.4f}   "
          f"solved {sum(1 for t in shared if a[t] <= 0)}/{len(shared)}")
    print(f"  {label_b:24s} mean regret {st.mean(b[t] for t in shared):.4f}   "
          f"solved {sum(1 for t in shared if b[t] <= 0)}/{len(shared)}")
    print(f"  moved: {len(better)} better, {len(worse)} worse, "
          f"{len(shared) - moved} tied"
          + (f"   sign test p = {pv:.4f}" if moved else ""))
    print(f"  -> this is PROPOSAL QUALITY at equal spend. The headline regret "
          f"also includes\n     the extra evaluations the fixed arm is able to "
          f"use, which is a real gain\n     but a different one.")
    return {"budget": budget, "n_shared": len(shared), "n_better": len(better),
            "n_worse": len(worse), "sign_p": pv,
            f"mean_{label_a}": st.mean(a[t] for t in shared),
            f"mean_{label_b}": st.mean(b[t] for t in shared)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="seqmut_iter_pll_rows.jsonl")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--arm", default="", help="restrict to one arm")
    ap.add_argument("--thresh", type=float, default=DISCRIMINATING_REGRET)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--max-evals", type=int, default=0,
                    help="the run's --max-evals, so the report can say whether "
                         "the budget or the generations were the limit")
    ap.add_argument("--compare", default="",
                    help="a second rows file; adds a PAIRED comparison over the "
                         "shared targets (use to separate an under-run from the "
                         "operator's ceiling)")
    ap.add_argument("--label", default="run A")
    ap.add_argument("--compare-label", default="run B")
    ap.add_argument("--exclude", default="",
                    help="comma-separated target_ids to drop from every table, "
                         "correlation and paired comparison")
    ap.add_argument("--exclude-artifacts", action="store_true",
                    help=f"also drop the curated purification-tag targets "
                         f"({', '.join(sorted(ARTIFACT_TARGETS))}): peptides "
                         f"that are expression tags, where there is no binding "
                         f"signal to recover")
    ap.add_argument("--traj", default="",
                    help="seqmut_iter_pll_traj.jsonl; adds the per-arm PROPOSAL "
                         "NOVELTY table, the only duplicate metric that is fair "
                         "to the EA arms (rows record dup only for LLM arms)")
    ap.add_argument("--compare-traj", default="",
                    help="a second trajectory file; with --at-budget, compares "
                         "one arm's regret in both runs at a MATCHED eval "
                         "budget, separating proposal quality from budget use")
    ap.add_argument("--at-budget", type=int, default=0,
                    help="the matched evaluation budget to read regret at")
    ap.add_argument("--compare-arms", default="",
                    help="two arms in THIS file to pair, as 'armA:armB' (e.g. "
                         "iter-ea:iter-llm). The matched-budget operator "
                         "comparison the LLM arm is run for.")
    a = ap.parse_args()
    exclude = {t.strip() for t in a.exclude.split(",") if t.strip()}
    if a.exclude_artifacts:
        exclude |= set(ARTIFACT_TARGETS)
    rc = report(a.rows, out_json=(a.out_json or None), thresh=a.thresh,
                n_perm=a.n_perm, max_evals=(a.max_evals or None),
                arm=(a.arm or None), exclude=exclude)
    if a.traj:
        print_novelty(a.traj, exclude=exclude)
    if a.traj and a.compare_traj and a.at_budget:
        compare_at_budget(a.compare_traj, a.traj, a.at_budget,
                          arm=(a.arm or "iter-llm"),
                          label_a=a.compare_label, label_b=a.label,
                          exclude=exclude)
    if a.compare:
        compare(a.rows, a.compare, label_a=a.label, label_b=a.compare_label,
                arm=(a.arm or None), exclude=exclude,
                max_evals=(a.max_evals or 0))
    if a.compare_arms:
        if ":" not in a.compare_arms:
            raise SystemExit("--compare-arms wants 'armA:armB'")
        aa, bb = a.compare_arms.split(":", 1)
        compare_arms(a.rows, arm_a=aa.strip(), arm_b=bb.strip(),
                     exclude=exclude)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
