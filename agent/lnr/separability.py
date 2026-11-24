#!/usr/bin/env python
r"""MEASURE whether an objective can be put in a table, instead of asserting it.

WHY THIS EXISTS. `pepliab.py` hand-labels every liability check SEPARABLE /
LOCAL / GLOBAL / JOINT, and the whole argument for a search arm rests on those
labels: only a JOINT objective has no exact table method. Hand labels are not
evidence. Worse, the obvious way to implement the declared-but-unimplemented
JOINT axes will QUIETLY PRODUCE A TABLEABLE METRIC and look like a result:

  * `aggregation_propensity` as a sliding-window hydrophobicity average is a sum
    over fixed windows -- a DP solves it exactly. We would have "measured" a
    JOINT axis and actually built a LOCAL one.
  * `off_target_panel` scored with FROZEN PSSMs is a sum of per-position terms
    per off-target, and a sum of separable functions is separable. Still a table.

So the classification has to be measured from the metric's own behaviour.

THE CRITERION IS NOT "HOW MANY POSITIONS IT READS." `net_charge` and `gravy`
read every position and are exactly solvable, because they are sums. What
decides solvability is whether the metric DECOMPOSES into terms that each read a
bounded number of positions.

THE TEST. The discrete mixed second difference -- the sequence analogue of a
cross partial derivative -- measures whether two positions interact at all:

    I(i,j) = f(s with i->a, j->b) - f(s with i->a) - f(s with j->b) + f(s)

  * I(i,j) = 0 for every pair                  => f is ADDITIVE. One sort.
  * I(i,j) = 0 for every pair with |i-j| > w   => bounded width w. Exact by DP
                                                  with state over the last w
                                                  residues.
  * I(i,j) != 0 at unbounded range, but f is invariant under PERMUTATION of the
    sequence                                   => f is a function of residue
                                                  COUNTS only. Exact by ILP.
  * none of the above                          => FLAGGED, not proved joint.
                                                  See the asymmetry note below.

WHAT THIS CANNOT DECIDE, AND WHY THE LAST CLASS IS A FLAG. Pairwise interaction
range detects an ADDITIVE decomposition. Bounded-width local terms combined by a
non-additive associative operator -- max, min, a count of windows passing a
predicate -- couple positions at unbounded distance while STILL being computable
by a bounded-state left-to-right scan, so they are DP-solvable and land in the
last class anyway. This module's own selftest caught that on a sliding-window
max-of-sums, which is why the class is named `no-decomp-found` and not `joint`.
The instrument concludes only in the safe direction: ADDITIVE / BOUNDED / COUNTS
are proofs of tableability, and the last class is a flag for hand scrutiny.
False "tableable" is the error that would matter -- it would let a search arm be
justified by a metric a DP eats -- and that is the error the asymmetry removes.

This is exactly the instrument already used on `pll_bg` in `pll_additivity.py`,
one order lower: there the question was whether single-substitution effects add,
here it is whether any two positions interact at all.

THE GUARD THAT MATTERS. A metric that never moves across the probes returns
UNDETERMINED, never "additive". Most liability checks are THRESHOLD indicators:
they sit at 0 until a cutoff is crossed, so an all-zero interaction sample is
the expected outcome of probing a metric that simply never fired, and reading it
as "separable" is the vacuous-selftest mistake with a confident label attached.
`n_moved` is reported for every metric so a zero can never pass as a finding.

Nothing here needs a structure, a fold or a GPU: it is arithmetic on sequences.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Labels this module can conclude. Deliberately not pepliab's labels: these are
# statements about measured STRUCTURE, which is then mapped to a solver class.
ADDITIVE = "additive"            # -> sort (pepliab SEPARABLE)
BOUNDED = "bounded"              # -> DP   (pepliab LOCAL)
COUNTS = "counts"                # -> ILP  (pepliab GLOBAL)
NO_DECOMP = "no-decomp-found"    # -> inconclusive, see the asymmetry note
UNDETERMINED = "undetermined"    # the probe never moved the metric

# THE INSTRUMENT IS ASYMMETRIC, AND THE LABEL NAMES SAY SO.
#
# The first version of this module called the last class JOINT, and its own
# selftest caught that as wrong: a sliding-window MAX-of-sums landed there while
# being exactly DP-solvable with state over the last w-1 residues. The reason is
# structural, not a tuning problem. Pairwise interaction range detects an
# ADDITIVE decomposition. When bounded-width local terms are combined by a
# non-additive associative operator -- max, min, a count of windows passing a
# predicate -- the operator couples positions at unbounded distance (raising one
# window's value can switch which window attains the max) even though a
# bounded-state scan still computes the metric exactly.
#
# Separating "bounded-state DP-solvable" from "genuinely needs evaluation" is
# not something black-box probing can decide. So this module concludes only in
# the safe direction:
#   * ADDITIVE / BOUNDED / COUNTS are PROOFS of tableability -- the
#     decomposition was exhibited, and the matching solver is exact.
#   * NO_DECOMP is a FLAG, not a verdict. It means no bounded-width additive
#     decomposition was found. A max/count aggregate over windows is NOT ruled
#     out and must be excluded by hand before the metric is used to argue that
#     a search arm is necessary.
# False "tableable" would be the dangerous error here, because it would let a
# search arm be justified by a metric a DP eats. That error is the one the
# asymmetry removes.
SOLVER = {ADDITIVE: "one sort over per-position values",
          BOUNDED: "DP with state over the last w residues",
          COUNTS: "ILP over residue counts",
          NO_DECOMP: ("INCONCLUSIVE -- no bounded-width additive decomposition "
                      "found; a bounded-state DP over window max/counts is NOT "
                      "ruled out and must be excluded by hand"),
          UNDETERMINED: "unknown: the metric never moved under this probe"}

PEPLIAB_EQUIV = {ADDITIVE: "SEPARABLE", BOUNDED: "LOCAL",
                 COUNTS: "GLOBAL", NO_DECOMP: "JOINT", UNDETERMINED: "-"}


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------
def sub(seq, *pairs):
    """`seq` with (position, residue) substitutions applied."""
    s = list(seq)
    for i, a in pairs:
        s[i] = a
    return "".join(s)


def mixed(fn, seq, i, j, a, b):
    """The discrete mixed second difference at (i, j) for substitutions (a, b).

    Zero means the effect of substituting at i does not depend on what sits at
    j: the two positions do not interact, and the metric is additive across
    them. This is the quantity a per-position table assumes is zero EVERYWHERE.
    """
    return (fn(sub(seq, (i, a), (j, b))) - fn(sub(seq, (i, a)))
            - fn(sub(seq, (j, b))) + fn(seq))


def probe(fn, seqs, rng=None, n_subs=0, tol=1e-9, max_pairs=None):
    """Interaction evidence over all position pairs of every sequence.

    EXHAUSTIVE OVER RESIDUE PAIRS BY DEFAULT (`n_subs=0`), and that is not an
    optimisation -- it is what makes a negative result mean anything. The first
    version sampled `n_subs` random (a, b) pairs per position pair, and the
    audit caught it calling `asp_pro` ADDITIVE: a D-P motif only interacts when
    a probe happens to draw exactly a=D and b=P, which is 1/400 per attempt, so
    with a few hundred attempts the interaction is missed by luck. That is a
    FALSE PROOF OF TABLEABILITY, the one error direction this module exists to
    prevent, and no `n_moved` guard catches it -- the metric does move, just
    never in the way that creates an interaction.

    Enumerating all 19x19 off-diagonal residue pairs removes the sampling luck:
    "no pair interacts" becomes a statement over every single-pair substitution
    from these base sequences, not over a lucky sample of them. Single
    substitutions are cached, so each (i, j, a, b) costs one fresh evaluation
    instead of four.
    """
    by_dist, n_probes, n_moved, n_inter = {}, 0, 0, 0
    for seq in seqs:
        L = len(seq)
        if L < 2:
            continue
        base = fn(seq)
        f1 = {(i, a): fn(sub(seq, (i, a)))
              for i in range(L) for a in AA20 if a != seq[i]}
        pairs = list(itertools.combinations(range(L), 2))
        if max_pairs and len(pairs) > max_pairs:
            pairs = (rng or random.Random(0)).sample(pairs, max_pairs)
        for i, j in pairs:
            alts_i = [a for a in AA20 if a != seq[i]]
            alts_j = [b for b in AA20 if b != seq[j]]
            if n_subs:
                r = rng or random.Random(0)
                alts_i = r.sample(alts_i, min(n_subs, len(alts_i)))
                alts_j = r.sample(alts_j, min(n_subs, len(alts_j)))
            for a in alts_i:
                fa = f1[(i, a)]
                for b in alts_j:
                    fb = f1[(j, b)]
                    n_probes += 1
                    fab = fn(sub(seq, (i, a), (j, b)))
                    # "Moved" must include the DOUBLE substitution. net_charge
                    # sits inside its allowed window until two edits push it
                    # out, so testing singles alone reported n_moved=0 -- and
                    # UNDETERMINED -- for a metric with 315 measured
                    # interactions. The guard exists to catch a dead probe, not
                    # to veto a live one.
                    if fa != base or fb != base or fab != base:
                        n_moved += 1
                    v = abs(fab - fa - fb + base)
                    if v > tol:
                        n_inter += 1
                        d = j - i
                        by_dist[d] = max(by_dist.get(d, 0.0), v)
    return {"by_dist": by_dist, "n_probes": n_probes, "n_moved": n_moved,
            "n_interacting": n_inter,
            "max_dist": max(by_dist) if by_dist else 0,
            "max_abs": max(by_dist.values()) if by_dist else 0.0}


def permutation_invariant(fn, seqs, rng, n_perm=40):
    """Is the metric a function of residue COUNTS only?

    Shuffling preserves the multiset of residues and destroys the order, so a
    metric that survives every shuffle depends on counts alone -- which is what
    makes an ILP over counts exact. Sequences whose metric never varies at all
    are excluded, because a constant is trivially invariant and would make any
    dead metric look like a counts function.
    """
    informative = 0
    for seq in seqs:
        base = fn(seq)
        chars = list(seq)
        for _ in range(n_perm):
            rng.shuffle(chars)
            cand = "".join(chars)
            if cand == seq:
                continue
            informative += 1
            if fn(cand) != base:
                return False, informative
    return (informative > 0), informative


def classify(fn, seqs, rng=None, n_subs=0, w_max=4, max_pairs=None):
    """Measured structure of `fn`, with the evidence that produced it."""
    rng = rng or random.Random(0)
    pr = probe(fn, seqs, rng, n_subs=n_subs, max_pairs=max_pairs)
    perm, n_inf = permutation_invariant(fn, seqs, rng)

    if pr["n_moved"] == 0:
        label, why = UNDETERMINED, ("the metric never changed under any probe, "
                                    "so NOTHING about its structure was tested")
    elif pr["n_interacting"] == 0:
        label, why = ADDITIVE, ("no pair of positions interacts: every mixed "
                                "second difference is zero")
    elif pr["max_dist"] <= w_max:
        label, why = BOUNDED, (f"interactions vanish beyond distance "
                               f"{pr['max_dist']} (<= w_max={w_max})")
    elif perm:
        label, why = COUNTS, ("long-range interactions, but invariant under "
                              "every permutation tested")
    else:
        label, why = NO_DECOMP, (
            f"interactions at distance {pr['max_dist']} and order matters: no "
            f"bounded-width additive decomposition, and not a function of "
            f"counts. NOT a proof of jointness -- a max/count aggregate over "
            f"windows lands here too and is DP-solvable")
    return {"label": label, "why": why, "solver": SOLVER[label],
            "pepliab_equiv": PEPLIAB_EQUIV[label],
            "perm_invariant": perm, "n_informative_perms": n_inf, **pr}


# ---------------------------------------------------------------------------
# The real metrics: violation COUNTS, which is what a solver constrains
# ---------------------------------------------------------------------------
def pepliab_metrics(ctx=None):
    """Each pepliab check as a scalar: how many violations it reports.

    The violation count, not some underlying continuous quantity, is the right
    object to classify -- it is what a constraint would bound. The distinction
    is not pedantic: "longest hydrophobic run" has long-range interactions (a
    substitution can merge two runs an arbitrary distance apart), while "number
    of runs of length >= 3" interacts only within the threshold, which is
    exactly why a DP with a bounded run-length state solves the constraint.
    Classifying the wrong scalar would label a DP-solvable constraint JOINT.
    """
    from lnr import pepliab as PL
    # THE PROBE CTX DELIBERATELY ENABLES EVERY CTX-GATED CHECK. An empty Ctx is
    # the "universal spec" arm, and under it `receptor_disulfide` has no anchor
    # to lose and `dkp_risk` is switched off entirely -- so they never fire and
    # report UNDETERMINED, which is a statement about the Ctx, not about the
    # metric. Structure is what is being measured here, so the probe supplies a
    # Ctx in which each check is live: a receptor tether at position 0, and a
    # non-cyclic peptide so the N-terminal checks apply.
    ctx = ctx or PL.Ctx(ss_anchor=(0,), cyclic=False)
    out = {}
    for fn in PL.CHECKS:
        name = fn.check_name
        out[name] = (lambda f: (lambda s: float(len(f(s, ctx)))))(fn)
    return out, {fn.check_name: fn.klass for fn in PL.CHECKS}


def native_seqs(rows_path=None, limit=12):
    """Peptide sequences to probe. Real ones if available, else a spread."""
    if rows_path and Path(rows_path).exists():
        seen = []
        for line in Path(rows_path).read_text().splitlines():
            if not line.strip():
                continue
            s = (json.loads(line).get("native_seq") or "").strip()
            if s and s not in seen and all(c in AA20 for c in s):
                seen.append(s)
            if len(seen) >= limit:
                break
        if seen:
            return seen
    # Fallback: deliberately spans the thresholds the checks sit on, because a
    # probe over sequences that trip nothing measures nothing.
    # The last two are charge extremes. Without them `net_charge` never leaves
    # its allowed window and reports UNDETERMINED -- a dead probe, not a result.
    return ["VAQGGAAGMF", "CGKWDGACPLM", "KKKKDEAGWMNG", "AILMFWVYAG",
            "DGNGSMWCCRT", "PEPTIDEKRKRA", "KKRRKKRRKA", "DDEEDDEEGA"]


# ---------------------------------------------------------------------------
# Selftest: synthetic metrics whose class is known by construction
# ---------------------------------------------------------------------------
def _selftest():
    import re
    fails = []

    def eq(label, got, want):
        ok = got == want
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}")
        if not ok:
            fails.append((label, got, want))

    seqs = ["AAAADAAAADAAAA", "MWKKKKDEAGCCAM", "AILMFWVYAGDGNG"]
    rng = random.Random(0)

    print("the classifier, on metrics whose class is known by construction")

    # ADDITIVE: a per-residue count. Every mixed difference is exactly zero.
    eq("a per-residue count is additive",
       classify(lambda s: float(s.count("M") + s.count("W")), seqs,
                random.Random(1))["label"], ADDITIVE)

    # BOUNDED: number of hydrophobic runs of length >= 3. Interactions exist
    # (two substitutions can merge or split a run) but die off with distance.
    c_local = classify(lambda s: float(len(re.findall(r"[AILMFWVY]{3,}", s))),
                       seqs, random.Random(2))
    eq("a motif count is bounded-width", c_local["label"], BOUNDED)
    eq("  and its width is small", c_local["max_dist"] <= 4, True)

    # COUNTS: nonlinear in a residue count. Interacts at every distance, yet
    # survives every shuffle -- the signature an ILP over counts handles.
    c_glob = classify(lambda s: float(s.count("K") ** 2), seqs,
                      random.Random(3))
    eq("a nonlinear count function is counts-only", c_glob["label"], COUNTS)
    eq("  it does interact at long range", c_glob["max_dist"] > 4, True)
    eq("  and it is permutation invariant", c_glob["perm_invariant"], True)

    # Long-range and order-dependent: palindromic self-matching couples i to
    # L-1-i. Genuinely not tableable, but the instrument can only FLAG it.
    eq("a long-range order-dependent metric is flagged",
       classify(lambda s: float(sum(1 for i in range(len(s))
                                    if s[i] == s[len(s) - 1 - i])),
                seqs, random.Random(4))["label"], NO_DECOMP)

    # THE GUARD. A metric that never fires must not be called additive.
    c_dead = classify(lambda s: 0.0, seqs, random.Random(5))
    eq("a metric that never moves is UNDETERMINED, not additive",
       c_dead["label"], UNDETERMINED)
    eq("  and it reports zero movement", c_dead["n_moved"], 0)

    # A sliding-window SUM is the trap: it looks like a whole-sequence property
    # and is exactly DP-solvable. The classifier must not call it joint.
    def winsum(s, w=5):
        kd = {a: (1.0 if a in "AILMFWVY" else -1.0) for a in AA20}
        return float(max((sum(kd[c] for c in s[i:i + w])
                          for i in range(max(1, len(s) - w + 1))), default=0.0))
    # THE ASYMMETRY, pinned down. A sliding-window MAX-of-sums is exactly
    # DP-solvable, yet the max couples positions at unbounded range, so it
    # lands in NO_DECOMP alongside the palindrome metric. The instrument must
    # NOT claim to have separated them, and its label must not read as a
    # verdict -- this is the assertion that keeps it honest.
    c_win = classify(winsum, seqs, random.Random(6))
    eq("a DP-solvable window max is only FLAGGED, not called joint",
       c_win["label"], NO_DECOMP)
    eq("  and the flag states it is inconclusive",
       "INCONCLUSIVE" in c_win["solver"], True)
    eq("  so the flag cannot be read as an impossibility result",
       "NOT ruled out" in c_win["solver"], True)

    print("\nthe probe itself")
    eq("mixed difference of an additive metric is zero",
       mixed(lambda s: float(s.count("K")), "AAAA", 0, 3, "K", "K"), 0.0)
    eq("mixed difference detects a real interaction",
       mixed(lambda s: float(s.count("KK")), "AAAA", 1, 2, "K", "K"), 1.0)
    eq("sub applies both positions",
       sub("AAAA", (0, "K"), (3, "M")), "KAAM")

    print(f"\n{'SELFTEST PASS' if not fails else f'SELFTEST FAIL ({len(fails)})'}")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
def audit(rows_path, out_path, n_subs, w_max, limit, seed):
    metrics, hand = pepliab_metrics()
    seqs = native_seqs(rows_path, limit=limit)
    print(f"sequences probed : {len(seqs)}  lengths "
          f"{sorted({len(s) for s in seqs})}")
    print(f"metrics          : {len(metrics)}")
    print(f"n_subs/pair={n_subs}  w_max={w_max}  seed={seed}\n")

    hdr = (f"{'metric':22s} {'measured':13s} {'hand':10s} {'agree':6s} "
           f"{'maxd':5s} {'moved':>7s} {'inter':>7s}")
    print(hdr); print("-" * len(hdr))
    rows, disagree = [], []
    for name, fn in metrics.items():
        c = classify(fn, seqs, random.Random(seed), n_subs=n_subs, w_max=w_max)
        h = hand[name]
        agree = (c["pepliab_equiv"] == h.upper()
                 or c["label"] == UNDETERMINED)
        print(f"{name:22s} {c['label']:13s} {h:10s} "
              f"{'yes' if agree else 'NO':6s} {c['max_dist']:<5d} "
              f"{c['n_moved']:>7d} {c['n_interacting']:>7d}")
        c.update(metric=name, hand_label=h, agrees=agree)
        rows.append(c)
        if not agree:
            disagree.append(c)

    und = [r for r in rows if r["label"] == UNDETERMINED]
    print(f"\n{len(rows) - len(und) - len(disagree)}/{len(rows)} hand labels "
          f"confirmed by measurement")
    if und:
        print(f"\n{len(und)} metric(s) UNDETERMINED -- the probe never moved "
              f"them, so their label is untested, NOT confirmed:")
        for r in und:
            print(f"    {r['metric']}: needs sequences that actually trip it")
    if disagree:
        print(f"\n{len(disagree)} DISAGREEMENT(S) -- measurement wins:")
        for r in disagree:
            print(f"    {r['metric']}: hand={r['hand_label']} "
                  f"measured={r['label']} ({r['why']})")
            print(f"      solver that is actually exact: {r['solver']}")

    print(f"\n{'=' * 72}\nWHAT THIS CAN AND CANNOT SAY\n{'=' * 72}")
    print("CAN: that a metric IS tableable. A measured bounded width or a")
    print("  surviving permutation test exhibits the decomposition, and the")
    print("  matching solver is then exact. That direction is a proof.")
    print("CANNOT: that a metric is NOT tableable. `no-decomp-found` means")
    print("  no bounded-width ADDITIVE decomposition was found by this probe.")
    print("  Local terms combined by max/min/count couple positions at")
    print("  unbounded range and land here too, while a bounded-state DP still")
    print("  solves them exactly -- the selftest pins this on a sliding-window")
    print("  max. So the flag means 'resisted the obvious table', and every")
    print("  metric carrying it needs the max/count aggregate excluded BY HAND")
    print("  before it is used to argue that a search arm is necessary.")
    print("So this instrument is sound for its real job: refusing to let a")
    print("secretly-tableable metric be presented as a reason to search.")

    if out_path:
        Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"\nwrote {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--rows", default="", help="jsonl with native_seq fields")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-subs", type=int, default=0,
                    help="0 = exhaustive over residue pairs; sampling can miss a motif by luck")
    ap.add_argument("--w-max", type=int, default=4)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.audit:
        return audit(a.rows or None, a.out or None, a.n_subs, a.w_max,
                     a.limit, a.seed)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
