r"""Does the pll table stay additive when the backbone is allowed to respond?

THIS TESTS THE BAR, NOT AN ARM. Every negative result about search in this
project rests on one property: `TargetPSSM.score` is a mean of independent
per-position terms, so `best_subs` is the EXACT optimum and any arm that misses
it is failing at a solved problem. That property is bought by FREEZING THE POSE.
The matrix is built once from one structure and never moves, so by construction
a substitution at position i cannot change the term at position j.

Real peptides do not work that way. A mutation changes the structure, and a
changed structure changes every position's marginal at once. If that coupling is
large enough, the frozen-pose table is a FIRST-ORDER APPROXIMATION and the
"exact optimum" is not the optimum -- at which point the bar that has been
killing every search arm stops existing, and the whole negative result reopens.

So this is the highest-value measurement available, and it is cheap: it runs on
structures ALREADY ON DISK from the earlier Boltz work. No GPU.

THE DOUBLE DIFFERENCE, AND WHY A RAW COMPARISON WOULD BE MEANINGLESS. An
own-pose PSSM is built from a BOLTZ-PREDICTED complex; the frozen table is built
from the CRYSTAL structure. Those two differ for reasons that have nothing to do
with the mutation -- predicted and deposited poses simply score differently. So
the quantity reported is

    gap = [pll_own(mut) - pll_own(nat)] - [pll_frozen(mut) - pll_frozen(nat)]
          \_________ what actually happens _______/   \__ what the table says __/

where `pll_own(nat)` is the NATIVE scored under its OWN re-predicted pose. The
crystal-vs-predicted offset cancels. A gap of zero means the table predicted the
change correctly; a gap that grows with k means the per-position terms are
coupled and additivity is an approximation.

THE NOISE FLOOR COMES FREE. Each Boltz call emitted `replicas` diffusion samples
from one trunk pass, and every one is a different structure for the SAME
sequence. Building a PSSM from each gives the spread of own-pose pll at fixed
sequence -- the floor any gap must clear to mean anything. This project has been
burned by exactly this omission before (the +-3 REU dG floor that turned out to
be fold sampling), so the verdict below refuses to call a gap real until it
exceeds the floor.

WHAT THIS PILOT CANNOT SAY. The cached folds were produced with a HELD template:
the backbone outside the released window `[a, b]` is pinned to the native
conformation, so only the window can move. Coupling through the held region is
suppressed by construction. Any non-additivity measured here is therefore a
LOWER BOUND, and a null result does NOT license "additivity holds" -- it licenses
"additivity holds when the backbone is mostly pinned", which is a much weaker
and much less interesting claim. Substitutions inside the window are flagged
`in_window` so the two regimes are never pooled.

Run:
  python agent/lnr/pll_additivity.py --selftest
  python agent/lnr/pll_additivity.py --measure --boltz-root .../lnr/boltz_val \
      --pssm-dir .../lnr/pssm --out pll_additivity.jsonl
  python agent/lnr/pll_additivity.py --report --rows pll_additivity.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Boltz writes the complex with the binder as chain B and the receptor as A --
# the convention boltz_inloop.fold_score_one scores with
# (`binder_chain="B", receptor_chains=["A"]`). Hard-coding it here would be a
# silent trap if that ever changed, so it is named once.
BINDER_CHAIN = "B"
RECEPTOR_CHAINS = ["A"]

# held_{target}_{a}_{b}_{hash}_model_{n}.pdb
STEM_RE = re.compile(
    r"^held_(?P<tid>[0-9a-zA-Z]{4})_(?P<a>\d+)_(?P<b>\d+)_(?P<hash>[0-9a-f]+)"
    r"(?:_d(?P<draw>\d+))?_model_(?P<model>\d+)\.pdb$")


def chain_seq(pdb_path, chain):
    """The one-letter sequence of `chain`, in file order, from a PDB.

    Read from the STRUCTURE rather than from a cache keyed by a hash: the
    predicted file is the authority on what was actually folded, and the
    variant hash is not invertible."""
    from lnr.env import _THREE_TO_ONE as T
    seen, out = set(), []
    for line in Path(pdb_path).read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
            continue
        if line[21] != chain:
            continue
        key = line[22:27]
        if key in seen:
            continue
        seen.add(key)
        out.append(T.get(line[17:20].strip(), "X"))
    return "".join(out)


def discover(boltz_root):
    """Group every surviving predicted PDB by (target, window, variant hash).

    Returns {(tid, a, b, hash, draw): [pdb, ...]} -- the diffusion samples of
    one sequence under one trunk pass, which is exactly the unit whose spread
    is the noise floor."""
    groups = {}
    for pdb in Path(boltz_root).rglob("*_model_*.pdb"):
        m = STEM_RE.match(pdb.name)
        if not m:
            continue
        key = (m.group("tid"), int(m.group("a")), int(m.group("b")),
               m.group("hash"), int(m.group("draw") or 0))
        groups.setdefault(key, []).append(pdb)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def subs_of(native, mutant):
    if len(native) != len(mutant):
        return None
    return [(i, b) for i, (a, b) in enumerate(zip(native, mutant)) if a != b]


def measure(boltz_root, pssm_dir, out_path, device="cpu", model_name=None,
            scope="all", metric=None, verbose=True):
    """Own-pose pll for every cached variant, against the frozen-table prediction."""
    from lnr import bindscore as bs

    model_name = model_name or bs.PRIMARY_MODEL
    interface_only = (scope == "iface")
    groups = discover(boltz_root)
    if verbose:
        print(f"cached variant groups: {len(groups)}  "
              f"pdbs: {sum(len(v) for v in groups.values())}")

    # One MPNN scorer reused for every structure: building it per PDB would
    # dominate the runtime and, worse, a per-call scorer is a chance for two
    # variants to be scored by different weights.
    scorer = bs.MPNNScorer(model_name=model_name, device=device)

    frozen, raw = {}, []
    for (tid, a, b, h, draw), pdbs in groups.items():
        fz = frozen.get(tid)
        if fz is None:
            hits = sorted(Path(pssm_dir).glob(f"pssm_{tid}_*_{model_name}.json"))
            if not hits:
                if verbose:
                    print(f"SKIP {tid}: no frozen PSSM in {pssm_dir}")
                frozen[tid] = False
                continue
            fz = frozen[tid] = bs.TargetPSSM.load(hits[0])
        if fz is False:
            continue

        per_pdb = []
        for pdb in pdbs:
            try:
                seq = chain_seq(pdb, BINDER_CHAIN)
                if len(seq) != fz.n or "X" in seq:
                    per_pdb.append({"pdb": str(pdb), "error":
                                    f"peptide length {len(seq)} != {fz.n} "
                                    f"or non-standard residue"})
                    continue
                own = bs.TargetPSSM.from_complex(
                    pdb, pep_chain=BINDER_CHAIN, rec_chains=RECEPTOR_CHAINS,
                    model_name=model_name, device=device, scorer=scorer)
                per_pdb.append({
                    "pdb": str(pdb), "seq": seq,
                    "pll_own": own.score(seq, metric=metric,
                                         interface_only=interface_only)})
            except Exception as exc:                      # noqa: BLE001
                per_pdb.append({"pdb": str(pdb),
                                "error": f"{type(exc).__name__}: {exc}"[:300]})
        ok = [r for r in per_pdb if "pll_own" in r]
        seqs = {r["seq"] for r in ok}
        row = {"target_id": tid, "window": [a, b], "hash": h, "draw": draw,
               "n_pdb": len(pdbs), "n_ok": len(ok),
               "native_seq": fz.pep_seq, "frozen_template": fz.template,
               "per_pdb": per_pdb}
        if len(seqs) == 1:
            seq = seqs.pop()
            row["seq"] = seq
            row["is_native"] = (seq == fz.pep_seq)
            sb = subs_of(fz.pep_seq, seq)
            row["subs"] = [[i, aa] for i, aa in sb] if sb is not None else None
            row["k"] = len(sb) if sb is not None else None
            row["in_window"] = (None if sb is None else
                                all(a <= i <= b for i, _ in sb))
            row["pll_frozen"] = fz.score(seq, metric=metric,
                                         interface_only=interface_only)
            vals = [r["pll_own"] for r in ok]
            row["pll_own_median"] = st.median(vals) if vals else None
            # THE NOISE FLOOR: spread across diffusion samples of ONE sequence.
            row["pll_own_sd"] = st.stdev(vals) if len(vals) > 1 else None
        elif len(seqs) > 1:
            # Different sequences under one hash means the stem is not the
            # identity we think it is. Say so; do not average across them.
            row["error"] = f"group holds {len(seqs)} distinct sequences"
        raw.append(row)
        if verbose:
            tag = ("native" if row.get("is_native") else
                   f"k={row.get('k')}" + ("" if row.get("in_window") else "*"))
            sd = row.get("pll_own_sd")
            print(f"{tid} [{a},{b}] d{draw} {tag:<8s} n_ok={len(ok)}/{len(pdbs)}"
                  f"  own={row.get('pll_own_median')}"
                  f"  sd={'n/a' if sd is None else f'{sd:+.4f}'}"
                  f"  frozen={row.get('pll_frozen')}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in raw:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(raw)} rows -> {out}")
    report(raw)
    return raw


def _own_folds(row):
    """The per-fold own-pose pll values behind one group's median."""
    return [r["pll_own"] for r in (row.get("per_pdb") or []) if "pll_own" in r]


def sign_test(xs):
    """Exact two-sided binomial test that the signs of `xs` are 50/50.

    The SIGN is the statistic to trust here, not the magnitude. `pll_bg` is a
    mean over peptide positions, so a fixed-k edit moves it by ~1/L and a gap on
    a 5-mer is not commensurable with one on a 19-mer (the same length confound
    this project already hit in pll_regret). A sign test is invariant to that
    rescaling, and it is also immune to the folding problem that makes a mean
    |gap| non-zero under the null."""
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "pos": 0, "p": 1.0}
    from math import comb
    k = max(pos, neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return {"n": n, "pos": pos, "neg": neg, "p": min(1.0, 2.0 * tail)}


def bootstrap_gap(mut_folds, nat_folds, d_frozen, n_boot=20000, seed=0):
    """Sampling distribution of one mutant's gap, resampling FOLDS.

    The gap is a difference of two medians each estimated from a handful of
    diffusion samples, so its own uncertainty is what decides whether it is
    real. `d_frozen` is deterministic (one table, one lookup) and contributes
    none. Returns the 95% interval and whether it excludes zero."""
    import random
    if len(mut_folds) < 2 or len(nat_folds) < 2:
        return None
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        m = st.median(rng.choices(mut_folds, k=len(mut_folds)))
        nt = st.median(rng.choices(nat_folds, k=len(nat_folds)))
        out.append((m - nt) - d_frozen)
    out.sort()
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out)) - 1]
    return {"lo": lo, "hi": hi, "excludes_zero": (lo > 0 or hi < 0),
            "sd": st.stdev(out)}


def _gaps(rows):
    """The double difference, per mutant, against its target's own-pose native.

    A mutant with no cached native fold is DROPPED, not compared against the
    crystal-pose score: that comparison would measure crystal-vs-Boltz, not
    additivity, and is the single easiest way to manufacture a fake result.

    DUPLICATE SEQUENCES ARE COLLAPSED. The designed sweep runs three arms
    (opt / pos / any) per (target, k), and they COLLIDE by construction: when the
    table offers only j positive-gain positions in the window, a random
    positive-gain subset of size k=j IS the argmax, so `pos` returns exactly what
    `opt` does. The fold is cached by sequence, so the collision costs no GPU --
    but it writes a second row whose own-pose folds and frozen score are
    BITWISE identical, and counting both would put two copies of one measurement
    into the sign test. That inflates n and makes the samples perfectly
    correlated, which is the one thing a binomial test cannot survive
    (cf. the pooled-pairs lesson: n is only n if the draws are independent).
    Dedupe on (target_id, seq) and keep the `opt` row when one is present, since
    that is the decision-relevant sequence; `arms_dropped` records what was
    collapsed so per-arm counts can be read honestly rather than silently."""
    nat = {}
    for r in rows:
        if r.get("is_native") and r.get("pll_own_median") is not None:
            nat.setdefault(r["target_id"], []).append(r)
    native_own = {t: st.median([x["pll_own_median"] for x in v])
                  for t, v in nat.items()}
    native_frozen = {}
    for r in rows:
        if r.get("is_native") and r.get("pll_frozen") is not None:
            native_frozen[r["target_id"]] = r["pll_frozen"]

    nat_folds = {t: [v for x in rows if x.get("is_native")
                     and x["target_id"] == t for v in _own_folds(x)]
                 for t in native_own}

    # Collapse identical sequences BEFORE any statistic touches them.
    uniq, dropped = {}, []
    for r in rows:
        if r.get("is_native") or r.get("pll_own_median") is None:
            continue
        # No `seq` means the row cannot be proven a duplicate, so it is kept:
        # a missing field must never silently merge two distinct measurements.
        key = ((r["target_id"], r["seq"]) if r.get("seq")
               else ("\0unkeyed", len(uniq)))
        prev = uniq.get(key)
        if prev is None:
            uniq[key] = r
        else:
            # Keep `opt`: it is the sequence `best_subs` actually returns.
            keep, drop = ((r, prev) if r.get("arm") == "opt"
                          and prev.get("arm") != "opt" else (prev, r))
            uniq[key] = keep
            dropped.append({"target_id": key[0], "k": drop.get("k"),
                            "arm": drop.get("arm"),
                            "duplicate_of": keep.get("arm")})

    out = []
    for r in uniq.values():
        t = r["target_id"]
        if t not in native_own or t not in native_frozen:
            continue
        d_own = r["pll_own_median"] - native_own[t]
        d_frozen = r["pll_frozen"] - native_frozen[t]
        g = {"target_id": t, "k": r.get("k"), "arm": r.get("arm"),
             "in_window": r.get("in_window"),
             "length": len(r.get("seq") or r.get("native_seq") or ""),
             "d_own": d_own, "d_frozen": d_frozen,
             "gap": d_own - d_frozen,
             "sd": r.get("pll_own_sd")}
        g["boot"] = bootstrap_gap(_own_folds(r), nat_folds.get(t, []), d_frozen)
        out.append(g)
    return out, native_own, native_frozen, dropped


def report(rows):
    if isinstance(rows, (str, Path)):
        rows = [json.loads(l) for l in Path(rows).read_text().splitlines()
                if l.strip()]
    print(f"\n{'=' * 74}\nIS THE pll TABLE ADDITIVE WHEN THE POSE RESPONDS?"
          f"\n{'=' * 74}")

    sds = [r["pll_own_sd"] for r in rows if r.get("pll_own_sd") is not None]
    floor = st.median(sds) if sds else None
    print(f"\nNOISE FLOOR (spread of own-pose pll across diffusion samples of "
          f"the SAME sequence)")
    if floor is None:
        print("  GATE: no group had >1 scorable fold. Without a floor no gap "
              "can be called real. This is not a result.")
    else:
        print(f"  groups with >=2 folds : {len(sds)}")
        print(f"  median within-group sd: {floor:.4f} nats/res"
              f"   (max {max(sds):.4f})")

    gaps, nat_own, nat_frozen, dropped = _gaps(rows)
    print(f"\nnative folds available for: {sorted(nat_own)}"
          f"   mutants comparable: {len(gaps)}")
    if dropped:
        # Printed, never silent: the reader has to see that the arm counts
        # below are smaller than the number of folds the sweep paid for.
        print(f"  collapsed {len(dropped)} duplicate sequence(s) -- an arm "
              f"returned a sequence another arm had already produced, so the "
              f"measurement is ONE measurement, not two:")
        for d in dropped:
            print(f"    {d['target_id']} k={d['k']} arm={d['arm']} "
                  f"== arm={d['duplicate_of']}")
    by_arm = {}
    for g in gaps:
        if g.get("arm"):
            by_arm.setdefault(g["arm"], []).append(g["gap"])
    if by_arm:
        print("\nby arm (after collapsing duplicates):")
        for a in ("opt", "pos", "any"):
            if a not in by_arm:
                continue
            s = sign_test(by_arm[a])
            print(f"  {a:4s} n={len(by_arm[a]):3d}  positive={s['pos']}/"
                  f"{s['n']}  p={s['p']:.4g}  "
                  f"mean={st.mean(by_arm[a]):+.4f}")
        print("  `opt` is the decision-relevant arm. If `any` shows the same "
              "bias, the pose absorbs edits in general and the effect is not a "
              "selection artefact of the argmax.")
    if not gaps:
        print("  GATE: no mutant has a cached native fold for its target, so "
              "no double difference is computable.")
        print("  A raw own-vs-frozen comparison would measure crystal-vs-Boltz "
              "and must NOT be substituted here.")
        return

    for label, sel in (("substitutions INSIDE the released window",
                        [g for g in gaps if g["in_window"]]),
                       ("substitutions touching the HELD region",
                        [g for g in gaps if not g["in_window"]])):
        if not sel:
            continue
        print(f"\n-- {label} (n={len(sel)}) --")
        print(f"  {'tgt':<6s} {'k':>2s} {'d_frozen':>9s} {'d_own':>9s} "
              f"{'gap':>9s}  {'fold sd':>8s}")
        for g in sorted(sel, key=lambda x: (x["k"] or 0, x["target_id"])):
            sd = g["sd"]
            sds_txt = "     n/a" if sd is None else f"{sd:>+8.4f}"
            print(f"  {g['target_id']:<6s} {g['k'] or 0:>2d} "
                  f"{g['d_frozen']:>+9.4f} {g['d_own']:>+9.4f} "
                  f"{g['gap']:>+9.4f}  {sds_txt}")
        ab = [abs(g["gap"]) for g in sel]
        print(f"  mean |gap| = {st.mean(ab):.4f}"
              + (f"   sd = {st.stdev(ab):.4f}" if len(ab) > 1 else ""))
        if floor:
            print(f"  mean |gap| / noise floor = {st.mean(ab) / floor:.2f}x")

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    # WHY NOT mean |gap| vs the noise floor. That comparison is invalid in two
    # directions at once and an earlier version of this report made it. |gap| is
    # FOLDED, so under a pure-noise null its mean is ~0.8*sigma_gap rather than
    # zero -- a null result would look like a violation. And the per-fold sd is
    # not sigma_gap: each own-pose number is a MEDIAN over several folds, so its
    # error is smaller, while the gap differences TWO such medians, which makes
    # it larger. The two errors do not cancel and neither is the right scale.
    # So: test the SIGN (scale-free, immune to folding), and bootstrap each
    # mutant's own gap over its folds.
    signed = [g["gap"] for g in gaps]
    stt = sign_test(signed)

    # CALIBRATION, not just direction. The sign test asks whether the table errs
    # one way; it cannot see that the table errs in OPPOSITE directions at its
    # optimum and on the edits it rejects, which averages to a null and hides
    # a single simple defect: the table's predicted delta is too LARGE in
    # whichever direction it points. Regressing realized delta on predicted
    # delta measures that directly, and a slope below 1 is the whole statement.
    #
    # The regressor is exact. `d_frozen` is a deterministic table lookup with
    # ZERO measurement error, so this slope is not attenuated by regression
    # dilution -- the usual reason a slope below 1 means nothing. Fold noise
    # lives in y, where it widens the interval but does not bias the estimate.
    # The native baseline enters every point as a common offset, so it moves the
    # intercept and not the slope: that is what the double difference buys.
    xs = [g["d_frozen"] for g in gaps]
    ys = [g["d_own"] for g in gaps]
    if len(xs) >= 4 and len(set(xs)) > 1:
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx
        rr = sxy / ((sxx * syy) ** 0.5) if syy > 0 else float("nan")
        print(f"\nCALIBRATION: realized delta vs the delta the table predicted")
        print(f"  realized = {my - b * mx:+.4f} {b:+.4f} * predicted"
              f"      r = {rr:+.3f}  (r^2 = {rr * rr:.2f})  n = {len(xs)}")
        print(f"  slope 1 = the table's magnitudes are right. slope 0 = the "
              f"table carries no\n  usable magnitude at all. Measured "
              f"{b:.2f} means the table INFLATES every predicted\n  change by "
              f"~{1 / b:.1f}x if it points anywhere at all.")
        print(f"  r^2 = {rr * rr:.2f}: the fraction of the realized change the "
              f"table can see.\n  The rest is pose response, invisible to a "
              f"matrix built once.")
        print(f"  CONSEQUENCE: a uniform rescaling preserves the per-position "
              f"ORDER, so this\n  does not by itself unseat `best_subs` as a "
              f"ranker -- but every margin in\n  nats/res shrinks by {b:.2f}, "
              f"so an indifference band of 0.05 realized is a band\n  of "
              f"{0.05 / b:.2f} in TABLE units. Bands quoted in table units are "
              f"too narrow by {1 / b:.1f}x.")
        for arm in ("opt", "pos", "any"):
            sel = [g for g in gaps if g.get("arm") == arm]
            if sel:
                print(f"    {arm:4s} n={len(sel):3d}  table promised "
                      f"{st.mean([g['d_frozen'] for g in sel]):+.3f}, pose "
                      f"delivered {st.mean([g['d_own'] for g in sel]):+.3f}")

    print(f"\nDIRECTION (the statistic that survives the length confound)")
    print(f"  gaps above zero : {stt['pos']}/{stt['n']}   "
          f"sign test p = {stt['p']:.4f}")
    print(f"  mean SIGNED gap : {st.mean(signed):+.4f} nats/res"
          + (f"   sd {st.stdev(signed):.4f}" if len(signed) > 1 else ""))
    print(f"  A POSITIVE gap means the frozen table OVER-predicts the damage a")
    print(f"  mutation does: the re-predicted pose absorbs part of the loss the")
    print(f"  table charged for. That is non-additivity in the direction that")
    print(f"  matters, because `best_subs` is then penalising edits the")
    print(f"  structure would have accommodated.")

    boots = [g for g in gaps if g.get("boot")]
    res = [g for g in boots if g["boot"]["excludes_zero"]]
    print(f"\nPER-MUTANT RESOLUTION (bootstrap over folds, 95%)")
    print(f"  mutants with a bootstrappable gap : {len(boots)}/{len(gaps)}")
    print(f"  whose 95% interval excludes zero  : {len(res)}")
    for g in sorted(res, key=lambda x: -abs(x["gap"]))[:8]:
        b = g["boot"]
        print(f"      {g['target_id']} k={g['k']} L={g['length']:<2d} "
              f"gap {g['gap']:+.4f}  [{b['lo']:+.4f}, {b['hi']:+.4f}]"
              f"{'' if g['in_window'] else '  (held region)'}")

    if stt["n"] and stt["p"] < 0.05 and len(res) >= 3:
        print(f"\n  => ADDITIVITY IS VIOLATED, DIRECTIONALLY AND REPRODUCIBLY.")
        print(f"     `best_subs` is a first-order approximation, not the")
        print(f"     optimum, so regret measured against it is not regret")
        print(f"     against the truth. The exact bar weakens and the search")
        print(f"     question re-opens.")
    elif stt["n"] and stt["p"] < 0.05:
        print(f"\n  => DIRECTIONAL SIGNAL, MAGNITUDE UNRESOLVED. The sign is")
        print(f"     consistent but few individual gaps clear their own fold")
        print(f"     noise. More FOLDS PER SEQUENCE is the lever, not more")
        print(f"     sequences.")
    else:
        print(f"\n  => NOT RESOLVED. This data cannot separate coupling from")
        print(f"     fold sampling. Do not report additivity as holding.")

    print(f"\nREAD WITH THESE THREE CAVEATS")
    print(f"  1. HELD TEMPLATE. Backbone outside the released window was pinned")
    print(f"     to native, so coupling is suppressed by construction. Every")
    print(f"     number here is a LOWER BOUND on non-additivity.")
    print(f"  2. LENGTH. `pll_bg` is a mean over positions, so a gap on a 5-mer")
    print(f"     and one on a 19-mer are not commensurable. Lengths present: "
          f"{sorted({g['length'] for g in gaps})}.")
    print(f"     The sign test is reported precisely because it is invariant to")
    print(f"     this; the means are not and must not be the headline.")
    print(f"  3. CACHED, NOT DESIGNED. These folds were produced for another")
    print(f"     experiment, so k is {sorted({g['k'] for g in gaps})} and the")
    print(f"     mutants are not a random or balanced sample. No k-slope is")
    print(f"     claimed. A designed sweep over k is the confirmation.")


def selftest():
    ok = True

    def eq(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {label}: {got!r}"
              + ("" if good else f" != {want!r}"))

    print("stem parsing")
    m = STEM_RE.match("held_1bjr_6_9_804158df929a_model_0.pdb")
    eq("plain stem", (m.group("tid"), m.group("a"), m.group("b"),
                      m.group("draw"), m.group("model")),
       ("1bjr", "6", "9", None, "0"))
    m2 = STEM_RE.match("held_1elr_0_1_deadbeefcafe_d2_model_3.pdb")
    eq("draw-suffixed stem", (m2.group("tid"), m2.group("draw"),
                              m2.group("model")), ("1elr", "2", "3"))
    eq("non-matching name", STEM_RE.match("1bjr_model_0.pdb"), None)

    print("subs")
    eq("no subs", subs_of("ACDE", "ACDE"), [])
    eq("two subs", subs_of("ACDE", "AWDG"), [(1, "W"), (3, "G")])
    eq("length mismatch refuses", subs_of("ACDE", "ACD"), None)

    print("chain sequence from a structure")
    import tempfile
    lines = []
    for i, nm in enumerate(["ALA", "CYS", "TRP"]):
        for at in ("N", "CA"):          # two atoms per residue: must dedupe
            lines.append(f"ATOM  {i * 2 + 1:5d}  {at:<3s} {nm} B{i + 1:4d}    "
                         f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00")
    # built with the SAME format string as the binder lines: hand-aligning a
    # PDB literal is how the first version of this test silently read column 21
    # as a space and found an empty receptor chain.
    lines.append(f"ATOM  {99:5d}  CA  GLY A{1:4d}    "
                 f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdb"
        p.write_text("\n".join(lines) + "\n")
        eq("binder chain only, deduped", chain_seq(p, "B"), "ACW")
        eq("receptor chain is separate", chain_seq(p, "A"), "G")

    print("double difference")
    rows = [
        {"target_id": "t", "is_native": True, "pll_own_median": -2.0,
         "pll_frozen": -1.0, "pll_own_sd": 0.01},
        # the table says +0.30; the pose says +0.30 -> gap 0
        {"target_id": "t", "is_native": False, "k": 1, "in_window": True,
         "pll_own_median": -1.7, "pll_frozen": -0.7, "pll_own_sd": 0.01},
        # the table says +0.30; the pose says +0.10 -> gap -0.20
        {"target_id": "t", "is_native": False, "k": 2, "in_window": True,
         "pll_own_median": -1.9, "pll_frozen": -0.7, "pll_own_sd": 0.01},
        # no native for target u -> must be dropped, not compared to crystal
        {"target_id": "u", "is_native": False, "k": 1, "in_window": True,
         "pll_own_median": -1.0, "pll_frozen": -0.5, "pll_own_sd": 0.01},
    ]
    g, _, _, drp = _gaps(rows)
    eq("rows with no seq are never merged", len(g), 2)
    eq("nothing is collapsed without evidence", drp, [])
    eq("mutants without a native fold are dropped",
       sorted({x["target_id"] for x in g}), ["t"])
    eq("crystal-vs-Boltz offset cancels",
       round(next(x["gap"] for x in g if x["k"] == 1), 6), 0.0)
    eq("real coupling survives",
       round(next(x["gap"] for x in g if x["k"] == 2), 6), -0.2)

    # A COLLIDING ARM MUST NOT BE COUNTED TWICE. `pos` returns the argmax
    # whenever the window has exactly k positive-gain positions, and the fold is
    # cached by sequence, so the second row is a bitwise copy. Two copies of one
    # measurement in a binomial test is a manufactured n.
    dup_rows = [
        {"target_id": "t", "is_native": True, "k": 0, "seq": "AAAA",
         "pll_own_median": -1.5, "pll_frozen": -0.5, "pll_own_sd": 0.01},
        {"target_id": "t", "is_native": False, "k": 3, "in_window": True,
         "arm": "pos", "seq": "AGGG",
         "pll_own_median": -1.9, "pll_frozen": -0.7, "pll_own_sd": 0.01},
        {"target_id": "t", "is_native": False, "k": 3, "in_window": True,
         "arm": "opt", "seq": "AGGG",
         "pll_own_median": -1.9, "pll_frozen": -0.7, "pll_own_sd": 0.01},
        {"target_id": "t", "is_native": False, "k": 3, "in_window": True,
         "arm": "any", "seq": "AHIE",
         "pll_own_median": -1.8, "pll_frozen": -0.7, "pll_own_sd": 0.01},
    ]
    g2, _, _, drp2 = _gaps(dup_rows)
    eq("an identical sequence is one measurement, not two", len(g2), 2)
    eq("the collapse is reported", len(drp2), 1)
    eq("the dropped row is the non-opt arm", drp2[0]["arm"], "pos")
    eq("`opt` is the arm kept",
       sorted(x["arm"] for x in g2), ["any", "opt"])

    print("statistics")
    eq("sign test, all positive", sign_test([1, 2, 3])["p"], 0.25)
    eq("sign test, balanced", sign_test([1, -1])["p"], 1.0)
    eq("sign test ignores zeros", sign_test([1, 2, 0, 0])["n"], 2)
    eq("17 of 22 is suggestive",
       round(sign_test([1] * 17 + [-1] * 5)["p"], 4), 0.0169)
    b = bootstrap_gap([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.5, n_boot=500)
    eq("a clean separation excludes zero", b["excludes_zero"], True)
    eq("and recovers the gap", round(b["lo"], 6), 0.5)
    b2 = bootstrap_gap([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 0.0, n_boot=2000)
    eq("identical noisy samples do NOT exclude zero", b2["excludes_zero"], False)
    eq("too few folds refuses to bootstrap",
       bootstrap_gap([1.0], [0.0], 0.0), None)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--boltz-root",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/boltz_val")
    ap.add_argument("--pssm-dir",
                    default="/zfsauton/scratch/yixiz/lp2cp_stage2/lnr/pssm")
    ap.add_argument("--out", default="pll_additivity.jsonl")
    ap.add_argument("--rows", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="")
    ap.add_argument("--scope", default="all", choices=("all", "iface"))
    ap.add_argument("--metric", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.measure:
        measure(a.boltz_root, a.pssm_dir, a.out, device=a.device,
                model_name=(a.model or None), scope=a.scope,
                metric=(a.metric or None))
        return 0
    if a.report:
        report(a.rows or a.out)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
