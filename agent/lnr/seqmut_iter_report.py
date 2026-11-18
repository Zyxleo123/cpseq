"""The BELIEVED-NUMBER report for the iterative-search mutation arms.

The firewall principle (why this file exists separate from the loop)
--------------------------------------------------------------------
The in-loop proxy (seqmut_iter.py) selects sequences with a cheap, fold-free,
zero-variance quantity (energy-weighted retention x native_dG). It is never the
reported result -- optimising against the same stochastic quantity you grade
with is exactly how the project's earlier "wins" evaporated (cpseq best-of-B:
-4.66 -> -0.06 REU under cross-fit; lnr: "LLM arms overfit the noisy verifier").
So: the search selects on the proxy; ONLY the report folds. No stochastic
component of the report's folding ever enters selection.

What the believed number is
---------------------------
Once a (sequence, window) finalist is picked, the reported score is the REAL
denoised Boltz-2 -> Rosetta interface dG: boltz_validate `--mode mut-held
--replicas 4` folds the finalist (and its matched nomut) 4 times from one trunk
pass; the headline is the MEDIAN refined dG_separated over the sequence/ring-valid
folds, and the error bar is the STANDARD ERROR of that median (std / sqrt(n_valid)),
not the single-fold spread. This module reads those folded rows and reports.

The references (per the design doc, decomposed -- N_lin is NOT the success bar)
-----------------------------------------------------------------------------
Per complex, more-negative dG = stronger binding:
  N_lin  : linear native (boltz_linear_ref.jsonl `linear_dG`). The ceiling the
           cyclic peptide structurally CANNOT reach; reported only as the top of
           the decomposition, never as the bar.
  ceiling: the best clash-free closing window, WT sequence. In this pipeline the
           committed window IS the clash-free contact/energy-max window (the
           energy-ranking "dead thread" showed energy-max == contact-max on clean
           closers), so the folded nomut (N_cyc) at that window IS the ceiling.
  N_cyc  : nomut cyclization in the SAME window as the finalist -- "did the edit
           help?". Here N_cyc == ceiling (same window).
Decomposition of the N_lin gap:
  irreducible (geometry) = N_cyc - N_lin            (binding lost to cyclisation)
  designable  (sequence) = N_cyc - achieved         (what the edit clawed back)
  captured fraction      = designable / irreducible (of the cyclisation cost)

The null the number must clear (non-negotiable)
-----------------------------------------------
Random mutation at MATCHED budget/rounds/compute: the `iter-ea` arm. The finalist
is a result ONLY if the denoised LLM finalist beats the denoised random-EA
finalist on the report metric, paired per complex (Wilcoxon). "More rounds helps
even random search", so nomut is not the bar and neither is the proxy -- iter-ea
is. Complex is the unit; nested budget rungs are never pooled.

Orthogonality (DEFERRED -- infra, not skipped)
----------------------------------------------
The doc also asks a different-lineage cyclic folder (AfCycDesign / HighFold)
confirm the sign/rank on finalists. No second folder is installed here, so this
report leaves an `orthogonal_dG` column that stays empty until one is; the
believed number is Boltz-only and labelled as such. This is the one piece of the
spec not yet met, and it is an infrastructure gap, not a modelling choice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RMSD_MAX = 8.0   # pair-quality guard: a fold >8 Å from the native interface is
#                  mislocalised, not a worse binder -- dropped, never pooled.


def _valid(r, rmsd_max=RMSD_MAX) -> bool:
    """A folded row counts if it scored, is the requested sequence, closed a ring,
    has a refined dG, and did not grossly mislocalise. Mirrors boltz_validate
    `_is_valid` + the held_rmsd pair guard."""
    if r is None:
        return False
    if not (r.get("score_status") == "ok" and r.get("sequence_matches") is True
            and r.get("ring_declared") is True and r.get("dG_refined") is not None):
        # legacy rows may lack the status fields; accept a present dG as a fallback
        if r.get("dG_refined") is None:
            return False
    hr = r.get("held_rmsd")
    return not (hr is not None and hr > rmsd_max)


def _sem(r):
    """Standard error of the median dG over valid folds: std / sqrt(n_valid).
    None when the spread/count was not recorded (legacy single-fold rows)."""
    sd, n = r.get("dG_refined_std"), r.get("n_valid_folds")
    if sd is None or not n or n < 1:
        return None
    return round(sd / math.sqrt(n), 3)


def load_folded(mut_held_path: Path) -> dict:
    """{(target_id, arm_label): row} from a boltz mut-held output (real dG)."""
    out = {}
    p = Path(mut_held_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                out[(r["target_id"], r["arm_label"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def load_linear(linear_ref_path: Path) -> dict:
    """{target_id: N_lin dG} from boltz_linear_ref.jsonl (`linear_dG`)."""
    out = {}
    p = Path(linear_ref_path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = r.get("linear_dG")
            if r.get("target_id") and v is not None:
                out[r["target_id"]] = float(v)
    return out


def _paired(folded, a_label, b_label, rmsd_max=RMSD_MAX):
    """Per-complex (achieved_a - achieved_b) on the real dG, over complexes where
    BOTH arms folded validly. More negative = a binds stronger than b."""
    tids = sorted({tid for (tid, lab) in folded if lab in (a_label, b_label)})
    rows = []
    for tid in tids:
        ra, rb = folded.get((tid, a_label)), folded.get((tid, b_label))
        if _valid(ra, rmsd_max) and _valid(rb, rmsd_max):
            rows.append((tid, ra["dG_refined"] - rb["dG_refined"],
                         ra["dG_refined"], rb["dG_refined"]))
    return rows


def _wilcoxon(deltas):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    xs = [d for d in deltas if abs(d) > 1e-9]
    if len(xs) < 1:
        return None
    try:
        return float(wilcoxon(xs).pvalue)
    except Exception:
        return None


def report(mut_held_path, linear_ref_path="", *, llm_arms=None, random_arm=None,
           nomut_label=None, k=2, model=None, rmsd_max=RMSD_MAX):
    """Print the believed-number report: real dG per arm (median +/- SEM), the
    N_lin -> ceiling -> achieved decomposition, and the matched-budget random
    null (LLM vs iter-ea, paired per complex)."""
    folded = load_folded(mut_held_path)
    nlin = load_linear(linear_ref_path) if linear_ref_path else {}
    if not folded:
        print(f"[iter-report] no folded rows at {mut_held_path} -- run the Boltz "
              f"mut-held fold first (--replicas 4).")
        return

    labels = sorted({lab for (_, lab) in folded})
    nomut_label = nomut_label or next((l for l in labels if l.startswith("nomut")), f"nomut:k{k}")
    random_arm = random_arm or next((l for l in labels if l.startswith("iter-ea")), None)
    if llm_arms is None:
        llm_arms = [l for l in labels if l.startswith("iter-llm")]
    arms = [a for a in (llm_arms + ([random_arm] if random_arm else [])) if a]
    tids = sorted({tid for (tid, _) in folded})

    def cell(tid, lab):
        r = folded.get((tid, lab))
        if not _valid(r, rmsd_max):
            return None
        return r

    # ---- per-complex table ------------------------------------------------
    print("\n" + "=" * 112)
    print("SEQMUT-ITER believed number -- denoised Boltz-2 x4 -> Rosetta refined "
          "dG_separated (REU); more negative = stronger binding.")
    print("dG shown as MEDIAN+/-SEM over sequence/ring-valid folds. CEIL = N_cyc "
          "(nomut at the committed clash-free window).")
    print("=" * 112)
    hdr = f"{'target':7s} {'N_lin':>9s} {'N_cyc/CEIL':>12s}"
    for a in arms:
        hdr += f"{a.replace('gpt-6-luna','luna').replace('gpt-6.1-terra','terra'):>22s}"
    print(hdr)
    for tid in tids:
        line = f"{tid:7s} "
        line += (f"{nlin[tid]:>9.2f}" if tid in nlin else f"{'-':>9s}")
        nc = cell(tid, nomut_label)
        line += (f"{nc['dG_refined']:>12.2f}" if nc else f"{'-':>12s}")
        for a in arms:
            c = cell(tid, a)
            if c:
                sem = _sem(c)
                s = f"{c['dG_refined']:+.2f}" + (f"±{sem:.2f}" if sem is not None else "")
            else:
                s = "-"
            line += f"{s:>22s}"
        print(line)

    # ---- decomposition (aggregate over complexes) -------------------------
    print("\n  N_lin gap decomposition (mean over complexes with all three folds):")
    for a in [x for x in arms]:
        irr, des, frac, n = [], [], [], 0
        for tid in tids:
            ca, nc = cell(tid, a), cell(tid, nomut_label)
            if not (ca and nc and tid in nlin):
                continue
            cyc_cost = nc["dG_refined"] - nlin[tid]          # irreducible (>0 = lost)
            gain = nc["dG_refined"] - ca["dG_refined"]       # designable (>0 = edit helped)
            irr.append(cyc_cost); des.append(gain)
            if abs(cyc_cost) > 1e-9:
                frac.append(gain / cyc_cost)
            n += 1
        if n:
            fr = f"{100*sum(frac)/len(frac):.0f}%" if frac else "n/a"
            print(f"    {a:28s} n={n:2d}  irreducible(N_cyc-N_lin)={sum(irr)/len(irr):+.2f}  "
                  f"designable(N_cyc-achieved)={sum(des)/len(des):+.2f} REU  "
                  f"(captured {fr} of the cyclisation cost)")

    # ---- the matched-budget random null (THE result gate) -----------------
    if random_arm:
        print(f"\n  KEY NULL -- LLM finalist vs random-EA ({random_arm}) at matched "
              f"budget, paired on the REAL dG (more negative Δ = LLM wins; a result "
              f"REQUIRES this to be negative & significant):")
        for a in [x for x in llm_arms if x != random_arm]:
            rows = _paired(folded, a, random_arm, rmsd_max)
            if not rows:
                print(f"    {a:28s} vs {random_arm}: no comparable folds yet")
                continue
            d = [x[1] for x in rows]
            p = _wilcoxon(d)
            better = sum(x < -1e-9 for x in d)
            ps = f"p={p:.3f}" if p is not None else "p=n/a"
            print(f"    {a:28s} n={len(d):2d} mean Δ={sum(d)/len(d):+.2f} REU "
                  f"(LLM better/worse={better}/{len(d)-better}) {ps}")
            print("      " + ", ".join(f"{t}:{dd:+.2f}" for t, dd, *_ in rows))
    # also the softer 'did the edit help' vs nomut
    print("\n  vs nomut (N_cyc) -- 'did the edit help' (softer than the null above):")
    for a in arms:
        rows = _paired(folded, a, nomut_label, rmsd_max)
        if rows:
            d = [x[1] for x in rows]
            better = sum(x < -1e-9 for x in d)
            print(f"    {a:28s} n={len(d):2d} mean Δ={sum(d)/len(d):+.2f} REU "
                  f"(better/worse={better}/{len(d)-better})")

    print("\n  NOTE: Boltz-only; a different-lineage cyclic folder (orthogonality) "
          "is not installed here -- that confirmation is DEFERRED, not met.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mut-held", required=True,
                    help="boltz mut-held output with the folded iter finalists + nomut")
    ap.add_argument("--linear-ref", default="",
                    help="boltz_linear_ref.jsonl for N_lin (optional)")
    ap.add_argument("--llm-arms", default="",
                    help="comma-sep LLM arm labels (default: all iter-llm*)")
    ap.add_argument("--random-arm", default="",
                    help="the matched-budget random-EA arm label (default: iter-ea*)")
    ap.add_argument("--nomut-label", default="")
    ap.add_argument("--rmsd-max", type=float, default=RMSD_MAX)
    args = ap.parse_args()
    report(args.mut_held, args.linear_ref,
           llm_arms=[a.strip() for a in args.llm_arms.split(",") if a.strip()] or None,
           random_arm=args.random_arm or None,
           nomut_label=args.nomut_label or None, rmsd_max=args.rmsd_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
