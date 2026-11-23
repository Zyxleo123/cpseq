"""The candidate binding metric on SKEMPI: does it call MUTATION EFFECTS, at scale?

Why this arm exists
-------------------
The CPSet gate is the right CHEMISTRY (cyclic peptides, head-to-tail) but a thin
instrument: 10 series, only 5 with n >= 3, so the per-series sign test cannot
produce a p-value below 1/32 however good the metric is. SKEMPI is the opposite
-- wrong chemistry (protein-protein interfaces), but thousands of measured
mutations across many complexes, i.e. hundreds of within-target series. The two
answer different halves:

    CPSet   does it transfer to head-to-tail cyclic chemistry?   (2 BB series)
    SKEMPI  does pose-conditioned likelihood track binding AT ALL? (high power)

SKEMPI was previously judged a dead end here, for two stated reasons. One of them
no longer applies:

  * "effects are ~1 kcal/mol, INSIDE our ~4 REU noise, so a null is ambiguous"
    -- that objection was about the Rosetta oracle's noise. THIS metric is
    exactly deterministic (`bindscore.selftest` asserts bitwise reproducibility),
    so there is no noise floor to hide a 1 kcal/mol effect under. The objection
    is void for a noiseless scorer.
  * "wrong chemistry" -- still true, and why this is a CONFIRMATION arm and not
    the head-to-tail claim. It is reported separately and never pooled with CPSet.

It is also newly affordable. The Rosetta gate managed 44 mutations (30% of its
tasks segfaulted) because each needed relax+score. Here a mutation is TWO TABLE
LOOKUPS into a position x amino-acid matrix that the pose already determined, so
the whole benchmark runs in minutes on CPU.

The quantity
------------
With the mutated position -- and ONLY that position -- marked "designed", so its
own side chain is hidden from the model while every other side chain in the
complex stays as context:

    dscore_cplx = log p(mut | complex pose) - log p(wt | complex pose)
    dscore_mono = log p(mut | that chain alone) - log p(wt | that chain alone)
    ddg_pred    = -dscore_cplx                      # the direct form
    ddg_pred_icg= -(dscore_cplx - dscore_mono)      # interface-attributable only

The ICG construction is far better posed here than on a 7-mer: the "monomer" is
a real folded protein, fully in distribution, so subtracting it genuinely removes
"is this residue good for the fold" and leaves "is it good for the INTERFACE".
On CPSet the same subtraction had to use a 7-residue peptide alone, which is not.

SIGN CONVENTION, stated once and asserted in tests. Measured ddG = RT ln(Kd_mut /
Kd_wt), so POSITIVE = the mutation WEAKENS binding. `ddg_pred` is negated above
so that positive also means weaker. A working metric therefore gives a POSITIVE
correlation -- the same convention as the Rosetta gate, so the numbers are
directly comparable to its rho +0.22 / sign 59%.

Usage
-----
  python -m lnr.bindscore_skempi --score --csv <skempi_v2.csv> --pdb-dir <pdb/> \
      --out skempi_bindscore.jsonl --max-records 4000
  python -m lnr.bindscore_skempi --report skempi_bindscore.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr.bindscore import MPNNScorer, PRIMARY_MODEL  # noqa: E402
from lnr.skempi_accuracy import select_records, _spearman, _pearson  # noqa: E402

# Reference numbers this arm is measured against. Printed beside our own so no
# result is ever read without its comparator.
REF = {
    "Rosetta dG_separated, single relax, 44 muts (this project's prior gate)": 0.22,
    "Flex ddG, 35-model ensemble (published, same benchmark)": 0.60,
    "random expectation": 0.00,
}


def _index_map(pose, scorer):
    """{(chain, resnum, icode): pose index} for a parsed pose.

    Built the same way LigandMPNN's own driver builds it, from R_idx and
    chain_letters, so a SKEMPI site resolves to the residue the crystal
    actually has rather than to an off-by-one.
    """
    letters = list(pose.protein_dict["chain_letters"])
    ridx = list(pose.protein_dict["R_idx"].cpu().numpy())
    return {(str(c), int(r)): i for i, (c, r) in enumerate(zip(letters, ridx))}


def score_mutations(scorer, selected, out_path, verbose=True, limit=0):
    """One row per mutation. Resumable; groups work so each encode is reused."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.is_file():
        for ln in out_path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            done.add((r["complex_id"], r["mut_str"], r["model"]))

    by_pdb = {}
    for s in selected:
        by_pdb.setdefault((s["pdb_path"], s["keep_chains"]), []).append(s)

    fh = out_path.open("a")
    n_new = n_fail = 0
    t0 = time.time()
    for (pdb_path, keep), muts in sorted(by_pdb.items()):
        if limit and n_new >= limit:
            break
        for s in muts:
            key = (s["complex_id"], s["mut_str"], scorer.model_name)
            if key in done:
                continue
            try:
                wt_i = scorer.aa_to_int[s["wt"]]
                mu_i = scorer.aa_to_int[s["mut"]]
                site = (s["mut_chain"], s["mut_resnum"])
                d_c = _delta_at(scorer, pdb_path, keep, wt_i, mu_i,
                                s["mut_chain"], site)
                mono = (s["chains1"] if s["mut_chain"] in s["chains1"]
                        else s["chains2"])
                d_m = _delta_at(scorer, pdb_path, mono, wt_i, mu_i,
                                s["mut_chain"], site)
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                if verbose and n_fail < 10:
                    print(f"  {s['complex_id']} {s['mut_str']}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                continue
            if d_c is None or d_m is None:
                n_fail += 1
                continue
            fh.write(json.dumps({
                "model": scorer.model_name,
                "complex_id": s["complex_id"], "pdb_id": s["pdb_id"],
                "mut_str": s["mut_str"], "wt": s["wt"], "mut": s["mut"],
                "mut_chain": s["mut_chain"], "mut_resnum": s["mut_resnum"],
                "location": s["location"], "measured_ddg": s["measured_ddg"],
                # Negated so POSITIVE = predicted weaker, matching the measured
                # convention and the prior Rosetta gate.
                "ddg_pred": -d_c,
                "ddg_pred_icg": -(d_c - d_m),
                "dscore_cplx": d_c, "dscore_mono": d_m,
            }) + "\n")
            fh.flush()
            n_new += 1
            if verbose and n_new % 200 == 0:
                print(f"  {n_new} scored ({time.time()-t0:.0f}s)", flush=True)
    fh.close()
    if verbose:
        print(f"wrote {n_new} rows ({n_fail} unscorable) in "
              f"{time.time()-t0:.1f}s", flush=True)
    return n_new


_POSE_CACHE = {}
_POSE_CACHE_MAX = 8


def _delta_at(scorer, pdb_path, chains, wt_i, mu_i, mut_chain, site):
    """log p(mut) - log p(wt) at one site. Returns None if the site is absent.

    The mutated CHAIN is the 'designed' one, exactly as the peptide is on the
    CPSet arm: its side chains are hidden from the model, the partner keeps
    hers as context. Keeping the two arms on one protocol is deliberate -- a
    per-site mask would be more informative but would make the SKEMPI number
    measure a different quantity than the CPSet number, and they are meant to
    be read together.

    That choice also makes this fast: the mask depends only on (complex, mutated
    chain), so the pose and its encoding are built once and every mutation in
    that chain is a lookup into the resulting position x amino-acid matrix.

    The index is ALWAYS resolved inside the pose being scored, never carried in
    from another parse: parse_PDB returns residues in file order, so an index
    taken from a differently-ordered parse would silently point at the wrong
    residue and still produce a plausible number.
    """
    chains = "".join(chains)
    ck = (pdb_path, chains, mut_chain)
    pose = _POSE_CACHE.get(ck)
    if pose is None:
        rec = [c for c in chains if c != mut_chain]
        pose = scorer.parse(pdb_path, pep_chain=mut_chain,
                            rec_chains=rec or None)
        pose._imap = _index_map(pose, scorer)
        pose._lp = None
        if len(_POSE_CACHE) >= _POSE_CACHE_MAX:
            _POSE_CACHE.clear()
        _POSE_CACHE[ck] = pose
    idx = pose._imap.get((site[0], site[1]))
    if idx is None:
        return None
    if pose._lp is None:
        pose._lp = {}
    lp = pose._lp.get(idx)
    if lp is None:
        pose.pep_idx = np.array([idx], dtype=int)
        lp = scorer.marginals(pose, see_other_seq=False)[0]
        pose._lp[idx] = lp
    return float(lp[mu_i] - lp[wt_i])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("measured_ddg") is None:
            continue
        rows.append(r)
    if not rows:
        print(f"no usable rows in {path}")
        return 1
    models = sorted({r["model"] for r in rows})
    print("=" * 78)
    print("CANDIDATE BINDING METRIC on SKEMPI v2 -- mutation effects, at scale")
    print("=" * 78)
    print("CONFIRMATION ARM ONLY. SKEMPI is protein-PROTEIN; the head-to-tail")
    print("claim rests on the CPSet BB tier and these are never pooled.")
    print("Sign: measured ddG > 0 = weaker. ddg_pred is negated to match, so a")
    print("WORKING metric gives a POSITIVE correlation.")
    print()
    for k, v in REF.items():
        print(f"   {k:62s} {v:+.3f}")
    print()
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        print("-" * 78)
        print(f"MODEL {model}:  {len(sub)} mutations, "
              f"{len({r['complex_id'] for r in sub})} complexes")
        print("-" * 78)
        y = [r["measured_ddg"] for r in sub]
        for field in ("ddg_pred", "ddg_pred_icg"):
            x = [r[field] for r in sub]
            rho, pr = _spearman(x, y), _pearson(x, y)
            sign = sum(1 for a, b in zip(x, y)
                       if (a > 0) == (b > 0)) / len(x)
            # Within-complex: the honest statistic, since pooling across
            # complexes imports the between-complex ddG range the way pooling
            # across CPSet series imports the cross-target pKd range.
            per = {}
            for r in sub:
                per.setdefault(r["complex_id"], []).append((r[field],
                                                            r["measured_ddg"]))
            wc = [(_spearman([p[0] for p in v], [p[1] for p in v]), len(v))
                  for v in per.values() if len(v) >= 5]
            wc = [(a, b) for a, b in wc if a is not None]
            med = st.median([a for a, _ in wc]) if wc else None
            pos = sum(1 for a, _ in wc if a > 0)
            p_sign = (sum(math.comb(len(wc), i)
                          for i in range(pos, len(wc) + 1)) / 2 ** len(wc)
                      if wc else None)
            print(f"  {field:14s} pooled rho {rho:+.3f}  pearson {pr:+.3f}  "
                  f"sign {sign:.3f}")
            print(f"  {'':14s} WITHIN-complex: median rho "
                  f"{med if med is None else round(med, 3)} over {len(wc)} "
                  f"complexes with n>=5; {pos}/{len(wc)} positive, "
                  f"sign-test p = "
                  f"{p_sign if p_sign is None else f'{p_sign:.2e}'}")
        # Stratify by how buried the site is: the interface claim should be
        # strongest at the interface core and weakest at the rim/surface.
        print()
        print("  by SKEMPI location (ddg_pred, pooled rho):")
        byloc = {}
        for r in sub:
            byloc.setdefault(r.get("location") or "?", []).append(r)
        for loc, v in sorted(byloc.items(), key=lambda kv: -len(kv[1]))[:6]:
            if len(v) < 10:
                continue
            rho = _spearman([r["ddg_pred"] for r in v],
                            [r["measured_ddg"] for r in v])
            print(f"    {loc:22s} n={len(v):5d}  rho {rho:+.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--report", metavar="JSONL")
    ap.add_argument("--csv", default="/zfsauton/scratch/yixiz/lp2cp_pilot/"
                                     "data_raw/skempi_v2.csv")
    ap.add_argument("--pdb-dir", default="/zfsauton/scratch/yixiz/lp2cp_pilot/pdb")
    ap.add_argument("--out", default="skempi_bindscore.jsonl")
    ap.add_argument("--models", default=PRIMARY_MODEL)
    ap.add_argument("--max-records", type=int, default=4000)
    ap.add_argument("--max-ca", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.report:
        return report(args.report)
    if args.score:
        from skempi import load_records
        records, rep = load_records(args.csv)
        selected, reasons = select_records(records, args.pdb_dir,
                                           max_records=args.max_records,
                                           max_ca=args.max_ca)
        print(f"[skempi-bind] {rep['csv_rows']} csv rows -> {len(records)} "
              f"parsed; selection {reasons}", flush=True)
        print(f"[skempi-bind] {len(selected)} mutations across "
              f"{len({s['complex_id'] for s in selected})} complexes", flush=True)
        total = 0
        for m in (x.strip() for x in args.models.split(",") if x.strip()):
            sc = MPNNScorer(model_name=m, device=args.device)
            print(f"=== model {m} ===", flush=True)
            total += score_mutations(sc, selected, args.out, limit=args.limit)
            _POSE_CACHE.clear()
        print(f"[skempi-bind] {total} rows written")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
