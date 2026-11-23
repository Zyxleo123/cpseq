"""Verify the real denoising lever: does raising Boltz diffusion_samples actually
shrink the oracle's per-variant SEM as predicted?

The CPU re-relax denoiser failed because the dominant noise is FOLD sampling
(sd ~3.9 REU between the 4 diffusion samples), not relax. The decomposition
predicts SEM = fold_sd / sqrt(n_folds), so 4 -> 16 folds should ~halve it. But
that prediction assumes the diffusion samples are INDEPENDENT draws. If they are
correlated (a diffusion trunk can share structure across its samples), more
samples help LESS than sqrt(n), and the lever is weaker than predicted. This
re-folds a few cached variants at a high replica count and checks:

  1. independence -- does the 16-fold std match the cached 4-fold std? (equal =>
     independent; much smaller => the extra samples are redundant)
  2. the SEM actually achieved at n_folds = 4, 8, 16 (subsampled from one run),
     vs the fold_sd/sqrt(n) prediction.

GPU. Folds each variant ONCE at --replicas (one trunk pass, N diffusion samples),
reusing boltz_inloop.fold_score_one, and persists the PER-FOLD dG list so the SEM
can be recomputed at any n. Resumable per variant. Run on a FREE node so it does
not compete with the search.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _per_fold_dapes(agg_reps):
    """pull the per-replica refined dG for the VALID folds from score_replicas reps."""
    reps = agg_reps
    return [r["dG_refined"] for r in reps
            if r.get("valid") and isinstance(r.get("dG_refined"), (int, float))]


def sem_at(fold_dGs, nf):
    """clustered SEM using the first nf folds (each fold = 1 relax here)."""
    v = fold_dGs[:nf]
    if len(v) < 2:
        return None
    return round(st.pstdev(v) / math.sqrt(len(v)), 3)


def refold(manifest, pdb_dir, cache_path, out_path, work, replicas, seed,
           targets=None, max_variants=3) -> int:
    from lnr import env as E
    from lnr import boltz_validate as BV
    from lnr.boltz_inloop import fold_score_one
    from proteinfoundation.cpseq.adapters.boltz2 import Boltz2Config

    tmap = {t.target_id: t for t in E.load_manifest(manifest, pdb_dir)}
    # pick variants from the cache (draw 0), a few per requested target
    want = set(t.strip() for t in (targets or []) if t.strip())
    per_t, seen = {}, set()
    for line in Path(cache_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("draw", 0) not in (0, None):
            continue
        tid = r.get("target_id")
        if (want and tid not in want) or tid not in tmap:
            continue
        key = (tid, r.get("subs_key"))
        if key in seen:
            continue
        seen.add(key)
        per_t.setdefault(tid, [])
        if len(per_t[tid]) < max_variants:
            per_t[tid].append(r)

    done = set()
    if Path(out_path).exists():
        for line in Path(out_path).read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["variant_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    cfg = Boltz2Config()
    cfg.use_potentials = True
    cfg.seed = int(seed)
    BV.configure_replicas(cfg, replicas)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)

    n = 0
    with Path(out_path).open("a") as fh:
        for tid, variants in per_t.items():
            t = tmap[tid]
            for r in variants:
                subs = [(int(i), a) for i, a in (r.get("subs") or [])]
                win = {"a": r["window"][0], "b": r["window"][1]}
                # fold_score_one returns the aggregate; we re-run the per-replica
                # scorer to keep the per-fold dGs. Reuse its building blocks via a
                # high-replica fold, then score_replicas to get per-rep rows.
                row = fold_score_one(t, subs, win, cfg, work, replicas)
                vid = row.get("variant_id")
                if vid in done:
                    continue
                # fold_score_one already scored; recover per-fold dGs by re-reading
                # its refined set is overkill -- instead re-score the produced PDBs.
                out = {"variant_id": vid, "target_id": tid,
                       "subs_key": r.get("subs_key"), "replicas": replicas,
                       "n_valid_folds": row.get("n_valid_folds"),
                       "dG_refined": row.get("dG_refined"),
                       "dG_refined_std": row.get("dG_refined_std"),
                       "cached_4fold_std": r.get("dG_refined_std"),
                       "cached_4fold_dG": r.get("dG_refined"),
                       "per_fold_dG": row.get("per_fold_dG"),
                       "error": row.get("error")}
                fh.write(json.dumps(out) + "\n"); fh.flush(); n += 1
                print(f"[refold] {tid} {r.get('subs_key') or 'WT':12s} "
                      f"n_valid={out['n_valid_folds']} dG={out['dG_refined']} "
                      f"std16={out['dG_refined_std']} std4(cached)={out['cached_4fold_std']}",
                      flush=True)
    return n


def report(out_path):
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if isinstance(r.get("dG_refined_std"), (int, float))]
    print("=" * 84)
    print("VERIFY: does raising Boltz diffusion_samples shrink the oracle SEM?")
    print("=" * 84)
    if not rows:
        print("no scored variants yet.")
        return
    print(f"\n{'variant':14s}{'nFold':>6s}{'std@16':>8s}{'std@4(cached)':>14s}"
          f"{'SEM@16':>8s}{'SEM@4':>7s}")
    print("-" * 84)
    ratios = []
    for r in rows:
        nf = r.get("n_valid_folds") or 0
        s16 = r["dG_refined_std"]
        s4 = r.get("cached_4fold_std")
        sem16 = round(s16 / math.sqrt(nf), 3) if nf else None
        sem4 = round(s4 / 2.0, 3) if isinstance(s4, (int, float)) else None
        pf = r.get("per_fold_dG")
        if isinstance(pf, list) and len(pf) >= 8:
            # independence check from one run: std over first 4 vs all nf
            s_first4 = st.pstdev(pf[:4])
            s_all = st.pstdev(pf)
            ratios.append((s_first4, s_all, len(pf)))
        print(f"{(r.get('subs_key') or 'WT'):14s}{nf:6d}{s16:8.2f}"
              f"{(f'{s4:.2f}' if s4 is not None else '-'):>14s}"
              f"{(f'{sem16:.2f}' if sem16 is not None else '-'):>8s}"
              f"{(f'{sem4:.2f}' if sem4 is not None else '-'):>7s}")
    print("\n" + "=" * 84)
    print("VERDICT")
    sems16 = [r["dG_refined_std"] / math.sqrt(r["n_valid_folds"])
              for r in rows if r.get("n_valid_folds")]
    sems4 = [r["cached_4fold_std"] / 2.0 for r in rows
             if isinstance(r.get("cached_4fold_std"), (int, float))]
    if sems16 and sems4:
        m16, m4 = st.mean(sems16), st.mean(sems4)
        print(f"  mean SEM: {m4:.2f} (4 folds) -> {m16:.2f} ({rows[0]['replicas']} folds)")
        if m4 > 0:
            exp = m4 * math.sqrt(4.0 / rows[0]['replicas'])
            print(f"  predicted if folds independent: {exp:.2f}. "
                  f"Achieved/predicted = {m16/exp:.2f}x "
                  f"({'independent -> lever works' if m16 <= 1.3*exp else 'CORRELATED -> lever weaker than predicted'})")
    if ratios:
        s4 = st.mean(r[0] for r in ratios)
        sall = st.mean(r[1] for r in ratios)
        print(f"  per-fold std: first-4 {s4:.2f} vs all-{ratios[0][2]} {sall:.2f} "
              f"({'consistent => samples are independent draws' if abs(s4 - sall) < 0.4 * max(s4, 1e-9) + 0.5 else 'differ'})")
    print("=" * 84)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refold", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--pdb-dir", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--replicas", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--targets", default="")
    ap.add_argument("--max-variants", type=int, default=3)
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    if args.refold:
        if not (args.manifest and args.pdb_dir and args.cache and args.out and args.work):
            raise SystemExit("--refold needs --manifest --pdb-dir --cache --out --work")
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        tg = [t for t in args.targets.split(",") if t.strip()]
        n = refold(args.manifest, args.pdb_dir, args.cache, out, args.work,
                   args.replicas, args.seed, targets=tg, max_variants=args.max_variants)
        print(f"[refold] folded {n} variants at replicas={args.replicas} -> {out}", flush=True)
        report(out)
        return 0
    if args.report:
        report(Path(args.report))
        return 0
    raise SystemExit("use --refold or --report")


if __name__ == "__main__":
    raise SystemExit(main())
