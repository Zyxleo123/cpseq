"""CPU re-relax denoiser: shrink the oracle's per-variant noise by re-scoring the
ALREADY-FOLDED Boltz structures R times, with no new GPU fold.

The idea (why this is free)
---------------------------
The ±3.4 REU spread on a per-variant dG is dominated by Rosetta FastRelax
stochasticity, not Boltz: the project measured the SAME Boltz structure (fixed
seed, identical coords) score -26.95 vs -24.28 on two independent relaxes. Boltz
is deterministic at a fixed seed; the scorer's `pyrosetta.init` sets no
`-constant_seed`, so every `score_structure` call is an INDEPENDENT relax draw.
So the dominant noise can be averaged down on CPU alone: re-relax the N cached
diffusion structures R times each and aggregate over all N x R valid draws. The
SEM of the median falls like 1/sqrt(N x R) on the Rosetta component.

Why it matters (what it unblocks)
---------------------------------
Nothing could be ranked -- not physics, not confidence, not sequence LMs --
because the ground-truth oracle's own SEM (~1.7 REU) was as large as the
between-variant differences (a few REU). A ranker can only be as good as the ruler
grading it. This denoises the ruler: it reports, per target, the within-variant
SEM and the between-variant RESOLUTION (between-variant SD / within SEM) BEFORE
(the cached N-fold estimate) and AFTER (N x R relaxes), so the payoff is the
resolution crossing ~1 -- the point at which the oracle can actually tell variants
apart.

Scope / safety
--------------
Reads the cached fold PDBs read-only and writes its own refined PDBs under a
dedicated dir, so it is safe to run WHILE the GPU search writes new folds. CPU
only, parallel across worker processes (each initialises PyRosetta once, then its
consecutive relaxes are independent draws). Resumable per (variant, model, repeat).

Usage
-----
  python -m lnr.rescore_denoise --score --work <boltz_inloop> \
      --cache <seqmut_iter_boltz_cache.jsonl> --out <denoise_scores.jsonl> \
      --refined-dir <scratch/denoise_refined> --repeats 6 --workers 8 \
      [--targets 1bjr,1elr] [--max-variants 12]
  python -m lnr.rescore_denoise --report <denoise_scores.jsonl> [--cache <...>]
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


# ---------------------------------------------------------------------------
# Pure aggregation / resolution math (unit-tested, no Rosetta)
# ---------------------------------------------------------------------------

def aggregate(vals) -> dict:
    """median, std, and naive (iid) SEM of a flat set of dG draws. NOTE: this
    treats every draw as independent. For the denoiser's own draws that is WRONG
    (they are clustered by fold); use `aggregate_clustered`. Kept for the simple
    case and the tests."""
    v = [float(x) for x in vals if isinstance(x, (int, float)) and x == x]
    if not v:
        return {"n": 0, "median": None, "std": None, "sem": None}
    sd = st.pstdev(v) if len(v) > 1 else 0.0
    return {"n": len(v), "median": round(st.median(v), 3),
            "std": round(sd, 3), "sem": round(sd / math.sqrt(len(v)), 3)}


def aggregate_clustered(by_model: dict) -> dict:
    """The CORRECT per-variant estimate when draws are R relaxes of each of N folds.

    The estimand is the variant's true dG. Its two noise sources are NOT
    exchangeable: re-relaxing a fold reduces the RELAX component, but the FOLD
    component (coordinate differences between diffusion samples) is fixed by how
    many folds exist, and re-relaxing does nothing to it. So:
      relax_sd   mean within-fold sd over relaxes (the CPU-reducible part)
      fold_sd    sd of the per-fold means (the GPU-only part)
      sem        sd(fold means) / sqrt(n_folds)  -- the honest SEM of the estimate
    Pooling all N*R draws and dividing by sqrt(N*R) (what `aggregate` does) treats
    correlated draws as independent and FABRICATES precision; that bug made an
    earlier run look like it halved the SEM when it did not."""
    means, relax_sds, all_draws = [], [], []
    for _m, vals in by_model.items():
        v = [float(x) for x in vals if isinstance(x, (int, float)) and x == x]
        if not v:
            continue
        means.append(st.mean(v))
        all_draws.extend(v)
        if len(v) > 1:
            relax_sds.append(st.pstdev(v))
    nf = len(means)
    if nf == 0:
        return {"median": None, "sem": None, "relax_sd": None, "fold_sd": None,
                "n_folds": 0, "n_draws": 0}
    fold_sd = st.pstdev(means) if nf > 1 else 0.0
    return {"median": round(st.median(means), 3),
            "sem": round(fold_sd / math.sqrt(nf), 3),
            "relax_sd": round(st.mean(relax_sds), 3) if relax_sds else 0.0,
            "fold_sd": round(fold_sd, 3),
            "n_folds": nf, "n_draws": len(all_draws)}


def resolution(variant_estimates) -> dict:
    """variant_estimates: list of {"median": dG, "sem": s}. Returns the between-
    variant SD, the mean within-variant SEM, and their ratio -- the discriminability
    of the oracle. Ratio >~1 means the oracle separates variants above its noise."""
    meds = [e["median"] for e in variant_estimates
            if isinstance(e.get("median"), (int, float))]
    sems = [e["sem"] for e in variant_estimates
            if isinstance(e.get("sem"), (int, float))]
    if len(meds) < 2 or not sems:
        return {"between_sd": None, "within_sem": None, "discrim": None, "n": len(meds)}
    between = st.pstdev(meds)
    within = st.mean(sems)
    return {"between_sd": round(between, 3), "within_sem": round(within, 3),
            "discrim": round(between / within, 3) if within else None,
            "n": len(meds)}


def denoised_vs_cached(rows, cache_by_vid) -> dict:
    """Per target: resolution from the DENOISED draws vs the cached N-fold estimate.
    `rows` are the per-(variant,model,repeat) score records this tool wrote;
    `cache_by_vid` maps variant_id -> the original cache row."""
    # group draws by variant AND fold-model, so the SEM is computed correctly
    by_var = {}
    for r in rows:
        if r.get("valid") and isinstance(r.get("dG_refined"), (int, float)):
            by_var.setdefault(r["variant_id"], {}).setdefault(r["model"], []).append(
                r["dG_refined"])
    by_target = {}
    for vid, by_model in by_var.items():
        c = cache_by_vid.get(vid, {})
        tid = c.get("target_id") or vid.split("_")[0]
        agg = aggregate_clustered(by_model)
        # cached estimate: median over the N diffusion folds (1 relax each). Its
        # SEM = sd over folds / sqrt(n_folds) -- the SAME clustered form, at R=1,
        # so it is a fair before/after.
        csem = None
        cstd, cn = c.get("dG_refined_std"), c.get("n_valid_folds")
        if isinstance(cstd, (int, float)) and isinstance(cn, int) and cn > 0:
            csem = cstd / math.sqrt(cn)
        by_target.setdefault(tid, []).append({
            "variant_id": vid,
            "denoised": {"median": agg["median"], "sem": agg["sem"],
                         "n": agg["n_draws"], "relax_sd": agg["relax_sd"],
                         "fold_sd": agg["fold_sd"], "n_folds": agg["n_folds"]},
            "cached": {"median": c.get("dG_refined"),
                       "sem": round(csem, 3) if csem is not None else None},
        })
    out = {}
    for tid, variants in by_target.items():
        den = resolution([v["denoised"] for v in variants])
        cac = resolution([{"median": v["cached"]["median"], "sem": v["cached"]["sem"]}
                          for v in variants])
        out[tid] = {"n_variants": len(variants), "cached": cac, "denoised": den,
                    "variants": variants}
    return out


# ---------------------------------------------------------------------------
# Discovery + scoring driver
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict:
    by_vid = {}
    for line in Path(cache_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = r.get("variant_id")
        # keep the primary (draw 0 / legacy None), not re-measurement draws
        if vid and r.get("draw", 0) in (0, None) and vid not in by_vid:
            by_vid[vid] = r
    return by_vid


def _model_pdbs(work: Path, vid: str) -> list:
    d = Path(work) / f"heldout_{vid}"
    if not d.is_dir():
        return []
    preds = list(d.glob("boltz_results_*/predictions/*"))
    pred = next((p for p in preds if p.is_dir()), None)
    if pred is None:
        return []
    return sorted(pred.glob("*_model_*.pdb"))


def _score_task(args):
    """Worker: one independent relax draw of one PDB. PyRosetta initialises once
    per worker process (module global), so the first task in a worker pays init and
    the rest are cheap; consecutive relaxes in a worker advance the global RNG and
    are therefore independent draws."""
    vid, model_k, rep, pdb, peptide, refined_path = args
    rec = {"variant_id": vid, "model": model_k, "repeat": rep, "pdb": pdb}
    try:
        from proteinfoundation.cpseq.scorer import score_structure
        sc = score_structure(pdb, binder_chain="B", receptor_chains=["A"],
                             chemistry="mainchain", requested_sequence=peptide,
                             refined_out_path=refined_path)
        rec["score_status"] = sc.get("status")
        rec["sequence_matches"] = sc.get("sequence_matches")
        rec["ring_declared"] = (sc.get("ring_declared_complex") or {}).get("declared")
        rec["dG_refined"] = (sc.get("refined") or {}).get("dG_separated")
        rec["valid"] = (rec["score_status"] == "ok"
                        and rec["sequence_matches"] is True
                        and rec["ring_declared"] is True
                        and rec["dG_refined"] is not None)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["valid"] = False
    return rec


def score(work: Path, cache_path: Path, out_path: Path, refined_dir: Path,
          repeats: int, workers: int, targets=None, max_variants: int = 0,
          _scorer=None) -> int:
    """Build the (variant, model, repeat) task list for the discovered variants and
    score them, resumable. `_scorer` overrides the worker fn (for tests)."""
    import concurrent.futures as cf

    cache = _load_cache(cache_path)
    refined_dir = Path(refined_dir)
    refined_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["variant_id"], r["model"], r["repeat"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    want = set(t.strip() for t in (targets or []) if t.strip())
    # group variants by target so --max-variants is per target, not global
    per_target = {}
    for vid, c in cache.items():
        tid = c.get("target_id") or vid.split("_")[0]
        if want and tid not in want:
            continue
        per_target.setdefault(tid, []).append((vid, c))

    tasks = []
    n_selected = 0
    for tid, variants in per_target.items():
        variants.sort(key=lambda vc: vc[0])
        if max_variants:
            variants = variants[:max_variants]
        n_selected += len(variants)
        for vid, c in variants:
            pep = c.get("peptide_sequence")
            if not pep:
                continue
            pdbs = _model_pdbs(work, vid)
            for k, pdb in enumerate(pdbs):
                for rep in range(repeats):
                    if (vid, k, rep) in done:
                        continue
                    rp = str(refined_dir / f"den_{vid}_m{k}_r{rep}.pdb")
                    tasks.append((vid, k, rep, str(pdb), pep, rp))

    print(f"[denoise] {len(cache)} variants in cache, "
          f"{n_selected} selected (<= {max_variants or 'all'}/target), "
          f"{len(tasks)} relax draws to run ({len(done)} already done)", flush=True)
    if not tasks:
        return 0

    scorer = _scorer or _score_task
    n = 0
    with out_path.open("a") as fh:
        if workers > 1 and _scorer is None:
            with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                for rec in ex.map(scorer, tasks, chunksize=1):
                    fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
                    if n % 25 == 0:
                        print(f"[denoise] {n}/{len(tasks)} draws done", flush=True)
        else:
            for t in tasks:
                fh.write(json.dumps(scorer(t)) + "\n"); fh.flush(); n += 1
    return n


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(rows_path: Path, cache_path: Path = None):
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    cache = _load_cache(Path(cache_path)) if cache_path else {}
    res = denoised_vs_cached(rows, cache)

    print("=" * 92)
    print("CPU RE-RELAX DENOISER -- does averaging more relaxes let the oracle")
    print("resolve variants? Compares the cached N-fold estimate to N x R relaxes.")
    print("=" * 92)

    # THE decomposition: where does the per-variant noise actually live?
    relax_sds = [v["denoised"]["relax_sd"] for d in res.values() for v in d["variants"]
                 if isinstance(v["denoised"].get("relax_sd"), (int, float))]
    fold_sds = [v["denoised"]["fold_sd"] for d in res.values() for v in d["variants"]
                if isinstance(v["denoised"].get("fold_sd"), (int, float))]
    relax = st.mean(relax_sds) if relax_sds else 0.0
    fold = st.mean(fold_sds) if fold_sds else 0.0
    print(f"\nWHERE THE NOISE LIVES (per-variant, mean over variants):")
    print(f"  relax noise (within one fold, re-relaxed)  : {relax:5.2f} REU  "
          f"<- what CPU re-relax reduces")
    print(f"  fold noise  (between the diffusion samples) : {fold:5.2f} REU  "
          f"<- GPU-only (more Boltz samples)")
    if fold and relax:
        print(f"  fold / relax = {fold/relax:.1f}x : the dominant source is "
              f"{'FOLD sampling' if fold > relax else 'the relax'}.")

    print(f"\nSEM is computed CLUSTERED (sd of fold means / sqrt(n_folds)) -- NOT by")
    print(f"pooling all relax draws, which would fake a sqrt(N*R) shrinkage that the")
    print(f"correlated draws do not earn.")
    print(f"\n{'target':8s}{'nVar':>5s}{'relaxes/var':>12s}  "
          f"{'cached SEM':>11s}{'denoised SEM':>13s}  "
          f"{'cached discrim':>15s}{'denoised discrim':>17s}")
    print("-" * 92)
    for tid, d in sorted(res.items()):
        nrel = [v["denoised"]["n"] for v in d["variants"] if v["denoised"]["n"]]
        mean_rel = round(st.mean(nrel), 1) if nrel else 0
        csem = d["cached"]["within_sem"]
        dsem = d["denoised"]["within_sem"]
        cdis = d["cached"]["discrim"]
        ddis = d["denoised"]["discrim"]
        print(f"{tid:8s}{d['n_variants']:5d}{mean_rel:12.1f}  "
              f"{(f'{csem:.2f}' if csem is not None else '-'):>11s}"
              f"{(f'{dsem:.2f}' if dsem is not None else '-'):>13s}  "
              f"{(f'{cdis:.2f}' if cdis is not None else '-'):>15s}"
              f"{(f'{ddis:.2f}' if ddis is not None else '-'):>17s}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("  discrim = between-variant SD / within-variant SEM. >~1 = the oracle can")
    print("  tell variants apart above its own noise (the prerequisite for ranking).")
    # did the HONEST (clustered) SEM actually improve?
    sem_gain = []
    for tid, d in res.items():
        c, dn = d["cached"]["within_sem"], d["denoised"]["within_sem"]
        if c is not None and dn is not None and c > 0:
            sem_gain.append(1.0 - dn / c)
    mean_gain = st.mean(sem_gain) if sem_gain else 0.0
    print()
    if fold > 1.5 * relax and mean_gain < 0.15:
        print(f"  CPU re-relax does NOT help: it cut the honest SEM by only "
              f"{100*mean_gain:.0f}%.")
        print(f"  The noise is dominated by FOLD sampling ({fold:.1f} vs {relax:.1f} REU "
              f"relax), which re-relaxing the SAME structures cannot reduce. An earlier")
        print(f"  run reported a halved SEM -- that was a pooling artifact (treating")
        print(f"  {res[list(res)[0]]['variants'][0]['denoised']['n']} correlated draws as independent), now fixed.")
        print(f"\n  THE REAL LEVER: more Boltz DIFFUSION SAMPLES (GPU). SEM scales as")
        print(f"  fold_sd / sqrt(n_folds), so going 4 -> 16 folds ~halves it; the pipeline")
        print(f"  already supports it (configure_replicas / --boltz-replicas, chunkfix")
        print(f"  shadow for >4). CPU re-relax past ~2/fold is wasted.")
    elif mean_gain >= 0.15:
        print(f"  CPU re-relax cut the honest (clustered) SEM by {100*mean_gain:.0f}% -- "
              f"worth keeping.")
        print(f"  Re-benchmark the cheap rankers against this denoised oracle.")
    else:
        print(f"  Inconclusive: SEM gain {100*mean_gain:.0f}%, fold {fold:.1f} vs relax "
              f"{relax:.1f} REU.")
    print("=" * 92)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--work", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--refined-dir", default="")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--targets", default="")
    ap.add_argument("--max-variants", type=int, default=0,
                    help="cap variants PER TARGET (0 = all)")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if args.score:
        if not (args.work and args.cache and args.out and args.refined_dir):
            raise SystemExit("--score needs --work --cache --out --refined-dir")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tg = [t for t in args.targets.split(",") if t.strip()]
        n = score(Path(args.work), Path(args.cache), out, Path(args.refined_dir),
                  repeats=args.repeats, workers=args.workers, targets=tg,
                  max_variants=args.max_variants)
        print(f"[denoise] wrote {n} new relax draws -> {out}", flush=True)
        report(out, Path(args.cache))
        return 0
    if args.report:
        report(Path(args.report), Path(args.cache) if args.cache else None)
        return 0
    raise SystemExit("use --score or --report")


if __name__ == "__main__":
    raise SystemExit(main())
