"""Interface-confidence metrics from the Boltz output we were already writing and
throwing away -- and the retrospective test of whether they rank peptide variants
better than the Rosetta interface dG does.

Why this exists
---------------
The in-loop fitness is Boltz-fold -> Rosetta InterfaceAnalyzer dG, used as an
AFFINITY REGRESSOR. The literature says that is the hardest possible use of it:
dedicated ddG methods cap at Pearson ~0.63 against experiment (Flex ddG, and only
WITH a 20-30 model ensemble; ~0.45 on a SKEMPI v2 test split), and this project's
own proxy-vs-true check measured rho ~ +0.29. Meanwhile Overath et al. 2025
(bioRxiv 2025.08.14.670059), the largest study of what actually predicts
experimental binding -- 3,766 designs, 436 binders vs 3,330 non-binders -- found
that INTERFACE-FOCUSED CONFIDENCE scores from the structure predictor beat the
physics energies, and that the best features pair a confidence score WITH a
physicochemical descriptor (interface dG/dSASA, shape complementarity).

Every Boltz prediction here already writes `confidence_*.json` (ipTM, per-chain
and PAIRWISE per-chain ipTM, complex ipLDDT, ipDE), the full `pae_*.npz` matrix
and `plddt_*.npz` -- and `boltz_inloop.fold_score_one` reads none of it: it takes
the PDB and hands it to Rosetta. So these metrics are recoverable for every fold
already on disk, retroactively, at CPU cost only.

What this can and CANNOT answer
-------------------------------
There is NO experimental ground truth anywhere in this project. So this is NOT a
test of which metric predicts affinity. It answers the three questions the data
can actually settle:

  (a) RESOLUTION -- can the metric tell this target's variants apart at all,
      above its own measurement noise? Reported as discriminability =
      (between-variant SD) / (within-variant SEM). Below ~1 the metric cannot
      rank anything, however well it might correlate with truth in principle.
  (b) REPRODUCIBILITY -- for variants folded twice with independent seeds, how
      far does the metric move? This is the quantity that sank the v1 search
      (dG moved ~4 REU on a redraw).
  (c) AGREEMENT and DRIFT -- does the metric order variants like dG does (if not,
      it carries independent information and a composite is worth trying), and
      does it flag the mislocalised folds (held_rmsd > 8 A) that the validator
      drops but the in-loop scorer happily scored?

Honest scope limit on the metrics themselves
--------------------------------------------
This computes an ipSAE-FAMILY of interface confidence scores, NOT Dunbrack's
exact ipSAE (which renormalises d0 per residue over the PAE-passing partner set).
What is computed is defined exactly in `model_metrics` and is reproducible from
these files. If the family shows ranking signal, implementing exact ipSAE is the
follow-up; if it shows none, exact ipSAE is unlikely to rescue it.

A second scope limit that matters: ipSAE and friends were validated on
BINDER-vs-NON-BINDER discrimination of de novo designs -- a coarse, large-effect
contrast. Resolving a 2-residue substitution on an already-binding peptide is a
much finer ask, and nothing in that literature says confidence scores can do it.
That is exactly what (a) measures here.

Usage
-----
  python -m lnr.confidence_metrics --extract --work <boltz_inloop dir> \
      --cache <seqmut_iter_boltz_cache.jsonl> --out <conf_metrics.jsonl>
  python -m lnr.confidence_metrics --report <conf_metrics.jsonl>

Resumable: a (variant_id, draw) already in --out is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The metrics compared against dG. Sign convention is normalised in `report` so
# every one is "more negative = better binder", matching dG.
CONF_METRICS = ("iptm", "iptm_min", "iptm_pep_to_rec", "iptm_rec_to_pep",
                "ipae_mean", "ipae_min_dir", "lis", "iplddt", "pep_plddt",
                "ptm", "complex_plddt", "ipde")
# Metrics computed over the WHOLE complex. In this setup the receptor is 281 of
# 291 residues and is IDENTICAL across variants, so these are dominated by a
# constant: they get a tiny within-variant SEM and therefore a flattering
# discriminability ratio while ordering variants on something that has nothing to
# do with the peptide. Precision is not relevance -- they are reported but never
# recommended, and the verdict says why.
WHOLE_COMPLEX = {"ptm", "complex_plddt"}
# metrics where HIGHER is better (confidence); the rest are error-like
HIGHER_IS_BETTER = {"iptm", "iptm_min", "iptm_pep_to_rec", "iptm_rec_to_pep",
                    "lis", "iplddt", "pep_plddt", "ptm", "complex_plddt"}

LIS_PAE_CUTOFF = 12.0   # Local Interaction Score cutoff (as published)


# ---------------------------------------------------------------------------
# Per-model extraction
# ---------------------------------------------------------------------------

def chain_lengths(pdb_path) -> list:
    """[(chain_id, n_residues)] in file order, counted by CA atoms. Authoritative
    for the PAE token split: Boltz emits one token per residue for protein-only
    complexes, in the chain order of the input YAML."""
    out, seen = [], {}
    order = []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16] == " CA ":
            ch = line[21]
            if ch not in seen:
                seen[ch] = 0
                order.append(ch)
            seen[ch] += 1
    for ch in order:
        out.append((ch, seen[ch]))
    return out


def model_metrics(conf_json, pae_npz, plddt_npz, pdb_path, pep_chain="B") -> dict:
    """Every interface-confidence number this fold already contains.

    Definitions (exact, so the numbers are reproducible):
      iptm              Boltz's complex ipTM (confidence json)
      iptm_pep_to_rec   pair_chains_iptm[pep][rec]   (asymmetric)
      iptm_rec_to_pep   pair_chains_iptm[rec][pep]
      iptm_min          min of the two -- the "weakest link" summary that
                        Overath et al. found best as ipSAE_min
      ipae_mean         mean PAE over BOTH inter-chain blocks (A->B and B->A)
      ipae_min_dir      the smaller of the two per-direction block means
      lis               mean over inter-chain pairs of max(0,(12-PAE)/12)
      iplddt            complex_iplddt from the json (interface pLDDT)
      pep_plddt         mean per-residue pLDDT over the peptide chain
    """
    import numpy as np

    rec = {}
    conf = json.loads(Path(conf_json).read_text())
    rec["iptm"] = conf.get("iptm")
    rec["ptm"] = conf.get("ptm")
    rec["complex_plddt"] = conf.get("complex_plddt")
    rec["iplddt"] = conf.get("complex_iplddt")
    rec["ipde"] = conf.get("complex_ipde")

    lens = chain_lengths(pdb_path)
    rec["chains"] = [[c, n] for c, n in lens]
    # chain INDEX (0-based, file order) of peptide and receptor
    idx = {c: i for i, (c, _n) in enumerate(lens)}
    pep_i = idx.get(pep_chain)
    rec_i = next((i for c, i in idx.items() if c != pep_chain), None)
    pci = conf.get("pair_chains_iptm") or {}
    if pep_i is not None and rec_i is not None:
        a = (pci.get(str(pep_i)) or {}).get(str(rec_i))
        b = (pci.get(str(rec_i)) or {}).get(str(pep_i))
        rec["iptm_pep_to_rec"] = a
        rec["iptm_rec_to_pep"] = b
        vals = [v for v in (a, b) if isinstance(v, (int, float))]
        rec["iptm_min"] = min(vals) if vals else None

    # token boundaries for the PAE split
    starts, pos = {}, 0
    for c, n in lens:
        starts[c] = (pos, pos + n)
        pos += n
    total = pos
    pae = np.load(pae_npz)["pae"]
    rec["pae_shape"] = list(pae.shape)
    rec["token_total"] = total
    if pae.shape[0] != total:
        # token count disagrees with residue count (non-standard residues,
        # ligands): refuse to guess a split rather than report a wrong interface
        rec["pae_usable"] = False
        return rec
    rec["pae_usable"] = True
    p0, p1 = starts[pep_chain]
    mask = np.zeros(total, dtype=bool)
    mask[p0:p1] = True
    blk_pr = pae[np.ix_(mask, ~mask)]      # peptide rows, receptor cols
    blk_rp = pae[np.ix_(~mask, mask)]
    rec["ipae_mean"] = float((blk_pr.mean() + blk_rp.mean()) / 2.0)
    rec["ipae_min_dir"] = float(min(blk_pr.mean(), blk_rp.mean()))
    inter = np.concatenate([blk_pr.ravel(), blk_rp.ravel()])
    rec["lis"] = float(np.clip((LIS_PAE_CUTOFF - inter) / LIS_PAE_CUTOFF,
                               0.0, None).mean())

    if Path(plddt_npz).exists():
        pl = np.load(plddt_npz)["plddt"]
        if pl.shape[0] == total:
            rec["pep_plddt"] = float(pl[p0:p1].mean())
    return rec


def _models_in(pred_dir: Path):
    """[(k, conf_json, pae_npz, plddt_npz, pdb)] for every diffusion sample."""
    out = []
    for cj in sorted(pred_dir.glob("confidence_*_model_*.json")):
        m = re.search(r"_model_(\d+)\.json$", cj.name)
        if not m:
            continue
        k = int(m.group(1))
        stem = cj.name[len("confidence_"):-len(f"_model_{k}.json")]
        pae = pred_dir / f"pae_{stem}_model_{k}.npz"
        pl = pred_dir / f"plddt_{stem}_model_{k}.npz"
        pdb = pred_dir / f"{stem}_model_{k}.pdb"
        if pae.exists() and pdb.exists():
            out.append((k, cj, pae, pl, pdb))
    return out


def _agg(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    if not v:
        return None, None, 0
    return (st.median(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v))


def extract(work: Path, cache_path: Path, out_path: Path, pep_chain="B") -> int:
    """One row per (variant, draw): every confidence metric aggregated over the
    diffusion samples, joined to that fold's cached dG."""
    cache = {}
    for line in Path(cache_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[r.get("cache_key") or r["variant_id"]] = r

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["cache_key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    n = 0
    with out_path.open("a") as fh:
        for d in sorted(Path(work).glob("heldout_*")):
            key = d.name[len("heldout_"):]
            if key in done:
                continue
            # Only read folds the search has FINISHED. The cache row is appended
            # after fold_score_one returns, so its presence is the completion
            # marker. Without this, a fold being written right now by the running
            # search would be recorded half-done (fewer diffusion samples) and
            # then skipped forever on resume, silently biasing its SD downward.
            if key not in cache:
                continue
            preds = list(d.glob("boltz_results_*/predictions/*"))
            pred_dir = next((p for p in preds if p.is_dir()), None)
            if pred_dir is None:
                continue
            models = _models_in(pred_dir)
            if not models:
                continue
            per = []
            for k, cj, pae, pl, pdb in models:
                try:
                    mm = model_metrics(cj, pae, pl, pdb, pep_chain=pep_chain)
                    mm["model"] = k
                    per.append(mm)
                except Exception as exc:  # noqa: BLE001
                    per.append({"model": k, "error": f"{type(exc).__name__}: {exc}"[:200]})
            cr = cache.get(key, {})
            row = {"cache_key": key, "variant_id": cr.get("variant_id"),
                   "target_id": cr.get("target_id"), "subs_key": cr.get("subs_key"),
                   "draw": cr.get("draw", 0), "window": cr.get("window"),
                   "dG_refined": cr.get("dG_refined"),
                   "dG_refined_std": cr.get("dG_refined_std"),
                   "n_valid_folds": cr.get("n_valid_folds"),
                   "held_rmsd": cr.get("held_rmsd"),
                   "n_models": len(per), "per_model": per}
            for mname in CONF_METRICS:
                med, sd, cnt = _agg([p.get(mname) for p in per])
                row[mname] = med
                row[f"{mname}_sd"] = sd
                row[f"{mname}_n"] = cnt
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# The retrospective comparison
# ---------------------------------------------------------------------------

def _oriented(name, v):
    """Flip so that MORE NEGATIVE = better binder, matching dG."""
    if v is None:
        return None
    return -v if name in HIGHER_IS_BETTER else v


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def report(rows_path: Path, min_n: int = 5):
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    base = [r for r in rows if r.get("draw", 0) == 0]
    targets = sorted({r["target_id"] for r in base if r.get("target_id")})
    metrics = ["dG_refined"] + list(CONF_METRICS)

    print("=" * 100)
    print("INTERFACE CONFIDENCE vs ROSETTA dG -- retrospective, on folds already computed")
    print(f"{len(rows)} fold-records ({len(base)} at draw 0), targets: {', '.join(targets)}")
    print("NOTE: no experimental ground truth exists here. This measures RESOLUTION,")
    print("REPRODUCIBILITY and AGREEMENT -- not which metric predicts real affinity.")
    print("=" * 100)

    # ---- (a) RESOLUTION -------------------------------------------------
    print("\n(a) RESOLUTION -- can the metric tell this target's variants apart?")
    print("    discrim = (SD across variants) / (mean within-variant SEM over diffusion")
    print("    samples). Below ~1 the metric cannot rank variants at all.\n")
    hdr = f"    {'metric':18s}" + "".join(f"{t:>12s}" for t in targets) + f"{'mean':>9s}"
    print(hdr)
    print("    " + "-" * (18 + 12 * len(targets) + 9))
    disc_table = {}
    for mname in metrics:
        cells, vals = [], []
        for t in targets:
            rs = [r for r in base if r["target_id"] == t
                  and isinstance(r.get(mname), (int, float))]
            if len(rs) < min_n:
                cells.append(f"{'-':>12s}")
                continue
            sig = st.stdev([r[mname] for r in rs])
            sems = []
            for r in rs:
                sd = r.get(f"{mname}_sd") if mname != "dG_refined" else r.get("dG_refined_std")
                nn = r.get(f"{mname}_n") if mname != "dG_refined" else r.get("n_valid_folds")
                if isinstance(sd, (int, float)) and isinstance(nn, int) and nn > 0:
                    sems.append(sd / math.sqrt(nn))
            if not sems or st.mean(sems) == 0:
                cells.append(f"{'inf':>12s}")
                continue
            d = sig / st.mean(sems)
            vals.append(d)
            cells.append(f"{d:12.2f}")
        mean = f"{st.mean(vals):9.2f}" if vals else f"{'-':>9s}"
        disc_table[mname] = st.mean(vals) if vals else None
        star = "  <<" if mname == "dG_refined" else ""
        print(f"    {mname:18s}" + "".join(cells) + mean + star)

    # ---- (b) REPRODUCIBILITY -------------------------------------------
    pairs = {}
    for r in rows:
        pairs.setdefault((r["target_id"], r.get("subs_key")), []).append(r)
    rep = {k: v for k, v in pairs.items() if len({x.get("draw", 0) for x in v}) > 1}
    repro = {}
    print(f"\n(b) REPRODUCIBILITY -- variants folded twice with INDEPENDENT seeds "
          f"(n={len(rep)})")
    if len(rep) < 3:
        print(f"    *** n={len(rep)}: ANECDOTAL, not a measurement. The v2 run's")
        print("    re-measurement draws are still landing; re-run --report as they do.")
    if not rep:
        print("    none yet.")
    else:
        print(f"    {'metric':18s}{'mean |d|':>10s}{'as % of variant spread':>26s}")
        print("    " + "-" * 54)
        for mname in metrics:
            ds, spreads = [], []
            for (t, _sk), v in rep.items():
                byd = {}
                for x in v:
                    if isinstance(x.get(mname), (int, float)):
                        byd[x.get("draw", 0)] = x[mname]
                if len(byd) > 1:
                    o = sorted(byd.items())
                    ds.append(abs(o[0][1] - o[-1][1]))
                    pool = [r[mname] for r in base if r["target_id"] == t
                            and isinstance(r.get(mname), (int, float))]
                    if len(pool) > 2:
                        spreads.append(st.stdev(pool))
            if ds:
                pct = (100 * st.mean(ds) / st.mean(spreads)) if spreads else float("nan")
                repro[mname] = pct
                print(f"    {mname:18s}{st.mean(ds):10.3f}{pct:25.0f}%")
        print("    (>100% = one independent redraw moves the metric further than the")
        print("     entire spread between variants, i.e. it cannot rank them at all)")

    # ---- (c) AGREEMENT + DRIFT -----------------------------------------
    print("\n(c) AGREEMENT with dG, and DRIFT detection (held_rmsd), within target.")
    print("    rho(dG) near 0 => the metric carries INDEPENDENT information.")
    print("    rho(rmsd) positive => the metric correctly penalises mislocalised folds.\n")
    print(f"    {'metric':18s}" + "".join(f"{t + ' rho(dG)':>16s}" for t in targets)
          + f"{'rho(rmsd) all':>15s}")
    print("    " + "-" * (18 + 16 * len(targets) + 15))
    for mname in CONF_METRICS:
        cells = []
        for t in targets:
            rs = [r for r in base if r["target_id"] == t
                  and isinstance(r.get(mname), (int, float))
                  and isinstance(r.get("dG_refined"), (int, float))]
            if len(rs) < min_n:
                cells.append(f"{'-':>16s}")
                continue
            rho = _spearman([_oriented(mname, r[mname]) for r in rs],
                            [r["dG_refined"] for r in rs])
            cells.append(f"{rho:16.2f}" if rho is not None else f"{'-':>16s}")
        rs = [r for r in base if isinstance(r.get(mname), (int, float))
              and isinstance(r.get("held_rmsd"), (int, float))]
        rr = _spearman([_oriented(mname, r[mname]) for r in rs],
                       [r["held_rmsd"] for r in rs]) if len(rs) >= min_n else None
        print(f"    {mname:18s}" + "".join(cells)
              + (f"{rr:15.2f}" if rr is not None else f"{'-':>15s}"))
    rs = [r for r in base if isinstance(r.get("dG_refined"), (int, float))
          and isinstance(r.get("held_rmsd"), (int, float))]
    rr = _spearman([r["dG_refined"] for r in rs], [r["held_rmsd"] for r in rs])
    print(f"    {'dG_refined':18s}" + "".join(f"{'(self)':>16s}" for _ in targets)
          + (f"{rr:15.2f}" if rr is not None else f"{'-':>15s}") + "  <<")

    # ---- verdict --------------------------------------------------------
    # A usable fitness needs THREE things, not one. Resolution alone is a trap:
    # a metric dominated by the invariant receptor has almost no noise and so a
    # flattering discriminability ratio, while ordering variants on nothing.
    dg = disc_table.get("dG_refined")
    dg_rep = repro.get("dG_refined")
    print("\n" + "=" * 100)
    if dg is None:
        print("VERDICT: not enough data to compare resolution.")
        print("=" * 100)
        return

    rows_v = []
    for m, d in disc_table.items():
        if m == "dG_refined" or d is None:
            continue
        res_ok = d > dg
        whole = m in WHOLE_COMPLEX
        r = repro.get(m)
        rep_ok = (r is not None and dg_rep is not None and r <= dg_rep)
        rows_v.append((m, d, res_ok, whole, r, rep_ok))

    passing = [x for x in rows_v if x[2] and not x[3] and x[5]]
    res_only = [x for x in rows_v if x[2] and not x[3] and not x[5]]
    artifact = [x for x in rows_v if x[2] and x[3]]

    print(f"VERDICT  (dG baseline: discrim {dg:.2f}"
          + (f", redraw moves it {dg_rep:.0f}% of the variant spread)" if dg_rep
             else ", reproducibility not yet measurable)"))
    if artifact:
        print("\n  DISCOUNTED as artifacts -- whole-complex metrics. The receptor is")
        print("  281 of 291 residues and identical in every variant, so these are")
        print("  mostly measuring a constant: tiny noise, flattering ratio, ordering")
        print("  unrelated to the peptide. Check their rho(dG)/rho(rmsd) above.")
        for m, d, _ro, _w, _r, _rk in sorted(artifact, key=lambda x: -x[1]):
            print(f"    {m:18s} discrim {d:5.2f}  -- receptor-dominated, ignore")
    if passing:
        print("\n  CANDIDATES -- resolve variants better than dG AND survive a redraw:")
        for m, d, _ro, _w, r, _rk in sorted(passing, key=lambda x: -x[1]):
            print(f"    {m:18s} discrim {d:5.2f} ({d/dg:.1f}x dG), redraw {r:.0f}%")
        print("  Worth testing as in-loop fitness, or as a composite with dG/dSASA")
        print("  (the pairing Overath et al. found best). Costs no extra GPU.")
    if res_only:
        print("\n  RESOLVE but DO NOT REPRODUCE -- precise within a run, unstable across")
        print("  independent fold seeds, so they cannot be selected on:")
        for m, d, _ro, _w, r, _rk in sorted(res_only, key=lambda x: -x[1]):
            rs = f"{r:.0f}%" if r is not None else "n/a"
            print(f"    {m:18s} discrim {d:5.2f}, redraw {rs} of the variant spread")
    if not passing:
        print("\n  BOTTOM LINE: no interface-confidence metric is a usable replacement")
        print("  for dG at THIS contrast (single substitutions on a peptide that")
        print("  already binds). That is consistent with where these scores were")
        print("  validated -- binder vs non-binder for de novo designs, a far coarser")
        print("  contrast. The lever stays ensemble averaging of the energy.")
    print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--work", default="", help="boltz_inloop dir with heldout_* subdirs")
    ap.add_argument("--cache", default="", help="seqmut_iter_boltz_cache.jsonl")
    ap.add_argument("--out", default="", help="conf_metrics.jsonl (resumable)")
    ap.add_argument("--pep-chain", default="B")
    ap.add_argument("--report", default="", help="report on an existing rows file")
    ap.add_argument("--min-n", type=int, default=5,
                    help="minimum variants per target to report a statistic")
    args = ap.parse_args()

    if args.extract:
        if not (args.work and args.cache and args.out):
            raise SystemExit("--extract needs --work, --cache and --out")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = extract(Path(args.work), Path(args.cache), out, pep_chain=args.pep_chain)
        print(f"[conf] extracted {n} new fold-records -> {out}", flush=True)
        report(out, min_n=args.min_n)
        return 0
    if args.report:
        report(Path(args.report), min_n=args.min_n)
        return 0
    raise SystemExit("use --extract or --report")


if __name__ == "__main__":
    raise SystemExit(main())
