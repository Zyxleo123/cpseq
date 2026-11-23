"""Grade ANY candidate binding metric on the CPSet within-target instrument.

This is the referee, kept separate from the thing being judged so that a new
metric cannot arrive with its own favourable statistics. Every number here is
computed by code IMPORTED from `cpset_accuracy` -- `within_series_pairs`,
`wilson95`, `pair_agreement`, `verdict`, `_spearman` -- so a metric graded here
is directly comparable to the Rosetta dG result it is trying to beat. A second
copy of Spearman would silently diverge and the two gates could no longer be
read side by side.

Four things it insists on, each because this project has been burned by its
absence:

1. ORIENTATION IS DECLARED, NEVER INFERRED. Every metric carries
   `lower_is_better` in `bindscore.METRICS`; values are flipped to the
   lower-is-stronger convention here, once, so `pair_agreement` is used
   unmodified and a sign slip cannot masquerade as a result.

2. THE SIZE NULL IS MANDATORY. `-peptide_length` scores cross-complex rho
   -0.376 on this benchmark, beating both Rosetta dG (-0.225) and AutoDock
   CrankPep (-0.316). Any cross-complex correlation is therefore printed with
   the partial correlation controlling for peptide length beside it, and the
   null scorers are printed in the same table as the candidate.

3. SELF-TEMPLATE AND CROSS-TEMPLATE ARE NEVER POOLED. Scoring a variant on its
   own crystal leaks: that backbone was solved for that sequence. `xtemplate`
   (mean over OTHER templates) is the headline; `self` is reported beside it and
   a large self-over-xtemplate margin is a leak warning, not a win.

4. NOISE IS MEASURED, NOT CLAIMED. Discriminability (between-variant spread /
   within-variant SEM over templates) is reported per metric, because a metric
   that needs many draws to separate two peptides is not usable in a loop --
   and because a flattering ratio can be bought with an invariant, the way
   `complex_plddt` bought one from a receptor that never changes.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys

import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr.cpset_accuracy import (  # noqa: E402
    TIER_OF, HEADLINE_BIN, DPKD_BINS, PAPER_REF,
    load_dataset1, within_series_pairs, pair_agreement, wilson95,
)
from lnr.skempi_accuracy import _spearman, _pearson  # noqa: E402


def _metrics_table():
    from lnr.bindscore import METRICS
    return METRICS


# ---------------------------------------------------------------------------
# Loading / aggregation
# ---------------------------------------------------------------------------
def load_rows(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def aggregate(rows, mode="xtemplate", model=None):
    """{metric: {pdb: value}} plus {metric: {pdb: [per-template values]}}.

    mode='xtemplate' averages over templates OTHER than the variant's own
    crystal -- the deployable, leak-free number. mode='self' uses only the
    variant's own crystal. mode='all' pools both and is provided for diagnosis
    only; it is never the headline.
    """
    per = {}
    for r in rows:
        if model and r.get("model") != model:
            continue
        if mode == "xtemplate" and r.get("is_self"):
            continue
        if mode == "self" and not r.get("is_self"):
            continue
        for k, v in (r.get("metrics") or {}).items():
            if k.startswith("_") or v is None:
                continue
            per.setdefault(k, {}).setdefault(r["pdb"], []).append(float(v))
    means = {k: {p: st.mean(vs) for p, vs in d.items()} for k, d in per.items()}
    return means, per


def orient(values, metric):
    """Flip to the lower-is-stronger convention used by `pair_agreement`."""
    tbl = _metrics_table()
    lower = (tbl.get(metric) or {}).get("lower_is_better", True)
    return values if lower else {k: -v for k, v in values.items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def partial_spearman(x, y, z):
    """Spearman of x,y controlling z. The size control this project requires on
    every cross-complex number: dG's apparent rho -0.225 against measured pKd
    collapses to -0.002 once peptide length is partialled out."""
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    if None in (rxy, rxz, ryz):
        return None
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den


def series_rhos(values, truth, tiers=None):
    """Per-series Spearman(oriented value, pKd). NEVER pooled across series:
    pooling imports the cross-target pKd range (3.55-10.29, ~9 kcal/mol) and
    inflates it into fake within-target skill."""
    by_series = {}
    for pdb, v in values.items():
        rec = truth.get(pdb)
        if not rec or not rec.get("series"):
            continue
        if tiers and TIER_OF.get(rec["series"]) not in tiers:
            continue
        by_series.setdefault(rec["series"], []).append((v, rec["pkd"]))
    out = {}
    for sid, pairs in sorted(by_series.items()):
        if len(pairs) < 2:
            continue
        out[sid] = {
            "n": len(pairs),
            "rho": _spearman([p[0] for p in pairs], [p[1] for p in pairs]),
        }
    return out


def discriminability(per_pdb_lists, values, truth, tiers=None):
    """between-variant spread / within-variant SEM over templates, per series.

    The quantity that decides whether a metric can be used in a loop without
    being run many times. Reported per series, because a ratio pooled across
    targets is bought by the cross-target range."""
    by_series = {}
    for pdb, vs in per_pdb_lists.items():
        rec = truth.get(pdb)
        if not rec or not rec.get("series"):
            continue
        if tiers and TIER_OF.get(rec["series"]) not in tiers:
            continue
        by_series.setdefault(rec["series"], []).append((pdb, vs))
    out = {}
    for sid, items in sorted(by_series.items()):
        means = [st.mean(vs) for _, vs in items if vs]
        sems = []
        for _, vs in items:
            if len(vs) >= 2:
                sems.append(st.stdev(vs) / math.sqrt(len(vs)))
        if len(means) < 2:
            continue
        between = st.stdev(means)
        within = st.mean(sems) if sems else 0.0
        out[sid] = {
            "between_sd": between,
            "within_sem": within,
            "ratio": (between / within) if within > 1e-12 else float("inf"),
            "n_variants": len(means),
        }
    return out


def permutation_test(values, truth, tiers=None, metric="?", min_dpkd=HEADLINE_BIN,
                     n_perm=20000, seed=0):
    """Within-series permutation null for the pooled pair agreement.

    WHY THIS EXISTS. The pooled pair count badly overstates the evidence: 13
    "independent pairs" in the BB/MIX tier come from only two informative
    series, so the pairs are strongly dependent and a Wilson CI computed on them
    is overconfident. The first run caught this in the act -- `null_random`, a
    single draw of pure noise, scored 11/13 = 0.846 on that tier, which a
    binomial would call a 1-in-90 event.

    The honest null permutes the metric's values WITHIN each series (leaving the
    series sizes, the pKd values, the |dpKd| stratification and the tie
    structure exactly as they are) and recomputes the same pooled statistic. Its
    spread is the real chance level.

    Returns the observed agreement, the null mean, the one-sided p-value, and
    the null's 95th percentile -- which is the threshold a metric actually has
    to clear.
    """
    o = orient(values, metric)
    pool = set(o)
    if tiers:
        pool = {p for p in pool
                if TIER_OF.get((truth.get(p) or {}).get("series")) in tiers}
    pairs = within_series_pairs(truth, min_dpkd=min_dpkd, pdbs=pool)
    obs = pair_agreement(pairs, o)
    if not obs["n_pairs"]:
        return {"observed": None, "p": None, "null_mean": None, "null_p95": None,
                "n_perm": 0, "n_series": 0}

    by_series = {}
    for pdb in pool:
        rec = truth.get(pdb)
        if rec and rec.get("series"):
            by_series.setdefault(rec["series"], []).append(pdb)
    members = {s: sorted(m) for s, m in by_series.items() if len(m) >= 2}

    rng = np.random.default_rng(seed)
    hits = 0
    draws = np.empty(n_perm)
    for t in range(n_perm):
        shuffled = {}
        for sid, mem in members.items():
            vals = [o[m] for m in mem]
            perm = rng.permutation(len(vals))
            for m, j in zip(mem, perm):
                shuffled[m] = vals[j]
        a = pair_agreement(pairs, shuffled)["agreement"]
        a = 0.0 if a is None else a
        draws[t] = a
        if a >= obs["agreement"]:
            hits += 1
    return {
        "observed": obs["agreement"],
        "n_pairs": obs["n_pairs"],
        "n_series": len(members),
        "null_mean": float(draws.mean()),
        "null_sd": float(draws.std()),
        "null_p95": float(np.percentile(draws, 95)),
        # +1 smoothing: a p-value of exactly 0 is not something 20k draws can
        # support, and reporting one would overstate the result.
        "p": (hits + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def series_sign_test(g):
    """Sign test over per-series rho, using series as the unit of evidence.

    Each series contributes ONE number, so unlike the pooled pair statistic this
    needs no independence assumption that the data does not support. Only series
    with n >= 3 are counted: a 2-member series has rho = +-1 by construction and
    carries no ranking information."""
    rhos = [v["rho"] for v in g["series_rho"].values()
            if v["n"] >= 3 and v["rho"] is not None]
    if not rhos:
        return {"n": 0, "n_right": 0, "p": None}
    k = sum(1 for r in rhos if r < 0)      # negative == working, after orientation
    n = len(rhos)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return {"n": n, "n_right": k, "p": p,
            "median_rho": st.median(rhos)}


def grade(values, per_lists, truth, tiers=None, metric="?", n_perm=10000):
    """All headline statistics for one metric in one mode."""
    o = orient(values, metric)
    pool = set(o)
    if tiers:
        pool = {p for p in pool
                if TIER_OF.get((truth.get(p) or {}).get("series")) in tiers}
    bins = {}
    for b in DPKD_BINS:
        prs = [pr for pr in within_series_pairs(truth, min_dpkd=b, pdbs=pool)]
        bins[b] = pair_agreement(prs, o)
    rh = series_rhos(o, truth, tiers=tiers)
    rhos = [v["rho"] for v in rh.values() if v["rho"] is not None]
    n3 = [v["rho"] for v in rh.values() if v["n"] >= 3 and v["rho"] is not None]
    disc = discriminability(per_lists, o, truth, tiers=tiers)
    perm = permutation_test(values, truth, tiers=tiers, metric=metric,
                            n_perm=n_perm)

    # Cross-complex, with the mandatory size control.
    pdbs = [p for p in sorted(o) if truth.get(p)]
    xs = [o[p] for p in pdbs]
    ys = [truth[p]["pkd"] for p in pdbs]
    ls = [truth[p]["pep_len"] for p in pdbs]
    out = {
        "metric": metric,
        "n_scored": len(o),
        "bins": bins,
        "headline": bins.get(HEADLINE_BIN),
        "perm": perm,
        "series_rho": rh,
        "mean_rho": st.mean(rhos) if rhos else None,
        "frac_right_signed_n3": (sum(1 for r in n3 if r < 0) / len(n3)) if n3 else None,
        "n_series_n3": len(n3),
        "disc": disc,
        "mean_disc": (st.mean([d["ratio"] for d in disc.values()
                               if d["ratio"] != float("inf")])
                      if any(d["ratio"] != float("inf") for d in disc.values())
                      else None),
        "cross_rho": _spearman(xs, ys) if len(xs) >= 3 else None,
        "cross_pearson": _pearson(xs, ys) if len(xs) >= 3 else None,
        "cross_rho_partial_len": (partial_spearman(xs, ys, ls)
                                  if len(xs) >= 4 else None),
        "cross_rho_len": _spearman(xs, ls) if len(xs) >= 3 else None,
    }
    out["sign_test"] = series_sign_test(out)
    return out


def verdict(g, best_null_lo=None):
    """The PRE-REGISTERED rule from docs/binding_metric_design.md. Same
    thresholds as the dG gate, plus one clause the dG gate did not need: the
    metric must beat the best NULL on the headline bin. A metric that ties a
    residue count has not worked, whatever its CI says."""
    head = g.get("headline") or {}
    lo = (head.get("ci95") or (0.0, 1.0))[0]
    mean_rho = g.get("mean_rho")
    frac = g.get("frac_right_signed_n3")
    reasons = [
        f"pair agreement |dpKd|>={HEADLINE_BIN}: {head.get('agreement')} "
        f"(n={head.get('n_pairs')}), CI95 lower bound = {lo:.3f} "
        f"(PASS needs >=0.65, FAIL at <=0.50)",
        f"mean per-series rho = "
        f"{mean_rho if mean_rho is None else round(mean_rho, 3)} "
        f"(PASS needs <=-0.40; negative = working, see orientation note)",
        f"fraction of n>=3 series right-signed = "
        f"{frac if frac is None else round(frac, 3)} (PASS needs >=0.60)",
    ]
    if best_null_lo is not None:
        reasons.append(
            f"best NULL headline CI95 lower bound = {best_null_lo:.3f} "
            f"(PASS needs the candidate to exceed it)")

    # The CI above treats the pooled pairs as independent, which they are not:
    # they come from a handful of series. These two clauses are the ones that
    # respect the data's actual structure, and a PASS now requires BOTH.
    perm = g.get("perm") or {}
    sgn = g.get("sign_test") or {}
    p_perm, p_sign = perm.get("p"), sgn.get("p")
    reasons.append(
        f"within-series permutation test: observed "
        f"{perm.get('observed')} vs null mean "
        f"{perm.get('null_mean') if perm.get('null_mean') is None else round(perm['null_mean'],3)} "
        f"(95th pct {perm.get('null_p95') if perm.get('null_p95') is None else round(perm['null_p95'],3)}), "
        f"p = {p_perm} (PASS needs p < 0.05)")
    reasons.append(
        f"per-series sign test: {sgn.get('n_right')}/{sgn.get('n')} series "
        f"right-signed, p = {p_sign if p_sign is None else round(p_sign, 4)} "
        f"(PASS needs p < 0.05; series are the independent unit)")

    if lo <= 0.50:
        return "FAIL", reasons
    beats_null = best_null_lo is None or lo > best_null_lo
    perm_ok = p_perm is not None and p_perm < 0.05
    sign_ok = p_sign is not None and p_sign < 0.05
    if (lo >= 0.65 and mean_rho is not None and mean_rho <= -0.40
            and frac is not None and frac >= 0.60 and beats_null
            and perm_ok and sign_ok):
        return "PASS", reasons
    return "WEAK", reasons


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(x, nd=3):
    return "   n/a" if x is None else f"{x:+.{nd}f}"


def report(path, data_dir, tiers=None, model=None):
    tbl = _metrics_table()
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    rows = load_rows(path)
    if not rows:
        print(f"no rows in {path}")
        return 1
    models = sorted({r.get("model") for r in rows})
    model = model or (models[0] if len(models) == 1 else models[0])

    print("=" * 78)
    print("CANDIDATE BINDING METRIC vs MEASURED Kd -- CPSet within-target gate")
    print("=" * 78)
    print(f"rows            : {len(rows)}   models: {models}   reporting: {model}")
    print(f"tier filter     : {tiers or 'ALL (BB, MIX and SS)'}")
    print("orientation     : all metrics flipped to LOWER = STRONGER, so a")
    print("                  WORKING metric has NEGATIVE rho against pKd, exactly")
    print("                  like the Rosetta dG gate it is compared against.")
    print()
    print("comparators (same benchmark, from the literature and this project):")
    for k, v in PAPER_REF.items():
        print(f"   {k:58s} {v:+.3f}")
    print(f"   {'null scorer: -peptide_length, cross-complex':58s} {-0.376:+.3f}")
    print(f"   {'Rosetta dG within-series pair agreement (prior result)':58s} "
          f"{0.424:.3f}")
    print()

    results = {}
    for mode in ("xtemplate", "self"):
        means, per = aggregate(rows, mode=mode, model=model)
        if not means:
            continue
        results[mode] = {}
        for metric in means:
            if metric not in tbl:
                continue
            results[mode][metric] = grade(means[metric], per[metric], truth,
                                          tiers=tiers, metric=metric)

    # --- the table --------------------------------------------------------
    order = [m for m in tbl if m in results.get("xtemplate", {})]
    fams = {"primary": "PRIMARY", "secondary": "secondary",
            "control": "control", "null": "NULL", "baseline": "baseline"}
    for mode in ("xtemplate", "self"):
        if mode not in results:
            continue
        tag = ("THE HEADLINE -- leak-free: every variant scored on OTHER "
               "members' crystals" if mode == "xtemplate"
               else "diagnostic only -- scored on its OWN crystal, which LEAKS")
        print("-" * 78)
        print(f"MODE: {mode}   ({tag})")
        print("-" * 78)
        print(f"{'metric':16s} {'family':10s} {'pairs':>11s} {'agree':>7s} "
              f"{'null':>6s} {'p_perm':>8s} {'sign':>6s} {'p_sgn':>7s} "
              f"{'mean_rho':>9s} {'x|len':>7s} {'disc':>6s}")
        for metric in order:
            g = results[mode].get(metric)
            if not g:
                continue
            h = g["headline"] or {}
            fam = fams.get((tbl[metric] or {}).get("family", ""), "?")
            lo = (h.get("ci95") or (None, None))[0]
            pm = g.get("perm") or {}
            sg = g.get("sign_test") or {}
            print(f"{metric:16s} {fam:10s} "
                  f"{str(h.get('n_correct'))+'/'+str(h.get('n_pairs')):>11s} "
                  f"{(h.get('agreement') if h.get('agreement') is not None else float('nan')):>7.3f} "
                  f"{(pm.get('null_mean') if pm.get('null_mean') is not None else float('nan')):>6.3f} "
                  f"{(pm.get('p') if pm.get('p') is not None else float('nan')):>8.4f} "
                  f"{str(sg.get('n_right'))+'/'+str(sg.get('n')):>6s} "
                  f"{(sg.get('p') if sg.get('p') is not None else float('nan')):>7.3f} "
                  f"{_fmt(g['mean_rho']):>9s} "
                  f"{_fmt(g['cross_rho_partial_len'], 2):>7s} "
                  f"{(g['mean_disc'] if g['mean_disc'] is not None else float('nan')):>6.1f}")
        print()
        print("  null  = MEAN agreement of the within-series permutation null.")
        print("          Not 0.5: ties and the series size structure move it, and")
        print("          with few series its spread is wide -- which is exactly why")
        print("          a single random draw scored 0.846 on the BB/MIX tier.")
        print("  p_perm= one-sided p against that null (the honest significance).")
        print("  sign  = series right-signed / series with n>=3; p_sgn its sign test.")
        print("          Series are the INDEPENDENT unit; pooled pairs are not.")
        print("  x|len = the SAME correlation with peptide length partialled out.")
        print("          dG scores -0.22 raw and -0.00 partial: that gap IS the")
        print("          size confound. A real metric keeps its signal here.")
        print("  disc  = between-variant spread / within-variant SEM across")
        print("          templates. Higher = fewer repeats needed.")
        print()

    # --- verdict ----------------------------------------------------------
    head = results.get("xtemplate", {})
    nulls = [m for m in head if (tbl[m] or {}).get("family") in ("null", "baseline")]
    best_null_lo = None
    best_null = None
    for m in nulls:
        h = head[m].get("headline") or {}
        lo = (h.get("ci95") or (None,))[0]
        if lo is not None and (best_null_lo is None or lo > best_null_lo):
            best_null_lo, best_null = lo, m
    print("=" * 78)
    primary = "icg"
    if primary in head:
        v, reasons = verdict(head[primary], best_null_lo)
        print(f"PRE-REGISTERED VERDICT on the primary metric '{primary}' "
              f"({'tier ' + ','.join(tiers) if tiers else 'all tiers'}): {v}")
        for r in reasons:
            print(f"   - {r}")
        if best_null is not None:
            print(f"   - best null was '{best_null}'")
    else:
        print(f"primary metric '{primary}' not present in rows")

    # --- the leak diagnostic ---------------------------------------------
    if "self" in results and "xtemplate" in results:
        print()
        print("LEAK DIAGNOSTIC (self minus xtemplate agreement on the headline bin):")
        for metric in order:
            a = ((results["xtemplate"].get(metric) or {}).get("headline") or {})
            b = ((results["self"].get(metric) or {}).get("headline") or {})
            if a.get("agreement") is None or b.get("agreement") is None:
                continue
            d = b["agreement"] - a["agreement"]
            flag = "  <-- large: the metric may be reading its own crystal" if d > 0.15 else ""
            print(f"   {metric:16s} self {b['agreement']:.3f} - "
                  f"xtemplate {a['agreement']:.3f} = {d:+.3f}{flag}")

    # --- per-series detail for the primary --------------------------------
    if primary in head:
        print()
        print(f"PER-SERIES detail, '{primary}', xtemplate "
              f"(negative rho = working):")
        g = head[primary]
        for sid, v in sorted(g["series_rho"].items()):
            d = g["disc"].get(sid, {})
            print(f"   {sid:18s} tier={TIER_OF.get(sid,'?'):4s} n={v['n']} "
                  f"rho={_fmt(v['rho'])}  "
                  f"disc={d.get('ratio', float('nan')):.1f} "
                  f"(between {d.get('between_sd', float('nan')):.4f} / "
                  f"within-SEM {d.get('within_sem', float('nan')):.4f})")
    return 0
