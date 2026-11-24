"""PUBLICATION FIGURES: `pll_bg` agrees with measured pKd on CPSet Dataset I.

One plot per file. Each figure answers exactly one question and carries only the
text needed to read it; nothing is a multi-panel composite.

  fig1_agreement   DOES THE PROXY AGREE, ACROSS EVERYTHING WE SCORED?
                   Every variant of every congeneric series, with each series
                   CENTERED on its own mean. Centering is what makes a single
                   pooled plot honest: the raw pKd range across targets is
                   3.55-10.29 (~9 kcal/mol), so an uncentered pool would import
                   that cross-target spread and show fake within-target skill
                   (the same trap that made Rosetta dG look like it worked --
                   its apparent rho was peptide SIZE). After centering, the only
                   thing left to correlate is the within-target ranking, which
                   is what a designer actually needs.
  fig2_s03         The showcase series on its raw axes, with the error bar that
                   template choice (the metric's only variability) contributes.
  fig3_concordance Is it chance? Pooled within-series pair concordance against
                   a WITHIN-SERIES PERMUTATION null, plus the wrong-receptor
                   specificity control.
  fig4_sar         Is it right for the right reason? Per-position log p for the
                   series whose measured SAR is known and local.
  fig5_series_rho  Per-series rho -- the unit of evidence that needs no
                   independence assumption -- for the candidate, the incumbent
                   and the composition control.

Redrawable with no recomputation: everything is read from the committed scoring
rows (`bindscore_v2.jsonl`, `bindscore_v2_decoy.jsonl`) and every statistic is
computed by `lnr.metric_gate`, the same referee module behind the table in
`docs/binding_metric_results.md`. Nothing is retyped from the doc, so a figure
cannot drift from the result; every plotted value is also dumped to
`<stem>_values.json` so a caption can cite it.

Design notes
------------
* Print figures: light surface, vector PDF + 600 dpi PNG, TrueType-embedded
  fonts (journals reject Type 3).
* Two validated categorical hues only -- blue #2a78d6 (candidate / head-to-tail)
  and orange #eb6834 (the Rosetta incumbent); all-pairs CVD dE 24.7,
  normal-vision 33.6, both >= 3:1 on white. Controls are neutral ink, never a
  third hue, and group identity always carries marker shape as well as color.

Usage
-----
  python -m lnr.fig_pll_cpset --rows ROWS.jsonl --decoy DECOY.jsonl \
      --data-dir CPSET_RAW --out-dir OUTDIR [--figures all] [--n-perm 20000]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless
import matplotlib.pyplot as plt             # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D         # noqa: E402
from matplotlib.patches import Patch        # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from lnr.cpset_accuracy import load_dataset1, TIER_OF, HEADLINE_BIN  # noqa: E402
from lnr.metric_gate import (  # noqa: E402
    load_rows, aggregate, orient, series_rhos, grade,
)
from lnr.skempi_accuracy import _spearman  # noqa: E402

PRIMARY_MODEL = "ligandmpnn_v_32_020_25"

# Palette: two validated categorical hues + neutral ink. See module docstring.
BLUE = "#2a78d6"
BLUE_DK = "#184f95"
BLUE_LT = "#9ec5f4"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND = "#edebe4"
SURFACE = "#ffffff"

# Diverging ramp (signed quantity -> two poles, neutral gray midpoint, equal
# steps per arm). A sequential ramp would hide the sign of the gains.
DIVERGING = LinearSegmentedColormap.from_list(
    "loss_gain",
    ["#6e1c1c", "#b32b2b", "#e34948", "#f3c9c8", "#f0efec",
     "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"],
)

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.4,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.4,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,                     # TrueType; journals reject Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

METRIC_LABEL = {
    "pll_bg": "pll_bg", "pll_cplx": "pll_cplx", "icg": "icg",
    "dg_rosetta": "Rosetta ΔG", "pll_shuf": "pll_shuf",
    "null_random": "random", "null_neglen": "−length",
}
# The role goes INTO the row label rather than beside it: a second text column
# collided with the tick labels and said nothing the label could not carry.
METRIC_ROW = {
    "pll_bg": "pll_bg", "pll_cplx": "pll_cplx", "icg": "icg",
    "dg_rosetta": "Rosetta ΔG  (incumbent)", "pll_shuf": "pll_shuf  (scrambled)",
    "null_random": "random", "null_neglen": "−length  (size null)",
}
SERIES_SUBTITLE = {
    "S03_keap1": "KEAP1 Kelch · head-to-tail 7-mers",
    "S05_6d3": "head-to-tail 14-mers",
    "S01_strep": "streptavidin · disulfide",
    "S02_qfdlstrrlk": "disulfide 12-mers",
}
HEAD_TO_TAIL = ("BB", "MIX")


# ---------------------------------------------------------------------------
# Numbers. Computed by the referee module, never typed in.
# ---------------------------------------------------------------------------
def rho_agree(values, truth, metric, tiers=None):
    """Per-series Spearman, oriented so +1 == agrees with measurement.

    `metric_gate.orient` flips every metric to the lower-is-stronger convention
    its pair test is written for, so a NEGATIVE rho there means agreement.
    Negating once here puts every scorer on one axis whose right-hand end means
    the same thing for all of them -- the only way a reader can compare pll_bg
    (higher = stronger) with Rosetta dG (lower = stronger) in one panel.
    """
    out = {}
    for sid, rec in series_rhos(orient(values, metric), truth, tiers=tiers).items():
        out[sid] = {"n": rec["n"],
                    "rho": None if rec["rho"] is None else -rec["rho"]}
    return out


def centered_points(values, truth, metric, tiers=None, min_n=2):
    """Within-series centered (score, pKd) for every scored variant.

    Each series is centered on its OWN means, so the cross-target pKd range
    cannot leak in. Scores are oriented first, then negated, so +x always means
    'this scorer calls it stronger' whatever its native sign convention.
    """
    o = orient(values, metric)
    by = {}
    for pdb, v in o.items():
        rec = truth.get(pdb) or {}
        sid = rec.get("series")
        if not sid:
            continue
        if tiers and TIER_OF.get(sid) not in tiers:
            continue
        by.setdefault(sid, []).append((pdb, -v, rec["pkd"]))
    pts = []
    for sid, items in sorted(by.items()):
        if len(items) < min_n:
            continue
        mv = st.mean([i[1] for i in items])
        mp = st.mean([i[2] for i in items])
        for pdb, v, p in items:
            # `v` is already flipped back to higher-is-stronger (see above), so
            # it is the scorer's own value for pll_bg and the NEGATED dG for
            # Rosetta -- which is what "raw" has to mean for a shared axis.
            pts.append({"pdb": pdb, "series": sid, "tier": TIER_OF.get(sid),
                        "dx": v - mv, "dy": p - mp, "raw": v, "pkd": p})
    return pts


def centered_stats(pts, n_perm=10000, seed=0):
    """Pooled centered Spearman + its WITHIN-SERIES permutation p.

    The pooled statistic is not 32 independent observations -- the variants come
    from 9 series and are strongly dependent, which is exactly how a single draw
    of pure noise once scored 11/13 on the pair test. The null therefore shuffles
    scores WITHIN each series (leaving series sizes, the pKd values and the tie
    structure untouched) and recomputes the same pooled number.
    """
    if len(pts) < 4:
        return {"rho": None, "p": None, "n": len(pts), "n_series": 0}
    x = np.array([p["dx"] for p in pts], float)
    y = np.array([p["dy"] for p in pts], float)
    obs = _spearman(list(x), list(y))
    idx = collections.defaultdict(list)
    for i, p in enumerate(pts):
        idx[p["series"]].append(i)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        xs = x.copy()
        for ii in idx.values():
            xs[ii] = x[rng.permutation(ii)]
        if _spearman(list(xs), list(y)) >= obs:
            hits += 1
    return {"rho": obs, "p": (hits + 1) / (n_perm + 1), "n": len(pts),
            "n_series": len(idx), "n_perm": n_perm}


def collect(rows_path, decoy_path, data_dir, model, n_perm, seed=0):
    truth = load_dataset1(Path(data_dir) / "dataset1.csv")
    rows = load_rows(rows_path)
    means, per = aggregate(rows, mode="xtemplate", model=model)

    dmeans = dper = None
    if decoy_path and Path(decoy_path).exists():
        dmeans, dper = aggregate(load_rows(decoy_path), mode="xtemplate",
                                 model=model)

    metrics = [m for m in ("pll_bg", "pll_cplx", "icg", "dg_rosetta",
                           "pll_shuf", "null_random", "null_neglen")
               if m in means]

    out = {"model": model, "rows": str(rows_path),
           "decoy": str(decoy_path or ""), "n_perm": n_perm,
           "headline_bin": HEADLINE_BIN, "metrics": {}, "centered": {},
           "points": {}, "sar": {}}

    for m in metrics:
        g_bb = grade(means[m], per[m], truth, tiers=HEAD_TO_TAIL, metric=m,
                     n_perm=n_perm)
        g_all = grade(means[m], per[m], truth, tiers=None, metric=m,
                      n_perm=n_perm)
        rec = {
            "bb": {"n_correct": g_bb["headline"]["n_correct"],
                   "n_pairs": g_bb["headline"]["n_pairs"],
                   "agreement": g_bb["headline"]["agreement"],
                   "perm_p": g_bb["perm"]["p"],
                   "null_mean": g_bb["perm"]["null_mean"],
                   "null_p95": g_bb["perm"]["null_p95"],
                   "n_series": g_bb["perm"]["n_series"]},
            "all": {"n_correct": g_all["headline"]["n_correct"],
                    "n_pairs": g_all["headline"]["n_pairs"],
                    "agreement": g_all["headline"]["agreement"],
                    "perm_p": g_all["perm"]["p"],
                    "sign_test": g_all["sign_test"],
                    "mean_rho": g_all["mean_rho"]},
            # The size control is mandatory on this project: a scorer that
            # counts peptide residues reaches cross-complex rho -0.376 here, so
            # a cross-complex number means nothing without the length-partialled
            # one beside it. Dumped even though the figures' claim is
            # within-series, where length is near-constant by construction.
            "cross": {"rho": g_all["cross_rho"],
                      "rho_partial_len": g_all["cross_rho_partial_len"],
                      "rho_vs_len": g_all["cross_rho_len"]},
            "rho_agree": rho_agree(means[m], truth, m),
        }
        if dmeans and m in dmeans:
            gd = grade(dmeans[m], dper[m], truth, tiers=HEAD_TO_TAIL, metric=m,
                       n_perm=n_perm)
            rec["decoy"] = {"n_correct": gd["headline"]["n_correct"],
                            "n_pairs": gd["headline"]["n_pairs"],
                            "agreement": gd["headline"]["agreement"],
                            "perm_p": gd["perm"]["p"]}
        out["metrics"][m] = rec

    # The headline plot's data: every scored variant, every series, centered.
    for m in ("pll_bg", "dg_rosetta", "pll_shuf"):
        if m not in means:
            continue
        pts = centered_points(means[m], truth, m)
        stats_all = centered_stats(pts, n_perm=min(n_perm, 10000))
        pts_bb = [p for p in pts if p["tier"] in HEAD_TO_TAIL]
        out["centered"][m] = {
            "all": stats_all,
            "head_to_tail": centered_stats(pts_bb, n_perm=min(n_perm, 10000)),
            "points": pts if m == "pll_bg" else [],
        }

    # Raw per-variant points, with the between-template SEM -- the only
    # variability this metric has (it is otherwise bitwise deterministic).
    for sid in ("S03_keap1", "S05_6d3", "S01_strep", "S02_qfdlstrrlk"):
        pts = []
        for pdb, v in means["pll_bg"].items():
            rec = truth.get(pdb) or {}
            if rec.get("series") != sid:
                continue
            vs = per["pll_bg"][pdb]
            pts.append({"pdb": pdb, "pkd": rec["pkd"], "pll_bg": v,
                        "sem": (st.stdev(vs) / math.sqrt(len(vs)))
                               if len(vs) > 1 else 0.0,
                        "n_templates": len(vs), "pep_len": rec["pep_len"]})
        pts.sort(key=lambda d: d["pkd"])
        out["points"][sid] = pts

    # Mechanism plot: per-position log p for S03, averaged over templates.
    acc = {}
    for r in rows:
        if r.get("model") != model or r.get("series") != "S03_keap1":
            continue
        if r.get("is_self"):
            continue
        pp = (r.get("metrics") or {}).get("_per_pos_cplx")
        if not pp:
            continue
        acc.setdefault(r["pdb"], {"seq": r["seq"], "vals": []})["vals"].append(pp)
    items = sorted(acc.items(), key=lambda kv: -truth[kv[0]]["pkd"])
    if items:
        ref_pdb, ref = items[0]
        ref_pos = np.mean(np.array(ref["vals"], float), axis=0)
        out["sar"] = {"series": "S03_keap1", "ref": ref_pdb,
                      "ref_seq": ref["seq"], "ref_pkd": truth[ref_pdb]["pkd"],
                      "ref_per_pos": [round(float(x), 4) for x in ref_pos],
                      "variants": []}
        for pdb, a in items[1:]:
            m_pos = np.mean(np.array(a["vals"], float), axis=0)
            d = m_pos - ref_pos
            out["sar"]["variants"].append({
                "pdb": pdb, "seq": a["seq"], "pkd": truth[pdb]["pkd"],
                "dpkd": truth[pdb]["pkd"] - truth[ref_pdb]["pkd"],
                "muts": [f"{ref['seq'][i]}{i + 1}{a['seq'][i]}"
                         for i in range(len(m_pos))
                         if a["seq"][i] != ref["seq"][i]],
                "delta": [round(float(x), 4) for x in d],
                "changed": [i for i in range(len(m_pos))
                            if a["seq"][i] != ref["seq"][i]]})
    return out, truth


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------
def zscore(vals):
    """Within-series z-score. The ONLY honest way to put a predicted score and a
    measured pKd on one axis: two y-scales would be a dual-axis chart, whose
    apparent agreement is an artefact of whatever ranges you chose. Indexing both
    to the series' own mean and spread removes the choice -- and removes the
    cross-target pKd range (3.55-10.29) that would otherwise manufacture
    within-target skill out of nothing."""
    m = st.mean(vals)
    s = st.pstdev(vals)
    return [(v - m) / s if s > 1e-12 else 0.0 for v in vals]


def _save(fig, stem, name, formats):
    paths = []
    for ext in formats:
        p = f"{stem}_{name}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
        paths.append(p)
    plt.close(fig)
    return paths


def _fmt_p(p):
    if p is None:
        return "—"
    return f"P = {p:.3f}" if p >= 0.001 else f"P < 0.001"


def place_labels(ax, xs, ys, texts, color=INK2, fontsize=6.6):
    """Direct labels, placed against the real renderer so none collide."""
    fig = ax.figure
    rend = fig.canvas.get_renderer()
    obstacles = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        obstacles.append(
            matplotlib.transforms.Bbox.from_bounds(px - 5, py - 5, 10, 10))
    cands = [(7, 2.5), (7, -8), (-7, 2.5), (-7, -8), (0, 8.5), (0, -11),
             (12, 0), (-12, 0)]
    for x, y, t in zip(xs, ys, texts):
        done = None
        for dx, dy in cands:
            ann = ax.annotate(
                t, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=fontsize, color=color, zorder=6,
                ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                va="bottom" if dy > 0 else ("top" if dy < 0 else "center"))
            bb = ann.get_window_extent(renderer=rend).expanded(1.06, 1.10)
            if (ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
                    and all(not bb.overlaps(o) for o in obstacles)):
                done = (ann, bb)
                break
            ann.remove()
        if done is None:                        # never drop a label
            ann = ax.annotate(t, (x, y), textcoords="offset points",
                              xytext=(7, 2.5), fontsize=fontsize, color=color,
                              zorder=6, ha="left", va="bottom")
            done = (ann, ann.get_window_extent(renderer=rend))
        obstacles.append(done[1])


# ---------------------------------------------------------------------------
# FIGURE: does the proxy TRACK the measurement, variant by variant?
#
# Variants are ordered by measured affinity within each series, and both
# quantities are z-scored within that series, so they share ONE axis in one unit
# (sigma). The measured trace is monotone by construction; the question the plot
# asks is whether the predicted trace descends with it. Every variant scored is
# here -- nothing is dropped to make the picture.
# ---------------------------------------------------------------------------
def fig_tracking(vals, stem, formats):
    pts = vals["centered"]["pll_bg"]["points"]
    by = {}
    for p in pts:
        by.setdefault(p["series"], []).append(p)
    sids = sorted(by)

    xpos, ticks, groups = [], [], []
    cur = 0.0
    for sid in sids:
        items = sorted(by[sid], key=lambda d: -d["pkd"])
        xs = [cur + i for i in range(len(items))]
        groups.append({
            "sid": sid, "tier": items[0]["tier"], "x": xs,
            "pdbs": [d["pdb"] for d in items],
            "z_pkd": zscore([d["pkd"] for d in items]),
            "z_pll": zscore([d["raw"] for d in items]),
            "rho": _spearman([d["raw"] for d in items],
                             [d["pkd"] for d in items]),
            "n": len(items),
        })
        xpos += xs
        ticks += [d["pdb"] for d in items]
        cur += len(items) + 1.15          # gap between series

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    # Limits and tick style FIRST: matplotlib only instantiates Tick objects for
    # ticks inside the current view, so labels styled before set_xlim are styled
    # on ticks that are then thrown away and recreated with defaults.
    ax.set_xlim(-0.9, cur - 0.65)
    ax.set_ylim(-2.4, 2.95)
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.0, length=2)
    ax.axhline(0, color=GRID, lw=0.7, zorder=0)
    # A small dodge: in a 2-member series both z-scores are exactly +-1, so an
    # agreeing prediction would sit EXACTLY under the measurement and the reader
    # could not tell agreement from a missing point.
    dx = 0.10
    for g in groups:
        x0, x1 = g["x"][0] - 0.5, g["x"][-1] + 0.5
        if g["tier"] in HEAD_TO_TAIL:     # deployment scope, shaded
            ax.axvspan(x0, x1, color="#f2f6fc", lw=0, zorder=0)
        ax.plot([x - dx for x in g["x"]], g["z_pkd"], "-", lw=1.3, color=INK2,
                zorder=2, marker="s", ms=4.0, mfc=INK2, mec=SURFACE, mew=0.8)
        ax.plot([x + dx for x in g["x"]], g["z_pll"], "-", lw=1.3, color=BLUE,
                zorder=3, marker="o", ms=4.6, mfc=BLUE, mec=SURFACE, mew=0.8)
        ax.text((x0 + x1) / 2, 2.52, g["sid"].split("_")[0], fontsize=7.2,
                color=INK, ha="center", va="bottom")
        if g["n"] >= 3 and g["rho"] is not None:
            ax.text((x0 + x1) / 2, 2.12, f"ρ {g['rho']:+.2f}", fontsize=6.8,
                    color=MUTED, ha="center", va="bottom")

    ax.set_xticks(xpos)
    ax.set_xticklabels(ticks)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylabel("within-series z-score  (σ)")
    ax.set_xlabel("peptide variant, ordered by measured affinity within each series")
    ax.spines["bottom"].set_position(("outward", 2))
    ax.grid(axis="y", lw=0.6, color=GRID)

    s_all = vals["centered"]["pll_bg"]["all"]
    handles = [
        Line2D([], [], color=INK2, lw=1.3, marker="s", ms=4.0, mec=SURFACE,
               mew=0.8, label="measured pK$_\\mathrm{d}$"),
        Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=4.6, mec=SURFACE,
               mew=0.8, label="pll_bg  (predicted)"),
        Patch(facecolor="#f2f6fc", edgecolor="none", label="head-to-tail"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, 1.02), handletextpad=0.4, columnspacing=1.4,
              borderpad=0.0)
    # Bottom right: the only empty quadrant, and the two numbers that matter --
    # the pooled one and the in-scope one. Quoting the pooled number alone would
    # understate it (it averages in two series of a chemistry not claimed);
    # quoting the in-scope one alone would overstate it.
    s_bb = vals["centered"]["pll_bg"]["head_to_tail"]
    ax.text(0.995, 0.055, f"ρ = {s_all['rho']:+.2f}   {_fmt_p(s_all['p'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6,
            weight="bold", color=INK)
    ax.text(0.995, 0.005,
            f"head-to-tail only: ρ = {s_bb['rho']:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
            color=MUTED)
    return _save(fig, stem, "tracking", formats)


# ---------------------------------------------------------------------------
# FIGURE 1 -- the headline: does the proxy agree, over everything scored?
# ---------------------------------------------------------------------------
def fig_agreement(vals, stem, formats):
    c = vals["centered"]["pll_bg"]
    pts = c["points"]
    s_all, s_bb = c["all"], c["head_to_tail"]

    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)

    x = np.array([p["dx"] for p in pts])
    y = np.array([p["dy"] for p in pts])
    # One least-squares line, on all points, purely as a reading aid for the
    # trend; the claim is the rank statistic printed beside it, not this fit.
    if len(x) >= 3:
        b, a0 = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 2)
        ax.plot(xr, a0 + b * xr, "-", lw=1.1, color=BLUE_LT, zorder=1)

    groups = [("head-to-tail", HEAD_TO_TAIL, BLUE, "o", 34, 1.0),
              ("disulfide", ("SS",), INK2, "s", 26, 0.85)]
    for lab, tiers, col, mk, sz, al in groups:
        gx = [p["dx"] for p in pts if p["tier"] in tiers]
        gy = [p["dy"] for p in pts if p["tier"] in tiers]
        ax.scatter(gx, gy, s=sz, marker=mk, facecolor=col, edgecolor=SURFACE,
                   linewidth=0.8, alpha=al, zorder=3,
                   label=f"{lab}  (n = {len(gx)})")

    ax.set_xlabel("pll_bg − series mean   (nats / residue)")
    ax.set_ylabel("measured pK$_\\mathrm{d}$ − series mean   (log units)")
    ax.set_title("Within-series agreement, all of CPSet Dataset I", loc="left",
                 pad=7)
    ax.margins(0.10)
    ax.grid(lw=0.6, color=GRID)

    ax.text(0.03, 0.965,
            f"ρ = {s_all['rho']:+.2f}   {_fmt_p(s_all['p'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.0,
            weight="bold", color=INK)
    ax.text(0.03, 0.885,
            f"{s_all['n']} peptides · {s_all['n_series']} series\n"
            f"head-to-tail only: ρ = {s_bb['rho']:+.2f}\n"
            f"Rosetta ΔG: ρ = {vals['centered']['dg_rosetta']['all']['rho']:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            color=MUTED, linespacing=1.45)
    ax.legend(loc="lower right", handletextpad=0.3, labelspacing=0.3,
              borderpad=0.2)
    return _save(fig, stem, "agreement", formats)


# ---------------------------------------------------------------------------
# FIGURE 2 -- the showcase series on raw axes
# ---------------------------------------------------------------------------
def fig_series_scatter(vals, stem, formats, sid="S03_keap1", name="s03"):
    pts = vals["points"][sid]
    rho = (vals["metrics"]["pll_bg"]["rho_agree"].get(sid) or {}).get("rho")

    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    xs = [p["pll_bg"] for p in pts]
    ys = [p["pkd"] for p in pts]
    es = [p["sem"] for p in pts]
    ax.grid(lw=0.6, color=GRID)
    # Variants joined in measured-affinity order: a monotone path IS the claim,
    # and every backtrack is a mis-ordered pair.
    ax.plot(xs, ys, "-", lw=1.1, color=BLUE_LT, zorder=1, solid_capstyle="round")
    ax.errorbar(xs, ys, xerr=es, fmt="none", ecolor=BLUE, elinewidth=0.8,
                capsize=2.0, capthick=0.8, alpha=0.9, zorder=2)
    ax.scatter(xs, ys, s=42, facecolor=BLUE, edgecolor=SURFACE, linewidth=0.9,
               zorder=3)
    ax.set_xlabel("pll_bg   (nats / residue)")
    ax.set_ylabel("measured pK$_\\mathrm{d}$")
    ax.set_title(f"{sid} · {SERIES_SUBTITLE.get(sid, '')}", loc="left", pad=7)
    ax.margins(0.16)
    ax.text(0.03, 0.96,
            "ρ = " + ("—" if rho is None else f"{rho:+.2f}")
            + f"   n = {len(pts)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9.0,
            weight="bold", color=INK)
    ax.text(0.03, 0.885,
            f"{max(ys) - min(ys):.1f} log units of measured affinity\n"
            "bars: ±1 SEM over templates",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.2,
            color=MUTED, linespacing=1.45)
    fig.canvas.draw()
    place_labels(ax, xs, ys, [p["pdb"] for p in pts])
    return _save(fig, stem, name, formats)


# ---------------------------------------------------------------------------
# FIGURE 3 -- is it chance? concordance vs the permutation null
# ---------------------------------------------------------------------------
def fig_concordance(vals, stem, formats):
    order = [m for m in ("pll_bg", "pll_cplx", "icg", "dg_rosetta",
                         "pll_shuf", "null_random", "null_neglen")
             if m in vals["metrics"]]
    n = len(order)
    ys = {m: n - 1 - i for i, m in enumerate(order)}
    fam = {"pll_bg": BLUE_DK, "pll_cplx": BLUE, "icg": BLUE,
           "dg_rosetta": ORANGE}
    hero = vals["metrics"]["pll_bg"]["bb"]

    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    ax.axvspan(hero["null_mean"], hero["null_p95"], color=BAND, lw=0, zorder=0)
    ax.axvline(hero["null_mean"], color=MUTED, lw=0.8, ls=(0, (2.6, 2.0)),
               zorder=1)

    for m, y in ys.items():
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
        rec = vals["metrics"][m]
        col = fam.get(m, INK2)
        a = rec["bb"]["agreement"] or 0.0
        hero_row = (m == "pll_bg")
        if (rec.get("decoy") or {}).get("agreement") is not None:
            d = rec["decoy"]["agreement"]
            ax.plot([d, a], [y, y], "-", lw=0.8, color="#d3d1c8", zorder=2)
            ax.scatter([d], [y], s=34, facecolor="none", edgecolor=col,
                       linewidth=1.0, zorder=4, clip_on=False)
        ax.scatter([a], [y], s=58 if hero_row else 38, facecolor=col,
                   edgecolor=SURFACE, linewidth=0.9, zorder=5, clip_on=False)
        ax.text(1.03, y, f"{rec['bb']['n_correct']}/{rec['bb']['n_pairs']}",
                fontsize=7.4, color=INK if hero_row else INK2, va="center",
                ha="left", weight="bold" if hero_row else "normal")

    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([METRIC_ROW.get(m, m) for m in order], fontsize=7.6)
    for t, m in zip(ax.get_yticklabels(), order):
        if m == "pll_bg":
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.set_ylim(-1.5, n - 0.4)          # room for the legend under the rows
    ax.set_xlim(-0.015, 1.0)            # a hair of air so the 0.00 dot clears
                                        # the row label it would otherwise touch
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("within-series pairs ordered correctly")
    ax.set_title(f"Head-to-tail tier · |ΔpK$_\\mathrm{{d}}$| ≥ "
                 f"{HEADLINE_BIN:g} · {hero['n_pairs']} pairs", loc="left",
                 pad=7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.text((hero["null_mean"] + hero["null_p95"]) / 2, -0.62,
            "permutation null (mean → 95th pct)", fontsize=6.8, color=MUTED,
            va="center", ha="center")
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=BLUE_DK,
               markeredgecolor=SURFACE, markersize=6, label="true receptor"),
        Line2D([], [], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=BLUE_DK, markeredgewidth=1.0, markersize=5.6,
               label="wrong-receptor pocket"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, 0.0), handletextpad=0.35,
              columnspacing=1.6, borderpad=0.1)
    return _save(fig, stem, "concordance", formats)


# ---------------------------------------------------------------------------
# FIGURE 4 -- right for the right reason
# ---------------------------------------------------------------------------
def fig_sar(vals, stem, formats):
    sar = vals.get("sar")
    if not sar:
        return []
    variants = sar["variants"]
    npos = len(sar["ref_per_pos"])
    M = np.full((len(variants), npos), np.nan)
    for i, v in enumerate(variants):
        for j in v["changed"]:
            M[i, j] = v["delta"][j]
    vmax = math.ceil(float(np.nanmax(np.abs(M))) * 2) / 2
    # Diverging only if both signs occur. Here every substitution is a loss, and
    # a symmetric scale would spend half its range on values that do not exist --
    # which reads as "the blue half is missing" rather than "nothing gained".
    one_sided = float(np.nanmax(M)) <= 0.0
    cmap = (LinearSegmentedColormap.from_list(
        "loss", ["#6e1c1c", "#b32b2b", "#e34948", "#f3c9c8", "#f7f6f2"])
        if one_sided else DIVERGING)
    vlo, vhi = (-vmax, 0.0) if one_sided else (-vmax, vmax)

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    im = ax.imshow(M, cmap=cmap, vmin=vlo, vmax=vhi, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(npos))
    ax.set_xticklabels([f"{i + 1}\n{sar['ref_seq'][i]}" for i in range(npos)],
                       fontsize=7.6, linespacing=1.3)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([f"{v['pdb']}  {'+'.join(v['muts'])}" for v in variants],
                       fontsize=7.4)
    ax.tick_params(length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, npos, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.8)
    ax.tick_params(which="minor", length=0)

    for i, v in enumerate(variants):
        for j in range(npos):
            if j in v["changed"]:
                d = v["delta"][j]
                r, g, b = cmap((d - vlo) / (vhi - vlo))[:3]
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, f"{d:+.1f}", ha="center", va="center",
                        fontsize=7.0, weight="bold",
                        color="#ffffff" if lum < 0.55 else INK)
            else:
                # A dot, so "no change" reads as measured rather than missing.
                ax.plot([j], [i], marker=".", ms=2.0, color="#cfcdc4")
        ax.text(npos - 0.3, i, f"{v['dpkd']:+.2f}", fontsize=7.4, color=INK2,
                va="center", ha="left")
    ax.text(npos - 0.3, -0.85, "measured\nΔpK$_\\mathrm{d}$", fontsize=6.8,
            color=MUTED, va="center", ha="left", linespacing=1.3)
    ax.set_xlabel("peptide position", labelpad=2)
    ax.set_title(f"Each substitution's cost lands on its own position\n"
                 f"{sar['series']} · change from {sar['ref']} {sar['ref_seq']}",
                 loc="left", pad=16, fontsize=8.4, linespacing=1.35)

    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035,
                      pad=0.13,
                      ticks=[vlo, vlo / 2, vhi] if one_sided else [-vmax, 0, vmax])
    cb.set_label("Δ log p(s$_i$ | bound complex)", fontsize=7.2, labelpad=3)
    cb.ax.tick_params(labelsize=7.0, length=2, width=0.7, color=INK2)
    cb.outline.set_linewidth(0.0)
    return _save(fig, stem, "sar", formats)


# ---------------------------------------------------------------------------
# FIGURE 5 -- per-series rho: the unit of evidence
# ---------------------------------------------------------------------------
def fig_series_rho(vals, stem, formats):
    order = ["S03_keap1", "S05_6d3", "S01_strep", "S02_qfdlstrrlk"]
    scorers = [("pll_bg", BLUE, "o", 44, "pll_bg"),
               ("dg_rosetta", ORANGE, "s", 36, "Rosetta ΔG"),
               ("pll_shuf", MUTED, "^", 34, "scrambled control")]
    ys = {sid: len(order) - 1 - i for i, sid in enumerate(order)}

    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    ax.axvspan(-1.1, 0, color="#f7f6f2", lw=0, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for y in ys.values():
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
    for m, col, mk, sz, _lab in scorers:
        rr = (vals["metrics"].get(m) or {}).get("rho_agree") or {}
        for sid, y in ys.items():
            r = (rr.get(sid) or {}).get("rho")
            if r is None:
                continue
            ax.scatter([r], [y], s=sz, marker=mk, facecolor=col,
                       edgecolor=SURFACE, linewidth=0.8, zorder=4)

    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(order, fontsize=7.8)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Spearman ρ with measured pK$_\\mathrm{d}$   "
                  "(+1 = agrees)")
    ax.set_title("Per-series, the independent unit of evidence", loc="left",
                 pad=7)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.axhline(1.5, color=INK2, lw=0.7, ls=(0, (2.4, 2.0)), zorder=2)
    ax.text(-1.06, 2.42, "head-to-tail", fontsize=7.0, color=INK2,
            style="italic", va="center", ha="left")
    ax.text(-1.06, 1.26, "disulfide", fontsize=7.0, color=MUTED,
            style="italic", va="center", ha="left")
    handles = [Line2D([], [], marker=mk, color="none", markerfacecolor=c,
                      markeredgecolor=SURFACE, markersize=5.6, label=lab)
               for _m, c, mk, _s, lab in scorers]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.02),
              handletextpad=0.3, labelspacing=0.3, borderpad=0.15)
    return _save(fig, stem, "series_rho", formats)


# ---------------------------------------------------------------------------
FIGURES = {
    "tracking": fig_tracking,
    "agreement": fig_agreement,
    "s03": fig_series_scatter,
    "concordance": fig_concordance,
    "sar": fig_sar,
    "series_rho": fig_series_rho,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--decoy", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="pll_cpset")
    ap.add_argument("--model", default=PRIMARY_MODEL)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--figures", default="all",
                    help="comma list of " + ",".join(FIGURES) + " or 'all'")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / a.stem)

    vals, _ = collect(a.rows, a.decoy, a.data_dir, a.model, a.n_perm)
    Path(f"{stem}_values.json").write_text(json.dumps(vals, indent=1) + "\n")

    ca = vals["centered"]["pll_bg"]["all"]
    cb = vals["centered"]["pll_bg"]["head_to_tail"]
    print(f"model: {vals['model']}")
    print(f"WITHIN-SERIES CENTERED (every scored variant, every series):")
    for m in ("pll_bg", "dg_rosetta", "pll_shuf"):
        c = vals["centered"].get(m)
        if not c:
            continue
        print(f"  {METRIC_LABEL.get(m, m):12s} all: rho {c['all']['rho']:+.3f} "
              f"p {c['all']['p']:.4f}  (n={c['all']['n']}, "
              f"{c['all']['n_series']} series)   head-to-tail: "
              f"rho {c['head_to_tail']['rho']:+.3f}")
    print(f"PAIR CONCORDANCE, head-to-tail tier, |dpKd| >= "
          f"{vals['headline_bin']:g}:")
    for m, rec in vals["metrics"].items():
        b = rec["bb"]
        dec = (rec.get("decoy") or {}).get("agreement")
        print(f"  {METRIC_LABEL.get(m, m):12s} {b['n_correct']:2d}/"
              f"{b['n_pairs']:<2d} agree {b['agreement']:.3f} "
              f"null {b['null_mean']:.3f} p {b['perm_p']:.4f}"
              + (f"  decoy {dec:.3f}" if dec is not None else ""))
    print("PER-SERIES rho (+1 = agrees):")
    for m in ("pll_bg", "dg_rosetta", "pll_shuf"):
        rr = vals["metrics"][m]["rho_agree"]
        cells = []
        for sid in ("S03_keap1", "S05_6d3", "S01_strep", "S02_qfdlstrrlk"):
            r = (rr.get(sid) or {}).get("rho")
            cells.append(f"{sid.split('_')[0]} "
                         + ("  n/a" if r is None else f"{r:+.2f}"))
        print(f"  {METRIC_LABEL.get(m, m):12s} " + "  ".join(cells))

    plt.rcParams.update(RC)
    want = (list(FIGURES) if a.figures.strip() == "all"
            else [f.strip() for f in a.figures.split(",") if f.strip()])
    formats = tuple(f.strip() for f in a.formats.split(",") if f.strip())
    for key in want:
        fn = FIGURES.get(key)
        if fn is None:
            print(f"  (no figure named {key})")
            continue
        for p in fn(vals, stem, formats):
            print(f"wrote {p}")
    print(f"wrote {stem}_values.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
