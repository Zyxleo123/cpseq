"""Redraw every Stage 0 figure from saved artefacts.

Reads only results.json and per_decision.jsonl, so a figure can be re-rendered
without a GPU, without a model and without rerunning an arm. That is the whole
reason the scorer writes per-decision rows.

Colour follows the ENTITY, not the rank: rule-based arms are blue, LLM arms
orange, controls and references grey, and a bar keeps its colour when the sort
order changes. The paired-difference panel is the one diverging encoding --
there the sign is the message, so blue/grey/red carries polarity.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e4e4e1"
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
ORANGE = "#eb6834"
GREY = "#9a9a96"
RED = "#d03b3b"

CONTROL_ARMS = {"most_measured", "first_listed", "max_dsasa", "random", "random_expected"}


def arm_colour(arm: str) -> str:
    if arm.startswith("llm_"):
        return ORANGE
    if arm in CONTROL_ARMS:
        return GREY
    return BLUE


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def fig_regret_by_arm(results: dict, slice_name: str, split: str, out: Path):
    rows = [
        v for v in results["summary"].values()
        if v["slice"] == slice_name and v["split"] == split and v.get("n_valid", 0) > 0
    ]
    if not rows:
        return None
    rows.sort(key=lambda v: v["mean_regret"])
    labels = [f"{v['arm']}" + (f" [{v['order']}]" if v["order"] != "main" else "")
              for v in rows]
    vals = [v["mean_regret"] for v in rows]
    cols = [arm_colour(v["arm"]) for v in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.33 * len(rows) + 1.7))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    # 4px-equivalent rounded data ends, anchored to the baseline at zero.
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    _style(ax)

    rnd = next((v["mean_regret"] for v in rows if v["arm"] == "random_expected"), None)
    if rnd is not None:
        ax.axvline(rnd, color=GREY, lw=1.4, ls=(0, (4, 3)), zorder=2)
        ax.text(rnd, -0.9, "  random", color=INK_MUTED, fontsize=7.5, va="center")
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.text(0, -0.9, "oracle  ", color=INK_MUTED, fontsize=7.5,
            va="center", ha="right")

    # Selective direct labels: the value at the end of each bar, and nothing else.
    span = max(vals) if vals else 1.0
    for yi, v in zip(y, vals):
        ax.text(v + span * 0.015, yi, f"{v:.2f}", va="center", fontsize=7.5,
                color=INK_MUTED)
    ax.set_xlim(0, span * 1.16)
    ax.set_xlabel("mean regret (kcal/mol of binding energy given up)",
                  fontsize=8.5, color=INK)
    n = rows[0]["n_valid"]
    ax.set_title(
        f"Where to spend the edit — {slice_name}, {split} split\n"
        f"lower is better · n={n} decisions · 0 = chose the best available position",
        fontsize=10, color=INK, loc="left", pad=12)

    # Only the classes actually present get a legend entry, and the legend sits
    # outside the axes: sorted ascending, the longest bar is the bottom one, so
    # an in-axes legend lands on top of it.
    present = {arm_colour(v["arm"]) for v in rows}
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=col, label=lab)
        for col, lab in ((BLUE, "rule-based"), (ORANGE, "LLM"),
                         (GREY, "control / reference"))
        if col in present
    ]
    if len(handles) > 1:
        fig.legend(handles=handles, frameon=False, fontsize=8,
                   loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.005), labelcolor=INK_MUTED)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
    else:
        fig.tight_layout()
    dest = out / f"regret_{slice_name}_{split}.png"
    fig.savefig(dest, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return dest


def fig_paired(per_decision: list[dict], arm_a: str, arm_b: str,
               slice_name: str, out: Path):
    """Per-decision paired difference, sorted. Diverging: sign is the message."""
    by = defaultdict(dict)
    for r in per_decision:
        if r["slice"] == slice_name and r.get("valid") and r.get("regret") is not None:
            by[r["decision_id"]][r["arm"]] = r["regret"]
    pairs = [
        (d, v[arm_a] - v[arm_b])
        for d, v in by.items() if arm_a in v and arm_b in v
    ]
    if len(pairs) < 6:
        return None
    pairs.sort(key=lambda p: p[1])
    diffs = np.array([p[1] for p in pairs])
    cols = [BLUE if d < -1e-9 else (RED if d > 1e-9 else GREY) for d in diffs]

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(diffs))
    ax.bar(x, diffs, color=cols, width=0.82, zorder=3)
    ax.axhline(0, color=INK, lw=1.2, zorder=4)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(diffs)} decisions, sorted by difference", fontsize=8.5,
                  color=INK)
    ax.set_ylabel("regret difference (kcal/mol)", fontsize=8.5, color=INK)
    wins = int((diffs < -1e-9).sum())
    losses = int((diffs > 1e-9).sum())
    med = float(np.median(diffs))
    ax.set_title(
        f"{arm_a}  vs  {arm_b} — {slice_name}\n"
        f"below zero = {arm_a} gave up less energy · "
        f"{wins} wins / {losses} losses · median {med:+.2f} kcal/mol",
        fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    dest = out / f"paired_{slice_name}_{arm_a}_vs_{arm_b}.png"
    fig.savefig(dest, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return dest


def fig_regret_distribution(per_decision: list[dict], slice_name: str, out: Path):
    """Per-decision regret spread, not just its mean.

    Added because the mean hid the result that mattered: on the full-interface
    arm the 8B matched the best heuristic's oracle-hit rate and median, and
    lost entirely in the tail. A bar of means cannot show that, and reading
    only the bar would have mis-diagnosed the failure as weak judgement rather
    than rare catastrophic error.
    """
    by_arm = defaultdict(list)
    for r in per_decision:
        if (r["slice"] == slice_name and r.get("valid")
                and r.get("regret") is not None and r["arm"] != "random_expected"):
            by_arm[r["arm"]].append(r["regret"])
    by_arm = {k: v for k, v in by_arm.items() if len(v) >= 10}
    if len(by_arm) < 2:
        return None
    order = sorted(by_arm, key=lambda k: float(np.mean(by_arm[k])))

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for i, arm in enumerate(order):
        vals = np.asarray(by_arm[arm], float)
        col = arm_colour(arm)
        # Every decision as a point, jittered; the mark carries the identity,
        # the summary bar sits on top of it rather than replacing it.
        ax.scatter(vals, np.full(len(vals), i) + rng.uniform(-0.17, 0.17, len(vals)),
                   s=11, color=col, alpha=0.38, linewidths=0, zorder=3)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=col, lw=2.4, solid_capstyle="round", zorder=4)
        ax.plot([med], [i], marker="|", ms=13, mew=2.2, color=SURFACE, zorder=5)
        ax.plot([float(np.mean(vals))], [i], marker="D", ms=5, color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=6)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.invert_yaxis()
    _style(ax)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_xlabel("regret per decision (kcal/mol)", fontsize=8.5, color=INK)
    ax.set_title(
        f"The tail, not the mean — {slice_name}\n"
        "one dot per decision · bar = IQR, notch = median, diamond = mean",
        fontsize=9.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    dest = out / f"regret_distribution_{slice_name}.png"
    fig.savefig(dest, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return dest


def fig_stakes(results: dict, bench_dir: Path, out: Path):
    """How much energy each decision actually puts at stake.

    Without this panel a small regret difference cannot be read: an arm losing
    0.3 kcal/mol on a decision whose whole range is 0.5 is doing badly, and the
    same 0.3 on a range of 8 is doing well.
    """
    oracle = json.loads((bench_dir / "decisions_oracle.json").read_text())
    by_slice = defaultdict(list)
    for o in oracle:
        by_slice[o["slice"]].append(o["spread"])
    slices = [s for s in ("ala_k1", "allsub_k1", "allsub_k2") if by_slice.get(s)]
    if not slices:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    fig.patch.set_facecolor(SURFACE)
    data = [by_slice[s] for s in slices]
    parts = ax.boxplot(data, orientation="vertical", widths=0.5, patch_artist=True,
                       medianprops=dict(color=SURFACE, lw=1.6),
                       flierprops=dict(marker="o", ms=3, mfc=GREY, mec="none"),
                       whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    for patch in parts["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor("none")
    ax.set_xticks(range(1, len(slices) + 1),
                  [f"{s}\nn={len(by_slice[s])}" for s in slices], fontsize=8)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("best-to-worst range (kcal/mol)", fontsize=8.5, color=INK)
    ax.set_title("What each decision puts at stake\n"
                 "the spread an allocator can win or lose within one complex",
                 fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    dest = out / "stakes_per_slice.png"
    fig.savefig(dest, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return dest


def fig_position_bias(results: dict, out: Path):
    bias = results.get("controls", {}).get("position_bias", {})
    rows = [(k, v) for k, v in bias.items() if v.get("n", 0) >= 6]
    if not rows:
        return None
    rows.sort(key=lambda kv: kv[1]["mean_relative_slot"])
    labels = [k.split("|")[0] for k, _ in rows]
    vals = [v["mean_relative_slot"] for _, v in rows]
    cols = [arm_colour(k.split("|")[0]) for k, _ in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(rows) + 1.8))
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.6, zorder=3)
    ax.axvline(0.5, color=INK, lw=1.2, zorder=4)
    ax.text(0.5, -0.9, "  no position bias", color=INK_MUTED, fontsize=7.5,
            va="center")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean relative slot of the chosen candidate "
                  "(0 = always first shown, 1 = always last)",
                  fontsize=8.5, color=INK)
    ax.set_title("Position-bias control\n"
                 "candidate order is a seeded shuffle, so a value away from 0.5 "
                 "is the model reading the list, not the evidence",
                 fontsize=10, color=INK, loc="left", pad=12)
    fig.tight_layout()
    dest = out / "position_bias.png"
    fig.savefig(dest, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res_dir = Path(args.results)
    results_path = res_dir / "results.json"
    if not results_path.exists():
        print(f"[fig] GATE FAILED: no results.json at {results_path}. "
              f"Run score.py first. Exiting 0.")
        return 0
    results = json.loads(results_path.read_text())
    per_decision = [
        json.loads(line)
        for line in (res_dir / "per_decision.jsonl").read_text().splitlines()
        if line.strip()
    ]
    out = Path(args.out or (res_dir / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for slice_name in sorted({v["slice"] for v in results["summary"].values()}):
        for split in ("all", "test"):
            d = fig_regret_by_arm(results, slice_name, split, out)
            if d:
                made.append(d)

    # The headline pairing: every LLM arm against the strongest rule-based bar
    # on that slice, chosen by mean regret among the non-control arms.
    for slice_name in sorted({v["slice"] for v in results["summary"].values()}):
        pool = [
            v for v in results["summary"].values()
            if v["slice"] == slice_name and v["split"] == "all"
            and v.get("n_valid", 0) > 0
        ]
        bars = [v for v in pool
                if not v["arm"].startswith("llm_") and v["arm"] not in CONTROL_ARMS]
        llms = [v for v in pool if v["arm"].startswith("llm_")]
        if not bars:
            continue
        best_bar = min(bars, key=lambda v: v["mean_regret"])["arm"]
        for v in llms:
            d = fig_paired(per_decision, v["arm"], best_bar, slice_name, out)
            if d:
                made.append(d)
        d = fig_paired(per_decision, best_bar, "random_expected", slice_name, out)
        if d:
            made.append(d)

    for slice_name in sorted({v["slice"] for v in results["summary"].values()}):
        d = fig_regret_distribution(per_decision, slice_name, out)
        if d:
            made.append(d)

    for fn in (lambda: fig_stakes(results, Path(args.bench), out),
               lambda: fig_position_bias(results, out)):
        d = fn()
        if d:
            made.append(d)

    print(f"[fig] wrote {len(made)} figures to {out}")
    for d in made:
        print(f"[fig]   {d.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
