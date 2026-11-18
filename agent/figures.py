"""Figures for Stage 2. Reads only saved artefacts; needs no GPU and no key.

Global rule for this project: figures must be redrawable without
recomputation. Everything here is a function of `results/per_campaign.jsonl`,
`results/results.json` and the benchmark directory, so a plot can be restyled
or a panel added without re-running a campaign -- which matters most for the
API arms, where re-running costs money.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A brand-neutral, colour-blind-safe sequence, dark enough to read on white.
C = {
    "oracle": "#1b3a5c",
    "llm": "#2f6f9f",
    "llm2": "#5ba3d0",
    "rule": "#c07a3e",
    "rule2": "#e0a96d",
    "floor": "#9a9a9a",
    "bad": "#a8324a",
    "ok": "#2e7d5b",
    # The two comparisons the figures exist to show are 8B-vs-70B and
    # Sonnet-vs-Haiku, so those four must not share a colour.
    "local_small": "#2e7d5b",
    "local_big": "#6b4f8a",
    "api_big": "#2f6f9f",
    "api_small": "#5ba3d0",
}


def colour_for(arm_key: str) -> str:
    k = arm_key.lower()
    if "oracle" in k:
        return C["oracle"]
    if k.startswith("rule:"):
        if any(x in k for x in ("random", "noop", "widest")):
            return C["floor"]
        return C["rule"]
    if "70b" in k:
        return C["local_big"]
    if "8b" in k or "llama" in k:
        return C["local_small"]
    if "opus" in k or "sonnet" in k:
        return C["api_big"]
    if "haiku" in k:
        return C["api_small"]
    return C["ok"]


def short(arm_key: str) -> str:
    k = arm_key.replace("rule:", "").replace("claude-", "")
    k = k.replace("Llama-3.1-", "L").replace("-Instruct", "")
    k = k.replace("|twocall", "").replace("|onecall", "/1c")
    k = k.replace("A1_FRESH", "A1").replace("A2_CONDITIONING", "A2")
    k = k.replace("A3_CONTRACT", "A3").replace("A4_PRIOR", "A4")
    k = k.replace("A5_REFLECTION", "A5")
    return k[:30]


def load(results: Path):
    rows = [
        json.loads(l) for l in (results / "per_campaign.jsonl").read_text().splitlines()
        if l.strip()
    ]
    summary = json.loads((results / "results.json").read_text())
    return rows, summary


def fig_regret(rows, summary, out: Path):
    """Mean regret per arm with bootstrap CIs, plus the oracle and noop lines."""
    arms = summary["arms"]
    order = sorted(arms, key=lambda k: arms[k]["mean_regret"])
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.34 * len(order) + 1.4)))
    ys = range(len(order))
    right = max(
        (arms[k]["regret_ci95"][1] for k in order
         if arms[k]["regret_ci95"][1] == arms[k]["regret_ci95"][1]),
        default=max(arms[k]["mean_regret"] for k in order),
    )
    for y, k in zip(ys, order):
        s = arms[k]
        lo, hi = s["regret_ci95"]
        ax.barh(y, s["mean_regret"], color=colour_for(k), height=0.68,
                edgecolor="white", linewidth=0.6)
        if lo == lo:
            ax.plot([lo, hi], [y, y], color="#2b2b2b", lw=1.4, solid_capstyle="butt")
        # Labels go in a reserved gutter past the widest CI, not at the bar
        # end, where they sat on top of the error bars and were unreadable.
        cap = f"  cap={s['n_truncated_by_spend_cap']}" if s.get(
            "n_truncated_by_spend_cap") else ""
        ax.text(right * 1.03, y,
                f"n={s['n']}{cap}  CHR={s['contract_holding_rate']:.2f}",
                va="center", fontsize=7.5, color="#444")
    ax.set_xlim(0, right * 1.30)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([short(k) for k in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean regret vs the enumerated oracle  (lower is better)\n"
                  "bars are 95% bootstrap CIs on the MEAN; they overlap across "
                  "every arm, so\nread the paired tests in results.json, which "
                  "remove parent-level variance")
    ax.set_title("Campaign regret by controller", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig1_regret_by_arm.png", dpi=170)
    plt.close(fig)


def fig_outcomes(rows, summary, out: Path):
    """Outcome mix per arm. The 2->3 ablation is visible as the split."""
    by = defaultdict(Counter)
    for r in rows:
        by[r["arm_key"]][r["best_outcome"] or "ERROR"] += 1
    order = sorted(by, key=lambda k: -by[k]["CONTRACT_HOLDING_REPAIR"] / max(sum(by[k].values()), 1))
    cats = ["CONTRACT_HOLDING_REPAIR", "DEFECT_RESOLVED_CONTRACT_BROKEN",
            "NO_CHANGE", "INFEASIBLE", "ERROR"]
    cols = [C["ok"], C["bad"], C["floor"], "#6b4f8a", "#000000"]
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.34 * len(order) + 1.6)))
    for y, k in enumerate(order):
        tot = max(sum(by[k].values()), 1)
        left = 0.0
        for cat, col in zip(cats, cols):
            v = by[k][cat] / tot
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=col, height=0.68,
                    edgecolor="white", linewidth=0.6)
            left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(k) for k in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of campaigns")
    ax.set_title("Campaign outcome mix  (green = repair that held the contract;\n"
                 "red = defect resolved by spending the interface)",
                 fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, [c.replace("_", " ").lower() for c in cats],
              fontsize=7.5, ncol=2, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig2_outcome_mix.png", dpi=170)
    plt.close(fig)


def fig_service(rows, summary, out: Path):
    """What the deterministic execution service caught. PROTEAN's bottlenecks."""
    arms = summary["arms"]
    order = [k for k in sorted(arms) if arms[k]["n_proposed"] > 0]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.34 * len(order) + 1.4)))
    w = 0.38
    for y, k in enumerate(order):
        s = arms[k]
        tot = max(s["n_proposed"], 1)
        ax.barh(y - w / 2, s["scope_leaks_caught"] / tot, height=w,
                color=C["bad"], edgecolor="white", linewidth=0.5)
        ax.barh(y + w / 2, s["budget_filtered"] / tot, height=w,
                color=C["rule"], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(k) for k in order], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of all backend proposals discarded by the service")
    ax.set_title("Step 6 (mask verification) and step 4 (budget filter) catches",
                 fontsize=11, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C["bad"]),
               plt.Rectangle((0, 0), 1, 1, color=C["rule"])]
    ax.legend(handles, ["scope leakage", "budget overrun"], fontsize=8,
              frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig3_service_catches.png", dpi=170)
    plt.close(fig)


def fig_heldout(rows, summary, out: Path):
    """Search-time vs held-out utility. The doc's reportable number."""
    by = defaultdict(list)
    for r in rows:
        if r.get("search_utility") is not None and r.get("heldout_utility") is not None:
            by[r["arm_key"]].append((r["search_utility"], r["heldout_utility"]))
    if not by:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = [-15, 115]
    ax.plot(lim, lim, color="#bbb", lw=1, ls="--", zorder=1)
    for k, pts in sorted(by.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=26, alpha=0.75, label=short(k),
                   color=colour_for(k), edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("search-time utility")
    ax.set_ylabel("held-out utility  (fresh seeds, second predictor)")
    ax.set_title("Search-time to held-out gap\npoints below the line lost "
                 "utility under the held-out protocol", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig4_heldout_gap.png", dpi=170)
    plt.close(fig)


def fig_bench(bench: Path, out: Path):
    """What the benchmark is: defect census, and where the conflict sits.

    The right panel is the figure that justifies the whole environment -- for
    each parent, how much of its MEASURED interface mass sits inside the
    closure neighbourhood. A parent high on that axis cannot have its ring
    repaired without spending binding, and that placement is read off a
    deposited structure rather than chosen.
    """
    pf = bench / "parents_full.json"
    if not pf.exists():
        return
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lp2cp.envbuild import surrogate as S
    from lp2cp.envbuild.build_parents import load_parents

    parents = load_parents(pf, with_labels=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    census = Counter(p.true_defect_class.value for p in parents)
    keys = [k for k, _ in census.most_common()]
    a1.bar(range(len(keys)), [census[k] for k in keys], color=C["llm"],
           edgecolor="white")
    a1.set_xticks(range(len(keys)))
    a1.set_xticklabels([k.replace("_", "\n").lower() for k in keys], fontsize=7)
    a1.set_ylabel("parents")
    a1.set_title(f"Defect-class census  (n={len(parents)}, balanced cells)",
                 fontsize=10, loc="left")
    a1.spines[["top", "right"]].set_visible(False)

    xs, ys, cs = [], [], []
    for p in parents:
        tot = sum(S.interface_mass(r) for r in p.residues)
        near = sum(S.interface_mass(r) * S.closure_weight(r, p.n_pos)
                   for r in p.residues)
        overlap = near / tot if abs(tot) > 1e-9 else 0.0
        xs.append(overlap)
        ys.append(p.scorecard0.max_strain)
        cs.append(C["bad"] if p.true_defect_class.value
                  == "BACKBONE_STRAIN_AT_CLOSURE" else C["llm"])
    a2.scatter(xs, ys, c=cs, s=38, alpha=0.8, edgecolor="white", linewidth=0.6)
    a2.axhline(4.0, color="#666", ls="--", lw=1)
    a2.text(min(xs), 4.08, " strain limit 4.0 REU", fontsize=7.5, color="#555")
    a2.set_xlabel("measured interface mass inside the closure neighbourhood\n"
                  "(fraction) -- high = closure and binding compete")
    a2.set_ylabel("parent max backbone strain")
    a2.set_title("Where each parent's conflict sits\n(measured from deposited "
                 "coordinates)", fontsize=10, loc="left")
    a2.spines[["top", "right"]].set_visible(False)
    a2.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "fig0_benchmark.png", dpi=170)
    plt.close(fig)


def fig_arms(rows, summary, out: Path):
    """The arm ladder per model: A1..A5, the doc's ablation."""
    # Arm keys are "<model>|<arm>|<scaffold>"; rule arms have no "|" and are
    # not part of the ladder. The scaffold stays in the series label because a
    # one-call and a two-call run of the same model are different scaffolds and
    # must not be drawn as one line.
    by = defaultdict(dict)
    for k, s in summary["arms"].items():
        parts = k.split("|")
        if len(parts) < 2:
            continue
        model, arm = parts[0], parts[1]
        scaffold = parts[2] if len(parts) > 2 else ""
        by[f"{model}|{scaffold}" if scaffold else model][arm] = s
    by = {m: v for m, v in by.items() if len(v) >= 2}
    if not by:
        return
    arms = ["A1_FRESH", "A2_CONDITIONING", "A3_CONTRACT", "A4_PRIOR",
            "A5_REFLECTION"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for i, (model, v) in enumerate(sorted(by.items())):
        xs = [j for j, a in enumerate(arms) if a in v]
        for ax, field, lab in ((a1, "mean_regret", "mean regret"),
                               (a2, "contract_holding_rate",
                                "contract-holding repair rate")):
            ax.plot(xs, [v[arms[j]][field] for j in xs], marker="o", ms=5,
                    lw=1.6, label=short(model),
                    color=[C["llm"], C["llm2"], C["ok"], C["rule"]][i % 4])
            ax.set_xticks(range(len(arms)))
            ax.set_xticklabels([a.split("_")[0] for a in arms], fontsize=9)
            ax.set_ylabel(lab)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.22, lw=0.6)
    a1.set_title("Arm ladder: regret  (lower is better)", fontsize=10, loc="left")
    a2.set_title("Arm ladder: contract-holding repairs", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig5_arm_ladder.png", dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res, bench, out = Path(args.results), Path(args.bench), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = load(res)
    print(f"[fig] {len(rows)} campaigns, {len(summary['arms'])} arms")
    fig_bench(bench, out)
    fig_regret(rows, summary, out)
    fig_outcomes(rows, summary, out)
    fig_service(rows, summary, out)
    fig_heldout(rows, summary, out)
    fig_arms(rows, summary, out)
    for f in sorted(out.glob("*.png")):
        print(f"[fig] wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
