#!/usr/bin/env python3
"""Stage 3 report: the TOOL action space against the mask-authoring arms.

Reads only saved artefacts -- the scorer's per-campaign rows plus the raw
campaign traces for the tool-choice histogram -- so every number here is
redrawable without re-running a campaign, a model or the environment.

    python report_stage3.py --results <dir> [--results <dir2> ...] \
                            [--campaigns <dir>] [--campaigns <dir2> ...]

Each --results directory is one BENCHMARK (one parent set, one environment
hash). They are never pooled into a single mean -- different parent sets have
different difficulty, so a pooled mean is not a quantity. What IS pooled is the
per-parent PAIRED DELTA, which is a difference measured within a parent and is
therefore comparable across parent sets; that pooling is what the combined test
at the end does, and it is reported alongside each benchmark on its own.

Identity of an arm. Rows carry `arm_key` as model|arm|scaffold for LLM arms and
rule:<name> for rules, and the scaffold string carries the action-space version
(`+menuv1` / `+menuv2`). The rule arms' arm_key does NOT carry it, so the key
used here appends the scaffold wherever it names a menu. Without that, the
deterministic router under two different action spaces would collapse into one
row -- which is the same de-duplication failure that once erased an entire
experiment from the results table.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import comb, erf, sqrt
from pathlib import Path
from statistics import mean, median

BAR = "rule:rim_first_closure"
CEILING = "rule:oracle_greedy"

TOOL_RULES = {
    "rule:headroom_repack": "repack_headroom",
    "rule:headroom_first": "substitute_headroom",
    "rule:greedy_interface": "substitute_cheap_rim",
    "rule:rim_first_closure": "narrow_near_closure",
    "rule:rebuild_cheap2": "rebuild_cheapest_2span",
    "rule:fixed_pipeline": "rebuild_at_strain",
    "rule:noop": "stop",
}


def key_of(row: dict) -> str:
    """Arm identity, including the action-space version where it applies."""
    k = row["arm_key"]
    sc = row.get("scaffold") or ""
    return f"{k}@{sc}" if "menu" in sc else k


def paired(a: dict[str, dict], b: dict[str, dict],
           skip: set[str] | None = None) -> list[float]:
    """a - b over shared parents. Negative means `a` has less regret.

    `skip` drops parents already contributed by an earlier benchmark. The two
    parent sets here overlap (17 of bench_v3's 48 are also in the 32-parent
    set), so pooling without that would count those parents twice and report a
    confidence interval narrower than the evidence supports.
    """
    ps = sorted(set(a) & set(b) - (skip or set()))
    return [a[p]["regret"] - b[p]["regret"] for p in ps]


def stats(deltas: list[float]) -> dict:
    """Wilcoxon signed-rank (exact sign test under n<6), plus a bootstrap CI.

    Ties are dropped, which is the standard treatment and the one the Stage 2
    scorer uses. The bootstrap CI is on the MEAN delta and is reported because
    a p-value alone does not say how big the effect is, and on this benchmark a
    single parent moving from no repair to a qualifying one is worth ~100
    regret points -- so a mean can be carried by two parents, and the interval
    makes that visible where the p-value does not.
    """
    n_all = len(deltas)
    nz = [d for d in deltas if abs(d) > 1e-9]
    n = len(nz)
    wins = sum(1 for d in deltas if d < -1e-9)
    losses = sum(1 for d in deltas if d > 1e-9)
    out = {
        "n": n_all, "wins": wins, "losses": losses, "ties": n_all - wins - losses,
        "mean_delta": mean(deltas) if deltas else 0.0,
        "median_delta": median(deltas) if deltas else 0.0,
        "p": None, "how": "no_data", "ci": (0.0, 0.0),
    }
    if not deltas:
        return out
    if n == 0:
        out["how"] = "all_tied"
        return out
    if n < 6:
        k = min(sum(1 for d in nz if d < 0), sum(1 for d in nz if d > 0))
        out["p"] = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
        out["how"] = f"sign_n{n}"
    else:
        order = sorted(range(n), key=lambda i: abs(nz[i]))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                ranks[order[t]] = avg
            i = j + 1
        w = min(sum(r for r, d in zip(ranks, nz) if d > 0),
                sum(r for r, d in zip(ranks, nz) if d < 0))
        mu = n * (n + 1) / 4.0
        sd = sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w - mu + 0.5) / sd if sd else 0.0
        out["p"] = min(1.0, 2.0 * 0.5 * (1.0 + erf(-abs(z) / sqrt(2.0))))
        out["how"] = f"wilcoxon_n{n}"
    # percentile bootstrap, deterministic seed so the interval is reproducible
    import random

    rng = random.Random(20261115)
    boots = sorted(
        mean(rng.choices(deltas, k=len(deltas))) for _ in range(4000))
    out["ci"] = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])
    return out


def mcnemar(a: dict[str, dict], b: dict[str, dict], field: str,
            skip: set[str] | None = None) -> dict:
    """Exact paired binary test on a per-campaign outcome flag.

    Regret on this benchmark is bimodal -- a parent either reaches a
    qualifying repair (regret near 0) or does not (regret near 100) -- so the
    mean is carried by a handful of parents and a rank test on it is weak.
    Whether a repair was FOUND is the quantity the metric is really tracking,
    and for a paired binary outcome the exact McNemar (a two-sided binomial on
    the discordant pairs) is the right instrument. Reported alongside the rank
    test, never instead of it.
    """
    ps = sorted(set(a) & set(b) - (skip or set()))
    a_only = sum(1 for p in ps if a[p].get(field) and not b[p].get(field))
    b_only = sum(1 for p in ps if b[p].get(field) and not a[p].get(field))
    n = a_only + b_only
    p_val = None
    if n:
        k = min(a_only, b_only)
        p_val = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {
        "n": len(ps), "a": sum(1 for p in ps if a[p].get(field)),
        "b": sum(1 for p in ps if b[p].get(field)),
        "a_only": a_only, "b_only": b_only, "p": p_val,
    }


def fmt_mc(m: dict, label: str, width: int = 56) -> str:
    p = "    n/a" if m["p"] is None else f"{m['p']:7.4f}"
    verdict = "tie"
    if m["p"] is not None and m["p"] < 0.05:
        verdict = "WINS" if m["a_only"] > m["b_only"] else "loses"
    return (f"{label:{width}s} n={m['n']:3d} repairs {m['a']:2d} vs {m['b']:2d} "
            f"(discordant {m['a_only']}/{m['b_only']}) p={p} {verdict}")


def fmt(s: dict, label: str, width: int = 56) -> str:
    p = "    n/a" if s["p"] is None else f"{s['p']:7.4f}"
    verdict = "tie"
    if s["p"] is not None and s["p"] < 0.05:
        verdict = "WINS" if s["mean_delta"] < 0 else "loses"
    return (f"{label:{width}s} n={s['n']:3d} W/L/T={s['wins']:2d}/{s['losses']:2d}/"
            f"{s['ties']:2d} mean={s['mean_delta']:+7.2f} "
            f"CI[{s['ci'][0]:+7.2f},{s['ci'][1]:+7.2f}] p={p} {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", action="append", required=True)
    ap.add_argument("--campaigns", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    args = ap.parse_args()

    benches: dict[str, dict[str, dict[str, dict]]] = {}
    for i, rd in enumerate(args.results):
        label = args.label[i] if i < len(args.label) else Path(rd).name
        arms: dict[str, dict[str, dict]] = defaultdict(dict)
        for line in (Path(rd) / "per_campaign.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            arms[key_of(r)][r["parent_id"]] = r
        benches[label] = arms

    for label, arms in benches.items():
        n_par = len(arms.get(BAR, {}))
        print("=" * 112)
        print(f"BENCHMARK {label}   ({n_par} parents)")
        print("=" * 112)
        print(f"{'arm':58s} {'n':>3s} {'regret':>7s} {'qual':>5s} {'CHR':>4s} "
              f"{'brk':>4s} {'calls':>6s} {'cost$':>7s}")
        for arm in sorted(arms, key=lambda a: mean(
                v["regret"] for v in arms[a].values())):
            v = list(arms[arm].values())
            tag = "  <- the bar" if arm == BAR else \
                  "  <- ceiling" if arm == CEILING else ""
            print(f"{arm:58s} {len(v):3d} {mean(x['regret'] for x in v):7.2f} "
                  f"{sum(1 for x in v if x.get('qualified')):5d} "
                  f"{sum(1 for x in v if x.get('latent_contract_holding')):4d} "
                  f"{sum(1 for x in v if x.get('resolved_broken')):4d} "
                  f"{sum(x.get('llm_calls') or 0 for x in v):6d} "
                  f"{sum(x.get('cost_usd') or 0.0 for x in v):7.3f}{tag}")
        print()

    # ---- the declared comparisons ---------------------------------------
    # Built by pattern rather than hard-coded arm names, so an arm that was not
    # run is simply absent instead of silently matching the wrong row.
    def find(arms, pred):
        return sorted(a for a in arms if pred(a))

    pooled: dict[str, list[float]] = defaultdict(list)
    pooled_mc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_parents: set[str] = set()
    for label, arms in benches.items():
        skip = set(seen_parents)
        print("=" * 112)
        print(f"COMPARISONS on {label}")
        print("=" * 112)

        tools = find(arms, lambda a: "toolselect" in a)
        masks = find(arms, lambda a: "onecall" in a or "twocall" in a
                     and not a.startswith("rule:"))
        masks = [m for m in masks if not m.startswith("rule:")]

        print("-- 1. the tool action space vs the bar " + "-" * 60)
        for t in tools:
            if BAR in arms:
                s = stats(paired(arms[t], arms[BAR]))
                print("   " + fmt(s, t))
                print("     " + fmt_mc(mcnemar(
                    arms[t], arms[BAR], "latent_contract_holding"), "^ repairs"))
                pooled[f"tool_vs_bar::{t.split('@')[-1]}"] += paired(
                    arms[t], arms[BAR], skip)
                _m = mcnemar(arms[t], arms[BAR], "latent_contract_holding", skip)
                pooled_mc[f"tool_vs_bar::{t.split('@')[-1]}"].append(
                    (_m["a_only"], _m["b_only"]))

        print("-- 2. SAME MODEL, tool space vs mask authoring " + "-" * 52)
        for t in tools:
            model = t.split("|")[0]
            for m in masks:
                if m.split("|")[0] != model:
                    continue
                s = stats(paired(arms[t], arms[m]))
                print("   " + fmt(s, f"{t}  vs  {m.split('|')[-1]}"))
                print("     " + fmt_mc(mcnemar(
                    arms[t], arms[m], "latent_contract_holding"), "^ repairs"))
                pooled["tool_vs_mask"] += paired(arms[t], arms[m], skip)
                _m = mcnemar(arms[t], arms[m], "latent_contract_holding", skip)
                pooled_mc["tool_vs_mask"].append((_m["a_only"], _m["b_only"]))

        print("-- 3. LLM selector vs the DETERMINISTIC router, same menu " + "-" * 40)
        for t in tools:
            menu = t.split("menu")[-1] if "menu" in t else ""
            router = [a for a in arms if a.startswith("rule:tool_router")
                      and (f"menu{menu}" in a or not menu)]
            router.sort(key=lambda a: ("router2" not in a.replace(
                "tool_router2", "router2"),))
            for rt in router:
                s = stats(paired(arms[t], arms[rt]))
                print("   " + fmt(s, f"{t}  vs  router@menu{menu}"))
                print("     " + fmt_mc(mcnemar(
                    arms[t], arms[rt], "latent_contract_holding"), "^ repairs"))
                pooled["llm_vs_router"] += paired(arms[t], arms[rt], skip)
                _m = mcnemar(arms[t], arms[rt], "latent_contract_holding", skip)
                pooled_mc["llm_vs_router"].append((_m["a_only"], _m["b_only"]))

        print("-- 4. the gain-aware depth instruction, mask space " + "-" * 48)
        for m in masks:
            if "symdepth" not in m:
                continue
            base = m.replace("+symdepth", "")
            if base in arms:
                s = stats(paired(arms[m], arms[base]))
                print("   " + fmt(s, f"{m}  vs  same without symdepth"))
                pooled["symdepth"] += paired(arms[m], arms[base], skip)

        seen_parents |= set(arms.get(BAR, {}))
        print("-- 5. new rules vs the bar " + "-" * 72)
        for a in find(arms, lambda a: a.startswith(
                ("rule:headroom", "rule:rebuild_cheap2", "rule:tool_router"))):
            if BAR in arms:
                print("   " + fmt(stats(paired(arms[a], arms[BAR])), a))
        print()

    # ---- pooled across benchmarks ---------------------------------------
    if len(benches) > 1 and pooled:
        print("=" * 112)
        print("POOLED ACROSS BENCHMARKS -- per-parent paired deltas, parents "
              "de-duplicated (a parent in two benchmarks counts once)")
        print("=" * 112)
        for name, d in sorted(pooled.items()):
            if d:
                print("   " + fmt(stats(d), name, width=48))
                mc = pooled_mc.get(name) or []
                if mc:
                    ao = sum(x for x, _ in mc)
                    bo = sum(y for _, y in mc)
                    n = ao + bo
                    pv = (min(1.0, 2.0 * sum(comb(n, i) for i in range(
                        min(ao, bo) + 1)) / (2 ** n)) if n else None)
                    pvs = "    n/a" if pv is None else f"{pv:7.4f}"
                    vd = "tie" if pv is None or pv >= 0.05 else (
                        "WINS" if ao > bo else "loses")
                    print(f"     {'^ repairs (discordant pairs)':48s} "
                          f"{ao} vs {bo}   p={pvs} {vd}")
        print()

    # ---- tool choices ----------------------------------------------------
    for cd in args.campaigns:
        for f in sorted(Path(cd).glob("campaigns__*.jsonl")):
            if "toolselect" not in f.name and "tool_router" not in f.name:
                continue
            picks: Counter = Counter()
            terms: Counter = Counter()
            n = 0
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                picks.update((r.get("usage") or {}).get("tool_choices") or [])
                terms[r.get("termination") or "-"] += 1
            if not n:
                continue
            total = sum(picks.values()) or 1
            print(f"  {f.name}  (n={n})")
            print("    " + "  ".join(
                f"{k}={v}({100*v/total:.0f}%)" for k, v in picks.most_common()))
            print(f"    terminations: {dict(terms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
