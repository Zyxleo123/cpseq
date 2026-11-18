"""Score the arms: regret, paired tests, power, and the confound checks.

Regret is in kcal/mol of binding energy given up relative to the best
candidate the agent could have chosen:

    regret(d, arm) = value(chosen) - value(oracle)   >= 0

Design choices that matter for reading the output:

Top-1 only. The agentic system's output is meant to be trusted without further
search, so an arm is scored on the single candidate it committed to. Top-3 is
computed as a diagnostic and never as the headline -- quoting best-of-3 would
be reporting a search the system does not get to run.

Invalid answers are dropped pairwise and counted, never imputed. A model that
fails to emit a usable action on a third of decisions is not a model with good
regret on the rest, and filling those rows with a default or a null that later
arithmetic treats as zero would say exactly that.

An analytic `random_expected` arm is computed here rather than sampled: the
expected regret of a uniform pick is the mean candidate value minus the oracle,
exactly, with no seed noise to confuse a paired test.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

MIN_DECISIONS_TESTABLE = 20
ALPHA = 0.05


def load_oracle(bench: Path) -> dict[str, dict]:
    out = {}
    for o in json.loads((bench / "decisions_oracle.json").read_text()):
        # The full-interface arm lists every interface position, and the ones
        # nobody assayed carry value None. They stay in the candidate POOL (the
        # agent could pick them) but out of `values` (they cannot be scored).
        vals = {
            c["cand_id"]: c["value"]
            for c in o["candidates"] if c.get("value") is not None
        }
        pool = [c["cand_id"] for c in o["candidates"]]
        counts = {
            c["cand_id"]: c["n_records"]
            for c in o["candidates"] if c.get("value") is not None
        }
        order = sorted(vals.values())
        out[o["decision_id"]] = {
            "values": vals,
            "counts": counts,
            "oracle": o["oracle_value"],
            "worst": o["worst_value"],
            "median": o["median_value"],
            "spread": o["spread"],
            "mean": sum(vals.values()) / len(vals),
            "sorted": order,
            "slice": o["slice"],
            "split": o["split"],
            "k": o["k"],
            "n_candidates": len(vals),
            "n_pool": len(pool),
            "measured_fraction": o.get(
                "measured_fraction", 1.0 if pool else 0.0
            ),
            "epistasis": {
                c["cand_id"]: c.get("epistasis") for c in o["candidates"]
            },
            "additive": {
                c["cand_id"]: c.get("additive_from_singles")
                for c in o["candidates"]
            },
        }
    return out


def read_rows(arms_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(arms_dir.glob("choices__*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated or NUL-corrupted tail is reported, not skipped
                # silently: it means two writers shared this file.
                print(f"[score] WARNING: unparseable line in {path.name}")
    return rows


def per_decision(rows: list[dict], oracle: dict) -> dict[tuple, dict]:
    """(arm_key, decision_id) -> scored record."""
    out: dict[tuple, dict] = {}
    for r in rows:
        did = r["decision_id"]
        ora = oracle.get(did)
        if ora is None:
            continue
        arm_key = (r["arm"], r.get("order", "main"), r.get("model_id", ""))
        rec = {
            "decision_id": did,
            "slice": ora["slice"],
            "split": ora["split"],
            "k": ora["k"],
            "n_candidates": ora["n_candidates"],
            "valid": bool(r.get("choice_valid")),
            # valid  = the model emitted a usable action
            # scorable = that action landed on a position SKEMPI measured
            # Conflating them would score an unmeasured pick as a model
            # failure, which it is not -- it is a gap in the ground truth.
            "scorable": bool(r.get("choice_valid")) and r.get("cand_id") in ora["values"],
            "n_pool": ora["n_pool"],
            "measured_fraction": ora["measured_fraction"],
            "status": r.get("status", ""),
            "presented_index": r.get("presented_index"),
            "cand_id": r.get("cand_id"),
            "seconds": r.get("seconds"),
            "n_retries": r.get("n_retries", 0),
        }
        if rec["valid"] and r["cand_id"] in ora["values"]:
            v = ora["values"][r["cand_id"]]
            rec.update(
                value=v,
                regret=v - ora["oracle"],
                norm_regret=(
                    (v - ora["oracle"]) / ora["spread"] if ora["spread"] > 1e-9 else 0.0
                ),
                # Fraction of candidates strictly better than the one chosen.
                rank_pct=float(
                    sum(1 for x in ora["values"].values() if x < v - 1e-9)
                ) / ora["n_candidates"],
                is_oracle=abs(v - ora["oracle"]) < 1e-9,
                in_top3=v <= ora["sorted"][min(2, len(ora["sorted"]) - 1)] + 1e-9,
                n_records_chosen=ora["counts"].get(r["cand_id"]),
                epistasis=ora["epistasis"].get(r["cand_id"]),
            )
        else:
            rec.update(value=None, regret=None, norm_regret=None, rank_pct=None,
                       is_oracle=None, in_top3=None, n_records_chosen=None,
                       epistasis=None)
        out[(arm_key, did)] = rec
    return out


def synth_random(oracle: dict, restrict_to: set[str] | None = None) -> dict[tuple, dict]:
    """Exact expected regret of a uniform pick, as its own arm.

    Restricted to the decisions the real arms actually answered. Computing it
    over every decision in the benchmark would put a different denominator in
    the same summary table as the arms it is there to be compared against,
    which reads as a difference in quality rather than in coverage.
    """
    out = {}
    key = ("random_expected", "main", "analytic")
    for did, ora in oracle.items():
        if restrict_to is not None and did not in restrict_to:
            continue
        out[(key, did)] = {
            "decision_id": did, "slice": ora["slice"], "split": ora["split"],
            "k": ora["k"], "n_candidates": ora["n_candidates"], "valid": True,
            "scorable": True, "n_pool": ora["n_pool"],
            "measured_fraction": ora["measured_fraction"],
            "status": "analytic", "presented_index": None, "cand_id": None,
            "value": ora["mean"], "regret": ora["mean"] - ora["oracle"],
            "norm_regret": (
                (ora["mean"] - ora["oracle"]) / ora["spread"]
                if ora["spread"] > 1e-9 else 0.0
            ),
            "rank_pct": float(
                sum(1 for x in ora["values"].values() if x < ora["mean"] - 1e-9)
            ) / ora["n_candidates"],
            "is_oracle": False,
            "in_top3": ora["mean"] <= ora["sorted"][min(2, len(ora["sorted"]) - 1)] + 1e-9,
            "n_records_chosen": None, "epistasis": None, "seconds": None,
            "n_retries": 0,
        }
    return out


def summarise(recs: list[dict]) -> dict:
    answered = [r for r in recs if r["valid"]]
    valid = [r for r in recs if r["valid"] and r["regret"] is not None]
    base = {
        "n": len(recs),
        "n_answered": len(answered),
        "invalid_rate": round(1 - len(answered) / len(recs), 4) if recs else 0.0,
        # Coverage only means something where the pool is wider than the
        # measured set, i.e. the full-interface arm.
        "coverage_of_measured": (
            round(len(valid) / len(answered), 4) if answered else None
        ),
        "blind_pick_coverage": (
            round(float(np.mean(
                [r.get("measured_fraction", 1.0) for r in answered]
            )), 4) if answered else None
        ),
    }
    if not valid:
        return {**base, "n_valid": 0}
    reg = np.array([r["regret"] for r in valid])
    nrm = np.array([r["norm_regret"] for r in valid])
    return {
        **base,
        "n_valid": len(valid),
        "mean_regret": round(float(reg.mean()), 4),
        "median_regret": round(float(np.median(reg)), 4),
        "sd_regret": round(float(reg.std(ddof=1)) if len(reg) > 1 else 0.0, 4),
        "mean_norm_regret": round(float(nrm.mean()), 4),
        # Tail statistics. An arm can match another's oracle-hit rate and still
        # have much worse mean regret, because its misses are bigger -- which
        # is exactly the 8B profile on the full-interface arm. Mean alone
        # cannot distinguish "usually slightly off" from "usually right,
        # occasionally catastrophic", and those call for different fixes.
        "p90_regret": round(float(np.percentile(reg, 90)), 4),
        "max_regret": round(float(reg.max()), 4),
        "frac_regret_gt_2kcal": round(float((reg > 2.0).mean()), 4),
        "frac_regret_gt_half_spread": round(float((nrm > 0.5).mean()), 4),
        "oracle_hit_rate": round(
            float(np.mean([bool(r["is_oracle"]) for r in valid])), 4
        ),
        "top3_rate": round(float(np.mean([bool(r["in_top3"]) for r in valid])), 4),
        "mean_rank_pct": round(
            float(np.mean([r["rank_pct"] for r in valid])), 4
        ),
        "mean_seconds": (
            round(float(np.mean([r["seconds"] for r in valid if r["seconds"]])), 2)
            if any(r["seconds"] for r in valid) else None
        ),
        "retry_rate": round(
            float(np.mean([1.0 if r["n_retries"] else 0.0 for r in recs])), 4
        ),
    }


def paired_test(a: list[dict], b: list[dict]) -> dict:
    """Paired comparison on the decisions where BOTH arms produced a choice."""
    by_a = {r["decision_id"]: r for r in a if r["valid"] and r["regret"] is not None}
    by_b = {r["decision_id"]: r for r in b if r["valid"] and r["regret"] is not None}
    shared = sorted(set(by_a) & set(by_b))
    n_dropped = len(set(by_a) | set(by_b)) - len(shared)
    if len(shared) < 6:
        return {"n_pairs": len(shared), "n_dropped": n_dropped,
                "verdict": "too few pairs to test"}
    da = np.array([by_a[d]["regret"] for d in shared])
    db = np.array([by_b[d]["regret"] for d in shared])
    diff = da - db  # negative => arm A gives up less energy => A is better

    try:
        w_stat, w_p = stats.wilcoxon(da, db, zero_method="zsplit")
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    t_stat, t_p = stats.ttest_rel(da, db)

    # Bootstrap CI on the median paired difference, which is what the Wilcoxon
    # is testing; the mean is reported too but is the more outlier-sensitive of
    # the two on a bounded-below quantity like regret.
    rng = np.random.default_rng(0)
    boot = np.median(
        diff[rng.integers(0, len(diff), size=(4000, len(diff)))], axis=1
    )
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    # Minimum detectable effect, two-sided alpha=0.05, 80% power, paired.
    mde = 2.8 * sd / math.sqrt(len(diff)) if len(diff) else float("nan")

    return {
        "n_pairs": len(shared),
        "n_dropped": n_dropped,
        "median_paired_diff": round(float(np.median(diff)), 4),
        "median_paired_diff_ci95": [
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4),
        ],
        "mean_paired_diff": round(float(diff.mean()), 4),
        "wins": int((diff < -1e-9).sum()),
        "losses": int((diff > 1e-9).sum()),
        "ties": int((np.abs(diff) <= 1e-9).sum()),
        "wilcoxon_p": round(float(w_p), 5),
        "ttest_p": round(float(t_p), 5),
        "sd_paired_diff": round(sd, 4),
        "mde_80pct_power": round(float(mde), 4),
        "significant": bool(w_p < ALPHA),
        "direction": (
            "A better" if np.median(diff) < 0 else
            "B better" if np.median(diff) > 0 else "tie"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="dev,test")
    args = ap.parse_args()

    bench = Path(args.bench)
    oracle = load_oracle(bench)
    rows = read_rows(Path(args.arms))
    print(f"[score] {len(rows)} choice rows over {len(oracle)} decisions")
    if not rows:
        print("[score] GATE FAILED: no arm output found. "
              "The upstream arms either never ran or wrote nowhere. Exiting 0.")
        return 0

    scored = per_decision(rows, oracle)
    answered = {did for (_, did) in scored}
    scored.update(synth_random(oracle, restrict_to=answered))

    by_arm_slice: dict[tuple, list[dict]] = defaultdict(list)
    for (arm_key, _), rec in scored.items():
        by_arm_slice[(arm_key, rec["slice"], rec["split"])].append(rec)
        by_arm_slice[(arm_key, rec["slice"], "all")].append(rec)

    summary = {}
    for (arm_key, slice_name, split), recs in sorted(
        by_arm_slice.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])
    ):
        arm, order, model = arm_key
        summary[f"{arm}|{order}|{model}|{slice_name}|{split}"] = {
            "arm": arm, "order": order, "model_id": model,
            "slice": slice_name, "split": split, **summarise(recs),
        }

    # ---------------- paired comparisons ----------------
    # Every LLM arm against every rule-based bar, plus the bar against random,
    # on each slice and split. The headline for an LLM arm is its result
    # against the STRONGEST bar on that slice.
    arms_present = sorted({k[0] for k in by_arm_slice})
    comparisons = {}
    for slice_name in sorted({k[1] for k in by_arm_slice}):
        for split in [s for s in args.splits.split(",") if s] + ["all"]:
            pool = {
                arm_key: by_arm_slice.get((arm_key, slice_name, split), [])
                for arm_key in arms_present
            }
            pool = {k: v for k, v in pool.items() if v}
            if len(pool) < 2:
                continue
            n_dec = max(len(v) for v in pool.values())
            testable = n_dec >= MIN_DECISIONS_TESTABLE
            for a_key, a_recs in pool.items():
                for b_key, b_recs in pool.items():
                    if a_key >= b_key:
                        continue
                    res = paired_test(a_recs, b_recs)
                    res["testable_slice"] = testable
                    if not testable:
                        res["caveat"] = (
                            f"slice has only {n_dec} decisions "
                            f"(<{MIN_DECISIONS_TESTABLE}); report as a trend, not a test"
                        )
                    label = (
                        f"{a_key[0]}|{a_key[1]} VS {b_key[0]}|{b_key[1]}"
                        f"|{slice_name}|{split}"
                    )
                    comparisons[label] = res

    # ---------------- controls ----------------
    controls: dict = {}

    # 1. Position bias: does an arm prefer early slots, and does it make the
    #    same choice when the list is permuted?
    pos_bias = {}
    for arm_key in arms_present:
        recs_main = [
            r for (k, _), r in scored.items()
            if k == arm_key and r["presented_index"] is not None
        ]
        if not recs_main:
            continue
        idx = np.array([r["presented_index"] for r in recs_main], float)
        ncand = np.array([r["n_candidates"] for r in recs_main], float)
        rel = idx / np.maximum(ncand - 1, 1)
        # Uniform choice gives mean relative slot 0.5.
        pos_bias[f"{arm_key[0]}|{arm_key[1]}"] = {
            "n": len(rel),
            "mean_relative_slot": round(float(rel.mean()), 4),
            "frac_first_slot": round(float((idx == 0).mean()), 4),
            "p_vs_uniform_half": round(
                float(stats.ttest_1samp(rel, 0.5).pvalue), 5
            ) if len(rel) > 2 else None,
        }
    controls["position_bias"] = pos_bias

    # 2. Order stability: same arm, main vs alt presentation order.
    stability = {}
    arms_by_name: dict[str, dict[str, dict]] = defaultdict(dict)
    for (arm_key, did), rec in scored.items():
        arms_by_name[arm_key[0]].setdefault(arm_key[1], {})[did] = rec
    for arm, orders in arms_by_name.items():
        if "main" in orders and "alt" in orders:
            shared = sorted(set(orders["main"]) & set(orders["alt"]))
            if not shared:
                continue
            same = [
                orders["main"][d]["cand_id"] == orders["alt"][d]["cand_id"]
                for d in shared
                if orders["main"][d]["valid"] and orders["alt"][d]["valid"]
            ]
            rm = [orders["main"][d]["regret"] for d in shared
                  if orders["main"][d]["valid"] and orders["alt"][d]["valid"]]
            ra = [orders["alt"][d]["regret"] for d in shared
                  if orders["main"][d]["valid"] and orders["alt"][d]["valid"]]
            stability[arm] = {
                "n": len(same),
                "same_choice_rate": round(float(np.mean(same)), 4) if same else None,
                "mean_regret_main": round(float(np.mean(rm)), 4) if rm else None,
                "mean_regret_alt": round(float(np.mean(ra)), 4) if ra else None,
            }
    controls["order_stability"] = stability

    # 3. Data-density confound: does picking a more-measured candidate pay?
    #    Computed on the all-substitution slice, where the value is a minimum
    #    over however many substitutions happened to be assayed.
    dens = {}
    for slice_name in ("allsub_k1", "allsub_k2"):
        pairs = [
            (r["n_records_chosen"], r["regret"])
            for (_, _), r in scored.items()
            if r["slice"] == slice_name and r["valid"]
            and r["n_records_chosen"] is not None and r["regret"] is not None
        ]
        if len(pairs) > 10:
            x = np.array([p[0] for p in pairs], float)
            y = np.array([p[1] for p in pairs], float)
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                dens[slice_name] = {
                    "n": len(pairs),
                    "spearman_nrecords_vs_regret": None,
                    "reading": "constant input: correlation undefined, not zero",
                }
                continue
            rho, p = stats.spearmanr(x, y)
            dens[slice_name] = {
                "n": len(pairs),
                "spearman_nrecords_vs_regret": round(float(rho), 4),
                "p": round(float(p), 5),
                "reading": (
                    "negative rho => more-measured candidates look better, so "
                    "the all-substitution value model rewards assay density; "
                    "prefer the ala_k1 slice for the headline"
                ),
            }
    controls["data_density_confound"] = dens

    # 4. Is the value function really measuring interface damage? The
    #    deliberate anti-baseline must be worse than random, or the oracle is
    #    not what we think it is.
    sanity = {}
    for slice_name in sorted({k[1] for k in by_arm_slice}):
        anti = by_arm_slice.get((("max_dsasa", "main", "heuristic:max_dsasa"),
                                 slice_name, "all"), [])
        rnd = by_arm_slice.get((("random_expected", "main", "analytic"),
                                slice_name, "all"), [])
        if anti and rnd:
            res = paired_test(anti, rnd)
            sanity[slice_name] = {
                "max_dsasa_vs_random": res,
                "expected": "max_dsasa should be WORSE (positive median diff)",
                "passes": res.get("median_paired_diff", 0) > 0,
            }
    controls["value_function_sanity"] = sanity

    # 5. Coordination headroom. The design doc's open question is whether an
    #    allocator can find JOINT moves that a greedy per-position controller
    #    misses. That is only worth asking if such moves exist in the data, so
    #    it is measured rather than assumed: how far is the best measured pair
    #    from the pair a perfect greedy controller would have picked?
    coord = {}
    for slice_name in ("allsub_k2", "allsub_k3"):
        rows_c = []
        for did, ora in oracle.items():
            if ora["slice"] != slice_name:
                continue
            additive = {
                c: v for c, v in ora["additive"].items()
                if v is not None and c in ora["values"]
            }
            if len(additive) < 2:
                continue
            greedy = min(additive, key=lambda c: (additive[c], c))
            rows_c.append({
                "greedy_value": ora["values"][greedy],
                "headroom": ora["values"][greedy] - ora["oracle"],
                "greedy_is_oracle": abs(ora["values"][greedy] - ora["oracle"]) < 1e-9,
                "epistasis": [v for v in ora["epistasis"].values() if v is not None],
            })
        if not rows_c:
            continue
        head = np.array([r["headroom"] for r in rows_c])
        eps = np.array([e for r in rows_c for e in r["epistasis"]])
        coord[slice_name] = {
            "n_decisions": len(rows_c),
            "greedy_is_oracle_rate": round(
                float(np.mean([r["greedy_is_oracle"] for r in rows_c])), 4),
            "mean_coordination_headroom": round(float(head.mean()), 4),
            "median_coordination_headroom": round(float(np.median(head)), 4),
            "max_coordination_headroom": round(float(head.max()), 4),
            "n_pairs_with_epistasis": int(len(eps)),
            "mean_abs_epistasis": (
                round(float(np.abs(eps).mean()), 4) if len(eps) else None),
            "frac_epistasis_gt_1kcal": (
                round(float((np.abs(eps) > 1.0).mean()), 4) if len(eps) else None),
            "reading": (
                "headroom is the energy a perfect GREEDY controller gives up to "
                "the best measured pair. Near zero => joint moves do not matter "
                "in this data, and no arm can demonstrate coordination here "
                "whatever it scores."),
        }
    controls["coordination_headroom"] = coord

    # 6. Arm agreement. A win means little if the LLM simply reproduces a rule
    #    we already have -- and a LOSS means something different if it is
    #    choosing idiosyncratically rather than tracking any known signal. So
    #    the fraction of decisions on which each pair of arms picked the SAME
    #    candidate is reported alongside regret. High agreement with one
    #    heuristic plus matching regret is the signature of a reimplemented
    #    rule; low agreement with everything is a genuinely different policy,
    #    whatever its score.
    agree = {}
    choices_by_arm: dict[str, dict[str, str]] = defaultdict(dict)
    for (arm_key, did), rec in scored.items():
        if rec["valid"] and rec.get("cand_id"):
            choices_by_arm[f"{arm_key[0]}|{arm_key[1]}"][did] = rec["cand_id"]
    names = sorted(choices_by_arm)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(choices_by_arm[a]) & set(choices_by_arm[b])
            if len(shared) < 10:
                continue
            same = sum(
                1 for d in shared if choices_by_arm[a][d] == choices_by_arm[b][d]
            )
            agree[f"{a} VS {b}"] = {
                "n": len(shared),
                "same_choice_rate": round(same / len(shared), 4),
            }
    controls["arm_agreement"] = agree



    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "bench_meta": json.loads((bench / "bench_meta.json").read_text()),
        "n_choice_rows": len(rows),
        "summary": summary,
        "comparisons": comparisons,
        "controls": controls,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    # Per-decision rows, so figures and any re-analysis never need the models.
    with (out / "per_decision.jsonl").open("w") as fh:
        for (arm_key, _), rec in sorted(scored.items(), key=lambda kv: str(kv[0])):
            fh.write(json.dumps({
                "arm": arm_key[0], "order": arm_key[1], "model_id": arm_key[2], **rec
            }) + "\n")

    # ---------------- console report ----------------
    print("\n=== REGRET BY ARM (kcal/mol given up vs the best candidate) ===")
    for slice_name in sorted({v["slice"] for v in summary.values()}):
        rows_s = [
            v for v in summary.values()
            if v["slice"] == slice_name and v["split"] == "all"
        ]
        if not rows_s:
            continue
        rows_s.sort(key=lambda v: v.get("mean_regret", 9e9))
        print(f"\n-- slice {slice_name} (all splits) --")
        print(f"  {'arm':26s} {'n':>4s} {'inval':>6s} {'mean':>7s} {'med':>7s} "
              f"{'norm':>6s} {'hit':>6s} {'rank%':>6s}")
        for v in rows_s:
            print(f"  {v['arm'][:24]+'|'+v['order'][:1]:26s} {v['n_valid']:4d} "
                  f"{v.get('invalid_rate', 0):6.2f} "
                  f"{v.get('mean_regret', float('nan')):7.3f} "
                  f"{v.get('median_regret', float('nan')):7.3f} "
                  f"{v.get('mean_norm_regret', float('nan')):6.3f} "
                  f"{v.get('oracle_hit_rate', float('nan')):6.2f} "
                  f"{v.get('mean_rank_pct', float('nan')):6.3f}")

    print("\n=== CONTROLS ===")
    for name, blob in controls.items():
        print(f"-- {name}")
        print("   " + json.dumps(blob, sort_keys=True)[:900])
    print(f"\n[score] wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
