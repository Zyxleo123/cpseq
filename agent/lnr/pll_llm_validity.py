"""Is the `iter-llm` arm actually an LLM arm?

This exists because of a specific, silent failure mode. `llm_offspring` falls
back to a RANDOM-EA child whenever the model declines, malforms, errors, or the
spend cap is reached -- deliberately, so one dead generation does not stall a
run. The consequence is that an `iter-llm` row whose budget ran out is a
random-EA row wearing an LLM label. It is still written, still graded, and its
regret still enters the mean.

That matters most in the direction the comparison is most likely to go. A
contaminated LLM arm converges on the EA arm's result, so the report says
"no difference" -- and the honest reading of a true no-difference here is
"the objective is separable and the EA already saturates it", which is a
scientific conclusion. Mistaking an unfunded run for that conclusion is the
cheapest available way to get the science wrong.

The cap makes this easy to hit. The client -- and therefore the ledger -- is
built ONCE in `seqmut_iter.main()` and reused for every target, so
`--max-spend-usd` is a budget for the WHOLE RUN, not per target. At one call
per generation, a 29-target run at `n_gen=80` needs on the order of 2,300
calls, and the cap is spent in target order: the early targets get a real LLM
and the tail gets random-EA. Target order then correlates with arm quality,
which no amount of pairing can fix.

So: never read the regret of an `iter-llm` run without reading this first.

Run:
  python agent/lnr/pll_llm_validity.py --rows seqmut_iter_pll_rows.LLM.jsonl
  python agent/lnr/pll_llm_validity.py --rows ... --cap 1.0 --out-json v.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# A status that means the model did NOT produce a usable proposal for that
# generation, so the generation was filled by random-EA instead. `ok_none` is
# NOT here: "I choose not to mutate" is a real answer from the model.
FALLBACK_STATUSES = ("budget_cap", "error", "declined", "malformed")

# Reaching the spend cap is categorically worse than a malformed reply: it is
# not random across targets, it is absorbing and ordered. Once spent, every
# later target is random-EA, so it invalidates the tail rather than adding noise.
FATAL_STATUSES = ("budget_cap",)

# Above this fraction of fallback generations, the arm is not meaningfully an
# LLM arm on that target. It is a judgement call, stated once here rather than
# buried in a report: at half the generations random, the regret is a blend of
# two operators and attributing it to either is wrong.
FALLBACK_FRAC_SUSPECT = 0.5


def load(path, arm="iter-llm"):
    """The LLM-arm rows with their per-run LLM accounting."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if arm and r.get("mut_arm") != arm:
            continue
        it = r.get("iter") or {}
        sc = dict(it.get("status_counts") or {})
        n_gen = int(it.get("n_gen_run") or 0)
        fb_gens = int(it.get("fallback_gens") or 0)
        out.append({
            "target_id": r.get("target_id"),
            "arm": r.get("mut_arm"),
            "regret": it.get("regret"),
            "n_gen_run": n_gen,
            "llm_calls": int(it.get("llm_calls") or 0),
            "cost_usd": float(it.get("llm_cost_usd") or 0.0),
            "status_counts": sc,
            "fallback_gens": fb_gens,
            "fallback_frac": (fb_gens / n_gen) if n_gen else 0.0,
            "n_fatal": sum(sc.get(s, 0) for s in FATAL_STATUSES),
            "n_ok": sum(v for k, v in sc.items() if k.startswith("ok")),
            "n_error": int(sc.get("error", 0)),
            "error_detail": (it.get("llm_error_detail") or "")[:300],
        })
    return out


def verdict(rows, cap=None):
    """A fleet-level judgement, plus the per-target facts behind it."""
    n = len(rows)
    spend = round(sum(r["cost_usd"] for r in rows), 6)
    fatal = [r for r in rows if r["n_fatal"] > 0]
    # Zero USABLE proposals, whether because no call was made or because every
    # call failed. Both mean the row is a random-EA row under an LLM label, so
    # the predicate is `n_ok == 0` rather than `llm_calls == 0`: in job 119773
    # 6qg8 and 6seo each made 80 calls that ALL errored (quota exhausted, $0.00
    # billed), and counting calls instead of successes filed them as merely
    # "suspect" when they carried no LLM signal whatsoever.
    never = [r for r in rows if r["n_ok"] == 0]
    never_ids = {r["target_id"] for r in never}
    suspect = [r for r in rows
               if r["fallback_frac"] > FALLBACK_FRAC_SUSPECT
               and not r["n_fatal"] and r["target_id"] not in never_ids]
    clean = [r for r in rows if r["fallback_gens"] == 0]
    out = {
        "n_targets": n,
        "total_spend_usd": spend,
        "cap_usd": cap,
        "n_clean": len(clean),
        "n_fatal_budget_cap": len(fatal),
        "n_suspect_fallback": len(suspect),
        "n_never_called": len(never),
        "fatal_targets": [r["target_id"] for r in fatal],
        "suspect_targets": [r["target_id"] for r in suspect],
        "never_called_targets": [r["target_id"] for r in never],
        # Ready to paste into `pll_regret.py --exclude`: the targets that must
        # come out of any LLM-vs-EA comparison built from this file.
        "exclude_arg": ",".join(sorted(
            {r["target_id"] for r in fatal} | never_ids)),
        "error_details": sorted({r["error_detail"] for r in rows
                                 if r["error_detail"]}),
        "mean_fallback_frac": (round(sum(r["fallback_frac"] for r in rows) / n, 4)
                               if n else 0.0),
        "mean_calls": (round(sum(r["llm_calls"] for r in rows) / n, 1) if n else 0.0),
        "mean_cost_per_target": (round(spend / n, 6) if n else 0.0),
    }
    # The gradation is about what a reader may conclude, not about neatness.
    if n == 0:
        out["verdict"] = "NO_ROWS"
    elif fatal or never:
        out["verdict"] = "INVALID_TAIL"
    elif suspect:
        out["verdict"] = "PARTIAL"
    else:
        out["verdict"] = "VALID"
    if cap and spend >= 0.98 * float(cap):
        out["verdict"] = "INVALID_TAIL"
        out["cap_exhausted"] = True
    return out


def projected_cap(rows, n_targets_total):
    """What the cap needed to be, from the targets that actually ran.

    Reported because the useful output of a capped run is the number that makes
    the next one valid, and the per-target cost is only knowable by measuring.
    """
    paid = [r for r in rows if r["llm_calls"] > 0 and r["cost_usd"] > 0]
    if not paid:
        return None
    per = sum(r["cost_usd"] for r in paid) / len(paid)
    return {"measured_targets": len(paid),
            "mean_cost_per_target": round(per, 6),
            "n_targets_total": n_targets_total,
            "projected_total_usd": round(per * n_targets_total, 4),
            "recommended_cap_usd": round(per * n_targets_total * 1.3, 2)}


def report(path, cap=None, out_json=None, arm="iter-llm", n_expected=0):
    rows = load(path, arm=arm)
    v = verdict(rows, cap=cap)
    print(f"LLM-ARM VALIDITY -- {path}")
    print(f"arm={arm}  targets={v['n_targets']}  spend=${v['total_spend_usd']:.4f}"
          + (f"  cap=${cap}" if cap else ""))
    if not rows:
        print("GATE: no rows for this arm. If the job printed "
              "'GATE: no OPENAI_API_KEY' it exited 0 having done nothing.")
        if out_json:
            Path(out_json).write_text(json.dumps({"verdict": "NO_ROWS"}, indent=2))
        return 1

    print(f"\n  {'target':>8s} {'regret':>8s} {'calls':>6s} {'cost$':>8s} "
          f"{'fbGens':>7s} {'fb%':>6s}  statuses")
    for r in sorted(rows, key=lambda r: -r["fallback_frac"]):
        flag = "  <-- BUDGET CAP" if r["n_fatal"] else (
            "  <-- mostly random" if r["fallback_frac"] > FALLBACK_FRAC_SUSPECT else "")
        reg = "None" if r["regret"] is None else f"{r['regret']:.4f}"
        sc = ",".join(f"{k}:{v}" for k, v in sorted(r["status_counts"].items()))
        print(f"  {r['target_id']:>8s} {reg:>8s} {r['llm_calls']:>6d} "
              f"{r['cost_usd']:>8.4f} {r['fallback_gens']:>7d} "
              f"{100*r['fallback_frac']:>5.1f}%  {sc}{flag}")

    for d in v.get("error_details", []):
        print(f"\n  ERROR REPORTED BY THE API: {d}")
    print(f"\n  clean (no fallback at all) : {v['n_clean']}/{v['n_targets']}")
    print(f"  mean fallback fraction     : {v['mean_fallback_frac']:.1%}")
    print(f"  mean LLM calls per target  : {v['mean_calls']}")
    print(f"  mean cost per target       : ${v['mean_cost_per_target']:.4f}")

    proj = projected_cap(rows, n_expected or v["n_targets"])
    if proj:
        print(f"\n  measured ${proj['mean_cost_per_target']:.4f}/target over "
              f"{proj['measured_targets']} paid target(s)")
        print(f"  -> {proj['n_targets_total']} targets projects to "
              f"${proj['projected_total_usd']:.2f}; "
              f"set --max-spend-usd {proj['recommended_cap_usd']} (30% headroom)")
        v["projection"] = proj

    print(f"\n  VERDICT: {v['verdict']}")
    if v["verdict"] == "INVALID_TAIL":
        print("  The spend cap was reached, or some target never called the "
              "model. Those targets ran as RANDOM-EA under an LLM label, and\n"
              "  the cap is spent in TARGET ORDER, so the contamination is not "
              "random -- it is the whole tail of the run.\n"
              "  Do NOT report an LLM-vs-EA regret difference from this file, "
              "and in particular do NOT read a tie as 'the objective is\n"
              "  separable': an unfunded arm converges on the EA arm by "
              "construction. Raise the cap and rerun the affected targets.")
        if v["fatal_targets"]:
            print(f"  budget-capped: {', '.join(v['fatal_targets'])}")
        if v["never_called_targets"]:
            print(f"  no usable LLM proposal at all: "
                  f"{', '.join(v['never_called_targets'])}")
        if v.get("exclude_arg"):
            print(f"\n  To grade the FUNDED subset, drop these targets:\n"
                  f"    --exclude {v['exclude_arg']}")
    elif v["verdict"] == "PARTIAL":
        print("  Some targets fell back for more than half their generations "
              "without hitting the cap (declines/malformed replies).\n"
              "  Their regret is a blend of two operators; grade with those "
              "targets excluded as well as included.")
        print(f"  suspect: {', '.join(v['suspect_targets'])}")
    else:
        print("  Every target called the model and no target exhausted the "
              "budget, so an LLM-vs-EA difference from this file is "
              "attributable to the operator.")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps({"summary": v, "targets": rows},
                                             indent=2))
        print(f"\nwrote {out_json}")
    # Non-zero only for a hard gate failure, so a caller can branch on it.
    return 2 if v["verdict"] == "INVALID_TAIL" else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--arm", default="iter-llm")
    ap.add_argument("--cap", type=float, default=0.0,
                    help="the run's --max-spend-usd, so the report can say "
                         "whether the budget was the binding constraint")
    ap.add_argument("--n-expected", type=int, default=0,
                    help="targets the run was supposed to cover; used to "
                         "project the cap a valid run would need")
    ap.add_argument("--out-json", default="")
    a = ap.parse_args()
    return report(a.rows, cap=(a.cap or None), out_json=(a.out_json or None),
                  arm=a.arm, n_expected=a.n_expected)


if __name__ == "__main__":
    raise SystemExit(main())
