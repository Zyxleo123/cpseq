"""What an API arm will cost, measured before it is run.

Two stages, because the two halves of the bill behave differently:

  Input tokens are known exactly and for free. The prompts are deterministic,
  so `messages.count_tokens` gives the true count without running inference.

  Output tokens cannot be known in advance. On the current Opus and Sonnet
  generation thinking is always on and bills as output, so the output side is
  dominated by how much the model thinks -- which depends on the effort level
  and on the prompt. The only honest way to get it is to run a few real calls
  and measure. `--calibrate N` does exactly that, for N decisions, and reports
  what it spent doing so.

Everything else in this file is arithmetic on those two numbers.

    python estimate_cost.py --bench ... --features ... --conservation ... \
        --slice ala_k1 --models claude-opus-5-5,claude-sonnet-5-5 --calibrate 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from pricing import PRICES_AS_OF, PRICES_PER_MTOK, cost_usd
from prompts import build_prompt


def load(bench: Path, features: Path, conservation: Path, slice_name: str,
         splits: set[str]):
    decisions = [
        d for d in json.loads((bench / "decisions_public.json").read_text())
        if d["slice"] == slice_name and d["split"] in splits
    ]
    feats = json.loads((features / "features.json").read_text())
    cons_path = conservation / "conservation.json"
    cons = json.loads(cons_path.read_text()) if cons_path.exists() else {}
    decisions = [d for d in decisions if d["complex_id"] in feats]
    decisions.sort(key=lambda d: d["decision_id"])
    return decisions, feats, cons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--conservation", required=True)
    ap.add_argument("--slice", default="ala_k1")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--evidence-arm", default="full")
    ap.add_argument("--models", default="claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="run this many REAL calls to measure output tokens "
                         "(costs money; the amount is reported)")
    ap.add_argument("--count-model", default="claude-opus-5-5",
                    help="model used for the free token count; tokenisation is "
                         "shared across the current generation")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    decisions, feats, cons = load(
        Path(args.bench), Path(args.features), Path(args.conservation),
        args.slice, {s for s in args.splits.split(",") if s},
    )
    if not decisions:
        print(f"[cost] GATE: no decisions for slice {args.slice}. Exiting 0.")
        return 0

    prompts = [
        build_prompt(
            d, feats.get(d["complex_id"], {}), cons.get(d["complex_id"], {}),
            args.evidence_arm, d["presentation_order"],
        )
        for d in decisions
    ]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    from envload import require_api_key
    if not require_api_key():
        print("[cost] GATE FAILED: no ANTHROPIC_API_KEY in the environment or "
              "the dotenv file. Nothing was called. Exiting 0.")
        raise SystemExit(0)
    import anthropic
    client = anthropic.Anthropic()

    # ---- exact input tokens, free ----
    print(f"[cost] slice={args.slice} evidence={args.evidence_arm} "
          f"decisions={len(decisions)}")
    print(f"[cost] counting input tokens with {args.count_model} ...")
    counts = []
    for i, (system, brief) in enumerate(prompts, 1):
        r = client.messages.count_tokens(
            model=args.count_model, system=system,
            messages=[{"role": "user", "content": brief}],
        )
        counts.append(int(r.input_tokens))
        if i % 25 == 0 or i == len(prompts):
            print(f"[cost]   {i}/{len(prompts)}")
    total_in = sum(counts)
    mean_in = total_in / len(counts)
    print(f"[cost] input tokens: total {total_in:,}  mean {mean_in:.0f}  "
          f"min {min(counts)}  max {max(counts)}")

    # ---- measured output tokens, costs money ----
    measured: dict[str, dict] = {}
    calib_spend = 0.0
    if args.calibrate:
        rng = random.Random(0)
        idx = rng.sample(range(len(decisions)), min(args.calibrate, len(decisions)))
        for model in models:
            outs, ins = [], []
            for j in idx:
                system, brief = prompts[j]
                resp = client.messages.create(
                    model=model, max_tokens=2000, system=system,
                    messages=[{"role": "user", "content": brief}],
                    output_config={"effort": args.effort},
                )
                u = resp.usage
                ins.append(int(u.input_tokens))
                outs.append(int(u.output_tokens))
                calib_spend += cost_usd(model, int(u.input_tokens), int(u.output_tokens))
            measured[model] = {
                "n": len(outs),
                "mean_output_tokens": sum(outs) / len(outs),
                "max_output_tokens": max(outs),
                "mean_input_tokens": sum(ins) / len(ins),
            }
            print(f"[cost] {model}: measured mean output "
                  f"{measured[model]['mean_output_tokens']:.0f} tokens "
                  f"(max {max(outs)}) over {len(outs)} calls at effort={args.effort}")
        print(f"[cost] calibration spent ${calib_spend:.4f}")

    # ---- the table ----
    print(f"\n=== PROJECTED COST PER ARM ({len(decisions)} decisions, "
          f"slice {args.slice}, effort {args.effort}) ===")
    print(f"prices as of {PRICES_AS_OF}; output includes thinking tokens\n")
    header = (f"  {'model':20s} {'in/dec':>7s} {'out/dec':>8s} "
              f"{'$/dec':>8s} {'$/arm':>8s} {'arms per $5':>12s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = {}
    for model in models:
        if model not in PRICES_PER_MTOK:
            print(f"  {model:20s}  no price on file - skipped")
            continue
        est_out = (
            measured[model]["mean_output_tokens"] if model in measured
            else float(args.calibrate and 0 or 400)
        )
        per_dec = cost_usd(model, mean_in, est_out)
        per_arm = cost_usd(model, total_in, est_out * len(decisions))
        rows[model] = {
            "mean_input_tokens": round(mean_in, 1),
            "mean_output_tokens": round(est_out, 1),
            "output_is_measured": model in measured,
            "usd_per_decision": round(per_dec, 6),
            "usd_per_arm": round(per_arm, 4),
            "decisions": len(decisions),
        }
        flag = "" if model in measured else "  (output ESTIMATED, not measured)"
        print(f"  {model:20s} {mean_in:7.0f} {est_out:8.0f} "
              f"{per_dec:8.4f} {per_arm:8.3f} {5.0 / per_arm:12.1f}{flag}")

    payload = {
        "slice": args.slice,
        "evidence_arm": args.evidence_arm,
        "effort": args.effort,
        "n_decisions": len(decisions),
        "prices_as_of": PRICES_AS_OF,
        "input_tokens_total": total_in,
        "input_tokens_mean": round(mean_in, 1),
        "calibration_calls": args.calibrate,
        "calibration_spend_usd": round(calib_spend, 6),
        "measured": measured,
        "projected": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(f"\n[cost] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
