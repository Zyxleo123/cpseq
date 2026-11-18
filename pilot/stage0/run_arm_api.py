"""Run one API arm over the frozen decision set, through the Batch API.

Batching because this workload is exactly what it is for: a fixed, known set of
prompts with no latency requirement. It bills at 50% of the synchronous rate,
which on a fixed pilot budget is the difference between affording one arm and
affording two.

The prompt, the evidence table, the candidate order and the `<choice>` parsing
are identical to the local-weights arm -- see `AnthropicProposer` for why that
matters.

Three budget-safety properties, in order of how much they would cost to get
wrong:

1. The batch id is written to disk BEFORE any polling. If this process dies,
   a re-run resumes that batch instead of submitting a second one. Without
   this, a crash loop silently bills the whole arm again each time.
2. A projected cost is computed from measured token counts and compared with
   `--max-spend-usd` BEFORE submission. Over budget, nothing is sent.
3. Every row carries its own measured `cost_usd` and `usage`, and an arm-level
   ledger is written beside the rows, so the spend is recoverable from the
   artefacts rather than from a console that has scrolled away.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from envload import require_api_key
from pricing import PRICES_AS_OF, cost_usd
from prompts import RETRY_REASON, build_prompt, parse_choice

POLL_SECONDS = 20


def build_requests(decisions, feats, cons, evidence_arm, order_key, model,
                   effort, max_tokens):
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    reqs, meta = [], {}
    for d in decisions:
        order = d.get(order_key) or [c["cand_id"] for c in d["candidates"]]
        system, brief = build_prompt(
            d, feats.get(d["complex_id"], {}), cons.get(d["complex_id"], {}),
            evidence_arm, order,
        )
        # custom_id must survive a round trip and be matched back by value:
        # batch results arrive in ANY order, never by position. The API
        # enforces ^[a-zA-Z0-9_-]{1,64}$, so decision ids are sanitised rather
        # than passed through, and collisions after truncation are resolved
        # here -- two decisions sharing a custom_id would silently map both
        # results onto one row.
        cid = "".join(
            ch if (ch.isalnum() or ch in "_-") else "-" for ch in d["decision_id"]
        )[:64]
        if cid in meta:
            n = 1
            while f"{cid[:60]}-{n}" in meta:
                n += 1
            cid = f"{cid[:60]}-{n}"
        meta[cid] = {"decision": d, "order": order}
        reqs.append(
            Request(
                custom_id=cid,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": brief}],
                    output_config={"effort": effort},
                ),
            )
        )
    return reqs, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--conservation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default="ala_k1")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--order", default="main", choices=["main", "alt"])
    ap.add_argument("--evidence-arm", default="full")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--max-spend-usd", type=float, default=0.0)
    ap.add_argument("--est-output-tokens", type=float, default=420.0,
                    help="measured mean output tokens, for the pre-submit "
                         "budget check (see estimate_cost.py)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-wait-min", type=float, default=240.0)
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    if not require_api_key():
        print("[api] GATE FAILED: no ANTHROPIC_API_KEY available. "
              "Nothing submitted, nothing spent. Exiting 0.")
        return 0

    bench = Path(args.bench)
    decisions = json.loads((bench / "decisions_public.json").read_text())
    feats = json.loads((Path(args.features) / "features.json").read_text())
    cons_path = Path(args.conservation) / "conservation.json"
    cons = json.loads(cons_path.read_text()) if cons_path.exists() else {}

    keep_s = {s for s in args.slices.split(",") if s}
    keep_p = {s for s in args.splits.split(",") if s}
    decisions = [
        d for d in decisions
        if d["slice"] in keep_s and d["split"] in keep_p and d["complex_id"] in feats
    ]
    decisions.sort(key=lambda d: d["decision_id"])
    if args.limit:
        decisions = decisions[: args.limit]
    if not decisions:
        print("[api] GATE FAILED: no decisions selected. Exiting 0.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}__{args.order}"
    dest = out_dir / f"choices__{tag}.jsonl"
    batch_file = out_dir / f"batch__{tag}.json"
    ledger_file = out_dir / f"cost__{tag}.json"

    done: set[str] = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["decision_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [d for d in decisions if d["decision_id"] not in done]
    print(f"[api] arm={args.arm} model={args.model} effort={args.effort}")
    print(f"[api] decisions={len(decisions)} done={len(done)} todo={len(todo)}")
    if not todo:
        print("[api] nothing to do")
        return 0

    import anthropic
    client = anthropic.Anthropic(max_retries=4, timeout=120.0)

    order_key = "presentation_order" if args.order == "main" else "presentation_order_alt"
    reqs, meta = build_requests(
        todo, feats, cons, args.evidence_arm, order_key, args.model,
        args.effort, args.max_tokens,
    )

    # ---- resume an existing batch rather than paying for a second one ----
    batch_id = None
    if batch_file.exists():
        try:
            saved = json.loads(batch_file.read_text())
            if saved.get("model") == args.model and saved.get("arm") == args.arm:
                batch_id = saved.get("batch_id")
                print(f"[api] RESUMING existing batch {batch_id} "
                      f"(submitted {saved.get('submitted_at')}) - not resubmitting")
        except json.JSONDecodeError:
            pass

    if batch_id is None:
        # ---- pre-submit budget gate ----
        total_in = 0
        for r in reqs:
            p = r["params"] if isinstance(r, dict) else r.params
            sys_text = p["system"] if isinstance(p, dict) else p.system
            msg = (p["messages"] if isinstance(p, dict) else p.messages)[0]
            body = msg["content"] if isinstance(msg, dict) else msg.content
            total_in += client.messages.count_tokens(
                model=args.model, system=sys_text,
                messages=[{"role": "user", "content": body}],
            ).input_tokens
        projected_sync = cost_usd(
            args.model, total_in, args.est_output_tokens * len(reqs)
        )
        projected = projected_sync * 0.5  # Batch API bills at 50%
        print(f"[api] projected batch cost ${projected:.4f} "
              f"({len(reqs)} requests, {total_in:,} input tokens, "
              f"~{args.est_output_tokens:.0f} output tokens each, 50% batch rate)")
        if args.max_spend_usd and projected > args.max_spend_usd:
            print(f"[api] GATE FAILED: projected ${projected:.4f} exceeds "
                  f"--max-spend-usd ${args.max_spend_usd:.4f}. "
                  f"NOTHING SUBMITTED. Exiting 0.")
            return 0

        batch = client.messages.batches.create(requests=reqs)
        batch_id = batch.id
        # Written BEFORE any polling: a crash after this point resumes.
        batch_file.write_text(json.dumps({
            "batch_id": batch_id, "arm": args.arm, "model": args.model,
            "evidence_arm": args.evidence_arm, "effort": args.effort,
            "n_requests": len(reqs), "projected_usd": round(projected, 6),
            "input_tokens": total_in, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=1))
        print(f"[api] submitted batch {batch_id}; id saved to {batch_file}")

    # ---- poll ----
    deadline = time.time() + args.max_wait_min * 60
    status = None
    while time.time() < deadline:
        b = client.messages.batches.retrieve(batch_id)
        status = b.processing_status
        counts = getattr(b, "request_counts", None)
        print(f"[api] status={status} counts={counts}", flush=True)
        if status == "ended":
            break
        time.sleep(POLL_SECONDS)
    if status != "ended":
        print(f"[api] batch {batch_id} not finished within "
              f"{args.max_wait_min} min. The id is saved; rerun this job to "
              f"resume polling. Exiting 0 so dependents can report the same.")
        return 0

    # ---- collect ----
    spent = 0.0
    n_ok = n_bad = 0
    tot_in = tot_out = 0
    with dest.open("a") as fh:
        for item in client.messages.batches.results(batch_id):
            cid = item.custom_id
            info = meta.get(cid)
            if info is None:
                continue
            d = info["decision"]
            order = info["order"]
            valid = {c["cand_id"] for c in d["candidates"]}
            cand, status_s, raw = None, "batch_error", ""
            usage = {"input_tokens": 0, "output_tokens": 0,
                     "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
            row_cost = 0.0

            if item.result.type == "succeeded":
                msg = item.result.message
                for k in usage:
                    usage[k] = int(getattr(msg.usage, k, 0) or 0)
                # Batch results bill at 50% of the standard rate.
                row_cost = 0.5 * cost_usd(
                    args.model, usage["input_tokens"], usage["output_tokens"],
                    usage["cache_read_input_tokens"],
                    usage["cache_creation_input_tokens"],
                )
                spent += row_cost
                tot_in += usage["input_tokens"]
                tot_out += usage["output_tokens"]
                if getattr(msg, "stop_reason", None) == "refusal":
                    det = getattr(msg, "stop_details", None)
                    status_s = f"refusal_{getattr(det, 'category', None) or 'unknown'}"
                else:
                    raw = "".join(
                        b.text for b in msg.content
                        if getattr(b, "type", "") == "text"
                    )
                    cand, status_s = parse_choice(raw, valid)
                    if cand is None:
                        status_s = f"invalid_{status_s}"
            else:
                status_s = f"batch_{item.result.type}"

            row = {
                "arm": args.arm,
                "evidence_arm": args.evidence_arm,
                "order": args.order,
                "model_id": args.model,
                "decision_id": d["decision_id"],
                "slice": d["slice"],
                "split": d["split"],
                "complex_id": d["complex_id"],
                "k": d["k"],
                "n_candidates": d["n_candidates"],
                "cand_id": cand,
                "choice_valid": cand is not None,
                "status": status_s,
                "n_retries": 0,
                "presented_index": order.index(cand) if cand in order else None,
                "seconds": None,
                "temperature": None,
                "effort": args.effort,
                "usage": usage,
                "cost_usd": round(row_cost, 8),
                "billing": "batch_50pct",
            }
            if args.save_raw:
                row["raw"] = raw
            fh.write(json.dumps(row) + "\n")
            n_ok += int(cand is not None)
            n_bad += int(cand is None)
    try:
        os.fsync(dest.open("a").fileno())
    except OSError:
        pass

    ledger = {
        "arm": args.arm, "model": args.model, "batch_id": batch_id,
        "evidence_arm": args.evidence_arm, "effort": args.effort,
        "prices_as_of": PRICES_AS_OF, "billing": "batch_50pct",
        "requests": len(reqs), "valid_choices": n_ok, "invalid_choices": n_bad,
        "input_tokens": tot_in, "output_tokens": tot_out,
        "mean_input_tokens": round(tot_in / max(len(reqs), 1), 1),
        "mean_output_tokens": round(tot_out / max(len(reqs), 1), 1),
        "spent_usd": round(spent, 6),
        "usd_per_decision": round(spent / max(len(reqs), 1), 6),
    }
    ledger_file.write_text(json.dumps(ledger, indent=1))
    print(f"[api] wrote {dest}")
    print(f"[api] valid {n_ok} / invalid {n_bad}")
    print(f"[api] SPENT ${spent:.4f} "
          f"(${ledger['usd_per_decision']:.5f}/decision, batch rate)")
    print(f"[api] ledger -> {ledger_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
