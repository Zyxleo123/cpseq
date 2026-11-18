"""Run one arm over the frozen decision set.

One arm, one output file. Concurrent appends from parallel jobs to a shared
file interleave and NUL-corrupt rows, so the filename carries the arm, the
order tag and the slice, and nothing else writes to it.

Resumable: already-answered decision_ids are skipped, so hitting a time limit
costs a resubmission rather than the pass.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from proposers import CONTROL_ARMS, HEURISTIC_ARMS, HeuristicProposer, LocalHFProposer


def load_inputs(bench: Path, features: Path, conservation: Path):
    decisions = json.loads((bench / "decisions_public.json").read_text())
    feats = json.loads((features / "features.json").read_text())
    cons_path = conservation / "conservation.json"
    cons = json.loads(cons_path.read_text()) if cons_path.exists() else {}
    return decisions, feats, cons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--conservation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default="ala_k1,allsub_k1,allsub_k2")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--order", default="main", choices=["main", "alt"])
    ap.add_argument("--model-path", default="")
    ap.add_argument("--evidence-arm", default="full")
    ap.add_argument("--device-map", default="")
    ap.add_argument("--load-4bit", action="store_true",
                    help="NF4 double-quantised load; records a distinct model_id")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    bench = Path(args.bench)
    decisions, feats, cons = load_inputs(
        bench, Path(args.features), Path(args.conservation)
    )

    keep_slices = {s for s in args.slices.split(",") if s}
    keep_splits = {s for s in args.splits.split(",") if s}
    decisions = [
        d
        for d in decisions
        if d["slice"] in keep_slices and d["split"] in keep_splits
    ]
    decisions.sort(key=lambda d: d["decision_id"])
    if args.limit:
        decisions = decisions[: args.limit]

    # Decision gate: a missing feature table means the agent would be reading
    # dashes. Report and exit 0 -- a non-zero exit here strands every dependent
    # job on DependencyNeverSatisfied with no explanation of which input was
    # absent.
    missing = [d["decision_id"] for d in decisions if d["complex_id"] not in feats]
    if missing:
        print(f"[run] GATE: {len(missing)} decisions have no structural features; "
              f"dropping them. First: {missing[:5]}")
        decisions = [d for d in decisions if d["complex_id"] in feats]
    if not decisions:
        print("[run] GATE FAILED: no decisions left to run. "
              "Check that features.json covers this slice. Exiting 0 so that "
              "dependent jobs can report the same gate rather than vanish.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}__{args.order}"
    dest = out_dir / f"choices__{tag}.jsonl"

    done: set[str] = set()
    if dest.exists():
        for line in dest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["decision_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [d for d in decisions if d["decision_id"] not in done]
    print(f"[run] arm={args.arm} order={args.order} "
          f"decisions={len(decisions)} already_done={len(done)} todo={len(todo)}")
    if not todo:
        print("[run] nothing to do")
        return 0

    if args.arm in HEURISTIC_ARMS or args.arm in CONTROL_ARMS:
        lookup = None
        if args.arm in CONTROL_ARMS:
            # Control arms are the only ones permitted to read the oracle
            # file, and only for n_records and the additive single-mutation
            # prediction -- never for a candidate's own measured value.
            ora = json.loads((bench / "decisions_oracle.json").read_text())
            lookup = {
                o["decision_id"]: {
                    c["cand_id"]: {
                        "n_records": c.get("n_records", 0),
                        "additive": c.get("additive_from_singles"),
                    }
                    for c in o["candidates"]
                }
                for o in ora
            }
        proposer = HeuristicProposer(args.arm, oracle_lookup=lookup)
        model_id = f"heuristic:{args.arm}"
    else:
        if not args.model_path:
            print("[run] GATE FAILED: --model-path required for an LLM arm. Exiting 0.")
            return 0
        proposer = LocalHFProposer(
            model_path=args.model_path,
            evidence_arm=args.evidence_arm,
            name=args.arm,
            temperature=args.temperature,
            max_retries=args.max_retries,
            device_map=args.device_map or None,
            load_4bit=args.load_4bit,
        )
        print(f"[run] loading {args.model_path} ...")
        t0 = time.time()
        proposer.load()
        print(f"[run] loaded in {time.time() - t0:.1f}s")
        # Derived from the weight directory, not from a constant: a constant
        # model id stamped on every row makes an arm comparison unreadable the
        # moment a second model runs.
        # Quantisation is part of the model's identity, not a runtime detail:
        # pooling 4-bit rows with bf16 rows under one model_id would make an
        # arm comparison unreadable.
        model_id = Path(args.model_path).name + ("-4bit" if args.load_4bit else "")

    order_key = "presentation_order" if args.order == "main" else "presentation_order_alt"
    t_start = time.time()
    n_invalid = 0
    with dest.open("a") as fh:
        for i, d in enumerate(todo, 1):
            order = d.get(order_key) or [c["cand_id"] for c in d["candidates"]]
            t0 = time.time()
            try:
                choice = proposer.choose(
                    d, feats.get(d["complex_id"], {}), cons.get(d["complex_id"], {}),
                    order,
                )
            except Exception as exc:  # noqa: BLE001
                choice = {
                    "cand_id": None,
                    "status": f"error_{type(exc).__name__}",
                    "n_retries": 0,
                    "raw": str(exc)[:2000],
                    "scores": {},
                }
            row = {
                "arm": args.arm,
                "evidence_arm": args.evidence_arm if proposer.needs_llm else args.arm,
                "order": args.order,
                "model_id": model_id,
                "decision_id": d["decision_id"],
                "slice": d["slice"],
                "split": d["split"],
                "complex_id": d["complex_id"],
                "k": d["k"],
                "n_candidates": d["n_candidates"],
                "cand_id": choice["cand_id"],
                "choice_valid": choice["cand_id"] is not None,
                "status": choice["status"],
                "n_retries": choice.get("n_retries", 0),
                "presented_index": (
                    order.index(choice["cand_id"]) if choice["cand_id"] in order else None
                ),
                "seconds": round(time.time() - t0, 2),
                "temperature": args.temperature,
            }
            if args.save_raw:
                row["raw"] = choice.get("raw", "")
            if choice.get("scores"):
                row["scores"] = choice["scores"]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if not row["choice_valid"]:
                n_invalid += 1
            if i % 10 == 0 or i == len(todo) or proposer.needs_llm:
                rate = (time.time() - t_start) / i
                print(f"[run] {i:4d}/{len(todo)} {d['decision_id']:28s} "
                      f"-> {row['cand_id']} ({row['status']}) "
                      f"{row['seconds']:5.1f}s  eta {rate * (len(todo) - i) / 60:5.1f}m",
                      flush=True)

    print(f"[run] wrote {dest}")
    print(f"[run] invalid choices: {n_invalid}/{len(todo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
