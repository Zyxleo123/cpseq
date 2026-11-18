"""Run one arm end-to-end over the parent set. One arm, one output file.

Resumable: campaign_ids already in the output file are skipped, so hitting a
time limit costs a resubmission rather than the pass. Concurrent appends from
parallel jobs to a shared file interleave and NUL-corrupt rows, so the filename
carries the arm, the controller and the model and nothing else writes to it.

Startup refuses to run on a changed benchmark. `assert_frozen` compares the
on-disk qualification manifest against the live code and raises if they differ:
rows produced under two qualification functions are not poolable, and the
failure would otherwise be invisible in the results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lp2cp.baselines import BASELINES, RuleController, load_oracle_rows  # noqa: E402
from lp2cp.controller import Controller  # noqa: E402
from lp2cp.enums import ARM_ORDER, Arm  # noqa: E402
from lp2cp.envbuild.build_parents import load_parents  # noqa: E402
from lp2cp.memory import Archive  # noqa: E402
from lp2cp.qualification import assert_frozen  # noqa: E402


def _tools_rev() -> str:
    """The tool registry revision, for the scaffold tag."""
    from lp2cp import tools as _t

    return _t.TOOLS_REV


def make_llm(args):
    if args.backend == "scripted":
        from lp2cp.llm.anthropic_api import ScriptedClient

        replies = json.loads(Path(args.scripted_replies).read_text()) \
            if args.scripted_replies else []
        return ScriptedClient(replies, model_id=args.model_id or "scripted")
    if args.backend == "local":
        from lp2cp.llm.local_hf import LocalHFClient

        return LocalHFClient(
            model_path=args.model_path,
            temperature=args.temperature,
            load_4bit=args.load_4bit,
            device_map=args.device_map or None,
            max_new_tokens=args.max_new_tokens,
        )
    if args.backend == "openai":
        from lp2cp.llm.openai_api import OpenAIClient

        return OpenAIClient(
            model=args.model_path,
            effort=args.effort,
            max_spend_usd=args.max_spend_usd,
            max_tokens=args.max_new_tokens,
        )
    if args.backend == "api":
        from lp2cp.llm.anthropic_api import AnthropicClient

        return AnthropicClient(
            model=args.model_path,
            effort=args.effort,
            max_spend_usd=args.max_spend_usd,
            max_tokens=args.max_new_tokens,
        )
    raise ValueError(f"unknown backend {args.backend!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parents", required=True, help="parents_public.json")
    ap.add_argument("--qualification", required=True)
    ap.add_argument("--out", required=True, help="campaigns directory")
    ap.add_argument("--env-manifest", default="")
    ap.add_argument("--arm", default="A3_CONTRACT",
                    help="arm name, or 'all' for every arm in order")
    ap.add_argument("--rule", default="", help=f"a baseline instead of an LLM: {BASELINES}")
    ap.add_argument("--oracle-table", default="", help="needed only for oracle_greedy")
    ap.add_argument("--backend", default="scripted",
                    choices=["scripted", "local", "api", "openai"])
    ap.add_argument("--model-path", default="", help="weights dir, or an API model id")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--scripted-replies", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-new-tokens", type=int, default=1400,
                help="for an API model this is max_tokens, and THINKING TOKENS "
                     "COUNT TOWARD IT: a ceiling that the thinking block alone "
                     "can reach returns stop_reason=refusal with category "
                     "reasoning_extraction and no answer. Measured: 3000 was too "
                     "low for this scaffold on sonnet-5-5.")
    ap.add_argument("--max-spend-usd", type=float, default=0.0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device-map", default="")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed-budget", type=int, default=60)
    ap.add_argument("--default-seeds", type=int, default=5)
    ap.add_argument("--max-eval-seeds", type=int, default=20)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--one-call-specify", action="store_true",
                    help="skip Agent Rosetta's operator-selection call (halves "
                         "the SPECIFY cost; records a different scaffold id)")
    ap.add_argument("--tool-select", action="store_true",
                    help="LLM selects from the lp2cp.tools action space instead "
                         "of authoring a mask (PDAgent's division of labour)")
    ap.add_argument("--tool-menu", default="v2", choices=["v1", "v2"],
                    help="which lp2cp.tools action space: v1 is the six tools "
                         "written before the oracle table was read, v2 adds "
                         "rebuild_cheapest_2span, whose shape came FROM it")
    ap.add_argument("--symmetric-depth", action="store_true",
                    help="SPECIFY depth instruction names the GAIN columns "
                         "(burf, mind) as well as the cost columns")
    ap.add_argument("--inject-contract", action="store_true",
                    help="restate the contract's protected positions inside the "
                         "SPECIFY brief; records a distinct scaffold tag")
    ap.add_argument("--backend-fidelity", default="declared",
                    choices=["declared", "perfect"])
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--archive", default="", help="shared episodic archive jsonl")
    args = ap.parse_args()

    assert_frozen(args.qualification)
    env_hash = ""
    if args.env_manifest and Path(args.env_manifest).exists():
        env_hash = json.loads(Path(args.env_manifest).read_text()).get("env_hash", "")

    parents = load_parents(args.parents, with_labels=False)
    keep = {s for s in args.splits.split(",") if s}
    parents = [p for p in parents if p.split in keep]
    parents.sort(key=lambda p: p.parent_id)
    if args.limit:
        parents = parents[: args.limit]
    if not parents:
        print("[run] GATE FAILED: no parents after filtering. Exiting 0 so that "
              "dependent jobs report the same gate rather than vanishing.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rule:
        arms = [None]
        oracle_rows = (
            load_oracle_rows(args.oracle_table)
            if args.rule == "oracle_greedy" and args.oracle_table else {}
        )
        if args.rule == "oracle_greedy" and not oracle_rows:
            print("[run] GATE FAILED: oracle_greedy needs --oracle-table. Exiting 0.")
            return 0
        tag = f"rule_{args.rule}"
    else:
        arms = list(ARM_ORDER) if args.arm == "all" else [Arm(args.arm)]
        oracle_rows = {}
        model_tag = (args.model_id or Path(args.model_path).name
                     or args.backend) + ("-4bit" if args.load_4bit else "")
        tag = f"llm_{model_tag}"

    # The menu version is part of the scaffold tag for every arm that draws on
    # it -- the LLM tool arm AND the tool rules. A row that does not say which
    # action space produced it cannot be compared with one that does, and
    # de-duplication keyed on a tag that omits it would collapse two different
    # experiments into one (the fifth instrument bug, in a new place).
    uses_menu = args.tool_select or args.rule in (
        "tool_router", "headroom_repack", "headroom_first", "rebuild_cheap2")
    scaffold = (
        ("toolselect" if args.tool_select
         else "onecall" if args.one_call_specify else "twocall")
        + ("+inject" if args.inject_contract and not args.tool_select else "")
        + ("+symdepth" if args.symmetric_depth and not args.tool_select else "")
        + (f"+menu{args.tool_menu}{_tools_rev()}" if uses_menu else "")
    )
    from lp2cp import tools as _tools

    _tools.set_menu(args.tool_menu)
    llm = None
    if not args.rule:
        # Pick the reasoning-format slot BEFORE any prompt is built. API models
        # in the current Opus/Sonnet generation emit native thinking blocks and
        # must not also be asked for <reasoning> tags -- the two channels
        # collide and the call comes back as a reasoning_extraction refusal.
        # Local instruct weights have no native channel and do need the tags.
        from lp2cp.prompts import set_reasoning_style

        set_reasoning_style(
            native_thinking=(args.backend in ("api", "openai")))
        # Both OpenAI reasoning models and the current Claude generation emit a
        # native thinking channel; asking either for <reasoning> tags as well
        # collides the two channels, which is what produced Stage 2's
        # reasoning_extraction refusals.
        from lp2cp.prompts import set_depth_style

        set_depth_style(symmetric=args.symmetric_depth)
        llm = make_llm(args)
        print(f"[run] loading model backend={args.backend} path={args.model_path!r}")
        t0 = time.time()
        llm.load()
        print(f"[run] loaded in {time.time() - t0:.1f}s as model_id={llm.model_id}")

    total = 0
    for arm in arms:
        arm_name = args.rule if args.rule else arm.value
        dest = out_dir / f"campaigns__{tag}__{arm_name}__{scaffold}.jsonl"
        done: set[str] = set()
        if dest.exists():
            for line in dest.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line)["campaign_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        # Arms 4 and 5 condition on an archive built from earlier campaigns.
        # Shared across parents within an arm, and retrieval excludes the
        # current parent's own cluster so no arm reads outcomes measured on the
        # target it is designing against.
        archive = Archive(args.archive or None)
        todo = [p for p in parents if not any(
            d.startswith(f"{p.parent_id}__") for d in done)]
        print(f"[run] arm={arm_name} parents={len(parents)} done={len(done)} "
              f"todo={len(todo)} -> {dest.name}")

        with dest.open("a") as fh:
            for i, p in enumerate(todo, 1):
                t0 = time.time()
                if args.rule:
                    ctl = RuleController(
                        rule=args.rule, max_iters=args.max_iters,
                        patience=args.patience, default_seeds=args.default_seeds,
                        seed_budget=args.seed_budget,
                        max_eval_seeds=args.max_eval_seeds,
                        backend_fidelity=args.backend_fidelity,
                        oracle_rows=oracle_rows,
                    )
                elif args.tool_select:
                    from lp2cp.toolselect import ToolSelectController

                    ctl = ToolSelectController(
                        llm=llm, arm=arm, max_iters=args.max_iters,
                        patience=args.patience, max_retries=args.max_retries,
                        default_seeds=args.default_seeds,
                        seed_budget=args.seed_budget,
                        max_eval_seeds=args.max_eval_seeds,
                        backend_fidelity=args.backend_fidelity,
                        save_raw=args.save_raw,
                    )
                else:
                    ctl = Controller(
                        llm=llm, arm=arm, archive=archive,
                        max_iters=args.max_iters, patience=args.patience,
                        max_retries=args.max_retries,
                        two_call_specify=not args.one_call_specify,
                        default_seeds=args.default_seeds,
                        seed_budget=args.seed_budget,
                        max_eval_seeds=args.max_eval_seeds,
                        backend_fidelity=args.backend_fidelity,
                        save_raw=args.save_raw,
                        inject_contract=args.inject_contract,
                    )
                try:
                    rec = ctl.run(p, env_hash=env_hash)
                    row = rec.to_row()
                except Exception as exc:  # noqa: BLE001
                    import traceback

                    row = {
                        "campaign_id": f"{p.parent_id}__{arm_name}__ERROR",
                        "parent_id": p.parent_id, "cluster_id": p.cluster_id,
                        "split": p.split, "arm": arm_name,
                        "controller": args.rule or f"llm:{llm.model_id}",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                    print(f"[run] ERROR on {p.parent_id}: {row['error']}")
                row["scaffold"] = scaffold
                row["backend_fidelity"] = args.backend_fidelity
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                total += 1
                print(
                    f"[run] {i:3d}/{len(todo)} {p.parent_id:28s} "
                    f"{row.get('best_outcome', '-'):32s} "
                    f"q={row.get('n_qualified', 0)} "
                    f"it={row.get('n_iterations', 0)} "
                    f"calls={row.get('total_llm_calls', 0)} "
                    f"${row.get('total_cost_usd', 0):.4f} "
                    f"{time.time() - t0:5.1f}s",
                    flush=True,
                )
        # Per-arm spend ledger, written next to the campaigns.
        if llm is not None and hasattr(llm, "summary"):
            (out_dir / f"cost__{tag}__{arm_name}__{scaffold}.json").write_text(
                json.dumps(llm.summary(), indent=2)
            )
            print(f"[run] spend so far: ${llm.ledger.spent_usd:.4f} "
                  f"over {llm.ledger.calls} calls")

    print(f"[run] wrote {total} campaigns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
