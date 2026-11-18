#!/usr/bin/env bash
#SBATCH --job-name=s2_llm8b4
#SBATCH --partition=legacy
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_21_llm8b4bit_%j.out
set -euo pipefail
# `set -euo pipefail` lives in the JOB BODY, not only in the preamble: if the
# source below silently fails, a preamble-only setting leaves the job running
# under the system python and exiting 0, which releases every afterok dependent
# behind a stage that did no work.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
# Slurm copies the batch script to /var/spool/slurm/job<id>/slurm_script, so
# $(dirname "${BASH_SOURCE[0]}") points at the spool dir and sourcing a shared
# preamble that way sources nothing at all.
. "$PRE" "$@"
banner

# WHY THIS JOB EXISTS, and why it is an rtx_2080_ti and not an a6000.
#
# The bf16 8B arm (s2_20) needs ~16 GB of weights before any KV cache, so it
# pins an a6000 -- and measured on this cluster the a6000s stayed saturated for
# hours, with two nodes showing idle cards and 2-4 GB of host RAM free. Meanwhile
# gpu12/13/17 sat IDLE with four free 2080 Tis each.
#
# NF4 double-quantised 8B is ~5.5 GB of weights, plus ~1-2 GB of KV cache for
# this scaffold's 2-3k token prompts: comfortably inside 11 GB. So the 4-bit
# rung runs NOW on hardware nobody is contending for, instead of queueing
# behind the bf16 rung indefinitely.
#
# The quantised model is a DIFFERENT model_id ("-4bit"), not a cheaper way of
# running the same weights, and run_campaign.py stamps that into every row --
# so these rows are never pooled with the bf16 arm's. That is the point of
# running both: the pair is a quantisation ablation at fixed scale, which is
# worth having next to the 8B-vs-70B scale comparison.
#
# Partition and GPU type were read off `myfree` at submission time (gpu17:
# legacy, rtx_2080_ti, 4/4 free, IDLE) rather than carried over from another
# script.

PY4BIT="${PY4BIT:-$PY_70B}"
MODEL4BIT="${MODEL4BIT:-$LLM8B_PATH}"
ARMS_4BIT="${ARMS_4BIT:-A1_FRESH A2_CONDITIONING A3_CONTRACT A4_PRIOR A5_REFLECTION}"

if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate (s2_00)"; then exit 0; fi
if [ ! -d "$MODEL4BIT" ]; then
  echo "GATE FAILED: no weights at $MODEL4BIT. Exiting 0."
  exit 0
fi
if [ ! -x "$PY4BIT" ]; then
  echo "GATE FAILED: no 4-bit venv at $PY4BIT (needs bitsandbytes + accelerate)."
  echo "  build it with: bash ${SUBMIT_DIR}/build_70b_env.sh"
  echo "  Exiting 0 so the scorer still runs on whatever arms exist."
  exit 0
fi

echo "python : $PY4BIT"
echo "model  : $MODEL4BIT  (NF4 4-bit)"
echo "arms   : $ARMS_4BIT"
"$PY4BIT" -c "import bitsandbytes, accelerate, torch; print('bnb', bitsandbytes.__version__, 'accel', accelerate.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p "$CAMPAIGNS_DIR" "$ARCHIVE_DIR"
for ARM in $ARMS_4BIT; do
  echo "=== arm $ARM"
  # The episodic archive is PER ARM: arms 4 and 5 read it, and sharing one file
  # across arms would let an earlier arm's outcomes condition a later one and
  # make the 3->4->5 ablation a measurement of run order.
  # Resumable: run_campaign.py skips campaign_ids already in its own output
  # file, so a time limit costs a resubmission rather than the pass.
  "$PY4BIT" "$AGENT/run_campaign.py" \
    --parents "$BENCH_DIR/parents_public.json" \
    --qualification "$BENCH_DIR/qualification.json" \
    --env-manifest "$BENCH_DIR/env_manifest.json" \
    --out "$CAMPAIGNS_DIR" \
    --arm "$ARM" --backend local --model-path "$MODEL4BIT" --load-4bit \
    --temperature 0.0 --max-new-tokens 900 \
    --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
    --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
    --max-eval-seeds "$MAX_EVAL_SEEDS" \
    --backend-fidelity "$BACKEND_FIDELITY" \
    --archive "$ARCHIVE_DIR/archive__8B4bit__${ARM}.jsonl" \
    --save-raw --limit "$LIMIT" ${SCAFFOLD_FLAG:-}
done
echo "DONE llm8b4bit"
