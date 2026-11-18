#!/usr/bin/env bash
#SBATCH --job-name=s2_llm70b
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=16:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_22_llm70b_%j.out
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

# ONE a6000, not four. Llama-3.1-70B is ~132 GB on disk and ~140 GB in bf16,
# which needs four 48 GB cards; measured on this cluster, no node had even two
# free a6000s, so a bf16 request queues indefinitely. NF4 double quantisation
# brings it to ~40 GB, which fits one card and schedules.
#
# The 4-bit model is recorded as a DIFFERENT model_id ("-4bit"), not as a
# cheaper way of running the same weights, so its rows are never pooled with a
# bf16 run.
#
# Host RAM is requested generously because the loader streams 30 shards; an
# a6000 sitting on a node with 4 GB of host RAM free would fail the load with
# no CUDA error to explain it.
if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate (s2_00)"; then exit 0; fi
if [ ! -d "$LLM70B_PATH" ]; then
  echo "GATE FAILED: no 70B weights at $LLM70B_PATH. Exiting 0."
  exit 0
fi
if [ ! -x "$PY_70B" ]; then
  echo "GATE FAILED: no 4-bit overlay venv at $PY_70B"
  echo "  build it with: bash ${SUBMIT_DIR}/build_70b_env.sh"
  echo "  Exiting 0 so the scorer still runs on whatever arms exist."
  exit 0
fi

echo "python: $PY_70B"
echo "model : $LLM70B_PATH  (NF4 4-bit)"
echo "arms  : $ARMS_70B"
"$PY_70B" -c "import bitsandbytes, accelerate, torch; print('bnb', bitsandbytes.__version__, 'accel', accelerate.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p "$CAMPAIGNS_DIR" "$ARCHIVE_DIR"

for ARM in $ARMS_70B; do
  echo "=== arm $ARM"
  # The episodic archive is PER ARM. Arms 4 and 5 read it; sharing one file
  # across arms would let arm 2's outcomes condition arm 5 and make the
  # 3->4->5 ablation a measurement of whichever arm happened to run first.
  # run_campaign.py skips campaign_ids already in its own output file, so a
  # time limit costs a resubmission rather than the pass.
  "$PY_70B" "$AGENT/run_campaign.py" \
    --parents "$BENCH_DIR/parents_public.json" \
    --qualification "$BENCH_DIR/qualification.json" \
    --env-manifest "$BENCH_DIR/env_manifest.json" \
    --out "$CAMPAIGNS_DIR" \
    --arm "$ARM" --backend local --model-path "$LLM70B_PATH" --load-4bit \
    --temperature 0.0 --max-new-tokens 900 \
    --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
    --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
    --max-eval-seeds "$MAX_EVAL_SEEDS" \
    --backend-fidelity "$BACKEND_FIDELITY" \
    --archive "$ARCHIVE_DIR/archive__70B4bit__${ARM}.jsonl" \
    --save-raw --limit "$LIMIT" ${SCAFFOLD_FLAG:-}
done
echo "DONE llm70b"
