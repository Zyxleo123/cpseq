#!/usr/bin/env bash
#SBATCH --job-name=s2_llm8b
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_20_llm8b_%j.out
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

# Host RAM is 32G, not 64G. The weights are 4 safetensors shards of ~4 GB and
# transformers streams them one at a time, so peak host RAM during load is
# ~12 GB; 64 G was over-asking and it matters here, because the a6000s on this
# cluster are routinely GPU-free-but-RAM-full (measured: gpu29 with 2 GB and
# gpu30 with 4 GB free while both had idle cards). A smaller request lands on
# strictly more nodes for no benefit given up.
#
# GPU class is pinned deliberately. Llama-3.1-8B in bf16 is ~16 GB of weights
# before any KV cache, so an 11 GB rtx_2080_ti cannot hold it and a 16 GB v100
# is borderline once this scaffold's prompts (a residue table plus a scorecard,
# ~2-3k tokens) are in the cache. A CUDA OOM kills the process with no Python
# traceback, which reads as a mysterious silent FAILED, so the request names
# a6000 (48 GB) rather than leaving the card to chance.
if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate (s2_00)"; then exit 0; fi
if [ ! -d "$LLM8B_PATH" ]; then
  echo "GATE FAILED: no 8B weights at $LLM8B_PATH"
  echo "  set LLM8B_PATH in the env file. Exiting 0 so the scorer still runs."
  exit 0
fi

echo "model : $LLM8B_PATH"
echo "arms  : $LOCAL_ARMS"
mkdir -p "$CAMPAIGNS_DIR" "$ARCHIVE_DIR"

for ARM in $LOCAL_ARMS; do
  echo "=== arm $ARM"
  # The episodic archive is PER ARM. Arms 4 and 5 read it; sharing one file
  # across arms would let arm 2's outcomes condition arm 5 and make the
  # 3->4->5 ablation a measurement of whichever arm happened to run first.
  # run_campaign.py skips campaign_ids already in its own output file, so a
  # time limit costs a resubmission rather than the pass.
  "$PY" "$AGENT/run_campaign.py" \
    --parents "$BENCH_DIR/parents_public.json" \
    --qualification "$BENCH_DIR/qualification.json" \
    --env-manifest "$BENCH_DIR/env_manifest.json" \
    --out "$CAMPAIGNS_DIR" \
    --arm "$ARM" --backend local --model-path "$LLM8B_PATH" \
    --temperature 0.0 --max-new-tokens 900 \
    --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
    --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
    --max-eval-seeds "$MAX_EVAL_SEEDS" \
    --backend-fidelity "$BACKEND_FIDELITY" \
    --archive "$ARCHIVE_DIR/archive__8B__${ARM}.jsonl" \
    --save-raw --limit "$LIMIT" ${SCAFFOLD_FLAG:-}
done
echo "DONE llm8b"
