#!/usr/bin/env bash
#SBATCH --job-name=s0_llmfi
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_32_llm_fullif_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage0_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage0_preamble.sh
. "$PRE" "$@"
banner
export FULLIF_BENCH="${RUN}/bench_fullif"
export FULLIF_CONS="${RUN}/features_fullif"
export FULLIF_ARMS="${RUN}/arms_fullif"
export FULLIF_RESULTS="${RUN}/results_fullif"

if ! require_file "$FULLIF_BENCH/decisions_public.json" "no full-interface bench (s0_12)"; then exit 0; fi
if [ ! -d "$LLM_MODEL_PATH" ]; then
  echo "GATE FAILED: no model weights at $LLM_MODEL_PATH. Exiting 0."
  exit 0
fi
# Only the 'full' evidence condition here: the secondary arm exists to measure
# the effect of widening the candidate set, so varying evidence at the same
# time would make the two changes inseparable.
MODEL_TAG="$(basename "$LLM_MODEL_PATH")"
"$PY" "$STAGE0/run_arm.py" \
  --arm "llm_${MODEL_TAG}_full" --bench "$FULLIF_BENCH" --features "$FEATURES_DIR" \
  --conservation "$FULLIF_CONS" --out "$FULLIF_ARMS" \
  --slices fullif_ala_k1 --splits "$SPLITS" --order main \
  --model-path "$LLM_MODEL_PATH" --evidence-arm full \
  --temperature "$LLM_TEMPERATURE" --save-raw
echo "DONE fullif-llm"
