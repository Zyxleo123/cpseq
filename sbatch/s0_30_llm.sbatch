#!/usr/bin/env bash
#SBATCH --job-name=s0_llm
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_30_llm_%j.out
set -euo pipefail
# `set -euo pipefail` lives in the JOB BODY, not only in the preamble: if the
# source below silently fails, a preamble-only setting leaves the job running
# under the system python and exiting 0, which releases every afterok
# dependent behind a stage that did no work.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage0_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage0_preamble.sh
# Slurm copies the batch script to /var/spool/slurm/job<id>/slurm_script, so
# $(dirname "${BASH_SOURCE[0]}") points at the spool dir and sourcing a shared
# preamble that way sources nothing at all.
. "$PRE" "$@"
banner

# GPU class is pinned deliberately. Llama-3.1-8B in bf16 is ~16 GB of weights
# before any KV cache, so an 11 GB rtx_2080_ti cannot hold it and a 16 GB v100
# is borderline. A CUDA OOM kills the process with no Python traceback, which
# reads as a mysterious silent FAILED, so the request names a6000 (48 GB)
# rather than leaving the card to chance.
if ! require_file "$BENCH_DIR/decisions_public.json" "no benchmark (s0_00)"; then exit 0; fi
if ! require_file "$FEATURES_DIR/features.json" "no structural features (s0_10)"; then exit 0; fi
if [ ! -d "$LLM_MODEL_PATH" ]; then
  echo "GATE FAILED: no model weights at $LLM_MODEL_PATH"
  echo "  set LLM_MODEL_PATH in the env file. Exiting 0 so the scorer still runs."
  exit 0
fi

echo "model: $LLM_MODEL_PATH"
echo "evidence arms: $LLM_EVIDENCE_ARMS   orders: $LLM_ORDERS"

for EV in ${LLM_EVIDENCE_ARMS//,/ }; do
  for ORD in ${LLM_ORDERS//,/ }; do
    MODEL_TAG="$(basename "$LLM_MODEL_PATH")"
    ARM="llm_${MODEL_TAG}_${EV}"
    echo "=== arm $ARM order=$ORD"
    # Resumable: run_arm.py skips decision_ids already in its own output file,
    # so a time limit costs a resubmission and not the pass.
    "$PY" "$STAGE0/run_arm.py" \
      --arm "$ARM" --bench "$BENCH_DIR" --features "$FEATURES_DIR" \
      --conservation "$CONS_DIR" --out "$ARMS_DIR" \
      --slices "$SLICES" --splits "$SPLITS" --order "$ORD" \
      --model-path "$LLM_MODEL_PATH" --evidence-arm "$EV" \
      --temperature "$LLM_TEMPERATURE" \
      ${LLM_DEVICE_MAP:+--device-map "$LLM_DEVICE_MAP"} \
      --save-raw --limit "${LIMIT:-0}"
  done
done
echo "DONE llm"
