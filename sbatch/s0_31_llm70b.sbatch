#!/usr/bin/env bash
#SBATCH --job-name=s0_llm70b
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_31_llm70b_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage0_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage0_preamble.sh
. "$PRE" "$@"
banner

# The overlay venv, not the project venv: it is the only one with bitsandbytes
# and accelerate. Built by build_70b_env.sh; it chains to the project venv, so
# torch/transformers are byte-identical to every other stage.
PY70B="${PY70B:-/zfsauton/scratch/yixiz/envs/lp2cp_70b/.venv/bin/python}"
MODEL70B="${MODEL70B:-/zfsauton/scratch/yixiz/models/Meta-Llama-3.1-70B-Instruct}"

# One a6000, not four. Llama-3.1-70B is ~132 GB on disk and ~140 GB in bf16,
# which needs four 48 GB cards; measured on this cluster, no node had even two
# free a6000s, so a bf16 request queues indefinitely. NF4 double quantisation
# brings it to ~40 GB, which fits one card and schedules immediately.
#
# Host RAM is requested generously because the loader streams 30 shards; the
# one free a6000 seen during planning sat on a node with 4 GB of host RAM free,
# which would have failed the load with no CUDA error to explain it.
if [ ! -d "$MODEL70B" ]; then
  echo "GATE FAILED: no 70B weights at $MODEL70B. Exiting 0."; exit 0
fi
if [ ! -x "$PY70B" ]; then
  echo "GATE FAILED: no 70B overlay venv at $PY70B."
  echo "  build it with: bash ${SUBMIT_DIR}/build_70b_env.sh"
  echo "  Exiting 0 so the scorer still runs on whatever arms exist."
  exit 0
fi
if ! require_file "$BENCH_DIR/decisions_public.json" "no benchmark (s0_00)"; then exit 0; fi
if ! require_file "$FEATURES_DIR/features.json" "no features (s0_10)"; then exit 0; fi

echo "python : $PY70B"
echo "model  : $MODEL70B  (NF4 4-bit)"
"$PY70B" -c "import bitsandbytes, accelerate, torch; print('bnb', bitsandbytes.__version__, 'accel', accelerate.__version__, 'cuda', torch.cuda.is_available())"

# Primary slice only. At roughly a minute per decision the full 186-decision
# set would be a 3-hour arm; ala_k1 is the slice the headline rests on.
SLICES70B="${SLICES70B:-ala_k1}"
EV70B="${EV70B:-full}"

# Resumable: run_arm.py skips decision_ids already in its own output file, so a
# time limit costs a resubmission rather than the pass.
"$PY70B" "$STAGE0/run_arm.py" \
  --arm "llm_Llama-3.1-70B-Instruct-4bit_${EV70B}" \
  --bench "$BENCH_DIR" --features "$FEATURES_DIR" --conservation "$CONS_DIR" \
  --out "$ARMS_DIR" --slices "$SLICES70B" --splits "$SPLITS" --order main \
  --model-path "$MODEL70B" --evidence-arm "$EV70B" \
  --temperature 0.0 --load-4bit --save-raw
echo "DONE llm70b"
