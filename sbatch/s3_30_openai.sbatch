#!/usr/bin/env bash
#SBATCH --job-name=s3_oai
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s3_30_openai_%j.out
set -euo pipefail
# set -euo pipefail lives in the job body, not only the preamble: a silently
# failed source would otherwise leave the job running under the system python
# and exiting 0, releasing every afterok dependent behind a stage that did
# nothing.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# CPU partition: an API arm is an HTTP client, it waits on the network and not
# on a card. Uses $PY (the project venv) rather than $PY_API because the tool
# controller imports lp2cp.baselines, which imports the surrogate; the OpenAI
# client itself needs no package at all beyond the standard library.
if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate"; then exit 0; fi
if [ ! -x "$PY" ]; then
  echo "GATE FAILED: no project venv at $PY. Exiting 0."
  exit 0
fi
# Availability is checked without revealing the key: envload reads it inside
# the process, from the dotenv file or the raw key file, never from a command
# line and never into a log.
if ! "$PY" -c "
import sys
sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key(key='OPENAI_API_KEY') else 1)
"; then
  echo "GATE FAILED: no OPENAI_API_KEY available. Exiting 0."
  exit 0
fi

mkdir -p "$CAMPAIGNS_DIR" "$ARCHIVE_DIR"
echo "specs  : ${OAI_SPECS}"
echo "limit  : ${OAI_LIMIT}   iters: ${MAX_ITERS}   effort: ${API_EFFORT}"

# Each spec is model|mode|cap_usd. The cap is enforced per spec by the ledger
# inside OpenAIClient, which checks a rolling estimate BEFORE each call, so a
# run stops short of its cap rather than discovering it passed it.
#   tool          -- LLM picks a tool from lp2cp.tools; never names a position
#   mask          -- the Stage 2 scaffold: LLM authors the mask (A3 + inject)
#   mask_symdepth -- same, with the depth instruction naming the GAIN columns
for SPEC in ${OAI_SPECS}; do
  MODEL="${SPEC%%|*}"; REST="${SPEC#*|}"; MODE="${REST%%|*}"; CAP="${REST##*|}"
  echo "================================================================"
  echo "=== $MODEL / $MODE  (cap \$${CAP})"
  FLAGS=""
  ARM="A3_CONTRACT"
  # The tool arm enforces no preservation contract -- ToolSelectController
  # inherits the rule controllers' enforce_contract=False -- so it is labelled
  # A2_CONDITIONING, the same label the rule baselines carry. Labelling it
  # A3_CONTRACT would put a row in the table claiming a mechanism it does not
  # run.
  case "$MODE" in
    tool)          FLAGS="--tool-select"; ARM="A2_CONDITIONING" ;;
    mask)          FLAGS="--one-call-specify --inject-contract" ;;
    mask_symdepth) FLAGS="--one-call-specify --inject-contract --symmetric-depth" ;;
    *) echo "unknown mode $MODE -- skipping"; continue ;;
  esac
  "$PY" "$AGENT/run_campaign.py" \
    --parents "$BENCH_DIR/parents_public.json" \
    --qualification "$BENCH_DIR/qualification.json" \
    --env-manifest "$BENCH_DIR/env_manifest.json" \
    --out "$CAMPAIGNS_DIR" \
    --arm "$ARM" --backend openai --model-path "$MODEL" \
    --effort "$API_EFFORT" --max-spend-usd "$CAP" \
    --max-new-tokens "${OAI_MAX_TOKENS:-9000}" \
    --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
    --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
    --max-eval-seeds "$MAX_EVAL_SEEDS" \
    --backend-fidelity "$BACKEND_FIDELITY" \
    --archive "$ARCHIVE_DIR/archive__${MODEL}__${MODE}.jsonl" \
    --tool-menu "$TOOL_MENU" \
    --save-raw --limit "$OAI_LIMIT" $FLAGS
done

echo "--- spend recorded across this job's ledgers ---"
"$PY" "$AGENT/../sbatch/sum_openai_spend.py" "$CAMPAIGNS_DIR"
echo "DONE openai"
