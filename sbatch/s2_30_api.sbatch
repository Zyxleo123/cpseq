#!/usr/bin/env bash
#SBATCH --job-name=s2_api
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_30_api_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# CPU partition, not GPU. An API arm is an HTTP client: it waits on the network,
# not on a card. Asking for a GPU would hold a 48 GB a6000 idle for hours while
# real work queued behind it.
#
# Uses PY_API, the venv that has `anthropic`. The campaign runner needs no
# scientific stack on this path -- numpy is imported inside
# diagnostics.measure, which an API job never calls.
if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate (s2_00)"; then exit 0; fi
if [ ! -x "$PY_API" ]; then
  echo "GATE FAILED: no API venv at $PY_API. Exiting 0."
  exit 0
fi
# The key is read from the dotenv file by envload.py INSIDE the process; it is
# never passed on a command line and never printed. This checks availability
# without revealing it.
if ! "$PY_API" -c "
import sys
sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key() else 1)
"; then
  echo "GATE FAILED: no ANTHROPIC_API_KEY available to the API job."
  echo "  Put it in the dotenv file envload.py reads. Exiting 0."
  exit 0
fi

mkdir -p "$CAMPAIGNS_DIR" "$ARCHIVE_DIR"
echo "models : $API_MODELS"
echo "arms   : $API_ARMS"
echo "effort : $API_EFFORT"
echo "cap    : \$${MAX_SPEND_PER_ARM} per (model, arm)   parents: ${API_LIMIT}"

# Spend is the binding constraint here, not time. A campaign makes up to four
# LLM calls per iteration (diagnose, operator-select, specify, allocate) and
# runs up to MAX_ITERS iterations per parent, so an arm is roughly
# 4 x MAX_ITERS x API_LIMIT calls. The ledger inside AnthropicClient is checked
# BEFORE each call against a rolling estimate and stops the arm short of its
# cap rather than discovering afterwards that it passed it -- so the loop below
# is safe to leave running even if the estimate is off.
for MODEL in $API_MODELS; do
  for ARM in $API_ARMS; do
    echo "=== $MODEL / $ARM"
    "$PY_API" "$AGENT/run_campaign.py" \
      --parents "$BENCH_DIR/parents_public.json" \
      --qualification "$BENCH_DIR/qualification.json" \
      --env-manifest "$BENCH_DIR/env_manifest.json" \
      --out "$CAMPAIGNS_DIR" \
      --arm "$ARM" --backend api --model-path "$MODEL" \
      --effort "$API_EFFORT" --max-spend-usd "$MAX_SPEND_PER_ARM" \
      --max-new-tokens 6000 \
      --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
      --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
      --max-eval-seeds "$MAX_EVAL_SEEDS" \
      --backend-fidelity "$BACKEND_FIDELITY" \
      --archive "$ARCHIVE_DIR/archive__${MODEL}__${ARM}.jsonl" \
      --save-raw --limit "$API_LIMIT" ${SCAFFOLD_FLAG:-}
  done
done

echo "--- total spend recorded across this job's ledgers ---"
"$PY_API" - <<'PYEOF'
import json, os, glob
tot = 0.0
for f in sorted(glob.glob(os.path.join(os.environ["CAMPAIGNS_DIR"], "cost__*.json"))):
    d = json.loads(open(f).read())
    tot += d.get("spent_usd", 0.0)
    print(f"  {os.path.basename(f):60s} ${d.get('spent_usd', 0):.4f} "
          f"over {d.get('calls', 0)} calls")
print(f"  {'TOTAL':60s} ${tot:.4f}")
PYEOF
echo "DONE api"
