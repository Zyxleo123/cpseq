#!/usr/bin/env bash
# Submit the Stage 2 end-to-end agent. This script only calls sbatch -- it does
# no work of its own, so nothing here depends on running from a login node or
# inside an allocation.
#
#   bash submit_stage2.sh                 # bench -> baselines + 8B + API -> score
#   bash submit_stage2.sh --bench-only    # build the benchmark and stop
#   bash submit_stage2.sh --no-llm        # rule baselines only
#   bash submit_stage2.sh --no-api        # everything except the billed arms
#   bash submit_stage2.sh --llm-only      # 8B arm + scorer (bench must exist)
#   bash submit_stage2.sh --api-only      # API arms + scorer
#   bash submit_stage2.sh --70b           # the 4-bit 70B arm + scorer
#   bash submit_stage2.sh --smoke         # 4 parents, 2 iterations, end to end
#   bash submit_stage2.sh --set MAX_ITERS=8 --set API_LIMIT=6
#
# Configuration travels as a timestamped env FILE passed positionally, never
# through `sbatch --export`: any explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on
# the compute node, and the job is requeued and HELD, stranding its dependents.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${RUN}/env/stage2_${STAMP}.env"

BENCH_ONLY=0; WITH_LLM=1; WITH_API=1; LLM_ONLY=0; API_ONLY=0
WITH_70B=0; SMOKE=0; EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bench-only) BENCH_ONLY=1; shift ;;
    --no-llm)     WITH_LLM=0; WITH_API=0; shift ;;
    --no-api)     WITH_API=0; shift ;;
    --llm-only)   LLM_ONLY=1; shift ;;
    --api-only)   API_ONLY=1; shift ;;
    --70b)        WITH_70B=1; shift ;;
    --smoke)      SMOKE=1; shift ;;
    --set)        EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${RUN}/env" "${RUN}/logs"

{
  echo "# Stage 2 submission ${STAMP}"
  echo "export RUN=${RUN}"
  echo "export CODE=$(cd "$HERE/.." && pwd)"
  echo "export N_PARENTS=32"
  echo "export K_BUDGET=4"
  echo "export MAX_SCAN=900"
  echo "export SPLITS=\"dev,test\""
  echo "export MAX_ITERS=5"
  echo "export PATIENCE=2"
  echo "export SEED_BUDGET=60"
  echo "export MAX_EVAL_SEEDS=20"
  # The five arms cost five times the API budget. The default runs the 2->3
  # ablation only -- the one comparison the design doc says is "the only one
  # with no prior-art answer" -- and the local arms carry the full ladder.
  echo "export LOCAL_ARMS=\"A1_FRESH A2_CONDITIONING A3_CONTRACT A4_PRIOR A5_REFLECTION\""
  echo "export API_ARMS=\"A2_CONDITIONING A3_CONTRACT\""
  echo "export ARMS_70B=\"A3_CONTRACT\""
  echo "export API_MODELS=\"claude-opus-5-5 claude-sonnet-5-5\""
  echo "export API_EFFORT=low"
  echo "export MAX_SPEND_PER_ARM=0.60"
  echo "export API_LIMIT=10"
  if [ "$SMOKE" = "1" ]; then
    echo "export N_PARENTS=8"
    echo "export MAX_SCAN=150"
    echo "export LIMIT=4"
    echo "export MAX_ITERS=2"
    echo "export LOCAL_ARMS=\"A3_CONTRACT\""
    echo "export API_ARMS=\"A3_CONTRACT\""
    echo "export API_MODELS=\"claude-sonnet-5-5\""
    echo "export API_LIMIT=2"
    echo "export MAX_SPEND_PER_ARM=0.08"
  else
    echo "export LIMIT=0"
  fi
  for kv in "${EXTRA[@]:-}"; do
    [ -n "$kv" ] && echo "export $kv"
  done
} > "$ENV_FILE"

echo "env file: $ENV_FILE"
sed 's/^/    /' "$ENV_FILE"
echo

# Scrub SLURM_* before calling sbatch: submitting from inside an allocation
# otherwise leaks this job's resource view into the child's environment.
sb() { env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ') sbatch "$@"; }
jid() { awk '{print $NF}'; }

if [ "$LLM_ONLY" = "1" ]; then
  J=$(sb "$HERE/s2_20_llm8b.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_20_llm8b   -> $J"
  JS=$(sb --dependency=afterok:"$J" "$HERE/s2_40_score.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_40_score   -> $JS (afterok:$J)"
  exit 0
fi

if [ "$API_ONLY" = "1" ]; then
  J=$(sb "$HERE/s2_30_api.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_30_api     -> $J"
  JS=$(sb --dependency=afterok:"$J" "$HERE/s2_40_score.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_40_score   -> $JS (afterok:$J)"
  exit 0
fi

if [ "$WITH_70B" = "1" ]; then
  J=$(sb "$HERE/s2_22_llm70b.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_22_llm70b  -> $J"
  JS=$(sb --dependency=afterok:"$J" "$HERE/s2_40_score.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_40_score   -> $JS (afterok:$J)"
  exit 0
fi

J_PAR=$(sb "$HERE/s2_00_parents.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_00_parents   -> $J_PAR"
J_ENUM=$(sb --dependency=afterok:"$J_PAR" "$HERE/s2_05_enumerate.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_05_enumerate -> $J_ENUM (afterok:$J_PAR)"
if [ "$BENCH_ONLY" = "1" ]; then exit 0; fi

J_RULE=$(sb --dependency=afterok:"$J_ENUM" "$HERE/s2_10_baselines.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_10_baselines -> $J_RULE (afterok:$J_ENUM)"

# The early scorer depends ONLY on the baselines, so the rule-based bar and the
# oracle spread are readable without waiting on a GPU allocation or on spend.
J_EARLY=$(sb --dependency=afterok:"$J_RULE" "$HERE/s2_40_score.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_40_score(early) -> $J_EARLY (afterok:$J_RULE)"

DEPS="$J_RULE"
if [ "$WITH_LLM" = "1" ]; then
  # Sibling of the baselines stage, not a child: it queues for a GPU
  # concurrently instead of behind CPU work it does not need. It depends on the
  # PARENTS job, not on enumeration -- an LLM arm never reads the oracle table.
  J_8B=$(sb --dependency=afterok:"$J_PAR" "$HERE/s2_20_llm8b.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_20_llm8b     -> $J_8B (afterok:$J_PAR)"
  DEPS="$DEPS:$J_8B"
fi
if [ "$WITH_API" = "1" ]; then
  J_API=$(sb --dependency=afterok:"$J_PAR" "$HERE/s2_30_api.sbatch" "$ENV_FILE" | jid)
  echo "submitted s2_30_api       -> $J_API (afterok:$J_PAR)"
  DEPS="$DEPS:$J_API"
fi

J_SCORE=$(sb --dependency=afterok:"$DEPS" "$HERE/s2_40_score.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_40_score(full) -> $J_SCORE (afterok:$DEPS)"
J_FIG=$(sb --dependency=afterok:"$J_SCORE" "$HERE/s2_50_figures.sbatch" "$ENV_FILE" | jid)
echo "submitted s2_50_figures   -> $J_FIG (afterok:$J_SCORE)"

echo
echo "watch:  squeue -u \$USER -o '%.9i %.12j %.9P %.2t %.10M %.6D %R'"
echo "logs:   ${RUN}/logs/"
echo "result: ${RESULTS_DIR:-${RUN}/results}/results.json"
