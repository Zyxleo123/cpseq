#!/usr/bin/env bash
# Submit the Stage 0 pilot. This script only calls sbatch -- it does no work of
# its own, so nothing here depends on running from a login node or inside an
# allocation.
#
#   bash submit_stage0.sh                      # full chain, 8B LLM arm included
#   bash submit_stage0.sh --no-llm             # deterministic stages only
#   bash submit_stage0.sh --llm-only           # LLM arm + its scorer (bench must exist)
#   bash submit_stage0.sh --smoke              # 6 decisions per slice, end to end
#   bash submit_stage0.sh --fullif             # full-interface secondary arm only
#   bash submit_stage0.sh --set LLM_EVIDENCE_ARMS=full,structure_only
#
# Configuration travels as a timestamped env FILE passed positionally, never
# through `sbatch --export`: any explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd then fails to rebuild the login environment on
# the compute node, and the job is requeued and HELD, stranding its dependents.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_pilot}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENV_FILE="${RUN}/env/stage0_${STAMP}.env"

WITH_LLM=1
LLM_ONLY=0
SMOKE=0
FULLIF=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --no-llm)   WITH_LLM=0; shift ;;
    --llm-only) LLM_ONLY=1; shift ;;
    --smoke)    SMOKE=1; shift ;;
    --fullif)   FULLIF=1; shift ;;
    --set)      EXTRA+=("$2"); shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${RUN}/env" "${RUN}/logs"

{
  echo "# Stage 0 submission ${STAMP}"
  echo "export RUN=${RUN}"
  echo "export CODE=$(cd "$HERE/.." && pwd)"
  echo "export SLICES=\"ala_k1,allsub_k1,allsub_k2\""
  echo "export SPLITS=\"dev,test\""
  echo "export LLM_MODEL_PATH=/zfsauton/scratch/yixiz/models/Llama-3.1-8B-Instruct"
  # The evidence ablation: full table, structure only, conservation only,
  # aggregate only (the 2->3 analogue), and no evidence at all (the floor that
  # says how much of any win is the prompt rather than the diagnostics).
  echo "export LLM_EVIDENCE_ARMS=\"full,structure_only,conservation_only,aggregate_only,no_evidence\""
  echo "export LLM_ORDERS=\"main\""
  echo "export LLM_TEMPERATURE=0.0"
  if [ "$SMOKE" = "1" ]; then
    echo "export LIMIT=6"
    echo "export LLM_EVIDENCE_ARMS=\"full\""
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

if [ "$FULLIF" = "1" ]; then
  # The full-interface secondary arm. Needs features.json from the primary
  # chain, so it is submitted separately once that has landed rather than being
  # wired in as a dependent of a job that may already have finished.
  J_FI=$(sb "$HERE/s0_12_fullif.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_12_fullif       -> $J_FI"
  J_FC=$(sb --dependency=afterok:"$J_FI" "$HERE/s0_17_cons_fullif.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_17_cons_fullif  -> $J_FC (afterok:$J_FI)"
  J_FH=$(sb --dependency=afterok:"$J_FC" "$HERE/s0_22_heur_fullif.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_22_heur_fullif  -> $J_FH (afterok:$J_FC)"
  J_FL=$(sb --dependency=afterok:"$J_FC" "$HERE/s0_32_llm_fullif.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_32_llm_fullif   -> $J_FL (afterok:$J_FC)"
  J_FS=$(sb --dependency=afterok:"$J_FL":"$J_FH" "$HERE/s0_42_score_fullif.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_42_score_fullif -> $J_FS (afterok:$J_FL:$J_FH)"
  exit 0
fi

if [ "$LLM_ONLY" = "1" ]; then
  J_LLM=$(sb "$HERE/s0_30_llm.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_30_llm        -> $J_LLM"
  J_SCORE=$(sb --dependency=afterok:"$J_LLM" "$HERE/s0_40_score.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_40_score      -> $J_SCORE (afterok:$J_LLM)"
  exit 0
fi

J_BUILD=$(sb "$HERE/s0_00_build.sbatch" "$ENV_FILE" | jid)
echo "submitted s0_00_build      -> $J_BUILD"

J_FEAT=$(sb --dependency=afterok:"$J_BUILD" "$HERE/s0_10_features.sbatch" "$ENV_FILE" | jid)
echo "submitted s0_10_features   -> $J_FEAT (afterok:$J_BUILD)"

J_CONS=$(sb --dependency=afterok:"$J_FEAT" "$HERE/s0_15_conservation.sbatch" "$ENV_FILE" | jid)
echo "submitted s0_15_conserv    -> $J_CONS (afterok:$J_FEAT)"

J_HEUR=$(sb --dependency=afterok:"$J_CONS" "$HERE/s0_20_heuristics.sbatch" "$ENV_FILE" | jid)
echo "submitted s0_20_heuristics -> $J_HEUR (afterok:$J_CONS)"

# The early scorer depends ONLY on the heuristic arms, so the rule-based bar and
# the oracle spread are readable without waiting on a GPU allocation.
J_EARLY=$(sb --dependency=afterok:"$J_HEUR" "$HERE/s0_40_score.sbatch" "$ENV_FILE" | jid)
echo "submitted s0_40_score(early) -> $J_EARLY (afterok:$J_HEUR)"

if [ "$WITH_LLM" = "1" ]; then
  # Sibling of the heuristics stage, not a child: it queues for a GPU
  # concurrently instead of behind CPU work it does not need.
  J_LLM=$(sb --dependency=afterok:"$J_CONS" "$HERE/s0_30_llm.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_30_llm       -> $J_LLM (afterok:$J_CONS)"
  J_FULL=$(sb --dependency=afterok:"$J_LLM":"$J_HEUR" "$HERE/s0_40_score.sbatch" "$ENV_FILE" | jid)
  echo "submitted s0_40_score(full) -> $J_FULL (afterok:$J_LLM:$J_HEUR)"
else
  echo "(--no-llm: GPU arm not submitted; run with --llm-only once a card frees)"
fi

echo
echo "watch:  squeue -u \$USER -o '%.9i %.12j %.9P %.2t %.10M %.6D %R'"
echo "logs:   ${RUN}/logs/"
echo "result: ${RUN}/results/results.json"
