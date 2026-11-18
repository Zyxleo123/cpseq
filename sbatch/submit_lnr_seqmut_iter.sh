#!/usr/bin/env bash
# Submit the ITERATIVE-SEARCH mutation arms: the free random-EA control and the
# LLM evolutionary loop, as SIBLING jobs that share one scan cache and both reuse
# the single-shot run's committed windows + Rosetta scans. They write SEPARATE
# rows files (the single-shot "one rows file per job" convention), aggregated by
# `seqmut_iter.py --report-dir` for the KEY CONTROL: does the LLM operator beat
# the random operator at a MATCHED proxy-evaluation budget?
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export` (an export list
# makes sbatch set SLURM_GET_USER_ENV=1, the compute node's login-env rebuild
# fails, and the job is requeued+held). SLURM_* is scrubbed from this submitter's
# own environment first, since we may submit from inside an allocation.
#
# Usage:
#   sbatch/submit_lnr_seqmut_iter.sh            # EA control only (free, no API)
#   SEQITER_WITH_LLM=1 sbatch/submit_lnr_seqmut_iter.sh    # + the LLM arms
set -euo pipefail

SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"
mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# Matched evolutionary knobs shared by BOTH arms -- the matched-budget contract.
# Keep these identical across the two env files; the whole comparison rests on it.
KMUT="${SEQITER_K:-2}"
NSEED="${SEQITER_N_SEED:-8}"
NGEN="${SEQITER_N_GEN:-6}"
MOFF="${SEQITER_M_OFF:-6}"
MAXEVALS="${SEQITER_MAX_EVALS:-40}"
ELITE="${SEQITER_ELITE_FRAC:-0.3}"
LIMIT="${LNR_LIMIT:-8}"
# LLM arm (only submitted when SEQITER_WITH_LLM is set)
MODEL="${SEQITER_MODEL:-gpt-6-luna}"
BACKEND="${SEQITER_BACKEND:-openai}"
MAXSPEND="${SEQITER_MAX_SPEND:-2.0}"
LLM_ARMS="${SEQITER_LLM_ARMS:-iter-llm,iter-llm-ala}"

# --- EA control env file (free: no backend) --------------------------------
EA_ENV="${ENVDIR}/seqiter_ea_${STAMP}.env"
cat > "$EA_ENV" <<EOF
export SEQITER_ARMS="iter-ea"
export SEQITER_ROWS_FILE="seqmut_iter_ea_rows.jsonl"
export SEQITER_K="${KMUT}"
export SEQITER_N_SEED="${NSEED}"
export SEQITER_N_GEN="${NGEN}"
export SEQITER_M_OFF="${MOFF}"
export SEQITER_MAX_EVALS="${MAXEVALS}"
export SEQITER_ELITE_FRAC="${ELITE}"
export LNR_LIMIT="${LIMIT}"
EOF
echo "wrote $EA_ENV:"; sed 's/^/    /' "$EA_ENV"

# --- LLM-arm env file (same knobs; adds the backend) -----------------------
LLM_ENV="${ENVDIR}/seqiter_llm_${STAMP}.env"
cat > "$LLM_ENV" <<EOF
export SEQITER_ARMS="${LLM_ARMS}"
export SEQITER_ROWS_FILE="seqmut_iter_llm_${MODEL}_rows.jsonl"
export SEQITER_BACKEND="${BACKEND}"
export SEQITER_MODEL="${MODEL}"
export SEQITER_MAX_SPEND="${MAXSPEND}"
export SEQITER_K="${KMUT}"
export SEQITER_N_SEED="${NSEED}"
export SEQITER_N_GEN="${NGEN}"
export SEQITER_M_OFF="${MOFF}"
export SEQITER_MAX_EVALS="${MAXEVALS}"
export SEQITER_ELITE_FRAC="${ELITE}"
export LNR_LIMIT="${LIMIT}"
EOF
echo "wrote $LLM_ENV:"; sed 's/^/    /' "$LLM_ENV"

# scrub SLURM_* so nothing from an allocation leaks into the submitted jobs
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true

echo "+ sbatch ${SBATCH_DIR}/s4_lnr_seqmut_iter.sbatch ${EA_ENV}"
A=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_seqmut_iter.sbatch" "${EA_ENV}")
echo "submitted EA-control job: $A"

if [ -n "${SEQITER_WITH_LLM:-}" ]; then
  echo "+ sbatch ${SBATCH_DIR}/s4_lnr_seqmut_iter.sbatch ${LLM_ENV}"
  B=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_seqmut_iter.sbatch" "${LLM_ENV}")
  echo "submitted LLM-arm job: $B"
else
  echo "LLM arms NOT submitted (set SEQITER_WITH_LLM=1 to add them). Env file is"
  echo "  ready at $LLM_ENV -- submit with:"
  echo "    sbatch ${SBATCH_DIR}/s4_lnr_seqmut_iter.sbatch ${LLM_ENV}"
fi

echo
echo "report after they land (aggregates EA + LLM for the matched-budget control):"
echo "  PYTHONPATH=agent:pilot/stage0 \$PY -m lnr.seqmut_iter --report-dir ${RUN}/lnr"
echo "then fold the finalists for the TRUE 4-fold dG (no code change needed):"
echo "  LNR_BOLTZ_MUT_ARMS='iter-ea:k${KMUT},iter-llm:${MODEL}:k${KMUT},iter-llm-ala:${MODEL}:k${KMUT}' \\"
echo "    sbatch ${SBATCH_DIR}/s4_lnr_boltz.sbatch <mut-held env file>"
