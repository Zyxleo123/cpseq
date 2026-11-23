#!/usr/bin/env bash
# Submit the REGRET analysis for the pose-conditioned-likelihood search arms.
# Reads saved rows only; CPU, minutes. Config travels as a POSITIONAL ENV FILE,
# never `sbatch --export` (an explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1 and the job is requeued and HELD). Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_pll_regret.sh
#   PLLREGRET_ROWS=.../seqmut_iter_pll_rows.GEN80.jsonl PLLREGRET_MAX_EVALS=900 \
#     PLLREGRET_OUT=.../pll_regret_gen80.json sbatch/submit_lnr_pll_regret.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pll_regret_${STAMP}.env"
cat > "$ENVF" <<EOT
export LNR_OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
export PLLREGRET_ROWS="${PLLREGRET_ROWS:-${LNR_OUTDIR:-${RUN}/lnr}/seqmut_iter_pll_rows.jsonl}"
export PLLREGRET_OUT="${PLLREGRET_OUT:-${LNR_OUTDIR:-${RUN}/lnr}/pll_regret_summary.json}"
export PLLREGRET_N_PERM="${PLLREGRET_N_PERM:-20000}"
export PLLREGRET_THRESH="${PLLREGRET_THRESH:-0.05}"
export PLLREGRET_MAX_EVALS="${PLLREGRET_MAX_EVALS:-0}"
export PLLREGRET_ARM="${PLLREGRET_ARM:-}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_pll_regret.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_pll_regret.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_pll_regret_${J}.out"
