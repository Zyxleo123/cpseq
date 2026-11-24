#!/usr/bin/env bash
# Submit the ADDITIVITY pilot: does the frozen pll table predict the change in
# pll once the pose is re-predicted? Re-scores Boltz structures already on disk,
# so it costs no fold -- CPU, minutes. Config travels as a POSITIONAL ENV FILE,
# never `sbatch --export` (an explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1 and the job is requeued and HELD). Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_pll_additivity.sh
#   PLLADD_SCOPE=iface sbatch/submit_lnr_pll_additivity.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pll_additivity_${STAMP}.env"
OUTDIR_D="${LNR_OUTDIR:-${RUN}/lnr}"
cat > "$ENVF" <<EOT
export LNR_OUTDIR="${OUTDIR_D}"
# Boltz predictions the oracle already produced. Reusing them is why this pilot
# is free; it is also why it is a LOWER BOUND on non-additivity, since those
# folds held the backbone outside the released window.
export PLLADD_BOLTZ_ROOT="${PLLADD_BOLTZ_ROOT:-${OUTDIR_D}/boltz_val}"
export PLLADD_PSSM_DIR="${PLLADD_PSSM_DIR:-${OUTDIR_D}/pssm}"
export PLLADD_OUT="${PLLADD_OUT:-${OUTDIR_D}/pll_additivity.jsonl}"
# cpu: one LigandMPNN forward pass per structure, a few score of structures.
export PLLADD_DEVICE="${PLLADD_DEVICE:-cpu}"
export PLLADD_SCOPE="${PLLADD_SCOPE:-all}"
export PLLADD_METRIC="${PLLADD_METRIC:-}"
# 1 = recompute the statistics from saved rows without re-scoring structures.
export PLLADD_REPORT_ONLY="${PLLADD_REPORT_ONLY:-0}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_pll_additivity.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_pll_additivity.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_pll_additivity_${J}.out"
