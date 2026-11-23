#!/usr/bin/env bash
# Submit the CANDIDATE BINDING METRIC gate: pose-conditioned interface
# likelihood (ICG) vs measured pKd, within-target, on CPSet Dataset I's
# congeneric series. CPU only.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: any explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild
# the login environment on the compute node, and the job is requeued and HELD.
# Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_bindscore.sh
#   BINDSCORE_SERIES=S03_keap1 sbatch/submit_lnr_bindscore.sh
#   BINDSCORE_MODELS=ligandmpnn_v_32_020_25,proteinmpnn_v_48_020 sbatch/submit_lnr_bindscore.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/bindscore_${STAMP}.env"
cat > "$ENVF" <<EOT
export CPSET_DATA_DIR="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
export CPSET_WORK_DIR="${CPSET_WORK_DIR:-${RUN}/cpset_prepared}"
export CPSET_OUT="${CPSET_OUT:-${RUN}/lnr/cpset_dg.jsonl}"
export BINDSCORE_OUT="${BINDSCORE_OUT:-${RUN}/lnr/bindscore.jsonl}"
export BINDSCORE_MODELS="${BINDSCORE_MODELS:-ligandmpnn_v_32_020_25}"
export BINDSCORE_SERIES="${BINDSCORE_SERIES:-}"
export BINDSCORE_ORDER_SEED="${BINDSCORE_ORDER_SEED:-0}"
export LIGANDMPNN_HOME="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrubbed so a submission from inside an allocation does not inherit its job's
# SLURM_* and confuse the new job's own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_bindscore.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_bindscore.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_bindscore_${J}.out"
