#!/usr/bin/env bash
# Submit the CONFIRMATION arm: the candidate binding metric on SKEMPI v2.
# CPU only. Config travels as a POSITIONAL ENV FILE, never `sbatch --export`.
# Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_bindscore_skempi.sh
#   BINDSCORE_SKEMPI_MAXREC=8000 sbatch/submit_lnr_bindscore_skempi.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/bindscore_skempi_${STAMP}.env"
cat > "$ENVF" <<EOT
export SKEMPI_CSV="${SKEMPI_CSV:-/zfsauton/scratch/yixiz/lp2cp_pilot/data_raw/skempi_v2.csv}"
export SKEMPI_PDB_DIR="${SKEMPI_PDB_DIR:-/zfsauton/scratch/yixiz/lp2cp_pilot/pdb}"
export BINDSCORE_SKEMPI_OUT="${BINDSCORE_SKEMPI_OUT:-${RUN}/lnr/skempi_bindscore.jsonl}"
export BINDSCORE_MODELS="${BINDSCORE_MODELS:-ligandmpnn_v_32_020_25}"
export BINDSCORE_SKEMPI_MAXREC="${BINDSCORE_SKEMPI_MAXREC:-4000}"
export BINDSCORE_SKEMPI_MAXCA="${BINDSCORE_SKEMPI_MAXCA:-900}"
export LIGANDMPNN_HOME="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_bindscore_skempi.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_bindscore_skempi.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_bindscore_skempi_${J}.out"
