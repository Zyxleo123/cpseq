#!/usr/bin/env bash
# Submit THE accuracy gate: Rosetta interface ddG vs measured SKEMPI ddG. CPU only.
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`. Run from lov*.
# Usage: sbatch/submit_lnr_skempi_accuracy.sh   (SKEMPI_MAX_RECORDS=N to resize)
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/skempi_acc_${STAMP}.env"
cat > "$ENVF" <<EOT
export SKEMPI_CSV="${SKEMPI_CSV:-/zfsauton/scratch/yixiz/lp2cp_pilot/data_raw/skempi_v2.csv}"
export SKEMPI_PDB_DIR="${SKEMPI_PDB_DIR:-/zfsauton/scratch/yixiz/lp2cp_pilot/pdb}"
export SKEMPI_OUT="${SKEMPI_OUT:-${RUN}/lnr/skempi_ddg.jsonl}"
export SKEMPI_MAX_RECORDS="${SKEMPI_MAX_RECORDS:-60}"
export SKEMPI_REPLICAS="${SKEMPI_REPLICAS:-3}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_skempi_accuracy.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_skempi_accuracy.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_skempi_accuracy_${J}.out"
