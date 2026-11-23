#!/usr/bin/env bash
# Submit the CYCLIC-PEPTIDE accuracy gate: Rosetta interface dG vs measured pKd,
# within-target, on CPSet Dataset I's 10 congeneric series. CPU only.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: any explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild
# the login environment on the compute node, and the job is requeued and HELD.
# Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_cpset_accuracy.sh
#   CPSET_REPLICAS=5 sbatch/submit_lnr_cpset_accuracy.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/cpset_acc_${STAMP}.env"
cat > "$ENVF" <<EOT
export CPSET_DATA_DIR="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
export CPSET_OUT="${CPSET_OUT:-${RUN}/lnr/cpset_dg.jsonl}"
export CPSET_WORK_DIR="${CPSET_WORK_DIR:-${RUN}/cpset_prepared}"
export CPSET_REPLICAS="${CPSET_REPLICAS:-3}"
export CPSET_TIMEOUT="${CPSET_TIMEOUT:-1800}"
export CPSET_SCORER="${CPSET_SCORER:-rosetta}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrubbed so a submission from inside an allocation does not inherit its job's
# SLURM_* and confuse the new job's own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_cpset_accuracy.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_cpset_accuracy.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_cpset_accuracy_${J}.out"
