#!/usr/bin/env bash
#SBATCH --job-name=s2_parents
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_00_parents_%j.out
set -euo pipefail
# `set -euo pipefail` lives in the JOB BODY, not only in the preamble: if the
# source below silently fails, a preamble-only setting leaves the job running
# under the system python and exiting 0, which releases every afterok dependent
# behind a stage that did no work.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
# Slurm copies the batch script to /var/spool/slurm/job<id>/slurm_script, so
# $(dirname "${BASH_SOURCE[0]}") points at the spool dir and sourcing a shared
# preamble that way sources nothing at all.
. "$PRE" "$@"
banner

# CPU only. Measuring SASA, contacts and dihedrals over ~900 deposited
# structures is biotite on numpy; asking for a GPU to do it would just queue
# behind real work.
if [ ! -d "$CPSEA_ROOT" ]; then
  echo "GATE FAILED: no CPSea preprocessed data at $CPSEA_ROOT"
  echo "  This is the sibling project's output (preprocess_cpsea.py)."
  echo "  Exiting 0 so dependents report the same gate rather than vanishing."
  exit 0
fi
mkdir -p "$BENCH_DIR"
"$PY" -m lp2cp.envbuild.build_parents \
  --cpsea "$CPSEA_ROOT" --out "$BENCH_DIR" \
  --n-parents "$N_PARENTS" --k-budget "$K_BUDGET" \
  --splits "$CPSEA_SPLITS" --max-scan "$MAX_SCAN"
echo "DONE parents"
