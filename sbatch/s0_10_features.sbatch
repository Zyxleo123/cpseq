#!/usr/bin/env bash
#SBATCH --job-name=s0_feat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_10_features_%j.out
set -euo pipefail
# `set -euo pipefail` lives in the JOB BODY, not only in the preamble: if the
# source below silently fails, a preamble-only setting leaves the job running
# under the system python and exiting 0, which releases every afterok
# dependent behind a stage that did no work.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage0_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage0_preamble.sh
# Slurm copies the batch script to /var/spool/slurm/job<id>/slurm_script, so
# $(dirname "${BASH_SOURCE[0]}") points at the spool dir and sourcing a shared
# preamble that way sources nothing at all.
. "$PRE" "$@"
banner

if ! require_file "$BENCH_DIR/decisions_public.json" "stage s0_00_build did not produce a benchmark"; then
  exit 0
fi

# Resumable: an existing features.json is extended, not rebuilt.
"$PY" "$STAGE0/features.py" \
  --bench "$BENCH_DIR" --pdb-cache "$PDB_CACHE" --out "$FEATURES_DIR" \
  --limit "${LIMIT:-0}"
echo "DONE features"
