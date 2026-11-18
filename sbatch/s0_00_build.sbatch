#!/usr/bin/env bash
#SBATCH --job-name=s0_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_00_build_%j.out
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

if ! require_file "$SKEMPI_CSV" "download it first: curl -sL -o $SKEMPI_CSV https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"; then
  exit 0
fi

"$PY" "$STAGE0/build.py" --csv "$SKEMPI_CSV" --out "$BENCH_DIR"

# Egress probe. s0_10 fetches ~90 structures from RCSB, so whether a compute
# node can reach the internet decides whether that stage can work at all. The
# probe runs here, where it is cheap, instead of being discovered as 90
# download failures an hour later.
echo "--- RCSB egress probe from $(hostname) ---"
if curl -sS --max-time 25 -o /dev/null -w 'http=%{http_code} bytes=%{size_download}\n' \
     https://files.rcsb.org/download/1CSE.pdb; then
  echo "EGRESS OK: this partition can fetch structures."
else
  echo "EGRESS FAILED from this node."
  echo "  s0_10_features will record download_failed for every uncached entry."
  echo "  Remedy: pre-fill $PDB_CACHE from a host with egress, then resubmit s0_10"
  echo "  (features.py is resumable and uses the cache before the network)."
fi
echo "DONE build"
