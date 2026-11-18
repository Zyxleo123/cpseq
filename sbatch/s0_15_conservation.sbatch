#!/usr/bin/env bash
#SBATCH --job-name=s0_cons
#SBATCH --partition=debug
#SBATCH --nodelist=gpu12,gpu13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_15_conservation_%j.out
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

# ESM-2 650M in fp32 is ~2.6 GB, so an 11 GB 2080 Ti is the right card for it
# and leaves the a6000s for the LLM arms. Pinned by --nodelist because GPU
# model is part of the resource decision, not a detail.
if ! require_file "$BENCH_DIR/decisions_public.json" "no benchmark"; then
  exit 0
fi
if ! require_file "$FEATURES_DIR/features.json" "structures not featurised yet (s0_10)"; then
  echo "NOTE: conservation needs the PDB cache that s0_10 fills; exiting 0."
  exit 0
fi

"$PY" "$STAGE0/conservation.py" \
  --bench "$BENCH_DIR" --pdb-cache "$PDB_CACHE" --out "$CONS_DIR" \
  --limit "${LIMIT:-0}"
echo "DONE conservation"
