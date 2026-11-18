#!/usr/bin/env bash
#SBATCH --job-name=s0_consfi
#SBATCH --partition=debug
#SBATCH --nodelist=gpu12,gpu13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_17_cons_fullif_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage0_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage0_preamble.sh
. "$PRE" "$@"
banner
export FULLIF_BENCH="${RUN}/bench_fullif"
export FULLIF_CONS="${RUN}/features_fullif"
export FULLIF_ARMS="${RUN}/arms_fullif"
export FULLIF_RESULTS="${RUN}/results_fullif"

# The primary conservation pass masks only the ~16 measured candidates per
# complex. The full-interface arm offers 20-80 positions, so its tolerance
# numbers have to be computed over that wider pool or most of its candidates
# would show dashes where the measured ones show values -- which would mark
# the measured positions as surely as a flag.
if ! require_file "$FULLIF_BENCH/decisions_public.json" "no full-interface bench (s0_12)"; then exit 0; fi
"$PY" "$STAGE0/conservation.py" \
  --bench "$FULLIF_BENCH" --pdb-cache "$PDB_CACHE" --out "$FULLIF_CONS"
echo "DONE fullif-conservation"
