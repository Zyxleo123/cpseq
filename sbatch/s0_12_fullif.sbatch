#!/usr/bin/env bash
#SBATCH --job-name=s0_fullif
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_12_fullif_%j.out
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

if ! require_file "$FEATURES_DIR/features.json" "no structural features (s0_10)"; then exit 0; fi
"$PY" "$STAGE0/fullinterface.py" \
  --bench "$BENCH_DIR" --features "$FEATURES_DIR" --out "$FULLIF_BENCH" --slice ala_k1
echo "DONE fullif-build"
