#!/usr/bin/env bash
#SBATCH --job-name=s0_scorefi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_42_score_fullif_%j.out
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

if ! require_file "$FULLIF_BENCH/decisions_oracle.json" "no full-interface bench (s0_12)"; then exit 0; fi
"$PY" "$STAGE0/score.py" \
  --bench "$FULLIF_BENCH" --arms "$FULLIF_ARMS" --out "$FULLIF_RESULTS" --splits "$SPLITS"
"$PY" "$STAGE0/figures.py" --results "$FULLIF_RESULTS" --bench "$FULLIF_BENCH"
echo "DONE fullif-score"
