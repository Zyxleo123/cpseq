#!/usr/bin/env bash
#SBATCH --job-name=s0_heurfi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_22_heur_fullif_%j.out
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

if ! require_file "$FULLIF_BENCH/decisions_public.json" "no full-interface bench (s0_12)"; then exit 0; fi
for ARM in random first_listed min_dsasa min_contacts max_entropy min_wt_prob composite max_dsasa; do
  echo "--- arm $ARM"
  "$PY" "$STAGE0/run_arm.py" \
    --arm "$ARM" --bench "$FULLIF_BENCH" --features "$FEATURES_DIR" \
    --conservation "$FULLIF_CONS" --out "$FULLIF_ARMS" \
    --slices fullif_ala_k1 --splits "$SPLITS" --order main
done
echo "DONE fullif-heuristics"
