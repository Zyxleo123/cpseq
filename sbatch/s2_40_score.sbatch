#!/usr/bin/env bash
#SBATCH --job-name=s2_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_40_score_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# Reads only saved artefacts: campaign traces plus the oracle table. No GPU, no
# API key, no model. The whole point of the enumerated table is that a metric
# can be redefined and every campaign re-scored without re-running anything.
if ! require_file "$BENCH_DIR/oracle_table.jsonl" "no oracle table (s2_05)"; then exit 0; fi
if ! ls "$CAMPAIGNS_DIR"/campaigns__*.jsonl >/dev/null 2>&1; then
  echo "GATE FAILED: no campaigns under $CAMPAIGNS_DIR"
  echo "  Run s2_10_baselines / s2_20_llm8b / s2_30_api first. Exiting 0."
  exit 0
fi
mkdir -p "$RESULTS_DIR"
"$PY" "$AGENT/score_campaigns.py" \
  --campaigns "$CAMPAIGNS_DIR" \
  --parents "$BENCH_DIR/parents_full.json" \
  --oracle-table "$BENCH_DIR/oracle_table.jsonl" \
  --out "$RESULTS_DIR" \
  --baseline "${SCORE_BASELINE:-rule:rim_first_closure}"
echo "DONE score"
