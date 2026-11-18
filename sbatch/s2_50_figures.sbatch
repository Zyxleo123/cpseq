#!/usr/bin/env bash
#SBATCH --job-name=s2_figs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_50_figures_%j.out
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
if ! require_file "$RESULTS_DIR/per_campaign.jsonl" "no scored results (s2_40)"; then exit 0; fi
"$PY" "$AGENT/figures.py" --results "$RESULTS_DIR" --bench "$BENCH_DIR" \
  --out "$RESULTS_DIR/figures"
echo "DONE figures"
