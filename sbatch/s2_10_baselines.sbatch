#!/usr/bin/env bash
#SBATCH --job-name=s2_rules
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_10_rules_%j.out
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

# Every rule-based controller. No GPU and no API key: these are the bar, and
# they must be on disk before any model arm is interpreted.
if ! require_file "$BENCH_DIR/parents_public.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/oracle_table.jsonl" "no oracle table (s2_05)"; then exit 0; fi
mkdir -p "$CAMPAIGNS_DIR"
for R in $RULES; do
  echo "=== rule $R"
  "$PY" "$AGENT/run_campaign.py" \
    --parents "$BENCH_DIR/parents_public.json" \
    --qualification "$BENCH_DIR/qualification.json" \
    --env-manifest "$BENCH_DIR/env_manifest.json" \
    --oracle-table "$BENCH_DIR/oracle_table.jsonl" \
    --out "$CAMPAIGNS_DIR" --rule "$R" \
    --splits "$SPLITS" --max-iters "$MAX_ITERS" --patience "$PATIENCE" \
    --seed-budget "$SEED_BUDGET" --default-seeds "$DEFAULT_SEEDS" \
    --max-eval-seeds "$MAX_EVAL_SEEDS" \
    --backend-fidelity "$BACKEND_FIDELITY" --limit "$LIMIT" \
    --tool-menu "$TOOL_MENU"
done
echo "DONE baselines"
