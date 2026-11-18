#!/usr/bin/env bash
#SBATCH --job-name=s0_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_40_score_%j.out
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

if ! require_file "$BENCH_DIR/decisions_oracle.json" "no oracle (s0_00)"; then exit 0; fi

# Reads only saved artefacts, so every table and figure is redrawable without
# rerunning a model or touching a GPU.
"$PY" "$STAGE0/score.py" \
  --bench "$BENCH_DIR" --arms "$ARMS_DIR" --out "$RESULTS_DIR" --splits "$SPLITS"

# Figures read only results.json and per_decision.jsonl, so they redraw without
# a GPU, a model, or rerunning an arm.
"$PY" "$STAGE0/figures.py" --results "$RESULTS_DIR" --bench "$BENCH_DIR"
echo "DONE score"
