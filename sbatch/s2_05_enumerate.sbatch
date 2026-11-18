#!/usr/bin/env bash
#SBATCH --job-name=s2_enum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s2_05_enum_%j.out
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

if ! require_file "$BENCH_DIR/parents_full.json" "no parents (s2_00)"; then exit 0; fi
if ! require_file "$BENCH_DIR/qualification.json" "no frozen predicate (s2_00)"; then exit 0; fi
"$PY" -m lp2cp.envbuild.enumerate_actions \
  --parents "$BENCH_DIR/parents_full.json" --out "$BENCH_DIR" \
  --qualification "$BENCH_DIR/qualification.json" --max-cells "$MAX_CELLS"
echo "DONE enumerate"
