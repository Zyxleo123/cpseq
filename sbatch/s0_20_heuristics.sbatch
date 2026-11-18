#!/usr/bin/env bash
#SBATCH --job-name=s0_heur
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_pilot/logs/s0_20_heuristics_%j.out
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

if ! require_file "$BENCH_DIR/decisions_public.json" "no benchmark (s0_00)"; then exit 0; fi
if ! require_file "$FEATURES_DIR/features.json" "no structural features (s0_10)"; then exit 0; fi

# Conservation is optional here: without it the two ESM-based heuristics score
# every candidate identically and are reported as degenerate rather than
# silently becoming a tie-break on candidate id.
if [ ! -s "$CONS_DIR/conservation.json" ]; then
  echo "NOTE: no conservation.json yet; max_entropy / min_wt_prob / composite will be degenerate."
fi

for ARM in $HEUR_ARMS; do
  echo "--- arm $ARM"
  # One output file per arm. Parallel jobs appending to a shared file
  # NUL-corrupt each other's rows under O_APPEND.
  "$PY" "$STAGE0/run_arm.py" \
    --arm "$ARM" --bench "$BENCH_DIR" --features "$FEATURES_DIR" \
    --conservation "$CONS_DIR" --out "$ARMS_DIR" \
    --slices "$SLICES" --splits "$SPLITS" --order main --limit "${LIMIT:-0}"
done

# The position-bias control needs the permuted order too, but only for arms
# whose answer could depend on order at all.
for ARM in first_listed random; do
  "$PY" "$STAGE0/run_arm.py" \
    --arm "$ARM" --bench "$BENCH_DIR" --features "$FEATURES_DIR" \
    --conservation "$CONS_DIR" --out "$ARMS_DIR" \
    --slices "$SLICES" --splits "$SPLITS" --order alt --limit "${LIMIT:-0}"
done
echo "DONE heuristics"
