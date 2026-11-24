#!/usr/bin/env bash
#SBATCH --job-name=s4_pllo_score
# CPU partition, many workers, no GPU. Relaxing and scoring 300+ structures
# twice is embarrassingly parallel CPU work; asking for a GPU to run FastRelax
# would queue it behind real folding for nothing.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_oracle_score_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE PROJECT'S OWN SCORING PROTOCOL, BOTH VERSIONS, ON EVERY STRUCTURE FROM
# EITHER PREDICTOR.
#
#   v3 "bond"      harmonic AtomPairConstraint on the closing bond alone, then
#                  FastRelax, then InterfaceAnalyzer dG_separated on the relaxed
#                  complex. The requested protocol.
#   v5 "junction"  the same, with the closure angles and torsion restrained too.
#
# Both, because they disagree in a measured way: under a bond-only relax cpseq
# measured 37 of 48 closure angles out of tolerance AFTER relax against 0 of 48
# before, so a v3 energy difference can be the minimiser bending a ring open
# rather than a property of the sequence. Scoring the SAME structures under both
# costs CPU and no GPU, and makes the disagreement a measurement.
#
# Resumable by construction: every (structure, version) pair already in
# scores.jsonl is skipped, so a time limit means resubmitting this job.
OUT="${PLLO_OUT:-${RUN}/lnr/pll_oracle}"
VERSIONS="${PLLO_VERSIONS:-v3,v5}"
WORKERS="${PLLO_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
ARMS="${PLLO_ARMS:-}"

# PyRosetta is single-threaded per pose; the parallelism is the worker pool, so each
# worker must not also try to thread as well.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "out=$OUT versions=$VERSIONS workers=$WORKERS arms='${ARMS:-<all>}'"
if [ ! -s "$OUT/folds.jsonl" ]; then
  echo "GATE FAILED: no $OUT/folds.jsonl -- nothing has been folded yet."
  echo "  'upstream never ran', not 'folding failed'. Exit 0."
  exit 0
fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: pyrosetta missing in $PY. Exit 0."; exit 0
fi

ARGS=(--score --out-dir "$OUT" --versions "$VERSIONS" --workers "$WORKERS")
[ -n "$ARMS" ] && ARGS+=(--arms "$ARMS")
"$PY" "$AGENT/lnr/pll_oracle.py" "${ARGS[@]}"

echo "DONE pll_oracle score -> $OUT/scores.jsonl"
wc -l "$OUT/scores.jsonl" 2>/dev/null || true
