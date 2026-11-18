#!/usr/bin/env bash
#SBATCH --job-name=s3_rosval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s3_40_rosval_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# CPU only: PyRosetta FastRelax and InterfaceAnalyzer are single-threaded CPU
# work on a ~135-residue complex. No GPU, no API key, no spend.
#
# This is the one job in the project that uses a REAL energy function. It does
# not score any campaign: it tests whether the surrogate's declared response
# model agrees with Rosetta on the parents' own deposited structures. Read
# agent/validate_real_scorer.py before trusting anything it writes.
if ! require_file "$BENCH_DIR/parents_full.json" "no parents (s2_00)"; then exit 0; fi
if [ ! -d "$CPSEA_ROOT" ]; then
  echo "GATE FAILED: no CPSea processed structures at $CPSEA_ROOT"
  echo "  The parents' PDB files are read from there. Exiting 0."
  exit 0
fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in $PY. Exiting 0."
  exit 0
fi
mkdir -p "$(dirname "${ROSVAL_OUT}")"
echo "out        : ${ROSVAL_OUT}"
echo "parents    : ${BENCH_DIR}/parents_full.json"
echo "relax reps : ${ROSVAL_RELAX_REPEATS}   candidates: ${ROSVAL_CANDIDATES}"
echo "n_probe    : ${ROSVAL_NPROBE}   limit: ${ROSVAL_LIMIT}"

# Resumable: one JSONL row per parent, parents already present are skipped, so
# hitting the time limit costs a resubmission rather than the pass.
"$PY" "$AGENT/validate_real_scorer.py" \
  --parents "$BENCH_DIR/parents_full.json" \
  --pdb-root "$CPSEA_ROOT/processed" \
  --out "$ROSVAL_OUT" \
  --relax-repeats "$ROSVAL_RELAX_REPEATS" \
  --candidates "$ROSVAL_CANDIDATES" \
  --n-probe "$ROSVAL_NPROBE" \
  --limit "$ROSVAL_LIMIT"
echo "DONE rosetta_validate"
