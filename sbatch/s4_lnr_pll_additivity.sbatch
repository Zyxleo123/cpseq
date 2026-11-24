#!/usr/bin/env bash
#SBATCH --job-name=s4_pll_add
# CPU, and that is the whole point of running this pilot FIRST. It reads Boltz
# structures ALREADY ON DISK from the earlier oracle work and re-scores them with
# a LigandMPNN forward pass, so it costs no fold. The designed k-sweep that would
# follow needs a GPU; this does not, and it decides whether that GPU time is
# worth spending (CLAUDE.md: split jobs by resource).
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_additivity_%j.out
# `set -euo pipefail` in the job body, not only the preamble: a failed source
# would otherwise run this under the system python and still EXIT 0.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ===========================================================================
# THIS JOB TESTS THE BAR, NOT AN ARM.
#
# Every negative result this project has about search rests on one property:
# `TargetPSSM.score` is a mean of independent per-position terms, so `best_subs`
# is the EXACT optimum and an arm that misses it is failing at a solved problem.
# That property is bought by FREEZING THE POSE -- the matrix is built once from
# one structure, so a substitution at i cannot move the term at j.
#
# Real peptides coupled through structure. If the coupling is large, the frozen
# table is a FIRST-ORDER APPROXIMATION, `best_subs` is not the optimum, and the
# regret that has been killing every search arm was measured against the wrong
# zero. That is why this is worth more than another arm comparison.
#
# THE DOUBLE DIFFERENCE. An own-pose PSSM comes from a BOLTZ-PREDICTED complex;
# the frozen table comes from the CRYSTAL. Those differ for reasons unrelated to
# any mutation, so the reported quantity is
#     gap = [pll_own(mut) - pll_own(nat)] - [pll_frozen(mut) - pll_frozen(nat)]
# with the native scored under its OWN re-predicted pose. A mutant whose target
# has no cached native fold is DROPPED, never compared against the crystal pose:
# that substitution would measure crystal-vs-Boltz and manufacture a result.
#
# THE NOISE FLOOR IS FREE AND IS NOT OPTIONAL. Each cached Boltz call emitted
# several diffusion samples from one trunk pass -- different structures, same
# sequence. The spread of own-pose pll across them is the floor any gap must
# clear. This project already paid for omitting that once (the +-3 REU dG floor
# that turned out to be fold sampling, not signal), so the verdict refuses to
# call a gap real until it exceeds the floor and prints NOT RESOLVED instead.
#
# WHAT A NULL RESULT HERE DOES NOT LICENSE. The cached folds used a HELD
# template: backbone outside the released window [a,b] is pinned to native, so
# only the window can move and coupling through the held region is suppressed BY
# CONSTRUCTION. Measured non-additivity is therefore a LOWER BOUND. A null
# licenses "additivity holds when the backbone is mostly pinned" -- not
# "additivity holds". Substitutions inside the window are reported separately
# from ones touching the held region so the two regimes are never pooled.
# ===========================================================================
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
BOLTZ_ROOT="${PLLADD_BOLTZ_ROOT:-${OUTDIR}/boltz_val}"
PSSM_DIR="${PLLADD_PSSM_DIR:-${OUTDIR}/pssm}"
OUT_ROWS="${PLLADD_OUT:-${OUTDIR}/pll_additivity.jsonl}"
DEVICE="${PLLADD_DEVICE:-cpu}"
SCOPE="${PLLADD_SCOPE:-all}"
METRIC="${PLLADD_METRIC:-}"

echo "boltz_root=$BOLTZ_ROOT"
echo "pssm_dir=$PSSM_DIR  device=$DEVICE  scope=$SCOPE  metric='${METRIC:-<pll_bg>}'"
echo "out=$OUT_ROWS"

# GATES. Both inputs are artefacts of earlier jobs; a missing one means this
# pilot cannot run at all. Print which, and exit 0 so nothing downstream is
# stranded in DependencyNeverSatisfied with no record of the cause.
N_PDB="$(find "$BOLTZ_ROOT" -name '*_model_*.pdb' 2>/dev/null | wc -l || true)"
if [ "${N_PDB:-0}" -eq 0 ]; then
  echo "GATE FAILED: no cached Boltz structures under $BOLTZ_ROOT"
  echo "  this pilot re-scores folds the oracle already produced; with none,"
  echo "  run the designed GPU sweep instead of this job"
  echo "Exit 0 so no dependent job is stranded."; exit 0
fi
if [ ! -d "$PSSM_DIR" ] || [ -z "$(ls -A "$PSSM_DIR"/pssm_*.json 2>/dev/null)" ]; then
  echo "GATE FAILED: no frozen PSSMs under $PSSM_DIR"
  echo "  run sbatch/submit_lnr_seqmut_iter_pll.sh first -- it writes them there"
  echo "Exit 0 so no dependent job is stranded."; exit 0
fi
echo "cached structures: $N_PDB   frozen PSSMs: $(ls -1 "$PSSM_DIR"/pssm_*.json | wc -l)"

# Fast gate first: it exercises the stem parser, the chain reader and the double
# difference without touching a structure. A broken double difference would
# otherwise print a confident zero gap, which is the exact shape of result this
# job exists to avoid producing by accident.
echo; echo "=== selftest ==="
"$PY" "${AGENT}/lnr/pll_additivity.py" --selftest

# REPORT-ONLY path. The rows carry every per-fold value, so the statistics can
# be recomputed without re-scoring a single structure -- which is the point of
# writing them (CLAUDE.md: figures and tables must be redrawable without
# recomputation). Used whenever the ANALYSIS changes rather than the data.
if [ "${PLLADD_REPORT_ONLY:-0}" = "1" ]; then
  if ! require_file "$OUT_ROWS" "run this job without PLLADD_REPORT_ONLY first"; then
    echo "Exit 0 so no dependent job is stranded."; exit 0
  fi
  echo; echo "=== report only (no re-scoring) ==="
  "$PY" "${AGENT}/lnr/pll_additivity.py" --report --rows "$OUT_ROWS"
  echo "done: $(date -Is)"; exit 0
fi

echo; echo "=== measure ==="
"$PY" "${AGENT}/lnr/pll_additivity.py" --measure \
  --boltz-root "$BOLTZ_ROOT" --pssm-dir "$PSSM_DIR" \
  --device "$DEVICE" --scope "$SCOPE" ${METRIC:+--metric "$METRIC"} \
  --out "$OUT_ROWS"

echo; echo "rows are re-reportable without re-scoring:"
echo "  $PY ${AGENT}/lnr/pll_additivity.py --report --rows $OUT_ROWS"
echo "done: $(date -Is)"
