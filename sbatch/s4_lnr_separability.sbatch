#!/usr/bin/env bash
#SBATCH --job-name=s4_separab
# CPU only: this is arithmetic on sequences. No structure, no fold, no GPU --
# asking for one would just queue behind real work (CLAUDE.md). `cpu` has 5 idle
# nodes (lov[5-9]); `debug` is the wrong default and is GPU anyway.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_separab_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
echo "host     : $(hostname)"
echo "request  : partition=cpu cpus=2 mem=8G time=00:30:00"

# ===========================================================================
# IS EACH LIABILITY OBJECTIVE ACTUALLY TABLEABLE? MEASURE IT.
#
# The entire case for a search arm is that some objective is JOINT -- not a
# function of per-position values or residue counts -- because every other class
# has an exact table method (sort / DP / ILP). `pepliab.py` HAND-LABELS that
# class per check, and a hand label is not evidence.
#
# This job measures it from each metric's own behaviour, with the discrete mixed
# second difference I(i,j) = f(s|i->a,j->b) - f(s|i->a) - f(s|j->b) + f(s):
#   all I = 0                  -> additive        -> one sort
#   I = 0 beyond distance w    -> bounded width w -> DP
#   long range, permutation-invariant -> counts    -> ILP
#   none of the above          -> FLAGGED for hand scrutiny (see below)
#
# THE LAST CLASS IS A FLAG, NOT A VERDICT, and the module's own selftest is what
# forced that. Bounded-width local terms combined by MAX (or min, or a count of
# passing windows) couple positions at unbounded distance while remaining
# solvable by a bounded-state scan -- a sliding-window max-of-sums is exactly
# DP-solvable and lands in the last class anyway. Black-box probing cannot
# separate "bounded-state DP" from "needs evaluation", so this instrument
# concludes only in the safe direction: the first three classes are PROOFS of
# tableability, and the last means "resisted the obvious table".
#
# That asymmetry is deliberate. A false "tableable" is the dangerous error here,
# because it would let a search arm be justified by a metric a DP eats. A false
# flag only costs a hand check.
#
# WHY THIS RUNS BEFORE ANY JOINT AXIS IS IMPLEMENTED. The obvious
# implementations of the five declared-but-unmeasured axes are secretly
# tableable, and would look like a result:
#   * aggregation propensity as a sliding-window hydrophobicity SUM is LOCAL.
#   * an off-target panel scored with FROZEN PSSMs is a sum of per-position
#     terms per target, and a sum of separable functions is separable.
# Any new axis gets run through this gate before it is believed.
#
# TWO PROBE PROPERTIES THAT ARE LOAD-BEARING, both found by this job failing:
#  1. EXHAUSTIVE over the 19x19 residue pairs, never sampled. Sampling called
#     `asp_pro` ADDITIVE because a D-P motif interacts only when a draw happens
#     to be exactly (D, P) -- 1/400 per attempt. That is a false proof of
#     tableability that no movement guard catches.
#  2. The movement guard counts the DOUBLE substitution. `net_charge` sits
#     inside its allowed window until two edits push it out, so counting
#     singles alone reported "never moved" for a metric with hundreds of
#     measured interactions.
# ===========================================================================
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
ROWS="${SEPARAB_ROWS:-${OUTDIR}/pepliab_audit.jsonl}"
OUT="${SEPARAB_OUT:-${OUTDIR}/separability.jsonl}"
NSUBS="${SEPARAB_N_SUBS:-0}"
WMAX="${SEPARAB_W_MAX:-4}"
LIMIT="${SEPARAB_LIMIT:-12}"
SEED="${SEPARAB_SEED:-0}"
mkdir -p "$OUTDIR"

echo "rows (native seqs) : $ROWS"
echo "out                : $OUT"
echo "n_subs=$NSUBS (0 = exhaustive)  w_max=$WMAX  limit=$LIMIT  seed=$SEED"

# The selftest is the gate, and it is a real one: it classifies metrics whose
# class is known BY CONSTRUCTION, including the sliding-window max that must
# come back flagged rather than joint, and a dead metric that must come back
# UNDETERMINED rather than additive. If it fails, the audit below is noise.
echo; echo "=== selftest (metrics of known class) ==="
if ! "$PY" "${AGENT}/lnr/separability.py" --selftest; then
  echo "GATE FAILED: the classifier mis-classifies metrics whose class is known."
  echo "  The audit would be noise, so it is not run."
  echo "Exit 0 so no chained job is stranded."; exit 0
fi

# Native sequences are nice-to-have, not required: the module falls back to a
# spread chosen to trip every threshold. Missing rows must not fail the job.
if [ ! -s "$ROWS" ]; then
  echo; echo "NOTE: $ROWS absent -- using the built-in threshold-spanning"
  echo "  sequences. Not a gate failure: the probe needs sequences that TRIP"
  echo "  the checks, and the fallback set is chosen for exactly that."
  ROWS=""
fi

echo; echo "=== audit: measured structure vs the hand label ==="
"$PY" "${AGENT}/lnr/separability.py" --audit \
  ${ROWS:+--rows "$ROWS"} --out "$OUT" \
  --n-subs "$NSUBS" --w-max "$WMAX" --limit "$LIMIT" --seed "$SEED"

echo "done: $(date -Is)"
