#!/usr/bin/env bash
#SBATCH --job-name=s4_bindscore
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_bindscore_%j.out
# `set -euo pipefail` lives HERE, in the job body, not only in the preamble: if
# the source below fails, a preamble-only setting would leave this running under
# the system python and still EXIT 0, releasing every afterok dependent behind a
# stage that did no work.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE CANDIDATE BINDING METRIC: pose-conditioned interface likelihood (ICG).
#
# Every metric this project has tried failed for one shared reason -- none was
# conditioned on the sequence's fit to the specific BOUND POSE. Rosetta dG tracks
# buried area (partial rho controlling peptide length = -0.002); Boltz ipTM/LIS
# score whether SOME structure is confident; ESM-2 / PepMLM score whether the
# peptide is A PLAUSIBLE PEPTIDE. On CPSet S03_keap1, D2->A costs 2.33 log units
# of measured affinity and Rosetta calls that variant its STRONGEST binder.
#
# ICG = mean_i [ log p(s_i | receptor + peptide backbone)
#                - log p(s_i | peptide backbone alone) ]   nats/residue
#
# CPU only, and deliberately so: the structure encoder never reads the sequence,
# so a template is encoded once and any number of variants are scored by a few
# small decoder passes. The marginal is EXACTLY deterministic (augment_eps=0, and
# a position decoded first is provably independent of every other position's
# decoding order) -- which is the point: the incumbent oracle needs 16 GPU folds
# to reach SEM ~1.0 REU, this needs one CPU pass and no repeats.
#
# Pre-registration, including the verdict rule, is in
# docs/binding_metric_design.md and was committed BEFORE this ever ran.
WORK="${CPSET_WORK_DIR:-${RUN}/cpset_prepared}"
DATA="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
OUT="${BINDSCORE_OUT:-${OUTDIR}/bindscore.jsonl}"
DGJ="${CPSET_OUT:-${OUTDIR}/cpset_dg.jsonl}"
MODELS="${BINDSCORE_MODELS:-ligandmpnn_v_32_020_25}"
SERIES_ONLY="${BINDSCORE_SERIES:-}"
ORDER_SEED="${BINDSCORE_ORDER_SEED:-0}"
LMPNN="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"
export LIGANDMPNN_HOME="$LMPNN"
mkdir -p "$OUTDIR"

# Torch on a shared CPU node: without this one job can grab every core and the
# node's other jobs crawl. Also makes the timing numbers in the log meaningful.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "work=$WORK data=$DATA out=$OUT"
echo "models=$MODELS series=${SERIES_ONLY:-<all>} order_seed=$ORDER_SEED"
echo "ligandmpnn=$LMPNN"

# Gates print why and exit 0. A non-zero exit would strand every dependent job
# in DependencyNeverSatisfied with no record of which input was missing.
if [ ! -s "$DATA/dataset1.csv" ]; then
  echo "GATE FAILED: no ground truth at $DATA/dataset1.csv"
  echo "  fetch it:  \$PY $AGENT/lnr/cpset_accuracy.py --fetch --data-dir $DATA"
  exit 0
fi
if [ -z "$(ls -A "$WORK"/*_prepared.pdb 2>/dev/null)" ]; then
  echo "GATE FAILED: no prepared structures in $WORK"
  echo "  they are written by s4_lnr_cpset_accuracy.sbatch (--score)."
  exit 0
fi
if [ ! -f "$LMPNN/model_utils.py" ]; then
  echo "GATE FAILED: LigandMPNN not found at $LMPNN. Exit 0."
  exit 0
fi
if ! "$PY" -c "import torch, prody" 2>/dev/null; then
  echo "GATE FAILED: torch or prody missing in $PY. Exit 0."
  exit 0
fi
if [ ! -s "$DGJ" ]; then
  echo "NOTE: no $DGJ -- dg_rosetta and null_negdsasa will be absent from the"
  echo "      comparison table. Everything else still runs."
fi

# The self-test asserts the module's OWN claims against the real model before
# any science is done: that the primary mode is bitwise-identical across
# decoding-order seeds, that it does not see the peptide's own side chains or
# sequence, that removing the receptor DOES change it, and that the subset
# scorer still agrees exactly with upstream single_aa_score. A silently-flipped
# upstream flag would otherwise invert the metric with no error.
echo
echo "=== self-test (claims asserted against the real checkpoint) ==="
if ! "$PY" "$AGENT/lnr/bindscore.py" --selftest --work-dir "$WORK"; then
  echo "GATE FAILED: bindscore self-test did not pass. Not scoring."
  exit 0
fi

echo
echo "=== scoring: every variant against every same-length template ==="
"$PY" "$AGENT/lnr/bindscore.py" --score \
  --work-dir "$WORK" --data-dir "$DATA" --dg "$DGJ" --out "$OUT" \
  --models "$MODELS" --order-seed "$ORDER_SEED" --n-shuf "${BINDSCORE_NSHUF:-8}" \
  ${SERIES_ONLY:+--series "$SERIES_ONLY"}

# THE SPECIFICITY CONTROL. The first run left one objection standing: pll_pep
# (peptide scored with the receptor DELETED) reached 0.923 on the BB/MIX tier,
# so the metric was not yet shown to read the POCKET rather than the peptide's
# own bound conformation. Threading each series onto a same-length template from
# a DIFFERENT receptor settles it: ranking must COLLAPSE there and hold on the
# true receptor. Written to its own file; decoy rows are never pooled with real.
if [ "${BINDSCORE_DECOY:-1}" = "1" ]; then
  DECOY_OUT="${BINDSCORE_DECOY_OUT:-${OUT%.jsonl}_decoy.jsonl}"
  echo
  echo "=== specificity control: wrong-receptor templates -> $DECOY_OUT ==="
  "$PY" "$AGENT/lnr/bindscore.py" --decoy \
    --work-dir "$WORK" --data-dir "$DATA" --out "$DECOY_OUT" \
    --models "$MODELS" || true
  echo
  echo "=== report on DECOY receptors (ranking must COLLAPSE here) ==="
  "$PY" "$AGENT/lnr/bindscore.py" --report "$DECOY_OUT" --data-dir "$DATA" \
    --tier BB,MIX || true
fi

echo
echo "=== report: ALL TIERS ==="
"$PY" "$AGENT/lnr/bindscore.py" --report "$OUT" --data-dir "$DATA" || true

echo
echo "=== report: BB/MIX TIER ONLY (the head-to-tail claim) ==="
"$PY" "$AGENT/lnr/bindscore.py" --report "$OUT" --data-dir "$DATA" --tier BB,MIX || true

echo
echo "=== per-position SAR diagnostic: S03_keap1 (the known-SAR series) ==="
echo "The measured SAR is local and known: D2->A costs 2.33 log units, T5->A"
echo "costs 2.78, E3->P gains 0.5-0.9. If the metric is right for the right"
echo "reason, its log-probability drops AT POSITIONS 2 AND 5 and nowhere else."
"$PY" "$AGENT/lnr/bindscore.py" --sar "$OUT" --data-dir "$DATA" --series S03_keap1 || true

# THE DEPLOYABLE ARTEFACT. In marginal mode the metric is a position x
# amino-acid matrix fixed by the pose, so a target is compiled ONCE and every
# candidate sequence is a table lookup. This prints that table for a real
# head-to-tail complex, which is what the search would consume.
if [ "${BINDSCORE_PSSM:-1}" = "1" ]; then
  PSSM_PDB="${BINDSCORE_PSSM_PDB:-${WORK}/7K2H_prepared.pdb}"
  PSSM_OUT="${BINDSCORE_PSSM_OUT:-${OUTDIR}/pssm_7K2H.json}"
  if [ -s "$PSSM_PDB" ]; then
    echo
    echo "=== deployable artefact: pose-derived PSSM for $(basename "$PSSM_PDB") ==="
    "$PY" "$AGENT/lnr/bindscore.py" --pssm "$PSSM_PDB" \
      --pssm-out "$PSSM_OUT" --models "${MODELS%%,*}" || true
  fi
fi

echo "DONE lnr bindscore"
echo "re-render (no recompute):  \$PY $AGENT/lnr/bindscore.py --report $OUT --data-dir $DATA --tier BB,MIX"
