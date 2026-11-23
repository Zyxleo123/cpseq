#!/usr/bin/env bash
#SBATCH --job-name=s4_conf_rank
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_confidence_rank_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# RETROSPECTIVE: do interface-CONFIDENCE scores rank peptide variants better than
# the Rosetta interface dG we currently select on? Reads ONLY folds already on
# disk -- no GPU, no new folds, nothing recomputed.
#
# Why: the in-loop fitness uses Boltz->Rosetta dG as an affinity regressor, the
# hardest use of it (dedicated ddG methods cap at Pearson ~0.63 WITH a 20-30
# model ensemble; this project's proxy-vs-true rho was +0.29). Overath et al.
# 2025 (3,766 designs, 436 binders) found interface-focused CONFIDENCE scores
# beat the physics energies for predicting real binding, best when paired with a
# physicochemical descriptor. Boltz already writes confidence_*.json (ipTM,
# PAIRWISE per-chain ipTM, ipLDDT, ipDE), pae_*.npz and plddt_*.npz for every
# fold -- and boltz_inloop reads none of it. This recovers all of it.
#
# What it can answer, and what it cannot. There is NO experimental ground truth
# in this project, so this is NOT "which metric predicts affinity". It measures
# (a) RESOLUTION -- can the metric separate a target's variants above its own
# noise, (b) REPRODUCIBILITY across independent fold seeds, (c) AGREEMENT with dG
# and whether it catches the drifted folds the validator drops.
#
# CPU only, minutes. Resumable: a (variant,draw) already extracted is skipped.
# Safe to run WHILE the GPU search is writing new folds -- a fold with no cache
# row yet is still being written and is skipped until it completes.
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
WORK="${CONF_WORK:-${OUTDIR}/boltz_inloop}"
CACHE="${CONF_CACHE:-${OUTDIR}/seqmut_iter_boltz_cache.jsonl}"
OUT="${CONF_OUT:-${OUTDIR}/conf_metrics.jsonl}"
PEPCHAIN="${CONF_PEP_CHAIN:-B}"
MINN="${CONF_MIN_N:-5}"
mkdir -p "$OUTDIR"

echo "work=$WORK cache=$CACHE out=$OUT pep_chain=$PEPCHAIN min_n=$MINN"

# --- gates: print why and exit 0, never a non-zero that strands dependents ---
if [ ! -d "$WORK" ]; then
  echo "GATE FAILED: no Boltz work dir at $WORK. Exiting 0."; exit 0
fi
if [ ! -f "$CACHE" ]; then
  echo "GATE FAILED: no fold cache at $CACHE -- nothing to join against. Exiting 0."
  exit 0
fi
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "GATE FAILED: no numpy in \$PY ($PY). Exiting 0."; exit 0
fi
NF=$(find "$WORK" -maxdepth 1 -type d -name 'heldout_*' | wc -l)
if [ "$NF" -eq 0 ]; then
  echo "GATE FAILED: no heldout_* fold dirs under $WORK. Exiting 0."; exit 0
fi
echo "$NF fold dirs on disk; $(wc -l < "$CACHE") cache rows"

echo "=== confidence-metric extraction + retrospective ranking comparison ==="
"$PY" "$AGENT/lnr/confidence_metrics.py" \
  --extract --work "$WORK" --cache "$CACHE" --out "$OUT" \
  --pep-chain "$PEPCHAIN" --min-n "$MINN"

echo "DONE lnr confidence-rank"
echo "re-render the comparison with no recompute:"
echo "  \$PY $AGENT/lnr/confidence_metrics.py --report $OUT"
