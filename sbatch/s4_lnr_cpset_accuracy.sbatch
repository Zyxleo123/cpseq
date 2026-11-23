#!/usr/bin/env bash
#SBATCH --job-name=s4_cpset_acc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_cpset_accuracy_%j.out
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

# THE CYCLIC-PEPTIDE ACCURACY GATE. Does our Rosetta interface-dG scorer rank the
# MEASURED binding of cyclic peptides against the SAME receptor?
#
# The SKEMPI gate (rho +0.22, sign 59%) had two confounds it could not remove:
# wrong chemistry (linear protein-protein, not cyclic peptide) and effects
# typically ~1 kcal/mol, i.e. INSIDE our ~4 REU noise -- so its null was
# ambiguous. This gate removes both. Ground truth is CPSet Dataset I (Zhao et al.
# Brief Bioinform 2025, doi 10.1093/bib/bbaf632): 50 protein-cyclic-peptide
# crystal complexes with experimental Kd, of which 38 form 10 CONGENERIC SERIES
# (same receptor, peptide variants) -- a within-target instrument the paper
# itself never used, since its own validation is entirely cross-complex.
# Headline series: 7K2* = KEAP1 + six head-to-tail 7-mers spanning 3.3 log units
# (~4.5 kcal/mol), i.e. 3-4x the noise floor instead of inside it.
#
# Scores DEPOSITED crystal coordinates -- no Boltz fold -- so this isolates the
# scoring function, exactly as the SKEMPI gate did. CPU only; the verdict rule is
# PRE-REGISTERED in the module docstring and cannot be tuned to the result.
#
# Each task runs in its OWN short-lived subprocess: the SKEMPI run lost 102 of
# 363 tasks to uncatchable PyRosetta SEGFAULTS, and under a shared pool one crash
# is a BrokenProcessPool that discards every completed result. Resumable per
# (pdb, replica, scorer), so a time limit means resubmit, not restart.
DATA="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
OUT="${CPSET_OUT:-${OUTDIR}/cpset_dg.jsonl}"
WORK="${CPSET_WORK_DIR:-${RUN}/cpset_prepared}"
REPLICAS="${CPSET_REPLICAS:-3}"
WORKERS="${CPSET_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
TIMEOUT="${CPSET_TIMEOUT:-1800}"
SCORER="${CPSET_SCORER:-rosetta}"
mkdir -p "$OUTDIR" "$WORK"

echo "data=$DATA out=$OUT work=$WORK replicas=$REPLICAS workers=$WORKERS"
echo "timeout=$TIMEOUT scorer=$SCORER"

# Gates print why and exit 0. A non-zero exit here would strand every dependent
# job in DependencyNeverSatisfied with no record of which input was missing.
if [ ! -s "$DATA/dataset1.csv" ]; then
  echo "GATE FAILED: no ground truth at $DATA/dataset1.csv"
  echo "  fetch it first:  \$PY $AGENT/lnr/cpset_accuracy.py --fetch --data-dir $DATA"
  exit 0
fi
if [ ! -d "$DATA/dataset1" ]; then
  echo "GATE FAILED: no structures at $DATA/dataset1 (the extracted tarball)"
  echo "  fetch it first:  \$PY $AGENT/lnr/cpset_accuracy.py --fetch --data-dir $DATA"
  exit 0
fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in $PY. Exit 0."
  exit 0
fi

# The headline is a within-SERIES statistic, so a grouping that silently shifted
# would silently move the headline. Verify the frozen table against the actual
# structures BEFORE spending any CPU; a mismatch is a hard stop, because the
# right response is to update the frozen table deliberately, not to score
# against a grouping nobody has looked at.
echo "=== verifying the frozen series table against the structures ==="
if ! "$PY" "$AGENT/lnr/cpset_accuracy.py" --regroup --data-dir "$DATA"; then
  echo "GATE FAILED: derived series grouping does NOT match the frozen table."
  echo "  Update SERIES in agent/lnr/cpset_accuracy.py deliberately, then resubmit."
  exit 0
fi

echo "=== CPSet accuracy: Rosetta dG vs measured pKd (within-target) ==="
"$PY" "$AGENT/lnr/cpset_accuracy.py" \
  --score --data-dir "$DATA" --out "$OUT" --work-dir "$WORK" \
  --replicas "$REPLICAS" --workers "$WORKERS" --timeout "$TIMEOUT" \
  --scorer "$SCORER"

echo
echo "=== report ==="
"$PY" "$AGENT/lnr/cpset_accuracy.py" --report "$OUT" || true

echo "DONE lnr cpset-accuracy"
echo "re-render (no recompute):  \$PY $AGENT/lnr/cpset_accuracy.py --report $OUT"
