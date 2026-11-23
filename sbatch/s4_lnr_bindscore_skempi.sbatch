#!/usr/bin/env bash
#SBATCH --job-name=s4_bind_skempi
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_bindscore_skempi_%j.out
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

# THE CONFIRMATION ARM: the candidate binding metric on SKEMPI v2.
#
# The CPSet gate has the right chemistry but only 5 series with n>=3, so its
# per-series sign test cannot go below p = 1/32 however good the metric is.
# SKEMPI has the wrong chemistry (protein-protein) but thousands of measured
# mutations across many complexes. The two answer different halves, are reported
# separately, and are NEVER pooled.
#
# SKEMPI was previously judged a dead end for two reasons. One is now void: the
# "effects are inside our ~4 REU noise" objection was about the Rosetta oracle;
# THIS metric is exactly deterministic, so a 1 kcal/mol effect has no noise floor
# to hide under. The other -- wrong chemistry -- stands, which is why this
# confirms the scoring principle and not the head-to-tail claim.
#
# It is also newly affordable. The Rosetta gate managed 44 mutations (30% of its
# tasks segfaulted) because each needed a relax. Here the mutated chain's mask
# depends only on (complex, chain), so each complex is encoded ONCE and every
# mutation is two lookups into the resulting position x amino-acid matrix.
#
# Comparator, same benchmark, already measured by this project:
#     Rosetta dG_separated, single relax, 44 muts    rho +0.22, sign 59%
#     Flex ddG, 35-model ensemble (published)        rho ~ +0.60
CSV="${SKEMPI_CSV:-/zfsauton/scratch/yixiz/lp2cp_pilot/data_raw/skempi_v2.csv}"
PDBS="${SKEMPI_PDB_DIR:-/zfsauton/scratch/yixiz/lp2cp_pilot/pdb}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
OUT="${BINDSCORE_SKEMPI_OUT:-${OUTDIR}/skempi_bindscore.jsonl}"
MODELS="${BINDSCORE_MODELS:-ligandmpnn_v_32_020_25}"
MAXREC="${BINDSCORE_SKEMPI_MAXREC:-4000}"
MAXCA="${BINDSCORE_SKEMPI_MAXCA:-900}"
LMPNN="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"
export LIGANDMPNN_HOME="$LMPNN"
mkdir -p "$OUTDIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "csv=$CSV pdbs=$PDBS out=$OUT"
echo "models=$MODELS max_records=$MAXREC max_ca=$MAXCA"

# Gates print why and exit 0: a non-zero exit would strand dependents in
# DependencyNeverSatisfied with no record of which input was missing.
if [ ! -s "$CSV" ]; then
  echo "GATE FAILED: no SKEMPI csv at $CSV"
  exit 0
fi
if [ -z "$(ls -A "$PDBS"/*.pdb 2>/dev/null)" ]; then
  echo "GATE FAILED: no structures in $PDBS"
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

echo
echo "=== scoring SKEMPI mutations ==="
"$PY" "$AGENT/lnr/bindscore_skempi.py" --score \
  --csv "$CSV" --pdb-dir "$PDBS" --out "$OUT" \
  --models "$MODELS" --max-records "$MAXREC" --max-ca "$MAXCA"

echo
echo "=== report ==="
"$PY" "$AGENT/lnr/bindscore_skempi.py" --report "$OUT" || true

echo "DONE lnr bindscore-skempi"
echo "re-render (no recompute):  \$PY $AGENT/lnr/bindscore_skempi.py --report $OUT"
