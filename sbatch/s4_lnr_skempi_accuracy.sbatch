#!/usr/bin/env bash
#SBATCH --job-name=s4_skempi_acc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_skempi_accuracy_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE ACCURACY GATE. Does our Rosetta interface-dG scorer rank REAL mutations by
# their MEASURED binding change? Runs on SKEMPI v2 crystal structures (no Boltz
# fold -> isolates the scorer from fold noise, best case). Spearman/Pearson of
# Rosetta ddG vs measured ddG. Flex ddG (35-model ensemble) gets ~0.6 here; a
# single-relax scorer lands lower if it works at all. FAIL (~0) => ranking affinity
# is the wrong objective; change it. PASS (>=0.4) => scorer is fine, the project's
# problem is fold noise + cyclization.
#
# CPU only. Each task runs in its OWN short-lived subprocess (crash-isolated: an
# OOM/segfault/timeout on one big complex is one failed row, never the whole run --
# the ProcessPoolExecutor failure that killed the first attempt). Complexes over
# SKEMPI_MAX_CA residues are skipped (OOM-prone and far from our peptide scale).
# Resumable per (complex, mutation|WT, replica).
CSV="${SKEMPI_CSV:-/zfsauton/scratch/yixiz/lp2cp_pilot/data_raw/skempi_v2.csv}"
PDBDIR="${SKEMPI_PDB_DIR:-/zfsauton/scratch/yixiz/lp2cp_pilot/pdb}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
OUT="${SKEMPI_OUT:-${OUTDIR}/skempi_ddg.jsonl}"
MAXREC="${SKEMPI_MAX_RECORDS:-60}"
REPLICAS="${SKEMPI_REPLICAS:-3}"
WORKERS="${SKEMPI_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
MAXCA="${SKEMPI_MAX_CA:-600}"
TIMEOUT="${SKEMPI_TIMEOUT:-1800}"
mkdir -p "$OUTDIR"

echo "csv=$CSV pdb=$PDBDIR out=$OUT max_records=$MAXREC replicas=$REPLICAS workers=$WORKERS max_ca=$MAXCA timeout=$TIMEOUT"

if [ ! -f "$CSV" ]; then echo "GATE FAILED: no SKEMPI csv $CSV. Exit 0."; exit 0; fi
if [ ! -d "$PDBDIR" ]; then echo "GATE FAILED: no SKEMPI pdb dir $PDBDIR. Exit 0."; exit 0; fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then echo "GATE FAILED: no pyrosetta. Exit 0."; exit 0; fi
if ! "$PY" -c "import pandas" 2>/dev/null; then echo "GATE FAILED: no pandas (skempi loader). Exit 0."; exit 0; fi

echo "=== SKEMPI accuracy: Rosetta ddG vs measured ddG ==="
"$PY" "$AGENT/lnr/skempi_accuracy.py" \
  --score --csv "$CSV" --pdb-dir "$PDBDIR" --out "$OUT" \
  --max-records "$MAXREC" --replicas "$REPLICAS" --workers "$WORKERS" \
  --max-ca "$MAXCA" --timeout "$TIMEOUT"

echo "DONE lnr skempi-accuracy"
echo "re-render:  \$PY $AGENT/lnr/skempi_accuracy.py --report $OUT"
