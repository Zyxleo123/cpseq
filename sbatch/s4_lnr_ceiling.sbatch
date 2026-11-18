#!/usr/bin/env bash
#SBATCH --job-name=s4_ceil
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=6:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_ceiling_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# The realistic binding-energy ceiling (lnr/energy_ceiling.py): the clash-free
# closing window that keeps the most interface ENERGY, per complex, so the
# nomut/placement recovered dG has a ceiling to look up to. CPU only (torsion
# solver + the saved WT alanine scan; no Rosetta, no GPU).
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
SCANS="${CEIL_SCANS:-${OUTDIR}/rosetta_interface.jsonl}"
TARGETS="${CEIL_TARGETS:-}"
MAXSEC="${LNR_MAX_SECONDS:-12}"
TOP="${CEIL_CONFIRM_TOP:-40}"
LIMIT="${LNR_LIMIT:-0}"
mkdir -p "$OUTDIR"

echo "manifest=$MAN scans=$SCANS targets=${TARGETS:-<all scanned>} confirm_top=$TOP"
if [ ! -f "$MAN" ] || [ ! -f "$SCANS" ]; then
  echo "GATE FAILED: need manifest ($MAN) and WT scans ($SCANS). Exiting 0."; exit 0
fi
if ! "$PY" -c "import scipy, numpy" 2>/dev/null; then
  echo "GATE FAILED: no scipy/numpy in \$PY. Exiting 0."; exit 0
fi

"$PY" "$AGENT/lnr/energy_ceiling.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --scans "$SCANS" \
  --out "$OUTDIR/${CEIL_OUT_FILE:-energy_ceiling.jsonl}" \
  --targets "$TARGETS" --confirm-top "$TOP" --max-seconds "$MAXSEC" --limit "$LIMIT"

echo "DONE lnr ceiling"
