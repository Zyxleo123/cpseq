#!/usr/bin/env bash
#SBATCH --job-name=s4_plot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_plot_%j.out
set -euo pipefail
# set -euo pipefail in the body (not only the preamble): a silently failed
# source would otherwise run under the system python and exit 0.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# Redrawable case figures: native bound pose vs placement_search ring (baseline)
# vs gpt-6-luna mask ring (ours). CPU only -- a few torsion re-solves + matplotlib.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
TARGETS="${LNR_PLOT_TARGETS:-5l9b,1d4t,4w50,5njx}"
MODE="${LNR_PLOT_MODE:-all}"

echo "plot targets=$TARGETS mode=$MODE outdir=$OUTDIR"
if [ ! -f "$MAN" ]; then echo "GATE FAILED: no manifest $MAN. Exit 0."; exit 0; fi
if ! "$PY" -c "import matplotlib, scipy, numpy" 2>/dev/null; then
  echo "GATE FAILED: missing matplotlib/scipy/numpy in $PY. Exit 0."; exit 0
fi

"$PY" "$AGENT/lnr/plot_cases.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --out-dir "$OUTDIR" \
  --targets "$TARGETS" --mode "$MODE"
echo "DONE plot -> $OUTDIR/cases.png"
