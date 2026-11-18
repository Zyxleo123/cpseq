#!/usr/bin/env bash
#SBATCH --job-name=s4_rescore
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_rescore_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# Rung-3 rescore: FastRelax + InterfaceAnalyzer on the agent's committed clash-free
# rings, same PyRosetta protocol that produced native_dG. Answers "how much of the
# rung-2 proxy's gap to native is just the missing side-chain repack".
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
TARGETS="${LNR_RESCORE_TARGETS:-5l9b,1d4t,4w50,5njx}"
ARM="${LNR_RESCORE_ARM:-gpt-6-luna|llm-mask}"

echo "rescore targets=$TARGETS arm=$ARM outdir=$OUTDIR"
if [ ! -f "$MAN" ]; then echo "GATE FAILED: no manifest $MAN. Exit 0."; exit 0; fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in $PY. Exit 0."; exit 0
fi

"$PY" "$AGENT/lnr/rescore_rings.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --out-dir "$OUTDIR" \
  --targets "$TARGETS" --arm "$ARM" --relax-repeats "${LNR_RELAX_REPEATS:-1}"
echo "DONE rescore -> $OUTDIR/rescore_rings.jsonl"
