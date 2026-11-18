#!/usr/bin/env bash
#SBATCH --job-name=s4_linref
#SBATCH --partition=debug
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_linref_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
nvidia-smi -L || true

# Linear-peptide Boltz->Rosetta REFERENCE (the real-pipeline dG ceiling): fold the
# WT sequence LINEAR (cyclic flag off), whole peptide templated at native, score a
# plain two-chain interface dG. Same Boltz+Rosetta pipeline as the cyclic arms, so
# (linear_ref - nomut_cyclic) is the pure cyclisation cost. Driver under $PY
# (cpseq + pyrosetta); it subprocesses the fold to the boltz env on this GPU.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
TARGETS="${LINREF_TARGETS:-1bjr,1d4t,1ee5,1elr,1fiw,1jrr,1ky6,1ssc}"

echo "linref targets=$TARGETS"
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in \$PY. Exit 0."; exit 0
fi
if [ ! -d /zfsauton/scratch/yixiz/boltz_cache ]; then
  echo "GATE FAILED: no boltz cache. Exit 0."; exit 0
fi

"$PY" "$AGENT/lnr/boltz_linear_ref.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --out-dir "$OUTDIR" --targets "$TARGETS" \
  --relax-repeats "${LNR_RELAX_REPEATS:-1}"
echo "DONE linref"
