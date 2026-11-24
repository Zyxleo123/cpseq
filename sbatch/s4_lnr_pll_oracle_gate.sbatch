#!/usr/bin/env bash
#SBATCH --job-name=s4_pllo_gate
# GPU class is pinned by EXCLUSION, not by node: the 11 GB rtx_2080_ti nodes
# (gpu12-gpu19) are too small for AF2 with a ~380-residue receptor, and a CUDA
# OOM here dies with no Python traceback -- a silent FAILED that reads like a
# code bug. Everything left (legacy v100 16/32 GB, general a6000 48 GB) fits.
#SBATCH --partition=general,legacy
#SBATCH --exclude=gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_oracle_gate_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE GATE AfCycDesign HAS TO CLEAR BEFORE ITS OPINION COUNTS.
#
# Predict each target's NATIVE peptide with the templated ColabDesign AF2 path
# (cyclic offset on, receptor given as a full template, no peptide coordinates)
# and measure receptor-aligned peptide CA RMSD against the deposited pose.
#
# Why this is a gate and not a formality: on six crystal macrocycle complexes
# this project measured Boltz-2 at 1.42 A median and HighFold3 at 0.64-3.39 A,
# and afcycdesign at 34.67 A -- the peptide was not in the pocket at all, and it
# was discarded. A Rosetta interface energy computed on a peptide 34 A away is a
# number about empty space. Re-measuring it on THESE targets turns an inherited
# verdict into a measured one.
OUT="${PLLO_OUT:-${RUN}/lnr/pll_oracle}"
RECYCLES="${PLLO_RECYCLES:-6}"
AF2DIR="${AF2_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/ckpts/AF2}"

# JAX grabs ~90% of VRAM on first allocation by default, which turns a shared
# GPU into an exclusive one and makes an unrelated job OOM.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

echo "out=$OUT recycles=$RECYCLES af2_params=$AF2DIR"
if [ ! -s "$OUT/designs.json" ]; then
  echo "GATE FAILED: no $OUT/designs.json -- run --select first. Exit 0."
  exit 0
fi
if [ ! -s "$AF2DIR/params_model_1_ptm.npz" ]; then
  echo "GATE FAILED: no AF2 params at $AF2DIR. Exit 0."
  exit 0
fi
if ! "$PY" -c "import colabdesign, jax" 2>/dev/null; then
  echo "GATE FAILED: colabdesign/jax missing in $PY. Exit 0."
  exit 0
fi

"$PY" "$AGENT/lnr/pll_oracle.py" --gate-af2 \
  --out-dir "$OUT" --recycles "$RECYCLES" --af2-params-dir "$AF2DIR"

echo "DONE pll_oracle gate -> $OUT/af2_gate.jsonl"
