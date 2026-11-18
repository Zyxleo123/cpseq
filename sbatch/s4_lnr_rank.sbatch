#!/usr/bin/env bash
#SBATCH --job-name=s4_rank
#SBATCH --partition=debug
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_rank_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
nvidia-smi -L || true

# Within-complex RANKING: does the pose-preserving proxy rank mutations like the
# real 4-fold Boltz->Rosetta score? Per complex: WT + random single mutations +
# the LLM pick, each scored by PROXY (seqmut, CPU) and by TRUE (Boltz
# diffusion_samples=4 -> replica median, GPU+CPU). Resumable (skips scored
# variants), so a time limit costs a resubmission.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
TARGETS="${RANK_TARGETS:-1elr,1ee5,1ssc}"
NRAND="${RANK_NRANDOM:-7}"

echo "rank targets=$TARGETS n_random=$NRAND"
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in \$PY. Exit 0."; exit 0
fi
if [ ! -d /zfsauton/scratch/yixiz/boltz_cache ]; then
  echo "GATE FAILED: no boltz cache. Exit 0."; exit 0
fi

"$PY" "$AGENT/lnr/rank_proxy_vs_true.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --out-dir "$OUTDIR" \
  --targets "$TARGETS" --n-random "$NRAND" --k-mut "${SEQMUT_K:-2}" \
  --relax-repeats "${LNR_RELAX_REPEATS:-1}"
echo "DONE rank"
