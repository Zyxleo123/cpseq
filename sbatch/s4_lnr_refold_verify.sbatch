#!/usr/bin/env bash
#SBATCH --job-name=s4_refold16
#SBATCH --partition=debug
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_refold_verify_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
nvidia-smi -L || true

# VERIFY the real denoising lever: does raising Boltz diffusion_samples 4 -> 16
# actually halve the oracle SEM, AND are the samples independent (so sqrt(n)
# scaling holds)? Re-folds a few cached variants at --replicas and reports the
# achieved SEM vs the fold_sd/sqrt(n) prediction, plus an independence check.
#
# Pinned to gpu22 (debug, v100, was idle) so it does NOT compete with the search
# on gpu20. debug caps at 2h; 6 variants at 16 samples fit. Resumable per variant.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
CACHE="${REFOLD_CACHE:-${OUTDIR}/seqmut_iter_boltz_cache.jsonl}"
OUT="${REFOLD_OUT:-${OUTDIR}/refold_verify.jsonl}"
WORK="${REFOLD_WORK:-${OUTDIR}/refold_inloop}"
REPLICAS="${REFOLD_REPLICAS:-16}"
SEED="${REFOLD_SEED:-2026}"
TARGETS="${REFOLD_TARGETS:-1bjr,1elr}"
MAXVAR="${REFOLD_MAX_VARIANTS:-3}"
mkdir -p "$OUTDIR" "$WORK"

echo "cache=$CACHE out=$OUT replicas=$REPLICAS seed=$SEED targets=$TARGETS max_var=$MAXVAR"

if [ ! -f "$MAN" ]; then echo "GATE FAILED: no manifest $MAN. Exit 0."; exit 0; fi
if [ ! -f "$CACHE" ]; then echo "GATE FAILED: no cache $CACHE. Exit 0."; exit 0; fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then echo "GATE FAILED: no pyrosetta. Exit 0."; exit 0; fi
if [ ! -d /zfsauton/scratch/yixiz/boltz_cache ]; then echo "GATE FAILED: no boltz cache. Exit 0."; exit 0; fi

echo "=== refold at replicas=$REPLICAS: SEM(16) vs SEM(4) + independence check ==="
"$PY" "$AGENT/lnr/refold_verify.py" \
  --refold --manifest "$MAN" --pdb-dir "$PDB" \
  --cache "$CACHE" --out "$OUT" --work "$WORK" \
  --replicas "$REPLICAS" --seed "$SEED" \
  --targets "$TARGETS" --max-variants "$MAXVAR"

echo "DONE lnr refold-verify"
echo "re-render:  \$PY $AGENT/lnr/refold_verify.py --report $OUT"
