#!/usr/bin/env bash
#SBATCH --job-name=s4_boltz
#SBATCH --partition=debug
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_boltz_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
nvidia-smi -L || true

# Boltz-in-the-loop validation: fold native cyclic peptide sequences, score real
# binding, correlate vs LNR retention ceiling. Driver runs under $PY (has cpseq +
# pyrosetta); it subprocesses the fold to the boltz env, which sees this job's GPU.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
TARGETS="${LNR_BOLTZ_TARGETS:-1d4t,3pkn,5n85}"
MODE="${LNR_BOLTZ_MODE:-held}"
MUT_ARMS="${LNR_BOLTZ_MUT_ARMS:-llm:gpt-6-luna:k2,llm-ala:gpt-6.1-terra:k2}"
MUT_OUTPUT="${LNR_BOLTZ_MUT_OUTPUT:-boltz_mut_held.jsonl}"
# Which seqmut*_rows.jsonl to pull finalists from. DEFAULT (empty) globs the whole
# out-dir -- but run_mut_held indexes by (target, arm_label) and LATER-sorting files
# OVERWRITE earlier ones, so a stale seqmut_iter_ea/llm_rows.jsonl silently clobbers
# a real-score-in-loop finalist of the same label. Point this at a dir holding ONLY
# the intended rows file to make the finalist selection unambiguous.
SEQMUT_DIR="${LNR_BOLTZ_SEQMUT_DIR:-}"
# Fold each variant REPLICAS times (diffusion samples from one trunk pass) and
# score every one; the real dG is the median over the valid folds and the spread
# is the fold noise. >4 engages the boltz_chunkfix shadow inside boltz_validate.
REPLICAS="${LNR_BOLTZ_REPLICAS:-5}"

echo "boltz-validate mode=$MODE targets=$TARGETS replicas=$REPLICAS"
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in \$PY. Exit 0."; exit 0
fi
if [ ! -d /zfsauton/scratch/yixiz/boltz_cache ]; then
  echo "GATE FAILED: no boltz cache. Exit 0."; exit 0
fi

SEQMUT_ARG=()
[ -n "$SEQMUT_DIR" ] && { echo "finalist source dir: $SEQMUT_DIR"; SEQMUT_ARG=(--seqmut-dir "$SEQMUT_DIR"); }
"$PY" "$AGENT/lnr/boltz_validate.py" \
  --manifest "$MAN" --pdb-dir "$PDB" --out-dir "$OUTDIR" --targets "$TARGETS" \
  --mode "$MODE" --mut-arms "$MUT_ARMS" --mut-output "$MUT_OUTPUT" \
  "${SEQMUT_ARG[@]}" --replicas "$REPLICAS"
echo "DONE boltz-validate mode=$MODE"
