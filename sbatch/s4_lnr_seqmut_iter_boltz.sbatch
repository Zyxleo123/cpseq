#!/usr/bin/env bash
#SBATCH --job-name=s4_iter_boltz
# legacy (not debug): debug caps at 2h and this real-score loop runs ~2-3h.
# v100 is the measured fit (Boltz with potentials OOMs an 11 GB 2080 Ti), so the
# node is pinned to a legacy v100. Check myfree and override --nodelist if busy.
#SBATCH --partition=legacy
#SBATCH --nodelist=gpu20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_seqmut_iter_boltz_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
nvidia-smi -L || true

# REAL-SCORE-IN-LOOP: the iterative search selects on the denoised Boltz x4 ->
# Rosetta dG itself (not the pose-preserving proxy, which was shown to disagree
# with the truth). Driver runs under $PY (cpseq + pyrosetta); it subprocesses each
# fold to the boltz env, which sees this job's GPU. The loop folds with --boltz-seed;
# the believed-number report re-folds the finalist with a DIFFERENT seed (an
# independent draw -- the noise firewall). GPU-bound: folds run serially.
#
# Resumable: finished (target,arm) skipped; every fold cached by variant_id in
# seqmut_iter_boltz_cache.jsonl, so a time limit costs a resubmission, not a restart.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
ARMS="${SEQITER_ARMS:-iter-ea,iter-llm}"
TARGETS="${SEQITER_TARGETS:-1bjr,1elr}"      # small validation set by default
KMUT="${SEQITER_K:-2}"
NSEED="${SEQITER_N_SEED:-8}"
NGEN="${SEQITER_N_GEN:-4}"
MOFF="${SEQITER_M_OFF:-4}"
MAXEVALS="${SEQITER_MAX_EVALS:-15}"          # matched budget: distinct Boltz folds / (target,arm)
ELITE="${SEQITER_ELITE_FRAC:-0.3}"
BREPLICAS="${SEQITER_BOLTZ_REPLICAS:-4}"
BSEED="${SEQITER_BOLTZ_SEED:-1234}"          # loop seed (report re-folds with a different seed)
MAXSEC="${LNR_MAX_SECONDS:-8}"
SEED_WINDOWS="${SEQITER_SEED_WINDOWS:-${OUTDIR}/seqmut_rows.jsonl}"
ROWS_FILE="${SEQITER_ROWS_FILE:-seqmut_iter_boltz_rows.jsonl}"
# LLM arm backend (iter-llm needs it)
BACKEND="${SEQITER_BACKEND:-openai}"
MODEL="${SEQITER_MODEL:-gpt-6-luna}"
EFFORT="${LNR_EFFORT:-low}"
MAXSPEND="${SEQITER_MAX_SPEND:-1.0}"
MAXTOK="${LNR_MAX_TOKENS:-6000}"
RETRIES="${SEQITER_MAX_RETRIES:-1}"
mkdir -p "$OUTDIR"

echo "arms=$ARMS targets=$TARGETS k_mut=$KMUT n_seed=$NSEED n_gen=$NGEN m_off=$MOFF max_evals=$MAXEVALS boltz_replicas=$BREPLICAS seed=$BSEED"

# --- gates (print why, exit 0) ---
if [ ! -f "$MAN" ]; then echo "GATE: no manifest $MAN. Exit 0."; exit 0; fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then echo "GATE: no pyrosetta in \$PY. Exit 0."; exit 0; fi
if [ ! -d /zfsauton/scratch/yixiz/boltz_cache ]; then echo "GATE: no boltz cache. Exit 0."; exit 0; fi
SEED_ARG=()
[ -f "$SEED_WINDOWS" ] && { echo "reusing committed windows from $SEED_WINDOWS"; SEED_ARG=(--seed-windows "$SEED_WINDOWS"); } \
  || echo "NOTE: no $SEED_WINDOWS -- will commit a fresh window per target."

LLM_ARG=()
case ",$ARMS," in
  *,iter-llm,*|*,iter-llm-ala,*)
    if [ -z "$BACKEND" ]; then echo "GATE: iter-llm arm but no SEQITER_BACKEND. Exit 0."; exit 0; fi
    if [ "$BACKEND" = "openai" ] && ! "$PY" -c "
import sys; sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key(key='OPENAI_API_KEY') else 1)"; then
      echo "GATE: no OPENAI_API_KEY for iter-llm. Exit 0."; exit 0
    fi
    LLM_ARG=(--backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
             --max-spend-usd "$MAXSPEND" --max-tokens "$MAXTOK"
             --max-retries "$RETRIES" --save-raw)
    echo "iter-llm arm: backend=$BACKEND model=$MODEL cap=\$$MAXSPEND"
    ;;
esac

echo "=== seqmut-iter (REAL Boltz x4 in loop): LLM vs random-EA at matched fold budget ==="
"$PY" "$AGENT/lnr/seqmut_iter.py" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --out "$OUTDIR/${ROWS_FILE}" \
  --traj "$OUTDIR/${SEQITER_TRAJ_FILE:-seqmut_iter_boltz_traj.jsonl}" \
  --scan-cache "$OUTDIR/seqmut_iter_boltz_scancache.jsonl" \
  "${SEED_ARG[@]}" "${LLM_ARG[@]}" \
  --evaluator boltz --boltz-replicas "$BREPLICAS" --boltz-seed "$BSEED" \
  --boltz-work "$OUTDIR/boltz_inloop" \
  --boltz-cache "$OUTDIR/seqmut_iter_boltz_cache.jsonl" \
  --arms "$ARMS" --targets "$TARGETS" \
  --k-mut "$KMUT" --n-seed "$NSEED" --n-gen "$NGEN" --m-off "$MOFF" \
  --max-evals "$MAXEVALS" --elite-frac "$ELITE" --max-seconds "$MAXSEC"

echo "DONE seqmut-iter boltz-in-loop"
