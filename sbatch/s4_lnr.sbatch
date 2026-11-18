#!/usr/bin/env bash
#SBATCH --job-name=s4_lnr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_%j.out
set -euo pipefail
# set -euo pipefail in the body, not only the preamble: a silently failed
# source would otherwise run under the system python and exit 0.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# The REAL environment. No surrogate, no response model: closure is solved in
# torsion space and interface retention is a coordinate quantity, both from the
# sibling project's calibrated solver. CPU only -- scipy least_squares on a
# short peptide backbone, no GPU, and an OpenAI key only when LLM arms are run.
#
# LNR_MODE selects what this job does:
#   rules     the deterministic controls + frontier ceiling (no key, no spend)
#   frontier  just the ceiling
#   llm       the LLM mask/tool arms (needs OPENAI_API_KEY)
LNR_MODE="${LNR_MODE:-rules}"
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
BUDGET="${LNR_BUDGET:-6}"
MAXSEC="${LNR_MAX_SECONDS:-8}"
LIMIT="${LNR_LIMIT:-0}"
mkdir -p "$OUTDIR"

echo "mode=$LNR_MODE manifest=$MAN budget=$BUDGET max_seconds=$MAXSEC limit=$LIMIT"
if [ ! -f "$MAN" ]; then
  echo "GATE FAILED: no LNR manifest at $MAN. Exiting 0."; exit 0
fi
if ! "$PY" -c "import scipy, numpy" 2>/dev/null; then
  echo "GATE FAILED: no scipy/numpy in $PY. Exiting 0."; exit 0
fi

if [ "$LNR_MODE" = "frontier" ] || { [ "$LNR_MODE" = "rules" ] && [ "${LNR_SKIP_FRONTIER:-0}" != "1" ]; }; then
  echo "=== frontier (achievable-retention ceiling) ==="
  "$PY" "$AGENT/lnr/frontier.py" \
    --manifest "$MAN" --pdb-dir "$PDB" \
    --out "$OUTDIR/frontier.jsonl" \
    --confirm-top "${LNR_CONFIRM_TOP:-16}" --max-seconds "$MAXSEC" --limit "$LIMIT"
fi

if [ "$LNR_MODE" = "rules" ]; then
  echo "=== deterministic rule arms ==="
  "$PY" "$AGENT/lnr/run.py" \
    --manifest "$MAN" --pdb-dir "$PDB" \
    --out "$OUTDIR/${LNR_RULES_FILE:-rules.jsonl}" \
    --arms "${LNR_RULE_ARMS:-contact_core,termini,feasible,router,placement_search,greedy_frontier,random,bo}" \
    --budget "$BUDGET" --max-seconds "$MAXSEC" --limit "$LIMIT"
fi

if [ "$LNR_MODE" = "llm" ]; then
  if ! "$PY" -c "
import sys; sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key(key='OPENAI_API_KEY') else 1)
"; then
    echo "GATE FAILED: no OPENAI_API_KEY. Exiting 0."; exit 0
  fi
  for SPEC in ${LNR_LLM_SPECS}; do
    MODEL="${SPEC%%|*}"; REST="${SPEC#*|}"; ARMS="${REST%%|*}"; CAP="${REST##*|}"
    echo "=== LLM $MODEL arms=$ARMS cap=\$$CAP ==="
    "$PY" "$AGENT/lnr/run.py" \
      --manifest "$MAN" --pdb-dir "$PDB" \
      --out "$OUTDIR/llm_${MODEL}.jsonl" \
      --arms "$ARMS" --budget "$BUDGET" --max-seconds "$MAXSEC" --limit "$LIMIT" \
      --backend openai --model "$MODEL" --effort "${LNR_EFFORT:-low}" \
      --max-spend-usd "$CAP" --max-tokens "${LNR_MAX_TOKENS:-6000}" --save-raw
  done
fi
if [ "$LNR_MODE" = "rosif" ]; then
  echo "=== Rosetta interface energy + per-residue binding share ==="
  if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
    echo "GATE FAILED: no pyrosetta in $PY. Exiting 0."; exit 0
  fi
  "$PY" "$AGENT/lnr/rosetta_interface.py" \
    --manifest "$MAN" --pdb-dir "$PDB" \
    --out "$OUTDIR/rosetta_interface.jsonl" \
    --relax-repeats "${LNR_RELAX_REPEATS:-1}" --limit "$LIMIT"
fi
if [ "$LNR_MODE" = "clash" ]; then
  echo "=== clash re-grade (clean closure) ==="
  "$PY" "$AGENT/lnr/clashcheck.py" \
    --dir "$OUTDIR" --manifest "$MAN" --pdb-dir "$PDB" \
    --out "$OUTDIR/clashes.jsonl" --max-seconds "$MAXSEC"
fi
if [ "$LNR_MODE" = "clash_fine" ] || [ "$LNR_MODE" = "clashfine" ]; then
  echo "=== backbone-triplet clash re-grade (finer clean closure) ==="
  # 3x finer than the CA gate: N/CA/C on both sides, receptor at backbone
  # resolution, 2.5 A cutoff calibrated so every deposited native stays
  # clash-free. Writes clashes_bb.jsonl (never touches clashes.jsonl), and is
  # resumable, so a re-run after 111090 lands the luna-info windows just adds
  # them. Emits both clash_free (CA) and clash_free_bb per window.
  # (Accepts both LNR_MODE spellings: two sessions named it clash_fine/clashfine.)
  "$PY" "$AGENT/lnr/clashcheck_fine.py" \
    --dir "$OUTDIR" --manifest "$MAN" --pdb-dir "$PDB" \
    --out "$OUTDIR/clashes_bb.jsonl" --max-seconds "$MAXSEC" \
    --self-cut "${LNR_SELF_CUT:-2.5}" --rec-cut "${LNR_REC_CUT:-2.5}"
fi
echo "DONE lnr ($LNR_MODE)"
