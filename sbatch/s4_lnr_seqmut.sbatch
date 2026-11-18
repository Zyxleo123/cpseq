#!/usr/bin/env bash
#SBATCH --job-name=s4_seqmut
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_seqmut_%j.out
set -euo pipefail
# set -euo pipefail in the BODY, not only the preamble: a silently failed source
# would otherwise run under the system python and exit 0, releasing dependents.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# The SEQUENCE-MUTATION lever on the REAL ring-closure environment (agent/lnr/
# seqmut.py). For each target it commits a clean closing window with the strong
# deterministic closure controller (closure is backbone-only, so it is the same
# with or without a sequence edit), then for each mutation arm re-scans the
# MUTANT bound pose in Rosetta (clean -> substitute+repack -> relax -> alanine
# scan) and scores binding as energy-weighted retention x native_dG -- the
# project's binding metric, on the correct (mutant) complex, never the WT table.
#
# CPU only: scipy least_squares for closure + PyRosetta for the mutant scan.
# Resumable: run.py-style skip of (target,arm) already in the rows file, and the
# mutant-scan cache (seqmut_scans.jsonl) skips any (target,subs) already scored,
# so a time limit costs a resubmission, not a restart.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
MUT_ARMS="${SEQMUT_ARMS:-nomut,random_mut}"
CLOSURE_ARM="${SEQMUT_CLOSURE_ARM:-placement_search}"
BUDGET="${LNR_BUDGET:-6}"
KMUT="${SEQMUT_K:-2}"
MAXSEC="${LNR_MAX_SECONDS:-8}"
RELAX="${LNR_RELAX_REPEATS:-1}"
TIMEOUT="${SEQMUT_TIMEOUT:-900}"
LIMIT="${LNR_LIMIT:-8}"          # a small subset by default: validate the scorer first
SEED_WT="${SEQMUT_SEED_WT:-${OUTDIR}/rosetta_interface.jsonl}"
SEED_SCANS="${SEQMUT_SEED_SCANS:-}"      # comma-sep prior scan caches to reuse read-only
# LLM mutation arm (fork B): only used when SEQMUT_ARMS contains an llm arm.
BACKEND="${SEQMUT_BACKEND:-}"            # openai | api | scripted | (empty: det. only)
MODEL="${SEQMUT_MODEL:-gpt-6-luna}"
EFFORT="${LNR_EFFORT:-low}"
MAXSPEND="${SEQMUT_MAX_SPEND:-1.0}"      # hard spend cap; nothing is sent over it
MAXTOK="${LNR_MAX_TOKENS:-6000}"
RETRIES="${SEQMUT_MAX_RETRIES:-1}"
mkdir -p "$OUTDIR"

echo "manifest=$MAN mut_arms=$MUT_ARMS closure=$CLOSURE_ARM k_mut=$KMUT budget=$BUDGET limit=$LIMIT"

# --- gates: print why and exit 0, never a non-zero that strands dependents ----
if [ ! -f "$MAN" ]; then
  echo "GATE FAILED: no LNR manifest at $MAN. Exiting 0."; exit 0
fi
if ! "$PY" -c "import scipy, numpy" 2>/dev/null; then
  echo "GATE FAILED: no scipy/numpy in \$PY ($PY). Exiting 0."; exit 0
fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in \$PY ($PY) -- the mutant re-scan needs it. Exiting 0."; exit 0
fi

SEED_ARG=()
if [ -f "$SEED_WT" ]; then
  echo "seeding WT scans from $SEED_WT (makes nomut free)"
  SEED_ARG=(--seed-wt "$SEED_WT")
else
  echo "NOTE: no WT scan at $SEED_WT -- nomut will re-scan the WT pose itself."
fi
if [ -n "$SEED_SCANS" ]; then
  echo "reusing prior mutant scans from $SEED_SCANS"
  SEED_ARG+=(--seed-scans "$SEED_SCANS")
fi

# The LLM args are passed only when a backend is set; an LLM arm needs a key.
LLM_ARG=()
case ",$MUT_ARMS," in
  *,llm,*|*,llm-ala,*)
    if [ -z "$BACKEND" ]; then
      echo "GATE FAILED: SEQMUT_ARMS has an llm arm but SEQMUT_BACKEND is unset. Exiting 0."; exit 0
    fi
    if [ "$BACKEND" = "openai" ]; then
      if ! "$PY" -c "
import sys; sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key(key='OPENAI_API_KEY') else 1)
"; then
        echo "GATE FAILED: no OPENAI_API_KEY for the llm mutation arm. Exiting 0."; exit 0
      fi
    fi
    LLM_ARG=(--backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
             --max-spend-usd "$MAXSPEND" --max-tokens "$MAXTOK"
             --max-retries "$RETRIES" --save-raw)
    echo "llm arm: backend=$BACKEND model=$MODEL effort=$EFFORT cap=\$$MAXSPEND"
    ;;
esac

echo "=== seqmut: mutate -> close -> binding (retention x native_dG) ==="
"$PY" "$AGENT/lnr/seqmut.py" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --out "$OUTDIR/${SEQMUT_ROWS_FILE:-seqmut_rows.jsonl}" \
  --scan-cache "$OUTDIR/${SEQMUT_SCAN_FILE:-seqmut_scans.jsonl}" \
  "${SEED_ARG[@]}" "${LLM_ARG[@]}" \
  --mut-arms "$MUT_ARMS" --closure-arm "$CLOSURE_ARM" \
  --budget "$BUDGET" --k-mut "$KMUT" --relax-repeats "$RELAX" \
  --max-seconds "$MAXSEC" --per-target-timeout "$TIMEOUT" --limit "$LIMIT"

echo "DONE lnr seqmut"
