#!/usr/bin/env bash
#SBATCH --job-name=s4_seqmut_iter
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_seqmut_iter_%j.out
set -euo pipefail
# set -euo pipefail in the BODY, not only the preamble: a silently failed source
# would otherwise run under the system python and exit 0, releasing dependents.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ITERATIVE-SEARCH mutation arms on the REAL ring-closure environment
# (agent/lnr/seqmut_iter.py): an LLM evolutionary loop (iter-llm / iter-llm-ala)
# vs a matched random-EA control (iter-ea), both scored by the SAME cheap proxy
# the single-shot seqmut arms use (mutate -> close window -> energy-weighted
# retention x native_dG). The proxy is deterministic, so the loop never regresses
# the frontier and the ONLY honest question is whether the LLM operator beats
# the random operator at a MATCHED number of proxy evaluations (--max-evals).
# The best finalist per arm is
# written as a standard seqmut row, so s4_lnr_boltz.sbatch (mut-held) folds it for
# the true 4-fold dG with no change.
#
# RE-RUN NOTE (2026-11-18): the arms used to draw DIFFERENT random seed pools
# (evolve() keyed its RNG on the arm name), so this job's FIRST run is confounded
# as an operator comparison -- the arms started 7.66 REU apart on average, 19.7 on
# 1jrr, against a headline arm difference of ~1 REU. The seed pool is now shared
# across arms. Rows written before this date are not poolable with new ones.
# The noise knobs (--z-commit / --no-reassess) are no-ops here: the proxy's SEM is
# 0 by construction, so they matter only in the boltz-in-loop job.
#
# CPU only: scipy least_squares closure + PyRosetta mutant scan. Resumable: a
# finished (target,arm) is skipped, and every proxy scan is cached in
# seqmut_iter_scans.jsonl (pre-seeded read-only from the single-shot
# seqmut_scans.jsonl, so work already done is reused). A time limit costs a
# resubmission, not a restart.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
ARMS="${SEQITER_ARMS:-iter-ea}"              # validate the control first; add LLM arms via submitter
CLOSURE_ARM="${SEQITER_CLOSURE_ARM:-placement_search}"
BUDGET="${LNR_BUDGET:-6}"                     # closure campaign budget (window commit)
KMUT="${SEQITER_K:-2}"
NSEED="${SEQITER_N_SEED:-8}"                  # initial pool size
NGEN="${SEQITER_N_GEN:-6}"                    # generations
MOFF="${SEQITER_M_OFF:-6}"                    # offspring per generation
MAXEVALS="${SEQITER_MAX_EVALS:-40}"           # THE matched budget: distinct proxy evals / (target,arm)
ELITE="${SEQITER_ELITE_FRAC:-0.3}"
WORKERS="${SEQITER_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"   # parallel proxy scans/gen (<= cpus-per-task)
MAXSEC="${LNR_MAX_SECONDS:-8}"
RELAX="${LNR_RELAX_REPEATS:-1}"
TIMEOUT="${SEQITER_TIMEOUT:-900}"
LIMIT="${LNR_LIMIT:-8}"                        # a small subset by default: validate before scaling
# reuse the single-shot run's windows AND Rosetta scans so the loop is cheap/comparable
SEED_WT="${SEQITER_SEED_WT:-${OUTDIR}/rosetta_interface.jsonl}"
SEED_WINDOWS="${SEQITER_SEED_WINDOWS:-${OUTDIR}/seqmut_rows.jsonl}"
SEED_SCANS="${SEQITER_SEED_SCANS:-${OUTDIR}/seqmut_scans.jsonl}"
# LLM arm backend: only used when ARMS contains an iter-llm* arm.
BACKEND="${SEQITER_BACKEND:-}"                # openai | api | scripted | (empty: EA only)
MODEL="${SEQITER_MODEL:-gpt-6-luna}"
EFFORT="${LNR_EFFORT:-low}"
MAXSPEND="${SEQITER_MAX_SPEND:-2.0}"          # hard spend cap; nothing is sent over it
MAXTOK="${LNR_MAX_TOKENS:-6000}"
RETRIES="${SEQITER_MAX_RETRIES:-1}"
mkdir -p "$OUTDIR"

echo "arms=$ARMS closure=$CLOSURE_ARM k_mut=$KMUT n_seed=$NSEED n_gen=$NGEN m_off=$MOFF max_evals=$MAXEVALS limit=$LIMIT"

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
if [ -f "$SEED_WINDOWS" ]; then
  echo "reusing committed windows from $SEED_WINDOWS (same window as single-shot)"
  SEED_ARG+=(--seed-windows "$SEED_WINDOWS")
else
  echo "NOTE: no $SEED_WINDOWS -- will commit a fresh window per target."
fi
if [ -f "$SEED_SCANS" ]; then
  echo "reusing prior mutant scans from $SEED_SCANS"
  SEED_ARG+=(--seed-scans "$SEED_SCANS")
fi

# The LLM args are passed only when a backend is set; an LLM arm needs a key.
LLM_ARG=()
case ",$ARMS," in
  *,iter-llm,*|*,iter-llm-ala,*)
    if [ -z "$BACKEND" ]; then
      echo "GATE FAILED: ARMS has an iter-llm arm but SEQITER_BACKEND is unset. Exiting 0."; exit 0
    fi
    if [ "$BACKEND" = "openai" ]; then
      if ! "$PY" -c "
import sys; sys.path.insert(0, '$STAGE0')
from envload import require_api_key
sys.exit(0 if require_api_key(key='OPENAI_API_KEY') else 1)
"; then
        echo "GATE FAILED: no OPENAI_API_KEY for the iter-llm arm. Exiting 0."; exit 0
      fi
    fi
    LLM_ARG=(--backend "$BACKEND" --model "$MODEL" --effort "$EFFORT"
             --max-spend-usd "$MAXSPEND" --max-tokens "$MAXTOK"
             --max-retries "$RETRIES" --save-raw)
    echo "iter-llm arm: backend=$BACKEND model=$MODEL effort=$EFFORT cap=\$$MAXSPEND"
    ;;
esac

echo "=== seqmut-iter: evolutionary loop (LLM vs random-EA) at matched budget ==="
"$PY" "$AGENT/lnr/seqmut_iter.py" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --out "$OUTDIR/${SEQITER_ROWS_FILE:-seqmut_iter_rows.jsonl}" \
  --traj "$OUTDIR/${SEQITER_TRAJ_FILE:-seqmut_iter_traj.jsonl}" \
  --scan-cache "$OUTDIR/${SEQITER_SCAN_FILE:-seqmut_iter_scans.jsonl}" \
  "${SEED_ARG[@]}" "${LLM_ARG[@]}" \
  --arms "$ARMS" --closure-arm "$CLOSURE_ARM" \
  --budget "$BUDGET" --k-mut "$KMUT" \
  --n-seed "$NSEED" --n-gen "$NGEN" --m-off "$MOFF" \
  --max-evals "$MAXEVALS" --elite-frac "$ELITE" --n-workers "$WORKERS" \
  --relax-repeats "$RELAX" --max-seconds "$MAXSEC" \
  --per-target-timeout "$TIMEOUT" --limit "$LIMIT"

echo "DONE lnr seqmut-iter"
