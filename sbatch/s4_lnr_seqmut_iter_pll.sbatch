#!/usr/bin/env bash
#SBATCH --job-name=s4_iter_pll
# CPU PARTITION, DELIBERATELY. The fitness is a table lookup against a matrix
# built once per target by a CPU forward pass; asking for a GPU to run it would
# only queue this behind real folding work for no gain (CLAUDE.md: split jobs by
# resource). This is the whole economic point of the evaluator.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_seqmut_iter_pll_%j.out
# `set -euo pipefail` lives HERE, in the job body, not only in the preamble: if
# the source below fails, a preamble-only setting would leave this running under
# the system python and still EXIT 0, releasing every afterok dependent behind a
# stage that did no work.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ===========================================================================
# IN-LOOP FITNESS = POSE-CONDITIONED INTERFACE LIKELIHOOD (pll_bg)
#
# The third in-loop fitness, and the first one with evidence that it ranks.
#   proxy  (pose-preserving Rosetta retention) was measured to DISAGREE with the
#          real folded score -- on 1bjr it called the edits big wins while the
#          folded score ranked nomut best and every edit worse.
#   boltz  (real Boltz x4 -> Rosetta dG) is the number we report, but as an
#          in-loop FITNESS it is a draw: per-variant fold sd ~3.7 REU, the same
#          variant moved ~4 REU across runs, and 61% of the apparent gain vs
#          wild-type evaporated on an independent redraw. Each draw costs a GPU
#          fold, so the budget was 24 evaluations per (target, arm).
#   pll    ranks: 13/13 on the leak-free CPSet head-to-tail pairs (permutation
#          p = 0.0059; incumbent dG 8/13, p = 0.36), within-complex median
#          rho +0.34 over 2501 SKEMPI mutations across 82 complexes.
#          See docs/binding_metric_design.md and docs/binding_metric_results.md.
#
# WHAT CHANGES IN THE LOOP, AND WHAT TO READ IN THIS LOG
#   * NOISE IS ZERO, bitwise. Every noise defence the Boltz loop needs
#     (re-measurement, optimistic/pessimistic selection intervals) is switched
#     off automatically because the scorer declares stochastic = False. Expect
#     n_reassess=0, best_sem=0.0, and best_recovered to be MONOTONE -- unlike
#     the Boltz loop, where it can legitimately go up.
#   * THE BUDGET STOPS BINDING. ~30 us per candidate instead of ~2 min, so
#     max_evals here is in the hundreds rather than 24.
#   * THE OPTIMUM IS KNOWN IN CLOSED FORM. The metric is a sum of independent
#     per-position terms, so the exact best <= k_mut substitution set is read off
#     the table. Each arm therefore reports REGRET against a real zero, and
#     `evals_to_optimum`. This is the first time this project can say how good a
#     search actually is rather than only which arm scored better. A run where
#     iter-ea's regret is 0 everywhere means the instrument is saturated and
#     k_mut (or the target set) has to grow before the LLM arm can be graded.
#
# WHAT THIS JOB IS NOT. The fitness is not the believed number. The finalist is
# emitted as a standard seqmut row with recovered_binding_energy = null (nats are
# NOT REU and must not enter that column) and is re-folded by
# `boltz_validate.py --mode mut-held` for the REU number -- which is now a real
# cross-fit: selection and reporting share neither a measurement nor a model.
# ===========================================================================

MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
PLL_METRIC="${SEQITER_PLL_METRIC:-pll_bg}"       # the graded form
PLL_SCOPE="${SEQITER_PLL_SCOPE:-all}"            # all | window | iface
PLL_MODEL="${SEQITER_PLL_MODEL:-ligandmpnn_v_32_020_25}"
PLL_DEVICE="${SEQITER_PLL_DEVICE:-cpu}"
ARMS="${SEQITER_PLL_ARMS:-iter-ea}"              # add iter-llm to grade the LLM arm
TARGETS="${SEQITER_PLL_TARGETS:-}"               # empty = every target in the manifest
LIMIT="${SEQITER_PLL_LIMIT:-0}"
KMUT="${SEQITER_PLL_K:-2}"
# Budget: hundreds, not 24. A fold cost minutes; a lookup costs microseconds, so
# the search is no longer rationed and the arms can be compared where they
# actually differ instead of where they ran out of folds.
NSEED="${SEQITER_PLL_N_SEED:-20}"
NGEN="${SEQITER_PLL_N_GEN:-25}"
MOFF="${SEQITER_PLL_M_OFF:-12}"
MAXEVALS="${SEQITER_PLL_MAX_EVALS:-400}"
ELITE="${SEQITER_PLL_ELITE_FRAC:-0.3}"
MAXSEC="${LNR_MAX_SECONDS:-8}"
SEED_WINDOWS="${SEQITER_PLL_SEED_WINDOWS:-${OUTDIR}/seqmut_rows.jsonl}"
ROWS_FILE="${SEQITER_PLL_ROWS_FILE:-seqmut_iter_pll_rows.jsonl}"
NSHUF="${SEQITER_PLL_N_SHUF:-24}"
# LLM arm (only used if ARMS contains iter-llm)
BACKEND="${SEQITER_PLL_BACKEND:-openai}"
MODEL="${SEQITER_PLL_MODEL_ID:-gpt-6-luna}"
EFFORT="${LNR_EFFORT:-low}"
MAXSPEND="${SEQITER_PLL_MAX_SPEND:-1.0}"
MAXTOK="${LNR_MAX_TOKENS:-6000}"
RETRIES="${SEQITER_PLL_MAX_RETRIES:-1}"
mkdir -p "$OUTDIR" "${OUTDIR}/pssm"

# Torch on CPU will otherwise spawn one thread per visible core on the whole
# node, not per allocated core, and thrash against the other jobs on it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "metric=$PLL_METRIC scope=$PLL_SCOPE model=$PLL_MODEL device=$PLL_DEVICE"
echo "arms=$ARMS targets='${TARGETS:-<all>}' limit=$LIMIT k_mut=$KMUT"
echo "budget: n_seed=$NSEED n_gen=$NGEN m_off=$MOFF max_evals=$MAXEVALS elite=$ELITE"

# --- gates: print why, exit 0 (a non-zero exit strands afterok dependents) ---
if [ ! -f "$MAN" ]; then echo "GATE: no manifest $MAN. Exit 0."; exit 0; fi
if [ ! -d "$PDB" ]; then echo "GATE: no pdb dir $PDB. Exit 0."; exit 0; fi
if [ ! -f "${LIGANDMPNN_HOME}/model_utils.py" ]; then
  echo "GATE: no LigandMPNN at ${LIGANDMPNN_HOME}. Exit 0."; exit 0
fi
if [ ! -f "${LIGANDMPNN_HOME}/model_params/${PLL_MODEL}.pt" ]; then
  echo "GATE: no checkpoint ${PLL_MODEL}.pt. Exit 0."; exit 0
fi
if ! "$PY" -c "import torch" 2>/dev/null; then echo "GATE: no torch in \$PY. Exit 0."; exit 0; fi

# ===========================================================================
# GATE 1 -- THE SELFTEST, against the real checkpoint and the real complexes.
# This runs BEFORE any search because the search's numbers are only meaningful
# if these hold: the table is bitwise reproducible (which is what licenses
# switching the loop's noise defences off), the fitness sign is the one the
# population minimises, the cached table on disk scores identically to the one
# in memory, the native sequence beats its own scrambles on the same pose (so
# the metric is not reading composition), and -- the one that would otherwise
# fail SILENTLY -- the loop's target object and the table agree on which residue
# sits at each index. env.LNRTarget drops residues that are missing a backbone
# atom; LigandMPNN's parser does not. If those disagree by one dropped residue
# at equal length, every lookup reads the wrong row and nothing raises.
# ===========================================================================
echo
echo "=== GATE 1: pll selftest (determinism / sign / alignment / composition) ==="
SELFTEST_TARGETS="${SEQITER_PLL_SELFTEST_TARGETS:-$TARGETS}"
if "$PY" "$AGENT/lnr/pll_inloop.py" --selftest \
    --manifest "$MAN" --pdb-dir "$PDB" \
    --targets "$SELFTEST_TARGETS" \
    --metric "$PLL_METRIC" --scope "$PLL_SCOPE" \
    --model "$PLL_MODEL" --device "$PLL_DEVICE" \
    --pssm-dir "${OUTDIR}/pssm" --k-mut "$KMUT" --n-shuf "$NSHUF" \
    --min-comp-frac "${SEQITER_PLL_MIN_COMP_FRAC:-0.6}" \
    --max-unscorable-frac "${SEQITER_PLL_MAX_UNSCORABLE_FRAC:-0.25}"; then
  echo "GATE 1 PASSED"
  echo "NOTE: individual 'SKIPPED (data, not code)' and 'below chance' targets in"
  echo "the block above are reported, not fatal -- the gate is on the aggregate."
  echo "The loop skips the same unindexable targets via the evaluator's prepare()."
else
  echo
  echo "GATE 1 FAILED -- the search is NOT RUN and no rows are written."
  echo "Read the 'gate failed on:' line above. A HARD FAIL (sign, determinism,"
  echo "disk round-trip, lookup) is a CODE defect and must be fixed before any"
  echo "number from this evaluator is believed. The aggregate gates instead say"
  echo "something about the DATA: too few natives beating their own scrambles"
  echo "means the metric is reading composition on this set, and too many"
  echo "unindexable targets means the manifest or pose directory is wrong."
  echo "Exiting 0 so no dependent job is stranded in DependencyNeverSatisfied."
  exit 0
fi

SEED_ARG=()
[ -f "$SEED_WINDOWS" ] && { echo "reusing committed windows from $SEED_WINDOWS"; SEED_ARG=(--seed-windows "$SEED_WINDOWS"); } \
  || echo "NOTE: no $SEED_WINDOWS -- will commit a fresh window per target."

LLM_ARG=()
case ",$ARMS," in
  *,iter-llm,*|*,iter-llm-ala,*)
    if [ -z "$BACKEND" ]; then echo "GATE: iter-llm arm but no backend. Exit 0."; exit 0; fi
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

TARGET_ARG=()
[ -n "$TARGETS" ] && TARGET_ARG=(--targets "$TARGETS")
LIMIT_ARG=()
[ "$LIMIT" != "0" ] && LIMIT_ARG=(--limit "$LIMIT")

echo
echo "=== seqmut-iter (pose-conditioned likelihood in loop) ==="
# Resumable: a finished (target, arm) is skipped whole, per-target PSSM tables are
# cached under $OUTDIR/pssm, so a time limit costs a resubmission, not a restart.
"$PY" "$AGENT/lnr/seqmut_iter.py" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --out "$OUTDIR/${ROWS_FILE}" \
  --traj "$OUTDIR/${SEQITER_PLL_TRAJ_FILE:-seqmut_iter_pll_traj.jsonl}" \
  --scan-cache "$OUTDIR/seqmut_iter_pll_scancache.jsonl" \
  "${SEED_ARG[@]}" "${LLM_ARG[@]}" "${TARGET_ARG[@]}" "${LIMIT_ARG[@]}" \
  --evaluator pll \
  --pll-metric "$PLL_METRIC" --pll-scope "$PLL_SCOPE" \
  --pll-model "$PLL_MODEL" --pll-device "$PLL_DEVICE" \
  --pll-pssm-dir "$OUTDIR/pssm" \
  --pll-cache "$OUTDIR/seqmut_iter_pll_scores.jsonl" \
  --arms "$ARMS" \
  --k-mut "$KMUT" --n-seed "$NSEED" --n-gen "$NGEN" --m-off "$MOFF" \
  --max-evals "$MAXEVALS" --elite-frac "$ELITE" --max-seconds "$MAXSEC" \
  --n-workers "${SLURM_CPUS_PER_TASK:-8}"

echo "DONE seqmut-iter pll-in-loop"
echo "rows : $OUTDIR/${ROWS_FILE}"
echo "traj : $OUTDIR/${SEQITER_PLL_TRAJ_FILE:-seqmut_iter_pll_traj.jsonl}"
echo "every scored sequence: $OUTDIR/seqmut_iter_pll_scores.jsonl"
