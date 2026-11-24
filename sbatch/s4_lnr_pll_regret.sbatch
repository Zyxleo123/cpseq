#!/usr/bin/env bash
#SBATCH --job-name=s4_pll_regret
# Reads SAVED ROWS ONLY -- no model, no GPU, no structure prediction. Every
# number it prints is redrawable from the jsonl it reads, which is why it is a
# separate job from the search that produced them (CLAUDE.md: split by resource,
# and figures/tables must be redrawable without recomputation).
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_regret_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# REGRET: how much of the REACHABLE improvement each search arm actually
# captured. Only possible because the pll fitness has a known optimum.
#
# The headline is the NORMALISED regret, not the raw one. `pll_bg` is a mean over
# peptide positions, so a fixed-k edit is diluted by 1/L and the available gain
# shrinks with peptide length (measured rho -0.688 on 79 targets). That confound
# SUPPRESSES the raw regret-vs-length correlation, so reading the raw number
# alone understates the effect -- the same class of mistake as this project's
# "interface dG measures size, not affinity" result. The job prints the confound
# next to every correlation and gives each one a permutation null.
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
ROWS="${PLLREGRET_ROWS:-${OUTDIR}/seqmut_iter_pll_rows.jsonl}"
OUT_JSON="${PLLREGRET_OUT:-${OUTDIR}/pll_regret_summary.json}"
NPERM="${PLLREGRET_N_PERM:-20000}"
THRESH="${PLLREGRET_THRESH:-0.05}"
MAXEVALS="${PLLREGRET_MAX_EVALS:-0}"
ARM="${PLLREGRET_ARM:-}"

echo "rows=$ROWS n_perm=$NPERM thresh=$THRESH max_evals=$MAXEVALS arm='${ARM:-<all>}'"
if ! require_file "$ROWS" "run sbatch/submit_lnr_seqmut_iter_pll.sh first"; then
  echo "Exit 0 so no dependent job is stranded."; exit 0
fi

ARM_ARG=()
[ -n "$ARM" ] && ARM_ARG=(--arm "$ARM")

# ---------------------------------------------------------------------------
# GATE: is the iter-llm arm actually an LLM arm?
# ---------------------------------------------------------------------------
# `llm_offspring` falls back to a random-EA child on decline/malform/error/spend
# cap, so an underfunded iter-llm row is a random-EA row wearing an LLM label.
# That failure converges on the EA control, so it LOOKS like the real finding
# ("the objective is separable, the EA saturates it") rather than like a broken
# run. The cap is also per-RUN, not per-target -- one ledger is built before the
# target loop -- so it is spent in target order and poisons the tail, not a
# random subset.
#
# Gated here rather than in the submitter (CLAUDE.md: decision gates belong
# inside the jobs) and exits 0 on failure so no afterok dependent is stranded in
# DependencyNeverSatisfied with no explanation.
# EVERY llm arm present is gated, not just `iter-llm`. The old test was
# `grep -q '"iter-llm"'`, which does not match `"iter-llm-hi"` -- the closing
# quote differs -- so a run of the hi-effort arm skipped the funding gate
# altogether and would have been graded unvalidated. The arms are discovered
# from the rows so a new arm cannot be forgotten here again.
LLM_ARMS_PRESENT=$("$PY" - "$ROWS" <<'PYEOF'
import json, sys
arms = []
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        a = json.loads(line).get("mut_arm", "")
    except json.JSONDecodeError:
        continue
    if a.startswith("iter-llm") and a not in arms:
        arms.append(a)
print(" ".join(sorted(arms)))
PYEOF
)
echo "LLM arms present in rows: '${LLM_ARMS_PRESENT:-<none>}'"
GATE_FAILED=""
for LARM in $LLM_ARMS_PRESENT; do
  VAL_JSON="${OUT_JSON%.json}_${LARM}_validity.json"
  echo
  echo "--- funding gate: $LARM ---"
  set +e
  "$PY" "$AGENT/lnr/pll_llm_validity.py" --rows "$ROWS" --arm "$LARM" \
    --cap "${PLLREGRET_LLM_CAP:-0}" --n-expected "${PLLREGRET_N_EXPECTED:-0}" \
    --out-json "$VAL_JSON"
  VRC=$?
  set -e
  [ "$VRC" = "2" ] && GATE_FAILED="$GATE_FAILED $LARM"
done
if [ -n "$GATE_FAILED" ] && [ "${PLLREGRET_FORCE:-0}" != "1" ]; then
  echo
  echo "GATE FAILED: these LLM arms are not fully funded:$GATE_FAILED"
  echo "Refusing to print an LLM-vs-EA regret comparison that would read as a"
  echo "tie for a reason that has nothing to do with the operator."
  echo "Raise --max-spend-usd to the projected figure above, delete the"
  echo "affected rows from $ROWS (the run is resumable and SKIPS"
  echo "rows already present), and resubmit the search."
  echo "NOTE: with two effort levels the cap is SPLIT per ledger, so each arm"
  echo "gets half of SEQITER_PLL_MAX_SPEND -- raise it accordingly."
  echo "To grade anyway for inspection: PLLREGRET_FORCE=1"
  echo "Exiting 0 so no dependent job is stranded."
  exit 0
fi
[ -n "$GATE_FAILED" ] && echo "WARNING: PLLREGRET_FORCE=1 -- grading a run the gate rejected."

"$PY" "$AGENT/lnr/pll_regret.py" --rows "$ROWS" --out-json "$OUT_JSON" \
  --n-perm "$NPERM" --thresh "$THRESH" --max-evals "$MAXEVALS" "${ARM_ARG[@]}" \
  ${PLLREGRET_EXCLUDE:+--exclude "$PLLREGRET_EXCLUDE"} \
  ${PLLREGRET_EXCLUDE_ARTIFACTS:+--exclude-artifacts}

echo "DONE pll regret"
echo "summary: $OUT_JSON"
