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
"$PY" "$AGENT/lnr/pll_regret.py" --rows "$ROWS" --out-json "$OUT_JSON" \
  --n-perm "$NPERM" --thresh "$THRESH" --max-evals "$MAXEVALS" "${ARM_ARG[@]}"

echo "DONE pll regret"
echo "summary: $OUT_JSON"
