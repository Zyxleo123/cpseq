#!/usr/bin/env bash
#SBATCH --job-name=s4_iter_report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_iter_report_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# The BELIEVED-NUMBER report for the iterative-search arms: reads the REAL
# denoised Boltz x4 -> Rosetta dG of the finalists + nomut (mut-held output) and
# N_lin (boltz_linear_ref), prints the N_lin -> ceiling -> achieved decomposition
# and the matched-budget random-EA null. CPU only (reads jsonl, scipy Wilcoxon);
# no fold here -- the fold is the GPU job this chains behind.
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
MUT_HELD="${LNR_ITER_MUT_HELD:-${OUTDIR}/boltz_iter_mut_held.jsonl}"
LINREF="${LNR_LINEAR_REF:-${OUTDIR}/boltz_linear_ref.jsonl}"

# gate: print why and exit 0 (never strand a dependent)
if [ ! -s "$MUT_HELD" ]; then
  echo "GATE: no folded mut-held output at $MUT_HELD -- the Boltz fold produced"
  echo "nothing to report. Exiting 0."; exit 0
fi
LINARG=()
[ -s "$LINREF" ] && LINARG=(--linear-ref "$LINREF") || echo "NOTE: no $LINREF -- N_lin column blank."

echo "=== iter believed-number report ==="
"$PY" "$AGENT/lnr/seqmut_iter_report.py" --mut-held "$MUT_HELD" "${LINARG[@]}" \
  ${LNR_ITER_LLM_ARMS:+--llm-arms "$LNR_ITER_LLM_ARMS"} \
  ${LNR_ITER_RANDOM_ARM:+--random-arm "$LNR_ITER_RANDOM_ARM"}
echo "DONE iter report"
