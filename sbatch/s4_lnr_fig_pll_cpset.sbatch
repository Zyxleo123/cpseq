#!/usr/bin/env bash
#SBATCH --job-name=s4_fig_pll
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_fig_pll_cpset_%j.out
# `set -euo pipefail` lives HERE, in the job body, not only in the preamble: if
# the source below fails, a preamble-only setting would leave this running under
# the system python and still EXIT 0.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE PUBLICATION FIGURE for the binding-metric result: pll_bg vs measured pKd
# on CPSet's within-target congeneric series.
#
# CPU partition and no GPU on purpose. This job reads nothing but committed
# scoring rows -- asking for a GPU to draw a figure only queues it behind real
# work, and a figure that needs a GPU is a figure that cannot be redrawn.
#
# Nothing here recomputes a score. The rows were written by s4_lnr_bindscore
# (jobs 119321 / 119338) and every statistic in the figure is computed by
# lnr.metric_gate, the same referee module behind the table in
# docs/binding_metric_results.md -- so the figure cannot drift from the result.
ROWS="${FIG_PLL_ROWS:-${RUN}/lnr/bindscore_v2.jsonl}"
DECOY="${FIG_PLL_DECOY:-${RUN}/lnr/bindscore_v2_decoy.jsonl}"
DATA="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
OUTDIR="${FIG_PLL_OUTDIR:-${RUN}/lnr/figs}"
STEM="${FIG_PLL_STEM:-fig_pll_cpset}"
MODEL="${FIG_PLL_MODEL:-${BINDSCORE_MODELS%%,*}}"
NPERM="${FIG_PLL_NPERM:-20000}"
FORMATS="${FIG_PLL_FORMATS:-pdf,png}"

echo "rows=$ROWS"
echo "decoy=$DECOY"
echo "data=$DATA  out=$OUTDIR/$STEM  model=$MODEL  n_perm=$NPERM  fmt=$FORMATS"

# Gates print why and exit 0: a non-zero exit would strand any dependent job in
# DependencyNeverSatisfied with no record of which input was missing. Each gate
# distinguishes "upstream never ran" from "upstream ran and failed".
if [ ! -s "$ROWS" ]; then
  echo "GATE FAILED: no scoring rows at $ROWS -- s4_lnr_bindscore.sbatch has"
  echo "  not produced them. Nothing to draw. Exit 0."
  exit 0
fi
if [ ! -s "$DATA/dataset1.csv" ]; then
  echo "GATE FAILED: no CPSet ground truth at $DATA/dataset1.csv"
  echo "  fetch it:  \$PY $AGENT/lnr/cpset_accuracy.py --fetch --data-dir $DATA"
  exit 0
fi
if [ ! -s "$DECOY" ]; then
  echo "NOTE: no decoy rows at $DECOY -- panel d will omit the"
  echo "      wrong-receptor specificity control. Everything else still draws."
  DECOY=""
fi
if ! "$PY" -c "import matplotlib, numpy" 2>/dev/null; then
  echo "GATE FAILED: matplotlib/numpy missing in $PY. Exit 0."
  exit 0
fi

mkdir -p "$OUTDIR"
"$PY" "$AGENT/lnr/fig_pll_cpset.py" \
  --rows "$ROWS" ${DECOY:+--decoy "$DECOY"} \
  --data-dir "$DATA" --out-dir "$OUTDIR" --stem "$STEM" \
  --model "$MODEL" --n-perm "$NPERM" --formats "$FORMATS"

echo
echo "DONE fig_pll_cpset"
ls -la "$OUTDIR" | sed 's/^/  /'
echo "redraw (no recompute, ~20 s):"
echo "  \$PY $AGENT/lnr/fig_pll_cpset.py --rows $ROWS --decoy $DECOY \\"
echo "      --data-dir $DATA --out-dir $OUTDIR"
