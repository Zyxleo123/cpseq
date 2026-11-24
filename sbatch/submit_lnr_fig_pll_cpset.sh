#!/usr/bin/env bash
# Submit the publication figure for the binding-metric result: pll_bg vs
# measured pKd on CPSet's within-target congeneric series. CPU only, reads
# committed rows, recomputes no score.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: any explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to
# rebuild the login environment on the compute node, and the job is requeued and
# HELD. Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_fig_pll_cpset.sh
#   FIG_PLL_FORMATS=pdf,png,svg sbatch/submit_lnr_fig_pll_cpset.sh
#   FIG_PLL_NPERM=50000 FIG_PLL_STEM=fig1 sbatch/submit_lnr_fig_pll_cpset.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/fig_pll_cpset_${STAMP}.env"
cat > "$ENVF" <<EOT
export FIG_PLL_ROWS="${FIG_PLL_ROWS:-${RUN}/lnr/bindscore_v2.jsonl}"
export FIG_PLL_DECOY="${FIG_PLL_DECOY:-${RUN}/lnr/bindscore_v2_decoy.jsonl}"
export CPSET_DATA_DIR="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
export FIG_PLL_OUTDIR="${FIG_PLL_OUTDIR:-${RUN}/lnr/figs}"
export FIG_PLL_STEM="${FIG_PLL_STEM:-fig_pll_cpset}"
export FIG_PLL_MODEL="${FIG_PLL_MODEL:-ligandmpnn_v_32_020_25}"
export FIG_PLL_NPERM="${FIG_PLL_NPERM:-20000}"
export FIG_PLL_FORMATS="${FIG_PLL_FORMATS:-pdf,png}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrubbed so a submission from inside an allocation does not inherit its job's
# SLURM_* and confuse the new job's own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_fig_pll_cpset.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_fig_pll_cpset.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_fig_pll_cpset_${J}.out"
