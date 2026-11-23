#!/usr/bin/env bash
# Submit the retrospective "does interface confidence rank better than dG?" check.
# Reads only folds already on disk: no GPU, no refolding. Safe to run while the
# GPU search is still writing (incomplete folds are skipped until they land).
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`. Run from lov*.
#
# Usage: sbatch/submit_lnr_confidence_rank.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

ENVF="${ENVDIR}/conf_rank_${STAMP}.env"
cat > "$ENVF" <<EOT
export CONF_WORK="${CONF_WORK:-${RUN}/lnr/boltz_inloop}"
export CONF_CACHE="${CONF_CACHE:-${RUN}/lnr/seqmut_iter_boltz_cache.jsonl}"
export CONF_OUT="${CONF_OUT:-${RUN}/lnr/conf_metrics.jsonl}"
export CONF_PEP_CHAIN="${CONF_PEP_CHAIN:-B}"
export CONF_MIN_N="${CONF_MIN_N:-5}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"

unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_confidence_rank.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_confidence_rank.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_confidence_rank_${J}.out"
