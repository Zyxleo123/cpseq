#!/usr/bin/env bash
# Submit the CPU re-relax denoiser (lever #1). Re-relaxes cached Boltz structures
# R times to shrink the oracle's per-variant noise; no GPU. Reads folds read-only
# and writes its own refined PDBs, so it is safe to run while the GPU search runs.
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`. Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_denoise.sh                       # 10 var/target, R=6, all targets
#   DENOISE_TARGETS=1bjr,1elr DENOISE_REPEATS=8 sbatch/submit_lnr_denoise.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

ENVF="${ENVDIR}/denoise_${STAMP}.env"
cat > "$ENVF" <<EOT
export DENOISE_WORK="${DENOISE_WORK:-${RUN}/lnr/boltz_inloop}"
export DENOISE_CACHE="${DENOISE_CACHE:-${RUN}/lnr/seqmut_iter_boltz_cache.jsonl}"
export DENOISE_OUT="${DENOISE_OUT:-${RUN}/lnr/denoise_scores.jsonl}"
export DENOISE_REFINED="${DENOISE_REFINED:-${RUN}/scratch/denoise_refined}"
export DENOISE_REPEATS="${DENOISE_REPEATS:-6}"
export DENOISE_MAX_VARIANTS="${DENOISE_MAX_VARIANTS:-10}"
export DENOISE_TARGETS="${DENOISE_TARGETS:-}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"

unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_denoise.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_denoise.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_denoise_${J}.out"
