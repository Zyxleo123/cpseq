#!/usr/bin/env bash
# Submit the cheap-sequence-ranker vs Boltz-oracle comparison. CPU only; reads the
# existing rank_proxy_vs_true ground truth; weights already cached. Run from lov*.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`.
# Usage: sbatch/submit_lnr_seq_rankers.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

ENVF="${ENVDIR}/seq_rank_${STAMP}.env"
cat > "$ENVF" <<EOT
export SEQRANK_GT="${SEQRANK_GT:-${RUN}/lnr/rank_proxy_vs_true.jsonl}"
export SEQRANK_OUT="${SEQRANK_OUT:-${RUN}/lnr/seq_rankers.jsonl}"
export SEQRANK_MODELS="${SEQRANK_MODELS:-esm2,pepmlm}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"

unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_seq_rankers.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_seq_rankers.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_seq_rankers_${J}.out"
