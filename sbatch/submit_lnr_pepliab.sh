#!/usr/bin/env bash
# Submit the liability audit of the EXACT pll optimum. Reads cached PSSMs only;
# CPU, seconds. Config travels as a POSITIONAL ENV FILE, never `sbatch --export`
# (an explicit export list makes sbatch set SLURM_GET_USER_ENV=1 and the job is
# requeued and HELD). Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_pepliab.sh
#   PEPLIAB_K=6 sbatch/submit_lnr_pepliab.sh          # k_mut=2 barely moves
#   PEPLIAB_SCOPE=iface sbatch/submit_lnr_pepliab.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pepliab_${STAMP}.env"
OUTDIR_D="${LNR_OUTDIR:-${RUN}/lnr}"
cat > "$ENVF" <<EOT
export LNR_OUTDIR="${OUTDIR_D}"
# The PSSM cache the pll search already built. Reusing it is why this audit is
# free: nothing is refolded and no model forward pass runs.
export PEPLIAB_PSSM_DIR="${PEPLIAB_PSSM_DIR:-${OUTDIR_D}/pssm}"
# Needed only for SSBOND records -- the per-target disulfide topology that the
# needs_ctx half of the measurement depends on.
export PEPLIAB_PDB_DIR="${PEPLIAB_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
# k_mut matches the graded search budget by default. k_mut=2 is known to be too
# tight to move the metric much, so the free optimum in the same output is the
# more informative column -- raise this to see the damage at a realistic budget.
export PEPLIAB_K="${PEPLIAB_K:-2}"
export PEPLIAB_SCOPE="${PEPLIAB_SCOPE:-all}"
export PEPLIAB_METRIC="${PEPLIAB_METRIC:-}"
# Two outputs, one per spec arm. The difference between them IS the result;
# neither file alone answers whether per-target context buys anything.
export PEPLIAB_OUT_CTX="${PEPLIAB_OUT_CTX:-${OUTDIR_D}/pepliab_audit_ctx.jsonl}"
export PEPLIAB_OUT_UNI="${PEPLIAB_OUT_UNI:-${OUTDIR_D}/pepliab_audit_universal.jsonl}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_pepliab.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_pepliab.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_pepliab_${J}.out"
