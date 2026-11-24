#!/usr/bin/env bash
# Submit the separability audit: does each liability objective actually admit an
# exact table method, measured rather than hand-labelled?
#
# CPU, seconds of work, no dependencies. Config travels as a POSITIONAL ENV FILE
# and never via `sbatch --export` -- an explicit export list makes sbatch set
# SLURM_GET_USER_ENV=1, slurmd fails to rebuild the login environment on the
# compute node, and the job is requeued and HELD (measured, jobs 22184/22243).
#
# Usage:
#   sbatch/submit_lnr_separability.sh
#   SEPARAB_W_MAX=6 sbatch/submit_lnr_separability.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/separab_${STAMP}.env"
OUTDIR_D="${LNR_OUTDIR:-${RUN}/lnr}"
cat > "$ENVF" <<EOT
export LNR_OUTDIR="${OUTDIR_D}"
export SEPARAB_ROWS="${SEPARAB_ROWS:-${OUTDIR_D}/pepliab_audit.jsonl}"
export SEPARAB_OUT="${SEPARAB_OUT:-${OUTDIR_D}/separability.jsonl}"
# 0 = exhaustive over the 19x19 residue pairs. Do NOT raise this off zero to
# save time: sampling is what made the probe call the D-P motif additive.
export SEPARAB_N_SUBS="${SEPARAB_N_SUBS:-0}"
export SEPARAB_W_MAX="${SEPARAB_W_MAX:-4}"
export SEPARAB_LIMIT="${SEPARAB_LIMIT:-12}"
export SEPARAB_SEED="${SEPARAB_SEED:-0}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrub SLURM_* so submitting from inside an allocation does not leak the
# parent job's environment into the child.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_separability.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_separability.sbatch" "$ENVF")
echo "  job: ${J}"
echo
echo "watch: tail -f ${RUN}/logs/s4_lnr_separab_${J}.out"
echo "rows : ${OUTDIR_D}/separability.jsonl"
