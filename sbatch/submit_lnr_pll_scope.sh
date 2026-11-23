#!/usr/bin/env bash
# Submit the cross-scope comparison: does the scope choice (`all` / `window` /
# `iface`) change which sequence you would ship? CPU only, no search, no model
# -- every optimum is read off a cached PSSM table.
#
# Answers the FACTUAL half of the open scope question in docs/seqmut_notes.md.
# It does not pick a scope: that is a judgement about what the objective is.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: any explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to
# rebuild the login environment on the compute node, and the job is requeued and
# HELD, stranding every dependent on Dependency. Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_pll_scope.sh
#   PLLSCOPE_TARGETS=1bjr,1ee5 sbatch/submit_lnr_pll_scope.sh
#   PLLSCOPE_K=3 sbatch/submit_lnr_pll_scope.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pll_scope_${STAMP}.env"
cat > "$ENVF" <<EOT
export LNR_MANIFEST="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
export LNR_PDB_DIR="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
export LNR_OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
export PLLSCOPE_PSSM_DIR="${PLLSCOPE_PSSM_DIR:-${LNR_OUTDIR:-${RUN}/lnr}/pssm}"
export PLLSCOPE_SEED_WINDOWS="${PLLSCOPE_SEED_WINDOWS:-${LNR_OUTDIR:-${RUN}/lnr}/seqmut_iter_pll_rows.jsonl}"
export PLLSCOPE_OUT="${PLLSCOPE_OUT:-${LNR_OUTDIR:-${RUN}/lnr}/pll_scope.json}"
export PLLSCOPE_METRIC="${PLLSCOPE_METRIC:-pll_bg}"
export PLLSCOPE_K="${PLLSCOPE_K:-2}"
export PLLSCOPE_TARGETS="${PLLSCOPE_TARGETS:-}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrubbed so a submission from inside an allocation does not inherit that job's
# SLURM_* and confuse the new job's own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
echo "+ sbatch ${SBATCH_DIR}/s4_lnr_pll_scope.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_pll_scope.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_pll_scope_${J}.out"
