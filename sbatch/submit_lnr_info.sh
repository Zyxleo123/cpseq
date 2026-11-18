#!/usr/bin/env bash
# Submit the INFO-MATCHED LLM arms (llm-info-mask, llm-info-tool) and chain a
# clean-closure re-grade behind them.
#
# The question: the geometry-aware rules (feasible / placement_search / bo) read
# the analytic closure-feasibility bound for free; the plain LLM arms are denied
# it and trail placement_search. These arms show the LLM the SAME bound and
# nothing else different, so llm-info-* vs placement_search tests whether the
# LLM can close the information-asymmetry gap to the strong bar.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export` (an export
# list makes sbatch set SLURM_GET_USER_ENV=1, the compute node's login-env
# rebuild fails, and the job is requeued+held). SLURM_* is scrubbed from this
# submitter's own environment first, because we submit from inside an interactive
# allocation and stale SLURM_* would otherwise leak into the child job record.
set -euo pipefail

SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"
mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# --- matched to the completed full-n run (do not drift) --------------------
#   budget 6, max_seconds 8, effort low, max_tokens 6000: identical to the
#   mask/tool arms already in llm_<model>.jsonl, so info-vs-plain is a clean
#   paired contrast. Caps are per spec (per model) and enforced by the ledger;
#   the plain arms cost terra ~$1.4 / luna ~$0.28, so 3.0 each never binds.
# The env file is SOURCED by the preamble, so every value with spaces (the spec
# list) MUST be quoted: an unquoted `VAR=a b c` line sets VAR=a and tries to run
# `b c`. Values are exported so any child also sees them.
LLM_ENV="${ENVDIR}/lnr_info_llm_${STAMP}.env"
cat > "$LLM_ENV" <<'EOF'
export LNR_MODE="llm"
export LNR_BUDGET="6"
export LNR_MAX_SECONDS="8"
export LNR_EFFORT="low"
export LNR_MAX_TOKENS="6000"
export LNR_LLM_SPECS="gpt-6.1-terra|llm-info-mask,llm-info-tool|3.0 gpt-6-luna|llm-info-mask,llm-info-tool|3.0"
EOF
echo "wrote $LLM_ENV:"; sed 's/^/    /' "$LLM_ENV"

CLASH_ENV="${ENVDIR}/lnr_info_clash_${STAMP}.env"
cat > "$CLASH_ENV" <<'EOF'
export LNR_MODE="clash"
export LNR_MAX_SECONDS="8"
EOF
echo "wrote $CLASH_ENV:"; sed 's/^/    /' "$CLASH_ENV"

# scrub SLURM_* so nothing from this allocation leaks into the submitted jobs
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true

echo "+ sbatch ${SBATCH_DIR}/s4_lnr.sbatch ${LLM_ENV}"
A=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr.sbatch" "${LLM_ENV}")
echo "submitted LLM info-arm job: $A"

echo "+ sbatch --dependency=afterok:${A} ${SBATCH_DIR}/s4_lnr.sbatch ${CLASH_ENV}"
B=$(sbatch --parsable --dependency="afterok:${A}" \
      "${SBATCH_DIR}/s4_lnr.sbatch" "${CLASH_ENV}")
echo "submitted clash re-grade job (afterok:${A}): $B"

echo
echo "jobs: llm=$A  clash=$B"
echo "report after both land:"
echo "  PYTHONPATH=agent:pilot/stage0 \$PY -m lnr.report --dir ${RUN}/lnr               # contact retention"
echo "  PYTHONPATH=agent:pilot/stage0 \$PY -m lnr.report --dir ${RUN}/lnr --metric binding"
