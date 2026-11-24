#!/usr/bin/env bash
# Submit the iterative search with the POSE-CONDITIONED INTERFACE LIKELIHOOD
# (pll_bg) as the in-loop fitness. CPU only, deterministic, and the only in-loop
# fitness this project has whose optimum is known in closed form -- so each arm
# reports REGRET against a real zero rather than just a score.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: any explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to rebuild
# the login environment on the compute node, and the job is requeued and HELD,
# stranding every dependent on Dependency. Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_seqmut_iter_pll.sh
#   SEQITER_PLL_TARGETS=1bjr,1elr sbatch/submit_lnr_seqmut_iter_pll.sh
#   SEQITER_PLL_ARMS=iter-ea,iter-llm sbatch/submit_lnr_seqmut_iter_pll.sh   # spends API $
#   SEQITER_PLL_SCOPE=window SEQITER_PLL_K=3 sbatch/submit_lnr_seqmut_iter_pll.sh
#
# NOTE ON ARMS. The default is iter-ea ALONE: it costs nothing, is fully
# deterministic, and is what establishes the regret baseline the LLM arm has to
# beat. Add iter-llm explicitly when you want to spend API budget on the
# comparison.
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/iter_pll_${STAMP}.env"
cat > "$ENVF" <<EOT
export LNR_MANIFEST="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
export LNR_PDB_DIR="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
export LNR_OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
export LIGANDMPNN_HOME="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"
export SEQITER_PLL_METRIC="${SEQITER_PLL_METRIC:-pll_bg}"
export SEQITER_PLL_SCOPE="${SEQITER_PLL_SCOPE:-all}"
export SEQITER_PLL_MODEL="${SEQITER_PLL_MODEL:-ligandmpnn_v_32_020_25}"
export SEQITER_PLL_DEVICE="${SEQITER_PLL_DEVICE:-cpu}"
export SEQITER_PLL_ARMS="${SEQITER_PLL_ARMS:-iter-ea}"
export SEQITER_PLL_TARGETS="${SEQITER_PLL_TARGETS:-}"
export SEQITER_PLL_SELFTEST_TARGETS="${SEQITER_PLL_SELFTEST_TARGETS:-${SEQITER_PLL_TARGETS:-}}"
export SEQITER_PLL_LIMIT="${SEQITER_PLL_LIMIT:-0}"
export SEQITER_PLL_K="${SEQITER_PLL_K:-2}"
export SEQITER_PLL_N_SEED="${SEQITER_PLL_N_SEED:-20}"
export SEQITER_PLL_N_GEN="${SEQITER_PLL_N_GEN:-25}"
export SEQITER_PLL_M_OFF="${SEQITER_PLL_M_OFF:-12}"
export SEQITER_PLL_MAX_EVALS="${SEQITER_PLL_MAX_EVALS:-400}"
export SEQITER_PLL_ELITE_FRAC="${SEQITER_PLL_ELITE_FRAC:-0.3}"
export SEQITER_PLL_N_SHUF="${SEQITER_PLL_N_SHUF:-24}"
export SEQITER_PLL_ROWS_FILE="${SEQITER_PLL_ROWS_FILE:-seqmut_iter_pll_rows.jsonl}"
export SEQITER_PLL_TRAJ_FILE="${SEQITER_PLL_TRAJ_FILE:-seqmut_iter_pll_traj.jsonl}"
export SEQITER_PLL_SEED_WINDOWS="${SEQITER_PLL_SEED_WINDOWS:-${LNR_OUTDIR:-${RUN}/lnr}/seqmut_rows.jsonl}"
export SEQITER_PLL_BACKEND="${SEQITER_PLL_BACKEND:-openai}"
export SEQITER_PLL_MODEL_ID="${SEQITER_PLL_MODEL_ID:-gpt-6-luna}"
export SEQITER_PLL_MAX_SPEND="${SEQITER_PLL_MAX_SPEND:-1.0}"
export LNR_EFFORT="${LNR_EFFORT:-low}"
export LNR_MAX_TOKENS="${LNR_MAX_TOKENS:-6000}"
export LNR_MAX_SECONDS="${LNR_MAX_SECONDS:-8}"
# The iter-llm-hi arm. These have to be written into the env file like every
# other setting: the batch script reads its configuration from the file passed
# as a positional argument, so a variable exported only in the submitter's own
# shell silently keeps the job's default (and a hi arm that quietly runs at
# n_gen=NGEN is matched on generations, not dollars -- the exact confound the
# arm exists to avoid).
export SEQITER_PLL_HI_EFFORT="${SEQITER_PLL_HI_EFFORT:-high}"
export SEQITER_PLL_HI_N_GEN="${SEQITER_PLL_HI_N_GEN:-0}"
export SEQITER_PLL_PROBE="${SEQITER_PLL_PROBE:-0}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrubbed so a submission from inside an allocation does not inherit that job's
# SLURM_* and confuse the new job's own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true
# The LLM arm's wall time is set by the API, not by the CPU: one call per
# generation at n_gen=80 is ~5 min/target MEASURED (job 119761), so a 29-target
# run does not fit the script's 4 h default. Overridden on the COMMAND LINE,
# which a #SBATCH directive cannot read, and which is safe here -- the
# `--export` hazard in this project is specific to export lists, not to
# resource flags. The run is resumable either way, so a time limit costs a
# resubmission rather than the run.
TIME_ARG=()
[ -n "${SEQITER_PLL_TIME:-}" ] && TIME_ARG=(--time "${SEQITER_PLL_TIME}")
echo "+ sbatch ${TIME_ARG[*]} ${SBATCH_DIR}/s4_lnr_seqmut_iter_pll.sbatch ${ENVF}"
J=$(sbatch --parsable "${TIME_ARG[@]}" "${SBATCH_DIR}/s4_lnr_seqmut_iter_pll.sbatch" "${ENVF}")
echo "submitted: $J"
echo "watch: tail -f ${RUN}/logs/s4_lnr_seqmut_iter_pll_${J}.out"
