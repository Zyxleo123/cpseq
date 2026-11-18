#!/usr/bin/env bash
# Shared configuration for every Stage 2 job. Sourced, never executed.
#
# Configuration arrives as POSITIONAL ARGUMENTS, not via `sbatch --export`.
# Any explicit export list -- including `--export=ALL,K=1` and `--export=NONE`
# -- makes sbatch set SLURM_GET_USER_ENV=1; slurmd then tries to rebuild the
# login environment on the compute node, the lookup fails, and the job is
# requeued and HELD with "(user env retrieval failed requeued held)", which
# strands every afterok dependent on Dependency. Arguments travel through the
# job record untouched, so the submitter writes one timestamped env file per
# submission and passes its path.
#
# An argument that is a readable file is sourced; an argument of the form
# VAR=value is exported.

for _arg in "$@"; do
  if [ -f "$_arg" ]; then
    # shellcheck disable=SC1090
    . "$_arg"
  elif [[ "$_arg" == *=* ]]; then
    export "${_arg?}"
  fi
done

# ---------------------------------------------------------------------------
# Paths. Nothing is written under $HOME.
# ---------------------------------------------------------------------------
export CODE="${CODE:-/zfsauton2/home/yixiz/lp2cp-agent}"
export AGENT="${AGENT:-${CODE}/agent}"
export STAGE0="${STAGE0:-${CODE}/pilot/stage0}"
export PROJ_REPO="${PROJ_REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"

# Three interpreters, because three jobs need three different dependency sets.
#   PY     project venv: numpy/scipy/biotite/torch/transformers. Env build,
#          enumeration, baselines, local 8B, scoring, figures.
#   PY_API dedicated venv with `anthropic` and nothing else. The campaign
#          runner needs no scientific stack on that path (numpy is imported
#          lazily inside diagnostics.measure, which an API job never calls).
#   PY_70B overlay venv with bitsandbytes + accelerate, chained to the project
#          venv so torch/transformers are byte-identical to every other stage.
export PY="${PY:-${PROJ_REPO}/.venv/bin/python}"
export PY_API="${PY_API:-/zfsauton/scratch/yixiz/envs/lp2cp_api/.venv/bin/python}"
export PY_70B="${PY_70B:-/zfsauton/scratch/yixiz/envs/lp2cp_70b/.venv/bin/python}"

export RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
export BENCH_DIR="${BENCH_DIR:-${RUN}/bench}"
export CAMPAIGNS_DIR="${CAMPAIGNS_DIR:-${RUN}/campaigns}"
export RESULTS_DIR="${RESULTS_DIR:-${RUN}/results}"
export ARCHIVE_DIR="${ARCHIVE_DIR:-${RUN}/archive}"
export CPSEA_ROOT="${CPSEA_ROOT:-/zfsauton/scratch/yixiz/CPSea/CPSea_PDB/preprocessed}"

# The stage0 directory is on the path because the API client reuses its
# `pricing` table and `envload` dotenv reader rather than duplicating either --
# a second price table would silently diverge and mis-state cost.
export PYTHONPATH="${AGENT}:${STAGE0}:${PROJ_REPO}/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/zfsauton/scratch/yixiz/hf}"
export TOKENIZERS_PARALLELISM=false
# Python block-buffers when stdout is a file, so a job making steady progress
# shows nothing but the banner and is indistinguishable from one that hung.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Benchmark knobs
# ---------------------------------------------------------------------------
export N_PARENTS="${N_PARENTS:-32}"
export K_BUDGET="${K_BUDGET:-4}"
export MAX_SCAN="${MAX_SCAN:-900}"
export CPSEA_SPLITS="${CPSEA_SPLITS:-test,val}"
export MAX_CELLS="${MAX_CELLS:-150}"

# ---------------------------------------------------------------------------
# Campaign knobs
# ---------------------------------------------------------------------------
export SPLITS="${SPLITS:-dev,test}"
export MAX_ITERS="${MAX_ITERS:-5}"
export PATIENCE="${PATIENCE:-2}"
export SEED_BUDGET="${SEED_BUDGET:-60}"
export DEFAULT_SEEDS="${DEFAULT_SEEDS:-5}"
export MAX_EVAL_SEEDS="${MAX_EVAL_SEEDS:-20}"
export BACKEND_FIDELITY="${BACKEND_FIDELITY:-declared}"
export LIMIT="${LIMIT:-0}"

export RULES="${RULES:-oracle_greedy fixed_pipeline uniform_depth greedy_interface rim_first_closure widest_rebuild random_controller noop}"

export LLM8B_PATH="${LLM8B_PATH:-/zfsauton/scratch/yixiz/models/Llama-3.1-8B-Instruct}"
export LLM70B_PATH="${LLM70B_PATH:-/zfsauton/scratch/yixiz/models/Meta-Llama-3.1-70B-Instruct}"
export LOCAL_ARMS="${LOCAL_ARMS:-A2_CONDITIONING A3_CONTRACT}"
export ARMS_70B="${ARMS_70B:-A3_CONTRACT}"

# API. Spend is capped PER ARM and the ledger is checked before each call, so a
# run stops short of its cap rather than discovering it passed it.
export API_MODELS="${API_MODELS:-claude-opus-5-5 claude-sonnet-5-5}"
export API_ARMS="${API_ARMS:-A2_CONDITIONING A3_CONTRACT}"
export API_EFFORT="${API_EFFORT:-low}"
export MAX_SPEND_PER_ARM="${MAX_SPEND_PER_ARM:-0.60}"
export API_LIMIT="${API_LIMIT:-10}"
export SCAFFOLD_FLAG="${SCAFFOLD_FLAG:-}"

# OpenAI arms (s3_30_openai.sbatch). Each spec is model|mode|cap_usd; modes are
# tool / mask / mask_symdepth. Caps are per spec and enforced by the ledger
# before each call, so the total a submission can spend is the sum of its caps.
export OAI_SPECS="${OAI_SPECS:-gpt-6-luna|tool|0.20}"
export OAI_LIMIT="${OAI_LIMIT:-32}"
export OAI_MAX_TOKENS="${OAI_MAX_TOKENS:-9000}"
# Which lp2cp.tools action space the tool arms and the tool RULES select from.
# v1 is the six tools written before the oracle table was read; v2 adds
# rebuild_cheapest_2span, whose operator and span came from it. Recorded in
# every row's scaffold tag, so the two are never pooled by accident.
export TOOL_MENU="${TOOL_MENU:-v2}"

# Real-scorer validation (s3_40_rosetta_validate.sbatch). The only job here
# that runs a real energy function; it validates the surrogate's response
# model against PyRosetta on the parents' own deposited structures and scores
# no campaign.
export ROSVAL_OUT="${ROSVAL_OUT:-${RUN}/rosetta_validation/parents.jsonl}"
export ROSVAL_RELAX_REPEATS="${ROSVAL_RELAX_REPEATS:-1}"
export ROSVAL_CANDIDATES="${ROSVAL_CANDIDATES:-W,F,L,R}"
export ROSVAL_NPROBE="${ROSVAL_NPROBE:-2}"
export ROSVAL_LIMIT="${ROSVAL_LIMIT:-0}"

banner() {
  # Printed FIRST, before any pipeline that could take SIGPIPE: a job whose
  # first command is `find | sort | head` dies with exit 141 and a zero-byte
  # log, which is indistinguishable from never having started.
  echo "=============================================================="
  echo "job            : ${SLURM_JOB_NAME:-<none>} (${SLURM_JOB_ID:-<none>})"
  echo "host           : $(hostname)"
  echo "date           : $(date -Is)"
  echo "submit dir     : ${SLURM_SUBMIT_DIR:-<none>}"
  echo "partition      : ${SLURM_JOB_PARTITION:-<none>}"
  echo "cpus           : ${SLURM_CPUS_ON_NODE:-?}  mem: ${SLURM_MEM_PER_NODE:-?}M"
  echo "gres           : ${SLURM_JOB_GRES:-<none>}"
  echo "run dir        : ${RUN}"
  echo "splits         : ${SPLITS}   max_iters: ${MAX_ITERS}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L || true
    # VRAM, not just the card name. A model that does not fit is killed by CUDA
    # with no Python traceback, which reads as a mysterious silent FAILED, and
    # with no accounting record on this cluster the card is only recoverable
    # from this log. The pool is mixed -- 2080 Ti 11 GB, v100 16 OR 32 GB,
    # a5000 24 GB, a6000 48 GB, h200 141 GB -- and v100s of both sizes are
    # advertised identically by `myfree`, so the size has to be recorded here.
    echo "--- gpu memory / compute capability ---"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
      --format=csv,noheader 2>/dev/null || true
  else
    echo "gpus           : none visible (CPU job)"
  fi
  echo "=============================================================="
}

require_file() {
  # Gate helper. A failed gate prints why and the CALLER exits 0: a non-zero
  # exit strands every dependent job in DependencyNeverSatisfied with no record
  # of which input was missing.
  if [ ! -s "$1" ]; then
    echo "GATE FAILED: missing or empty required input: $1"
    echo "  ${2:-}"
    return 1
  fi
  return 0
}
