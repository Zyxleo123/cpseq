#!/usr/bin/env bash
# Shared configuration for every Stage 0 job. Sourced, never executed.
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
# VAR=value is exported. Both forms are accepted so a single knob can be
# overridden on the command line without writing a file.

for _arg in "$@"; do
  if [ -f "$_arg" ]; then
    # shellcheck disable=SC1090
    . "$_arg"
  elif [[ "$_arg" == *=* ]]; then
    export "${_arg?}"
  fi
done

# ---------------------------------------------------------------------------
# Paths. Nothing is written under $HOME: logs, env files, run directories and
# any artefact all live on scratch.
# ---------------------------------------------------------------------------
export CODE="${CODE:-/zfsauton2/home/yixiz/lp2cp-agent}"
export STAGE0="${STAGE0:-${CODE}/pilot/stage0}"
export PROJ_REPO="${PROJ_REPO:-/zfsauton2/home/yixiz/Proteina-Complexa}"
export PY="${PY:-${PROJ_REPO}/.venv/bin/python}"

export RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_pilot}"
export BENCH_DIR="${BENCH_DIR:-${RUN}/bench}"
export FEATURES_DIR="${FEATURES_DIR:-${RUN}/features}"
export CONS_DIR="${CONS_DIR:-${RUN}/features}"
export PDB_CACHE="${PDB_CACHE:-${RUN}/pdb}"
export ARMS_DIR="${ARMS_DIR:-${RUN}/arms}"
export RESULTS_DIR="${RESULTS_DIR:-${RUN}/results}"
export SKEMPI_CSV="${SKEMPI_CSV:-${RUN}/data_raw/skempi_v2.csv}"

# The project venv ships a broken torchaudio that `transformers` imports on the
# way to anything else; the stub shadows it. Same mechanism every other job in
# the sibling project uses.
export PYTHONPATH="${STAGE0}:${PROJ_REPO}/script_utils/_compat_stubs${PYTHONPATH:+:$PYTHONPATH}"

# Keep model downloads and caches off home.
export HF_HOME="${HF_HOME:-/zfsauton/scratch/yixiz/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
# Unbuffered stdout. Python block-buffers when stdout is a file, so a job that
# is making steady progress shows nothing but the banner in its .out and is
# indistinguishable from one that hung on its first import.
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Experiment knobs
# ---------------------------------------------------------------------------
export SLICES="${SLICES:-ala_k1,allsub_k1,allsub_k2}"
export SPLITS="${SPLITS:-dev,test}"
export LLM_MODEL_PATH="${LLM_MODEL_PATH:-/zfsauton/scratch/yixiz/models/Llama-3.1-8B-Instruct}"
export LLM_EVIDENCE_ARMS="${LLM_EVIDENCE_ARMS:-full}"
export LLM_ORDERS="${LLM_ORDERS:-main}"
export LLM_DEVICE_MAP="${LLM_DEVICE_MAP:-}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
export LIMIT="${LIMIT:-0}"
# Which rule/control arms the heuristic stage runs. Overridable so a newly
# added arm can be filled in without re-running the ones already on disk
# (run_arm.py writes one file per arm, so a supplementary run cannot collide
# with an arm that already completed).
export HEUR_ARMS="${HEUR_ARMS:-random first_listed min_dsasa min_contacts rim_first max_entropy min_wt_prob composite max_dsasa most_measured greedy_singles}"

banner() {
  # Printed FIRST, before any pipeline that could take SIGPIPE. A job whose
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
  echo "python         : ${PY}"
  echo "run dir        : ${RUN}"
  echo "slices         : ${SLICES}   splits: ${SPLITS}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L || true
  else
    echo "gpus           : none visible (CPU job)"
  fi
  echo "=============================================================="
}

require_file() {
  # Gate helper. A failed gate prints why and the CALLER exits 0: a non-zero
  # exit strands every dependent job in DependencyNeverSatisfied with no
  # record of which input was missing.
  if [ ! -s "$1" ]; then
    echo "GATE FAILED: missing or empty required input: $1"
    echo "  ${2:-}"
    return 1
  fi
  return 0
}
