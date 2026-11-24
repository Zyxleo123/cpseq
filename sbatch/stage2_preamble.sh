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

# ---------------------------------------------------------------------------
# CPSet cyclic-peptide accuracy gate (s4_lnr_cpset_accuracy.sbatch)
# ---------------------------------------------------------------------------
# Ground truth: CPSet Dataset I (Zhao et al. Brief Bioinform 2025,
# doi 10.1093/bib/bbaf632; data github.com/huifengzhao/MM_PBGBSA-CP) -- 50
# protein-cyclic-peptide crystal complexes with experimental Kd. 38 of them form
# 10 congeneric series (same receptor, peptide variants), which is the
# WITHIN-TARGET instrument the project has lacked; the paper's own validation is
# cross-complex only, so its Rp numbers calibrate nothing here but the
# cross-complex row.
export CPSET_DATA_DIR="${CPSET_DATA_DIR:-${RUN}/cpset_raw}"
export CPSET_OUT="${CPSET_OUT:-${RUN}/lnr/cpset_dg.jsonl}"
export CPSET_WORK_DIR="${CPSET_WORK_DIR:-${RUN}/cpset_prepared}"
export CPSET_REPLICAS="${CPSET_REPLICAS:-3}"
export CPSET_TIMEOUT="${CPSET_TIMEOUT:-1800}"
# Only `rosetta` exists today; MM/PBSA per the paper's tuned protocol
# (ff03 / implicit / eps_in=2 / no entropy / minimization-only) lands here if
# and only if the Rosetta baseline loses on this instrument.
export CPSET_SCORER="${CPSET_SCORER:-rosetta}"

# ---------------------------------------------------------------------------
# Candidate binding metric gate (s4_lnr_bindscore.sbatch)
# ---------------------------------------------------------------------------
# Pose-conditioned interface likelihood (ICG) -- the metric that answers "does
# THIS residue at THIS position suit THIS pocket", which is what the measured
# SAR is made of and what every prior metric (dG = buried area, ipTM = structure
# confidence, ESM-2/PepMLM = peptide plausibility) failed to ask. CPU only; the
# structure encoder never reads the sequence, so templates are encoded once and
# variants are nearly free. Pre-registration: docs/binding_metric_design.md.
export BINDSCORE_OUT="${BINDSCORE_OUT:-${RUN}/lnr/bindscore.jsonl}"
export BINDSCORE_MODELS="${BINDSCORE_MODELS:-ligandmpnn_v_32_020_25}"
export BINDSCORE_SERIES="${BINDSCORE_SERIES:-}"
export BINDSCORE_ORDER_SEED="${BINDSCORE_ORDER_SEED:-0}"
export LIGANDMPNN_HOME="${LIGANDMPNN_HOME:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/LigandMPNN}"

# ---------------------------------------------------------------------------
# Publication figure for that gate (s4_lnr_fig_pll_cpset.sbatch)
# ---------------------------------------------------------------------------
# Draws ONLY from the committed rows above -- no score is recomputed, so the
# figure is re-renderable in ~20 s on a CPU node after any styling change, and
# every statistic in it is computed by lnr.metric_gate rather than typed in.
# `_v2` is the 3-model run (119321); the primary model is the headline.
export FIG_PLL_ROWS="${FIG_PLL_ROWS:-${RUN}/lnr/bindscore_v2.jsonl}"
export FIG_PLL_DECOY="${FIG_PLL_DECOY:-${RUN}/lnr/bindscore_v2_decoy.jsonl}"
export FIG_PLL_OUTDIR="${FIG_PLL_OUTDIR:-${RUN}/lnr/figs}"
export FIG_PLL_STEM="${FIG_PLL_STEM:-fig_pll_cpset}"
export FIG_PLL_MODEL="${FIG_PLL_MODEL:-ligandmpnn_v_32_020_25}"
export FIG_PLL_NPERM="${FIG_PLL_NPERM:-20000}"
export FIG_PLL_FORMATS="${FIG_PLL_FORMATS:-pdf,png}"

# ---------------------------------------------------------------------------
# Pose-conditioned-likelihood IN-LOOP fitness (s4_lnr_seqmut_iter_pll.sbatch)
# ---------------------------------------------------------------------------
# The same metric as BINDSCORE_* above, deployed as the search's fitness instead
# of being graded against measured Kd. CPU only and bitwise deterministic, so
# the loop's noise defences switch themselves off; and exactly solvable, so each
# arm is graded by REGRET against the known optimum rather than by its score.
# `all` is the scope that was graded -- `window`/`iface` are recorded in every
# row regardless, so changing the choice never needs a rerun.
export SEQITER_PLL_METRIC="${SEQITER_PLL_METRIC:-pll_bg}"
export SEQITER_PLL_SCOPE="${SEQITER_PLL_SCOPE:-all}"
export SEQITER_PLL_MODEL="${SEQITER_PLL_MODEL:-ligandmpnn_v_32_020_25}"
export SEQITER_PLL_DEVICE="${SEQITER_PLL_DEVICE:-cpu}"
# iter-ea alone by default: free, deterministic, and the baseline the LLM arm is
# measured against. Adding iter-llm spends API budget.
export SEQITER_PLL_ARMS="${SEQITER_PLL_ARMS:-iter-ea}"
export SEQITER_PLL_MAX_EVALS="${SEQITER_PLL_MAX_EVALS:-400}"

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
