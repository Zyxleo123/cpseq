#!/usr/bin/env bash
# Submit the REAL-SCORE-IN-LOOP iterative search with the noise defences on.
#
# Why there is a v2 at all. The first real-score run's own firewall (loop folds
# with one Boltz seed, the report re-folds the finalist with another) showed that
# 61% of the apparent gain vs wild-type did not survive the redraw -- 1bjr
# iter-llm claimed -8.01 REU and realised -0.57. The cause was not the LLM: an
# in-loop fitness is a DRAW (per-variant fold sd 3.74 REU mean, 11.7 max; the
# same variant re-measured across runs moved ~4 REU) and the loop took the argmin
# of ~15 such draws, never re-measured the incumbent, and let the first lucky
# draw steer every later generation. Two further defects compounded it: the arms
# drew DIFFERENT random seed pools (mean gap 7.66 REU, i.e. the control was
# confounded before an operator proposed anything) and 53-62% of the matched
# budget went on that random pool.
#
# What v2 changes (all now the sbatch defaults; see s4_lnr_seqmut_iter_boltz.sbatch):
#   * shared seed pool across arms -> the LLM-vs-EA comparison is paired;
#   * incumbent re-measured once per generation with an independent fold seed;
#   * elite ranked optimistically (z-explore), finalist committed pessimistically
#     (z-commit), so we fold the genome that is good, not the one measured lucky;
#   * budget re-split 5 seed / 5 re-measure / 14 offspring of 24.
# On a known-truth simulation driven by the EMPIRICAL fold-noise distribution,
# this cuts the finalist's selection bias from -2.41 to -0.93 REU (28% -> 12% of
# the claim) while the finalist's TRUE fitness improves (-6.05 -> -6.52).
#
# v1 and v2 rows are NOT poolable (different budget, different seeding). v2
# writes its own rows/traj files; archive the v1 rows file before running
# `--report-dir`, which globs seqmut_iter*_rows.jsonl.
#
# The Boltz fold cache IS shared and should be: it is keyed by (variant, draw)
# and v2's draw-0 folds use the same seed as v1's, so every fold already computed
# is reused rather than re-run.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export` (an export
# list makes sbatch set SLURM_GET_USER_ENV=1, the compute node's login-env
# rebuild fails, and the job is requeued+held). Run from lov* (submits directly).
#
# Usage:
#   sbatch/submit_lnr_seqmut_iter_boltz.sh                 # arms + targets below
#   SEQITER_TARGETS=1bjr,1elr,1fiw sbatch/submit_lnr_seqmut_iter_boltz.sh
#   SEQITER_REASSESS=0 sbatch/submit_lnr_seqmut_iter_boltz.sh   # v1 behaviour (A/B)
set -euo pipefail

SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

ARMS="${SEQITER_ARMS:-iter-ea,iter-llm}"
TARGETS="${SEQITER_TARGETS:-1bjr,1elr}"
MODEL="${SEQITER_MODEL:-gpt-6-luna}"
BACKEND="${SEQITER_BACKEND:-openai}"
MAXSPEND="${SEQITER_MAX_SPEND:-1.0}"
K="${SEQITER_K:-2}"
# Matched across BOTH arms -- the whole comparison rests on these being equal.
NSEED="${SEQITER_N_SEED:-5}"
NGEN="${SEQITER_N_GEN:-5}"
MOFF="${SEQITER_M_OFF:-4}"
MAXEVALS="${SEQITER_MAX_EVALS:-24}"
ELITE="${SEQITER_ELITE_FRAC:-0.3}"
ZEXPLORE="${SEQITER_Z_EXPLORE:-1.0}"
ZCOMMIT="${SEQITER_Z_COMMIT:-1.0}"
REASSESS="${SEQITER_REASSESS:-1}"
TAG="${SEQITER_TAG:-v2}"
ROWS="seqmut_iter_boltz_${TAG}_rows.jsonl"
TRAJ="seqmut_iter_boltz_${TAG}_traj.jsonl"

ENVF="${ENVDIR}/seqiter_boltz_${TAG}_${STAMP}.env"
cat > "$ENVF" <<EOF
export SEQITER_ARMS="${ARMS}"
export SEQITER_TARGETS="${TARGETS}"
export SEQITER_ROWS_FILE="${ROWS}"
export SEQITER_TRAJ_FILE="${TRAJ}"
export SEQITER_BACKEND="${BACKEND}"
export SEQITER_MODEL="${MODEL}"
export SEQITER_MAX_SPEND="${MAXSPEND}"
export SEQITER_K="${K}"
export SEQITER_N_SEED="${NSEED}"
export SEQITER_N_GEN="${NGEN}"
export SEQITER_M_OFF="${MOFF}"
export SEQITER_MAX_EVALS="${MAXEVALS}"
export SEQITER_ELITE_FRAC="${ELITE}"
export SEQITER_Z_EXPLORE="${ZEXPLORE}"
export SEQITER_Z_COMMIT="${ZCOMMIT}"
export SEQITER_REASSESS="${REASSESS}"
EOF
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"

# scrub SLURM_* so nothing from an allocation leaks into the submitted job
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true

echo "+ sbatch ${SBATCH_DIR}/s4_lnr_seqmut_iter_boltz.sbatch ${ENVF}"
J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_seqmut_iter_boltz.sbatch" "${ENVF}")
echo "submitted real-score-in-loop job: $J"
echo
echo "watch:   squeue -j $J   /   tail -f ${RUN}/logs/s4_lnr_seqmut_iter_boltz_${J}.out"
echo "the per-generation diagnosis is redrawable with no recompute:"
echo "  ${RUN}/lnr/${TRAJ}   (incumbent_shift = the optimizer's curse, live)"
echo
echo "when it lands, fold the v2 finalists with an INDEPENDENT draw and report:"
echo "  SEQITER_SEQMUT_SRC=${RUN}/lnr/${ROWS} \\"
echo "  SEQITER_TARGETS=${TARGETS} \\"
echo "    ${SBATCH_DIR}/submit_iter_boltz_report.sh ${MODEL}"
