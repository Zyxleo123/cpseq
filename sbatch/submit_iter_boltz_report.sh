#!/usr/bin/env bash
# Fold the iterative-search FINALISTS with the REAL scorer, then report the
# believed number. This is the report tier of the firewall: the loop selected on
# the fold-free proxy; here -- and ONLY here -- do we fold.
#
# Two-tier budget (ALDE/EVOLVEpro pattern): per campaign the expensive Boltz x4
# -> Rosetta is spent on exactly the four things the paired comparison needs --
# the LLM finalist(s), their matched nomut (N_cyc, folded automatically by
# run_mut_held), and the random-EA finalist (the null). N_lin comes from the
# already-computed boltz_linear_ref.jsonl.
#
#   #folds = 4 (the Boltz-2 unit; no cross-fit at this depth, by design).
#
# GPU fold (s4_lnr_boltz.sbatch mut-held) -> CPU report (s4_lnr_iter_report.sbatch),
# chained afterok. Config travels as a positional env file (never sbatch
# --export). Run from lov* (submits directly).
#
# Usage: sbatch/submit_iter_boltz_report.sh [model]        (default model gpt-6-luna)
set -euo pipefail

SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL="${1:-${SEQITER_MODEL:-gpt-6-luna}}"
K="${SEQITER_K:-2}"
REPLICAS="${LNR_BOLTZ_REPLICAS:-4}"
# the folded things: the LLM finalist + the random-EA null; nomut (N_cyc) is added
# automatically per target by run_mut_held. (iter-llm-ala was not run in this
# validation, so it is omitted -- add it back when that arm is campaigned.)
ARMS="iter-ea:k${K},iter-llm:${MODEL}:k${K}"
MUT_OUT="boltz_iter_mut_held.jsonl"

FOLD_ENV="${ENVDIR}/iter_fold_${STAMP}.env"
TARGETS="${SEQITER_TARGETS:-1bjr,1elr}"   # the campaign's targets (NOT s4_lnr_boltz's default grid)
# Finalists come from ONE file (the real-score-in-loop rows). run_mut_held globs
# seqmut*_rows.jsonl and lets later-sorting files overwrite by (target,arm_label),
# so a stale proxy-loop seqmut_iter_ea/llm_rows.jsonl would clobber the real
# finalist. Stage a dir holding only the intended rows file and point at it.
# Overridable so a v2 (noise-defended) loop can be reported without touching v1.
SEQMUT_SRC="${SEQITER_SEQMUT_SRC:-${RUN}/lnr/seqmut_iter_boltz_rows.jsonl}"
STAGE="${RUN}/lnr/iter_boltz_finalists"
mkdir -p "$STAGE"
# Exactly ONE rows file may be staged: run_mut_held globs seqmut*_rows.jsonl
# and lets a later-sorting file overwrite by (target,arm_label), so a stale
# sibling here silently clobbers the intended finalist. Clear only the links
# this script itself plants -- never recurse, never touch a real file.
find "$STAGE" -maxdepth 1 -type l -name "seqmut*_rows.jsonl" -delete
ln -sf "$SEQMUT_SRC" "$STAGE/$(basename "$SEQMUT_SRC")"
echo "staged finalists: $SEQMUT_SRC"
cat > "$FOLD_ENV" <<EOF
export LNR_BOLTZ_MODE="mut-held"
export LNR_BOLTZ_TARGETS="${TARGETS}"
export LNR_BOLTZ_SEQMUT_DIR="${STAGE}"
export LNR_BOLTZ_MUT_ARMS="${ARMS}"
export LNR_BOLTZ_MUT_OUTPUT="${MUT_OUT}"
export LNR_BOLTZ_REPLICAS="${REPLICAS}"
EOF
echo "wrote $FOLD_ENV:"; sed 's/^/    /' "$FOLD_ENV"

RPT_ENV="${ENVDIR}/iter_report_${STAMP}.env"
cat > "$RPT_ENV" <<EOF
export LNR_ITER_MUT_HELD="${RUN}/lnr/${MUT_OUT}"
export LNR_ITER_LLM_ARMS="iter-llm:${MODEL}:k${K}"
export LNR_ITER_RANDOM_ARM="iter-ea:k${K}"
EOF
echo "wrote $RPT_ENV:"; sed 's/^/    /' "$RPT_ENV"

# The Boltz fold is a GPU job. s4_lnr_boltz.sbatch pins --nodelist=gpu22; before
# submitting, CHECK `myfree` and override the node on the sbatch line if gpu22 is
# not AVAILABLE (a v100 is the measured fit for these x4 folds). This script does
# NOT guess a node.
echo
echo "NEXT (run from lov*, after checking myfree for an AVAILABLE GPU node):"
echo "  # scrub SLURM_* if inside an allocation, then:"
echo "  A=\$(sbatch --parsable ${SBATCH_DIR}/s4_lnr_boltz.sbatch ${FOLD_ENV})"
echo "  sbatch --dependency=afterok:\$A ${SBATCH_DIR}/s4_lnr_iter_report.sbatch ${RPT_ENV}"
echo
echo "(If gpu22 is busy, add --nodelist=<AVAILABLE v100> to the first sbatch.)"
echo "env files are written and ready; this script does not submit so the GPU"
echo "node is chosen against live myfree, per the cluster rules."
