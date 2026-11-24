#!/usr/bin/env bash
#SBATCH --job-name=s4_pepliab
# Reads CACHED PSSMs ONLY -- no folding, no model forward pass, no GPU. Every
# number is a table lookup plus regex over a sequence, so this is seconds of CPU
# on artefacts already on disk (CLAUDE.md: split by resource, and figures/tables
# must be redrawable without recomputation). The `cpu` partition is the only
# correct target for it: `myfree` lists GPU nodes, and asking for one here would
# queue a regex behind real folding work.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pepliab_%j.out
# `set -euo pipefail` lives HERE, in the job body, not only in the preamble: a
# failed source would otherwise leave this running under the system python and
# still EXIT 0, releasing any afterok dependent behind a stage that did nothing.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ===========================================================================
# WHAT THE EXACT pll OPTIMUM COSTS ON AXES IT WAS NEVER TOLD ABOUT
#
# `best_subs` is the exact argmax of a mean of independent per-position terms,
# so no search beats it on pll -- measured, 6/6 on the free optimum. This job
# asks the only remaining question about that optimum: is the sequence it
# returns a peptide anyone could make or use?
#
# WHY THIS RUNS BEFORE ANY MULTI-OBJECTIVE SEARCH IS BUILT. The cheap null for
# the whole multi-objective angle is "annotate the table and run a DP/ILP" --
# exact, free, and strictly better than a search on every objective that is
# per-position, a local motif, or linear in residue counts. So a search arm is
# only defensible on the residue this job measures. If the exact optimum turns
# out to introduce almost nothing, the angle is weak and we learned it for
# seconds of CPU instead of a month of API spend. That is the point.
#
# THE TWO NUMBERS, AND DO NOT CONFUSE THEM
#   total introduced   motivates ANNOTATING THE TABLE. Every class counted here
#                      (separable / local / global) is solvable exactly and for
#                      free. A big number here is an argument AGAINST a search
#                      arm, not for one.
#   needs_ctx          the only number that argues for a per-target
#                      specification arm: violations whose correct RULE differs
#                      by target (a disulfide-cyclized peptide needs exactly two
#                      Cys at given positions; a linear one needs none), so no
#                      universal column set contains them.
#
# THE ARM COMPARISON IS THE PAIR OF RUNS, NOT EITHER ONE. The job runs the
# battery twice over the same targets:
#   --no-ctx   empty Ctx everywhere: the "one careful universal spec" arm.
#   default    Ctx built from each target's own SSBOND records: what a per-target
#              specifier would have to supply.
# The DIFFERENCE between those two files is the measurement. Reading only the
# ctx run would credit the specification arm with everything a hand-written
# column set already catches.
#
# WHAT THIS JOB DOES NOT MEASURE. Five JOINT axes (aggregation, permeability,
# immunogenicity, off-target, synthesis yield) are declared in `pepliab.py` and
# deliberately NOT implemented: none is a function of per-position values or
# residue counts, so there is no column to add and no ILP to write. They are
# where a search arm is structurally necessary and they all cost a real
# evaluation. A clean battery here is NOT a clean peptide, and the summary says
# so in its own output so a future reader cannot quote the number without it.
# ===========================================================================
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
PSSM_DIR="${PEPLIAB_PSSM_DIR:-${OUTDIR}/pssm}"
PDB_DIR="${PEPLIAB_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
KMUT="${PEPLIAB_K:-2}"
SCOPE="${PEPLIAB_SCOPE:-all}"
METRIC="${PEPLIAB_METRIC:-}"
OUT_CTX="${PEPLIAB_OUT_CTX:-${OUTDIR}/pepliab_audit_ctx.jsonl}"
OUT_UNI="${PEPLIAB_OUT_UNI:-${OUTDIR}/pepliab_audit_universal.jsonl}"

echo "pssm_dir=$PSSM_DIR"
echo "pdb_dir=$PDB_DIR  k_mut=$KMUT  scope=$SCOPE  metric='${METRIC:-<default pll_bg>}'"
echo "out(ctx)=$OUT_CTX"
echo "out(universal)=$OUT_UNI"

# GATE. The audit reads cached PSSMs; with none there is nothing to audit. Print
# why and exit 0 rather than non-zero, so a dependent job is never stranded in
# DependencyNeverSatisfied with no record of which input was missing.
if [ ! -d "$PSSM_DIR" ] || [ -z "$(ls -A "$PSSM_DIR"/pssm_*.json 2>/dev/null)" ]; then
  echo "GATE FAILED: no cached PSSMs under $PSSM_DIR"
  echo "  run sbatch/submit_lnr_seqmut_iter_pll.sh first -- it writes them there"
  echo "Exit 0 so no dependent job is stranded."
  exit 0
fi
echo "cached PSSMs: $(ls -1 "$PSSM_DIR"/pssm_*.json | wc -l)"

# The fast gate runs first and is CHEAP: it exercises every check and the
# native-relative set difference without touching a PSSM. A battery whose
# needs_ctx exemplar is broken would otherwise report a confident zero.
echo; echo "=== selftest ==="
"$PY" "${AGENT}/lnr/pepliab.py" --selftest

echo; echo "=== arm 1: ONE UNIVERSAL SPEC (empty Ctx on every target) ==="
"$PY" "${AGENT}/lnr/pepliab.py" --audit --no-ctx \
  --pssm-dir "$PSSM_DIR" --pdb-dir "$PDB_DIR" \
  --k-mut "$KMUT" --scope "$SCOPE" ${METRIC:+--metric "$METRIC"} \
  --out "$OUT_UNI"

echo; echo "=== arm 2: PER-TARGET CONTEXT (SSBOND topology from each structure) ==="
"$PY" "${AGENT}/lnr/pepliab.py" --audit \
  --pssm-dir "$PSSM_DIR" --pdb-dir "$PDB_DIR" \
  --k-mut "$KMUT" --scope "$SCOPE" ${METRIC:+--metric "$METRIC"} \
  --out "$OUT_CTX"

echo; echo "=== both files written; the DIFFERENCE between them is the result ==="
ls -la "$OUT_UNI" "$OUT_CTX"
echo "done: $(date -Is)"
