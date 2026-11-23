#!/usr/bin/env bash
#SBATCH --job-name=s4_pll_scope
# CPU ONLY and no search: `pll_bg` is additive over positions, so every optimum
# here is read off a CACHED PSSM table. No model forward pass, no pose
# prediction, no GPU -- asking for one would just queue behind real work
# (CLAUDE.md: split jobs by resource).
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_scope_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# THE SCOPE QUESTION, factual half. The in-loop fitness averages the per-residue
# likelihood over a position set, and three are defensible: `all` (what was
# graded), `window` (what the LNR objective asks for) and `iface` (the
# contacting residues). WHICH ONE IS RIGHT IS A JUDGEMENT AND IS NOT DECIDED
# HERE. What this job measures is whether the judgement changes the answer --
# the off-diagonal of a 3x3 regret matrix: the fraction of the true scope's
# reachable gain that is forgone by deploying another scope's optimum.
#
# Exact rather than sampled, and deliberately NOT computed from
# seqmut_iter_pll_scores.jsonl: every sequence in that file was proposed by a
# search optimising `all`, so correlating its three recorded scope columns would
# understate the divergence. Optima come off the table instead.
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
PSSM_DIR="${PLLSCOPE_PSSM_DIR:-${OUTDIR}/pssm}"
SEED_WINDOWS="${PLLSCOPE_SEED_WINDOWS:-${OUTDIR}/seqmut_iter_pll_rows.jsonl}"
OUT_JSON="${PLLSCOPE_OUT:-${OUTDIR}/pll_scope.json}"
METRIC="${PLLSCOPE_METRIC:-pll_bg}"
KMUT="${PLLSCOPE_K:-2}"
TARGETS="${PLLSCOPE_TARGETS:-}"

echo "manifest=$MAN pssm_dir=$PSSM_DIR metric=$METRIC k_mut=$KMUT targets='${TARGETS:-<all>}'"

# GATE: the cached tables are what makes this free. Without them every target
# would rebuild a PSSM from the pose, which is a different (much slower) job and
# almost certainly not what was intended.
if [ ! -d "$PSSM_DIR" ]; then
  echo "GATE FAILED: no cached PSSM directory at $PSSM_DIR."
  echo "Run sbatch/submit_lnr_seqmut_iter_pll.sh first -- it writes the tables."
  echo "Exiting 0 so no dependent job is stranded."; exit 0
fi
N_PSSM=$(find "$PSSM_DIR" -name '*.json' 2>/dev/null | wc -l)
echo "cached PSSM tables: $N_PSSM"

# Not fatal: without the committed windows, scope='window' silently falls back
# to the whole peptide and collapses onto 'all'. The analysis would then report
# agreement it never tested, so say so loudly and keep going (the `iface` vs
# `all` comparison is still meaningful).
if [ ! -f "$SEED_WINDOWS" ]; then
  echo "WARNING: no committed windows at $SEED_WINDOWS -- scope='window'"
  echo "         cannot discriminate and the window row/column will be a tie."
fi

TARGET_ARG=()
[ -n "$TARGETS" ] && TARGET_ARG=(--targets "$TARGETS")

"$PY" "$AGENT/lnr/pll_scope.py" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --pssm-dir "$PSSM_DIR" --seed-windows "$SEED_WINDOWS" \
  --metric "$METRIC" --k-mut "$KMUT" \
  --out-json "$OUT_JSON" "${TARGET_ARG[@]}"

echo "DONE pll scope"
echo "summary: $OUT_JSON"
