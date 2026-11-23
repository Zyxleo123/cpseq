#!/usr/bin/env bash
#SBATCH --job-name=s4_seq_rank
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_seq_rankers_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# CHEAP SEQUENCE RANKERS vs the expensive Boltz x4 -> Rosetta oracle.
# Does a fold-free sequence signal rank variants at least as well as the physics
# does, against the SAME ground truth (true_dG_median, already computed in
# rank_proxy_vs_true.jsonl)? This is the EVOLVEpro/ProtRL pattern -- a cheap
# ranker drives the search, the expensive oracle is spent sparingly -- tested on
# THIS problem before building a loop around it.
#
# Compared: proxy_recovered (current pose-preserving Rosetta proxy, BASELINE),
# ESM-2 650M peptide pseudo-log-likelihood (generic naturalness), PepMLM-650M
# conditional log-likelihood of the peptide GIVEN the receptor (a binder-specific
# LM -- the trained target->binder signal). Weights are already in HF_HOME.
#
# CPU only (650M on CPU, ~23 short peptides -- minutes), so it does NOT queue
# behind the running GPU search. Offline: no weight is downloaded. Resumable: a
# variant already scored is skipped.
#
# Caveat baked into the report: the LMs score LINEAR sequences; this peptide is
# head-to-tail CYCLIC, so this measures whether linear-sequence likelihood ranks
# cyclic variants -- the out-of-distribution question, settled empirically.
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
RANKFILE="${SEQRANK_GT:-${OUTDIR}/rank_proxy_vs_true.jsonl}"
OUT="${SEQRANK_OUT:-${OUTDIR}/seq_rankers.jsonl}"
MODELS="${SEQRANK_MODELS:-esm2,pepmlm}"
mkdir -p "$OUTDIR"

# fully offline HF; the weights are cached under HF_HOME (set in the preamble).
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "manifest=$MAN pdb=$PDB rank_file=$RANKFILE out=$OUT models=$MODELS HF_HOME=$HF_HOME"

# --- gates: print why and exit 0 (never strand dependents) -------------------
if [ ! -f "$RANKFILE" ]; then
  echo "GATE FAILED: no ground-truth file at $RANKFILE (run rank_proxy_vs_true first). Exit 0."
  exit 0
fi
if ! "$PY" -c "import torch, transformers" 2>/dev/null; then
  echo "GATE FAILED: no torch/transformers in \$PY ($PY). Exit 0."; exit 0
fi
case ",$MODELS," in
  *,esm2,*)   W="$HF_HOME/hub/models--facebook--esm2_t33_650M_UR50D"
              [ -d "$W" ] || { echo "GATE FAILED: no cached ESM-2 at $W. Exit 0."; exit 0; } ;;
esac
case ",$MODELS," in
  *,pepmlm,*) W="$HF_HOME/hub/models--ChatterjeeLab--PepMLM-650M"
              [ -d "$W" ] || { echo "GATE FAILED: no cached PepMLM at $W. Exit 0."; exit 0; } ;;
esac
echo "$(wc -l < "$RANKFILE") ground-truth variants to score"

echo "=== seq-rankers: cheap LM scoring + head-to-head vs the Boltz oracle ==="
"$PY" "$AGENT/lnr/seq_rankers.py" \
  --score --rank-file "$RANKFILE" \
  --manifest "$MAN" --pdb-dir "$PDB" \
  --out "$OUT" --models "$MODELS" --device cpu

echo "DONE lnr seq-rankers"
echo "re-render with no recompute:  \$PY $AGENT/lnr/seq_rankers.py --report $OUT"
