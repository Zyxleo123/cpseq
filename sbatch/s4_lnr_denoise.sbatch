#!/usr/bin/env bash
#SBATCH --job-name=s4_denoise
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_denoise_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# CPU RE-RELAX DENOISER (lever #1). Shrink the oracle's per-variant noise by
# re-scoring the ALREADY-FOLDED Boltz structures R times -- no new GPU fold. The
# ±3.4 REU per-variant spread is dominated by Rosetta FastRelax stochasticity
# (same Boltz coords score -26.95 vs -24.28 on two relaxes; the scorer sets no
# -constant_seed), so averaging N x R relaxes cuts it like 1/sqrt(N x R).
#
# Why it matters: NOTHING ranked these variants -- physics, confidence, sequence
# LMs alike -- because the oracle's own SEM (~1.7 REU) was as big as the
# between-variant differences. A ranker can only be as good as the ruler grading
# it. This denoises the ruler, and reports whether the between-variant RESOLUTION
# (between-var SD / within-var SEM) crosses ~1 -- the point the oracle can tell
# variants apart.
#
# CPU only, parallel across workers (each inits PyRosetta once; its consecutive
# relaxes are independent draws). Reads cached fold PDBs read-only, writes its own
# refined PDBs -> SAFE to run while the GPU search writes new folds. Resumable per
# (variant, model, repeat).
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
WORK="${DENOISE_WORK:-${OUTDIR}/boltz_inloop}"
CACHE="${DENOISE_CACHE:-${OUTDIR}/seqmut_iter_boltz_cache.jsonl}"
OUT="${DENOISE_OUT:-${OUTDIR}/denoise_scores.jsonl}"
REFINED="${DENOISE_REFINED:-${RUN}/scratch/denoise_refined}"
REPEATS="${DENOISE_REPEATS:-6}"
WORKERS="${DENOISE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
TARGETS="${DENOISE_TARGETS:-}"            # empty = all in the cache
MAXVAR="${DENOISE_MAX_VARIANTS:-10}"      # per target; validate before scaling
mkdir -p "$OUTDIR" "$REFINED"

echo "work=$WORK cache=$CACHE out=$OUT refined=$REFINED repeats=$REPEATS workers=$WORKERS targets=${TARGETS:-ALL} max_var=$MAXVAR"

# --- gates: print why, exit 0 (never strand dependents) ----------------------
if [ ! -d "$WORK" ]; then echo "GATE FAILED: no fold work dir $WORK. Exit 0."; exit 0; fi
if [ ! -f "$CACHE" ]; then echo "GATE FAILED: no cache $CACHE. Exit 0."; exit 0; fi
if ! "$PY" -c "import pyrosetta" 2>/dev/null; then
  echo "GATE FAILED: no pyrosetta in \$PY ($PY) -- the re-relax needs it. Exit 0."; exit 0
fi
if ! "$PY" -c "import proteinfoundation.cpseq.scorer" 2>/dev/null; then
  echo "GATE FAILED: cpseq scorer not importable in \$PY. Exit 0."; exit 0
fi
NF=$(find "$WORK" -maxdepth 1 -type d -name 'heldout_*' | wc -l)
echo "$NF fold dirs on disk; $(wc -l < "$CACHE") cache rows"

TG_ARG=()
[ -n "$TARGETS" ] && TG_ARG=(--targets "$TARGETS")

echo "=== re-relax denoiser: N x R relaxes per variant, resolution before/after ==="
"$PY" "$AGENT/lnr/rescore_denoise.py" \
  --score --work "$WORK" --cache "$CACHE" --out "$OUT" \
  --refined-dir "$REFINED" --repeats "$REPEATS" --workers "$WORKERS" \
  --max-variants "$MAXVAR" "${TG_ARG[@]}"

echo "DONE lnr denoise"
echo "re-render with no recompute:  \$PY $AGENT/lnr/rescore_denoise.py --report $OUT --cache $CACHE"
