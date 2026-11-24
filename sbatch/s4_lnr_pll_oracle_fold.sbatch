#!/usr/bin/env bash
#SBATCH --job-name=s4_pllo_fold
# One ARRAY TASK PER TARGET. The five targets are independent, so they are
# siblings rather than a loop: five short jobs queue and finish in the time one
# long job would take, and a time limit costs one target's folds, not all five.
#SBATCH --array=0-4
# GPU class pinned by exclusion -- see the gate job for why the 11 GB nodes are
# out (Boltz-2 with inference potentials at diffusion_samples=4 is sized for the
# 16 GB v100 this project has used throughout).
#SBATCH --partition=general,legacy
#SBATCH --exclude=gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_oracle_fold_%A_%a.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ARM 1: Boltz-2 with inference potentials, held native backbone as template.
# 4 designs x 4 INDEPENDENT runs x 4 diffusion samples = 64 structures per
# target. The 4 runs are separate trunk passes (different seeds); the 4 samples
# inside a run share one. Both levels are recorded per structure, because a
# single fold carries ~3-4 REU of noise and 61% of an apparent gain once
# evaporated on an independent redraw -- so one draw per sequence would measure
# the denoiser rather than the sequence.
#
# NO SCORING HERE. Folding is GPU work, Rosetta is CPU work, and the predictions
# are the irreplaceable artefact: re-scoring them under another protocol must
# never need another GPU hour.
OUT="${PLLO_OUT:-${RUN}/lnr/pll_oracle}"
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
NRUNS="${PLLO_RUNS:-4}"
REPLICAS="${PLLO_REPLICAS:-4}"

if [ ! -s "$OUT/designs.json" ]; then
  echo "GATE FAILED: no $OUT/designs.json -- run --select first. Exit 0."
  exit 0
fi
# The array index picks this task's target out of the committed design file, so
# the mapping is in the data rather than duplicated in the script.
TID=$("$PY" - "$OUT/designs.json" "${SLURM_ARRAY_TASK_ID:-0}" <<'EOF'
import json, sys
doc = json.loads(open(sys.argv[1]).read())
ids = [r["target_id"] for r in doc["targets"]]
i = int(sys.argv[2])
print(ids[i] if i < len(ids) else "")
EOF
)
if [ -z "$TID" ]; then
  echo "no target for array index ${SLURM_ARRAY_TASK_ID:-0}; nothing to do. Exit 0."
  exit 0
fi
echo "array task ${SLURM_ARRAY_TASK_ID:-0} -> target $TID"
echo "out=$OUT runs=$NRUNS replicas=$REPLICAS (=$((NRUNS * REPLICAS)) folds/design)"

if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "GATE FAILED: torch missing in $PY. Exit 0."; exit 0
fi
if [ ! -x "/zfsauton/scratch/yixiz/envs/boltz2/.venv/bin/python" ]; then
  echo "GATE FAILED: no boltz2 env. Exit 0."; exit 0
fi

"$PY" "$AGENT/lnr/pll_oracle.py" --fold-boltz \
  --out-dir "$OUT" --manifest "$MAN" --pdb-dir "$PDB" \
  --targets "$TID" --n-runs "$NRUNS" --replicas "$REPLICAS"

echo "DONE pll_oracle fold-boltz $TID -> $OUT/folds.jsonl"
grep -c "\"target_id\": \"$TID\"" "$OUT/folds.jsonl" 2>/dev/null || true
