#!/usr/bin/env bash
#SBATCH --job-name=s4_pllo_af2
#SBATCH --partition=general,legacy
#SBATCH --exclude=gpu12,gpu13,gpu14,gpu15,gpu16,gpu17,gpu18,gpu19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_oracle_af2_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner

# ARM 2: AfCycDesign (ColabDesign AF2 + cyclic offset), the predictor RFpeptides
# uses to filter macrocycle designs before Rosetta interface metrics.
#
# TWO parameter sets, not five: ColabDesign loads model_{1,2}_ptm when templates
# are on and model_{1..5}_ptm only when they are off, because AF2 models 3-5 were
# trained without templates. AfCycDesign's published five comes from a MONOMER
# benchmark, where there is no template to begin with. Re-seeding adds nothing --
# fixed sequence, no MSA, dropout off is a deterministic forward pass.
#
# THE DECISION GATE IS INSIDE THE JOB. --require-gate reads af2_gate.jsonl and
# refuses to fold if the crystal control did not pass, printing why and exiting
# 0: a non-zero exit would strand the scoring job in DependencyNeverSatisfied
# with no record of which input was missing.
OUT="${PLLO_OUT:-${RUN}/lnr/pll_oracle}"
MAN="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
RECYCLES="${PLLO_RECYCLES:-6}"
AF2DIR="${AF2_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/ckpts/AF2}"
REQUIRE_GATE="${PLLO_REQUIRE_GATE:-1}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=2

if [ ! -s "$OUT/designs.json" ]; then
  echo "GATE FAILED: no $OUT/designs.json. Exit 0."; exit 0
fi
if [ ! -s "$OUT/af2_gate.jsonl" ]; then
  echo "GATE FAILED: no $OUT/af2_gate.jsonl -- the crystal control has not run."
  echo "  This is 'upstream never ran', not 'the control failed'. Exit 0."
  exit 0
fi

ARGS=(--fold-af2 --out-dir "$OUT" --manifest "$MAN" --pdb-dir "$PDB"
      --recycles "$RECYCLES" --af2-params-dir "$AF2DIR")
[ "$REQUIRE_GATE" = "1" ] && ARGS+=(--require-gate)
"$PY" "$AGENT/lnr/pll_oracle.py" "${ARGS[@]}"

echo "DONE pll_oracle fold-af2 -> $OUT/folds.jsonl"
