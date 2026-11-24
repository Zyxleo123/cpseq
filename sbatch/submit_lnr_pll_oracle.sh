#!/usr/bin/env bash
# Submit the pll-table-vs-oracle experiment: does the expensive structure oracle
# agree that the pll_bg-optimal sequence beats random suboptimal ones?
#
#   stage          resource   what
#   gate           1 GPU      AfCycDesign crystal control on the native peptides
#   fold (array)   5 GPUs     Boltz-2, 16 structures per design, one task/target
#   af2            1 GPU      AfCycDesign on all designs, ONLY if the gate passed
#   score          32 CPUs    relax + Rosetta under v3 AND v5, every structure
#
# gate and fold are siblings (fold does not depend on the gate), so they queue
# concurrently; af2 waits on the gate; score waits on both folding stages.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export`: an explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1, slurmd then fails to
# rebuild the login environment on the compute node, and the job is requeued and
# HELD -- stranding every dependent job on Dependency. Run from lov*.
#
# Usage:
#   sbatch/submit_lnr_pll_oracle.sh            # the whole chain
#   PLLO_STAGES="gate" sbatch/submit_lnr_pll_oracle.sh
#   PLLO_STAGES="fold score" sbatch/submit_lnr_pll_oracle.sh
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pll_oracle_${STAMP}.env"
STAGES="${PLLO_STAGES:-gate fold af2 score}"

cat > "$ENVF" <<EOT
export PLLO_OUT="${PLLO_OUT:-${RUN}/lnr/pll_oracle}"
export LNR_MANIFEST="${LNR_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
export LNR_PDB_DIR="${LNR_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
export PLLO_RUNS="${PLLO_RUNS:-4}"
export PLLO_REPLICAS="${PLLO_REPLICAS:-4}"
export PLLO_RECYCLES="${PLLO_RECYCLES:-6}"
export PLLO_VERSIONS="${PLLO_VERSIONS:-v3,v5}"
export PLLO_REQUIRE_GATE="${PLLO_REQUIRE_GATE:-1}"
export AF2_DIR="${AF2_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/community_models/ckpts/AF2}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
echo "stages: $STAGES"

# Scrubbed so a submission from inside an allocation does not inherit its job's
# SLURM_* and confuse the new jobs' own resource view.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true

GATE=""; FOLD=""; AF2=""; SCORE=""
sub() { echo "+ sbatch $*" >&2; sbatch --parsable "$@"; }

case " $STAGES " in *" gate "*)
  GATE=$(sub "${SBATCH_DIR}/s4_lnr_pll_oracle_gate.sbatch" "$ENVF")
  echo "gate   : $GATE";; esac

case " $STAGES " in *" fold "*)
  FOLD=$(sub "${SBATCH_DIR}/s4_lnr_pll_oracle_fold.sbatch" "$ENVF")
  echo "fold   : $FOLD (array 0-4)";; esac

case " $STAGES " in *" af2 "*)
  DEP=()
  [ -n "$GATE" ] && DEP=(--dependency=afterok:"$GATE")
  AF2=$(sub "${DEP[@]}" "${SBATCH_DIR}/s4_lnr_pll_oracle_af2.sbatch" "$ENVF")
  echo "af2    : $AF2";; esac

case " $STAGES " in *" score "*)
  DEPS=()
  [ -n "$FOLD" ] && DEPS+=("afterany:$FOLD")
  [ -n "$AF2" ]  && DEPS+=("afterany:$AF2")
  DEP=()
  if [ ${#DEPS[@]} -gt 0 ]; then
    # afterany, not afterok: a target whose folding hit the time limit still
    # leaves usable structures on disk, and the scorer is resumable, so the
    # right behaviour is to score what exists rather than to strand the stage.
    DEP=(--dependency=$(IFS=,; echo "${DEPS[*]}"))
  fi
  SCORE=$(sub "${DEP[@]}" "${SBATCH_DIR}/s4_lnr_pll_oracle_score.sbatch" "$ENVF")
  echo "score  : $SCORE";; esac

echo
echo "watch:"
[ -n "$GATE" ]  && echo "  tail -f ${RUN}/logs/s4_lnr_pll_oracle_gate_${GATE}.out"
[ -n "$FOLD" ]  && echo "  tail -f ${RUN}/logs/s4_lnr_pll_oracle_fold_${FOLD}_0.out"
[ -n "$AF2" ]   && echo "  tail -f ${RUN}/logs/s4_lnr_pll_oracle_af2_${AF2}.out"
[ -n "$SCORE" ] && echo "  tail -f ${RUN}/logs/s4_lnr_pll_oracle_score_${SCORE}.out"
echo "  squeue -u $USER"
exit 0
