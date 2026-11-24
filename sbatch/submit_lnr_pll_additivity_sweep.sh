#!/usr/bin/env bash
# Submit the designed additivity fold sweep, CHAINED. 89 variants x 4 replicas,
# MEASURED on gpu22 (v100) in job 120912 at ~1.7 min per replica, so ~6.9 min
# per variant and ~10 HOURS for the sweep -- not the ~3 h this script first
# assumed. That is 6 links of the 2-hour `debug` cap, and the job is resumable
# (appends rows, skips completed keys), so this queues N links with
# --dependency=afterany: afterany, not afterok, because hitting the wall is the
# EXPECTED way a link ends and afterok would stop the chain on it.
#
# Config travels as a POSITIONAL ENV FILE, never `sbatch --export` (an explicit
# export list makes sbatch set SLURM_GET_USER_ENV=1 and the job is requeued and
# HELD). Run from lov*.
#
#
# NOTE ON THE 35-JOB QOS CAP. `QOSMaxSubmitJobPerUserLimit` counts PENDING jobs
# too, so 6 links cannot always be queued at once -- submit what is accepted and
# add the rest as slots free. Links must stay SEQUENTIAL: folds are cached by
# sequence in one shared work dir, so two concurrent links would plan the same
# remaining variants and race on a half-written fold.
#
# Usage:
#   sbatch/submit_lnr_pll_additivity_sweep.sh            # 6 chained links
#   PLLSWEEP_LINKS=5 sbatch/submit_lnr_pll_additivity_sweep.sh
#   PLLSWEEP_TARGETS="1bjr 1elr" PLLSWEEP_LINKS=1 sbatch/...sh   # a pilot
set -euo pipefail
SBATCH_DIR="/zfsauton2/home/yixiz/lp2cp-agent/sbatch"
RUN="${RUN:-/zfsauton/scratch/yixiz/lp2cp_stage2}"
ENVDIR="${RUN}/envs"; mkdir -p "$ENVDIR" "${RUN}/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ENVF="${ENVDIR}/pll_add_sweep_${STAMP}.env"
OUTDIR_D="${LNR_OUTDIR:-${RUN}/lnr}"
LINKS="${PLLSWEEP_LINKS:-6}"
cat > "$ENVF" <<EOT
export LNR_OUTDIR="${OUTDIR_D}"
export PLLSWEEP_MANIFEST="${PLLSWEEP_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
export PLLSWEEP_PDB_DIR="${PLLSWEEP_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
# Targets and released windows come from the committed windows of the seqmut
# rows: substitutions are drawn ONLY from inside the window, because the fold
# path pins the backbone everywhere else.
export PLLSWEEP_SEQMUT_ROWS="${PLLSWEEP_SEQMUT_ROWS:-${OUTDIR_D}/seqmut_rows.jsonl}"
export PLLSWEEP_PSSM_DIR="${PLLSWEEP_PSSM_DIR:-${OUTDIR_D}/pssm}"
export PLLSWEEP_OUT="${PLLSWEEP_OUT:-${OUTDIR_D}/pll_additivity_sweep.jsonl}"
# Its OWN work dir, not the oracle's: the pilot re-scores structures under
# boltz_val and must keep reading the same files it already reported on.
export PLLSWEEP_WORK="${PLLSWEEP_WORK:-${OUTDIR_D}/boltz_sweep}"
export PLLSWEEP_REPLICAS="${PLLSWEEP_REPLICAS:-4}"
# The native's error enters EVERY gap, so it is the one sequence worth folding
# repeatedly. The pilot's own conclusion was "more folds per sequence".
export PLLSWEEP_NATIVE_DRAWS="${PLLSWEEP_NATIVE_DRAWS:-3}"
export PLLSWEEP_KMAX="${PLLSWEEP_KMAX:-4}"
export PLLSWEEP_TARGETS="${PLLSWEEP_TARGETS:-}"
export PLLSWEEP_SCOPE="${PLLSWEEP_SCOPE:-all}"
export PLLSWEEP_METRIC="${PLLSWEEP_METRIC:-}"
EOT
echo "wrote $ENVF:"; sed 's/^/    /' "$ENVF"
# Scrub SLURM_* so a submission from inside an allocation does not leak the
# parent job's environment into the child.
unset $(compgen -v | grep '^SLURM_' || true) 2>/dev/null || true

PREV=""
for i in $(seq 1 "$LINKS"); do
  if [ -z "$PREV" ]; then
    echo "+ sbatch ${SBATCH_DIR}/s4_lnr_pll_additivity_sweep.sbatch ${ENVF}"
    J=$(sbatch --parsable "${SBATCH_DIR}/s4_lnr_pll_additivity_sweep.sbatch" "${ENVF}")
  else
    echo "+ sbatch --dependency=afterany:${PREV} ... (link ${i})"
    J=$(sbatch --parsable --dependency="afterany:${PREV}" \
          "${SBATCH_DIR}/s4_lnr_pll_additivity_sweep.sbatch" "${ENVF}")
  fi
  echo "  link ${i}: ${J}"
  PREV="$J"
done
echo
echo "chained ${LINKS} link(s); the last is ${PREV}"
echo "watch: tail -f ${RUN}/logs/s4_lnr_pll_add_sweep_*.out"
echo "rows : ${OUTDIR_D}/pll_additivity_sweep.jsonl  (resumable; appends)"
