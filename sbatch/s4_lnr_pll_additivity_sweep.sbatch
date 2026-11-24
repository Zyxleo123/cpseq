#!/usr/bin/env bash
#SBATCH --job-name=s4_pll_sweep
# GPU, because every variant costs a Boltz fold. Pinned to gpu22 (v100): the
# rtx_2080_ti nodes carry 11 GB and a Boltz allocation failure can kill the
# process with NO Python traceback, which reads as a mysterious silent FAILED
# (CLAUDE.md). gpu22 is the node the proven s4_lnr_boltz job already uses.
# `debug` caps at 02:00:00 and the full sweep does not fit, which is why this
# job is RESUMABLE and the submitter chains several with --dependency=afterany.
#SBATCH --partition=debug
#SBATCH --nodelist=gpu22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/zfsauton/scratch/yixiz/lp2cp_stage2/logs/s4_lnr_pll_add_sweep_%j.out
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/lp2cp-agent/sbatch}"
PRE="${SUBMIT_DIR}/stage2_preamble.sh"
[ -f "$PRE" ] || PRE=/zfsauton2/home/yixiz/lp2cp-agent/sbatch/stage2_preamble.sh
. "$PRE" "$@"
banner
# Which node and which GPU, printed into the .out at startup: with sacct dead
# there is no accounting record to recover this from afterwards (CLAUDE.md).
echo "host     : $(hostname)"
nvidia-smi -L || true
echo "request  : partition=debug nodelist=gpu22 gres=gpu:1 mem=24G time=02:00:00"

# ===========================================================================
# CONFIRMATION OF THE ADDITIVITY RESULT: DOES THE GAP GROW WITH k?
#
# The pilot (jobs 120901/120905) measured that the frozen pll table OVER-charges
# mutations: 17/22 gaps positive, sign test p = 0.0169, 7 bootstrap-resolved,
# mean signed gap +0.26 nats/res against a 0.05 indifference band. It could not
# measure a k-slope, because it reused folds made for another experiment -- k was
# {1,2} and the mutants were an unbalanced convenience sample.
#
# A k-slope is what turns "the table is biased" into "the table degrades
# predictably", and only the second says where `best_subs` stops being usable.
#
# THREE ARMS PER (target, k), because a gap at the optimum alone is ambiguous:
#   opt  the table's own exact argmax inside the window -- the decision-relevant
#        case, since this is the sequence `best_subs` actually returns
#   pos  a random positive-gain subset -- is the bias a property of the optimum,
#        or of edits in general?
#   any  a random subset including losing edits -- the table predicts a loss;
#        does the pose absorb that too?
# Without pos/any, a gap at the optimum could be pure selection effect: the
# argmax is where the table is most confident and so most exposed.
#
# NATIVE REPLICATES ARE THE EXPENSIVE PART AND THE POINT. Every gap is measured
# against the native's own-pose score, so the native's error enters EVERY
# comparison. The pilot's floor (median sd 0.228 nats/res) came from single
# draws, and "more folds per sequence, not more sequences" was its own stated
# lever. This folds the native at several independent draws per target.
#
# STILL A LOWER BOUND. This reuses `boltz_inloop.fold_score_one` verbatim --
# deliberately, so the job cannot fail for a new reason -- and that path folds
# against a HELD TEMPLATE, pinning the backbone outside the window. Coupling
# through the held region stays suppressed by construction. The free co-fold arm
# that would turn this bound into an estimate needs a yaml path without
# `_held_templates_block`; it is the follow-up, NOT this job. The designed-sweep
# label does not launder that caveat away.
#
# WHAT TO READ: this job only WRITES rows. The statistics live in
# `pll_additivity.report`, which already applies the sign test and the
# per-mutant bootstrap, and the rows are emitted in its shape so the pilot and
# the sweep stay comparable row for row.
# ===========================================================================
OUTDIR="${LNR_OUTDIR:-${RUN}/lnr}"
MAN="${PLLSWEEP_MANIFEST:-/zfsauton/scratch/yixiz/LNR/test.txt}"
PDB_DIR="${PLLSWEEP_PDB_DIR:-/zfsauton/scratch/yixiz/LNR/pdbs}"
SEQROWS="${PLLSWEEP_SEQMUT_ROWS:-${OUTDIR}/seqmut_rows.jsonl}"
PSSM_DIR="${PLLSWEEP_PSSM_DIR:-${OUTDIR}/pssm}"
OUT_ROWS="${PLLSWEEP_OUT:-${OUTDIR}/pll_additivity_sweep.jsonl}"
WORK="${PLLSWEEP_WORK:-${OUTDIR}/boltz_sweep}"
REPLICAS="${PLLSWEEP_REPLICAS:-4}"
NATIVE_DRAWS="${PLLSWEEP_NATIVE_DRAWS:-3}"
KMAX="${PLLSWEEP_KMAX:-4}"
TARGETS="${PLLSWEEP_TARGETS:-}"
SCOPE="${PLLSWEEP_SCOPE:-all}"
METRIC="${PLLSWEEP_METRIC:-}"

echo "manifest=$MAN"
echo "seqmut_rows=$SEQROWS  pssm_dir=$PSSM_DIR"
echo "out=$OUT_ROWS  work=$WORK"
echo "replicas=$REPLICAS native_draws=$NATIVE_DRAWS k_max=$KMAX scope=$SCOPE"
echo "targets='${TARGETS:-<all with a committed window>}'"

# GATES on inputs. Each exits 0 with the reason named, never non-zero: a
# non-zero exit strands every chained job in DependencyNeverSatisfied with no
# record of which input was missing (CLAUDE.md).
for f in "$MAN" "$SEQROWS"; do
  if ! require_file "$f" "this sweep needs it to choose targets and windows"; then
    echo "Exit 0 so no chained job is stranded."; exit 0
  fi
done
if [ ! -d "$PSSM_DIR" ] || [ -z "$(ls -A "$PSSM_DIR"/pssm_*.json 2>/dev/null)" ]; then
  echo "GATE FAILED: no frozen PSSMs under $PSSM_DIR"
  echo "Exit 0 so no chained job is stranded."; exit 0
fi

echo; echo "=== selftest ==="
"$PY" "${AGENT}/lnr/pll_additivity_sweep.py" --selftest

echo; echo "=== plan (no GPU) ==="
"$PY" "${AGENT}/lnr/pll_additivity_sweep.py" --plan \
  --manifest "$MAN" --pdb-dir "$PDB_DIR" --seqmut-rows "$SEQROWS" \
  --pssm-dir "$PSSM_DIR" --k-max "$KMAX" --native-draws "$NATIVE_DRAWS" \
  ${TARGETS:+--targets "$TARGETS"} ${METRIC:+--metric "$METRIC"}

# SMOKE GATE. One variant, then check it actually produced a scorable fold
# before committing ~3 GPU-hours. A Boltz path that is broken for this job's
# arguments would otherwise burn the whole allocation writing rows whose
# `n_ok` is 0, and the failure would only surface at analysis time. Skipped
# once the output already holds a scorable row, so a chained resume does not
# re-pay for it.
echo; echo "=== smoke gate: one variant ==="
SMOKE_NEEDED=1
if [ -s "$OUT_ROWS" ]; then
  if "$PY" - "$OUT_ROWS" <<'EOF'
import json, sys
ok = any(json.loads(l).get("n_ok", 0) > 0
         for l in open(sys.argv[1]) if l.strip())
sys.exit(0 if ok else 1)
EOF
  then
    echo "output already holds a scorable fold -- smoke gate not re-run"
    SMOKE_NEEDED=0
  fi
fi
if [ "$SMOKE_NEEDED" = "1" ]; then
  "$PY" "${AGENT}/lnr/pll_additivity_sweep.py" --run --limit 1 \
    --manifest "$MAN" --pdb-dir "$PDB_DIR" --seqmut-rows "$SEQROWS" \
    --pssm-dir "$PSSM_DIR" --out "$OUT_ROWS" --work "$WORK" \
    --replicas "$REPLICAS" --native-draws "$NATIVE_DRAWS" --k-max "$KMAX" \
    --scope "$SCOPE" ${TARGETS:+--targets "$TARGETS"} \
    ${METRIC:+--metric "$METRIC"}
  if ! "$PY" - "$OUT_ROWS" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok = [r for r in rows if r.get("n_ok", 0) > 0]
if not ok:
    print("  no row has a scorable own-pose fold.")
    for r in rows[-3:]:
        print(f"    {r.get('target_id')} {r.get('arm')} "
              f"status={r.get('predict_status')} err={r.get('fold_error')}")
sys.exit(0 if ok else 1)
EOF
  then
    echo "GATE FAILED: the smoke variant produced no scorable fold."
    echo "  Not spending the rest of the allocation. Fix the fold path first;"
    echo "  the rows above name the predict_status and error."
    echo "Exit 0 so no chained job is stranded."; exit 0
  fi
  echo "smoke gate PASSED"
fi

echo; echo "=== run (resumable: appends, skips completed keys) ==="
"$PY" "${AGENT}/lnr/pll_additivity_sweep.py" --run \
  --manifest "$MAN" --pdb-dir "$PDB_DIR" --seqmut-rows "$SEQROWS" \
  --pssm-dir "$PSSM_DIR" --out "$OUT_ROWS" --work "$WORK" \
  --replicas "$REPLICAS" --native-draws "$NATIVE_DRAWS" --k-max "$KMAX" \
  --scope "$SCOPE" ${TARGETS:+--targets "$TARGETS"} \
  ${METRIC:+--metric "$METRIC"}

echo; echo "=== statistics on what exists so far ==="
# Safe to run on a PARTIAL sweep: the report drops any mutant whose target has
# no native fold yet, so an incomplete chain under-reports rather than lies.
"$PY" "${AGENT}/lnr/pll_additivity.py" --report --rows "$OUT_ROWS" || true
echo "done: $(date -Is)"
