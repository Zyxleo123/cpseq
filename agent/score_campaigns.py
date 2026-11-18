"""Score every arm against the enumerated oracle. Reads only saved artefacts.

No GPU, no API key, no model. Everything here is a function of the campaign
traces and the oracle table, so a figure can be redrawn and a metric redefined
without re-running a single campaign -- which is the point of having the
enumerated table at all.

The headline comparisons are PAIRED over shared parents, not mean-vs-mean.
Stage 0's lesson: arms sit on different denominators once refusals, malformed
actions and compile rejections are counted, so a difference of means mixes "is
this controller better" with "did this model answer more often". The paired
Wilcoxon over the parents both arms completed answers the first question only.

The reportable number the design doc names is NOT the win rate: "The
reportable number is the search-time-to-held-out gap per method, not the win."
That gap is computed here per arm, because the environment has a declared
held-out protocol (fresh seeds, a second predictor with its own bias, extended
conformer sampling) and every campaign's best candidate is re-scored under it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Statistics, written out rather than imported, so the job needs no scipy
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, int, int, int]:
    """Two-sided Wilcoxon signed-rank p, plus win/loss/tie counts for arm A.

    `deltas` are (A's regret - B's regret) and LOWER REGRET IS BETTER, so a
    NEGATIVE delta is a win for A. Getting that sign backwards is the easiest
    way to report a headline the wrong way round, so the counts are named for
    what they mean rather than for the sign of the arithmetic.

    Normal approximation with a tie correction, which is what scipy uses above
    n=25 and is adequate here; exact enumeration below n=10 would change a p of
    0.03 to 0.04 and no conclusion in this project turns on that. Zero
    differences are DROPPED from the ranking (Wilcoxon's own convention) but
    reported as ties, because an arm that matches the bar on 40 of 60 parents
    and wins the rest is a different claim from one that wins 20 of 20.
    """
    nz = [d for d in deltas if abs(d) > 1e-12]
    wins = sum(1 for d in deltas if d < -1e-12)      # A had LOWER regret
    losses = sum(1 for d in deltas if d > 1e-12)     # A had HIGHER regret
    ties = len(deltas) - wins - losses
    n = len(nz)
    if n < 6:
        return float("nan"), wins, losses, ties
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_groups: list[int] = []
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        tie_groups.append(j - i + 1)
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4.0
    tie_corr = sum(t ** 3 - t for t in tie_groups)
    var = (n * (n + 1) * (2 * n + 1) - tie_corr / 2.0) / 24.0
    if var <= 0:
        return float("nan"), wins, losses, ties
    z = (w_plus - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return max(min(p, 1.0), 0.0), wins, losses, ties


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def boot_ci(xs: list[float], n_boot: int = 2000, seed: int = 0
            ) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean. Seeded, so the interval is stable."""
    import random

    xs = [x for x in xs if x == x]
    if len(xs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(mean(rng.choices(xs, k=len(xs))))
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def assert_env_frozen(env_manifest: Path) -> dict:
    """Refuse to score if the surrogate changed since the benchmark was built.

    Invariant 2 of the design doc freezes the QUALIFICATION function, and
    `qualification.assert_frozen` enforces that. The ENVIRONMENT needs the same
    guard and did not have it, which cost real numbers: the surrogate was
    edited from v1 to v2 after the oracle table and every campaign had been
    produced under v1, and the scorer went on recomputing held-out cards and
    per-iteration utilities under v2 against a v1 denominator. The resulting
    table looked completely normal -- the oracle's own mean regret moved from
    12.51 to 6.33 on campaign data that had not changed at all.

    A mismatch is a hard stop, not a warning. There is no correct way to pool
    rows from two response models, and the failure is invisible in the output.
    """
    if not env_manifest or not Path(env_manifest).exists():
        print("[score] WARNING: no env_manifest given; cannot verify that the "
              "surrogate matches the one the oracle table was built with.")
        return {}
    man = json.loads(Path(env_manifest).read_text())
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lp2cp.envbuild import surrogate as S

    live, frozen = S.provenance(), man.get("surrogate")
    if frozen and live != frozen:
        raise RuntimeError(
            "SURROGATE CHANGED since the benchmark was frozen.\n"
            f"  oracle table / campaigns were produced under: {frozen}\n"
            f"  the code in this tree is now:                 {live}\n"
            "Rows from the two are not poolable and every regret here would be "
            "measured against a denominator from the other model. Either:\n"
            "  (a) revert the surrogate to the frozen version and re-score, or\n"
            "  (b) rebuild the benchmark (s2_00, s2_05) and RE-RUN every arm "
            "under the new version -- note that re-running API arms costs "
            "spend, and that partial re-runs cannot be mixed.\n"
            "Pass --allow-env-drift ONLY to inspect, never to report."
        )
    return man


def load_campaigns(d: Path) -> list[dict]:
    """Every campaign trace, deduplicated by campaign_id and corruption-tolerant.

    Both defences are there because both have happened. One arm is meant to own
    one file, but two jobs writing the same arm (a Slurm requeue that reran a
    cancelled arm alongside its replacement) append concurrently: interleaved
    writes produce unparseable lines, and a reran parent produces a SECOND row
    with the same campaign_id. Unparseable lines are skipped with a count;
    duplicate ids keep the LAST complete row, since a resumed job rewrites a
    parent only when it is redoing it.

    Silently double-counting a parent would bias that arm's mean toward
    whichever parents happened to be rerun, and nothing downstream would show
    it -- so the counts are printed, not just handled.
    """
    rows: list[dict] = []
    n_corrupt = 0
    for f in sorted(d.glob("campaigns__*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_corrupt += 1
                continue
            r["_file"] = f.name
            rows.append(r)
    # De-duplicate on (arm_key, parent_id), NOT on campaign_id.
    #
    # campaign_id is "{parent}__{arm}__{model}" and carries no scaffold, so two
    # runs of the same arm under different scaffolds collide exactly. Measured:
    # the contract-injection experiment (scaffold "onecall+inject") shared every
    # campaign_id with its own A3 baseline, and keying on campaign_id collapsed
    # 80 rows and silently deleted the entire experiment from the results
    # table -- the run looked like it had never happened.
    #
    # (arm_key, parent_id) is the real identity of a campaign: one controller,
    # one arm, one scaffold, one parent. A genuine duplicate -- two jobs
    # appending the same arm file -- still collapses, which is what the warning
    # is for.
    by_id: dict[tuple, dict] = {}
    n_dup = 0
    for r in rows:
        key = (arm_key(r), r.get("parent_id") or r.get("_file"))
        if key in by_id:
            n_dup += 1
        by_id[key] = r
    if n_corrupt or n_dup:
        print(f"[score] NOTE: skipped {n_corrupt} unparseable line(s) and "
              f"collapsed {n_dup} duplicate (arm, parent) row(s) -- check for two "
              f"jobs writing the same arm file")
    return list(by_id.values())


def load_oracle(path: Path) -> dict[str, dict]:
    """Per-parent oracle facts: the best attainable utility and its cell."""
    best: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r["parent_id"]
        cur = best.get(pid)
        if cur is None or r["utility"] > cur["utility"]:
            best[pid] = r
    return best


def oracle_noop(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["family"] == "noop":
                out[r["parent_id"]] = r["utility"]
    return out


# ---------------------------------------------------------------------------
# Per-campaign metrics
# ---------------------------------------------------------------------------


def _n_api_error_steps(r: dict) -> int:
    """Steps that failed in TRANSPORT, not in the model's answer.

    Traces written before the controller separated the two report a billing or
    rate-limit failure as `n_malformed`, i.e. as the model emitting an action
    that could not be parsed. That is a slander on the model and it is not a
    small one: when the API credit balance ran out mid-run, two whole arms came
    out as 100% malformed with regret identical to doing nothing, which reads
    as "this model cannot do the task" rather than "the account was empty".
    Recovered here from the step records, which carry the HTTP error text.
    """
    n = 0
    for it in r.get("iterations") or []:
        for st in it.get("steps") or []:
            if st.get("status") != "ERROR":
                continue
            d = str(st.get("detail") or "")
            if any(t in d for t in ("BadRequestError", "credit balance",
                                    "RateLimit", "APIConnection", "Overloaded",
                                    "InternalServer", "APITimeout")):
                n += 1
    return n


def arm_key(r: dict) -> str:
    """One label per (controller, arm, scaffold). The unit of comparison."""
    ctl = r.get("controller", "?")
    if ctl.startswith("rule:"):
        return ctl
    return f"{r.get('model_id', '?')}|{r.get('arm', '?')}|{r.get('scaffold', '?')}"


def campaign_metrics(r: dict, oracle: dict, noop_u: dict,
                     noop_latent: dict | None = None) -> dict:
    """Everything derivable from one campaign trace."""
    pid = r["parent_id"]
    orc = oracle.get(pid, {})
    best_u_possible = orc.get("utility", float("nan"))
    card = r.get("best_card") or {}
    # The controller's achieved utility, recomputed from the frozen utility
    # function rather than read from the trace, so a change to the utility
    # definition re-scores every campaign already on disk.
    achieved = r.get("_achieved_utility")
    # Regret is measured on the LATENT value of what the arm committed to, so
    # that it is on the same footing as the oracle table it is subtracted from.
    truth = r.get("_latent_utility")
    if truth is None:
        # No committed intervention (the arm never acted): its latent value is
        # the parent's own, which is exactly the no-op cell.
        truth = noop_latent.get(pid)
    regret = (
        best_u_possible - truth
        if truth is not None and best_u_possible == best_u_possible
        else float("nan")
    )
    its = r.get("iterations") or []
    svc = r.get("service_ledger") or {}
    return {
        "campaign_id": r.get("campaign_id"),
        "parent_id": pid,
        "cluster_id": r.get("cluster_id"),
        "split": r.get("split"),
        "arm_key": arm_key(r),
        "arm": r.get("arm"),
        "model_id": r.get("model_id"),
        "controller": r.get("controller"),
        "scaffold": r.get("scaffold"),
        "error": r.get("error"),
        # A campaign the SPEND LEDGER stopped is not a measurement of the
        # controller: it is a stub. Measured on Sonnet A2, 21 of 32 campaigns
        # ended this way after exactly one iteration and never emitted a design
        # spec at all, which would have dragged the arm's mean regret from 11
        # real campaigns to 32 mostly-empty ones and reported it as a loss to
        # the rule bar. Excluded from every rate below and counted separately.
        "truncated": r.get("termination") == "budget_cap_reached",
        # A campaign whose LLM steps ALL failed at the transport layer is not a
        # measurement of the controller either. Measured: when the API credit
        # balance ran out, Haiku's A4_PRIOR and A5_REFLECTION arms each wrote
        # 32 campaigns in which every call returned HTTP 400 and every step
        # recorded StepStatus.ERROR -- at zero cost, because the requests were
        # rejected before billing. Those rows otherwise look like ordinary
        # "no_improvement" campaigns: an ERROR step increments neither the
        # decline counter nor the malformed counter, so the arm would have been
        # reported with a real regret it never earned.
        "no_llm_response": bool(
            (r.get("controller") or "").startswith("llm:")
            and (r.get("iterations") or [])
            and all(
                st.get("status") == "ERROR"
                for it in r["iterations"] for st in (it.get("steps") or [])
            )
        ),
        "qualified": int(r.get("n_qualified") or 0) > 0,
        "best_outcome": r.get("best_outcome"),
        "contract_holding": r.get("best_outcome") == "CONTRACT_HOLDING_REPAIR",
        "resolved_broken": r.get("best_outcome") == "DEFECT_RESOLVED_CONTRACT_BROKEN",
        "achieved_utility": achieved,
        "latent_utility": r.get("_latent_utility"),
        "latent_contract_holding": r.get("_latent_contract_holding"),
        "oracle_utility": best_u_possible,
        "noop_utility": noop_u.get(pid, float("nan")),
        "regret": regret,
        "n_iterations": int(r.get("n_iterations") or 0),
        "utility_by_iteration": r.get("_utility_by_iteration") or [],
        "spend_truncated": r.get("termination") == "budget_cap_reached",
        # Relaxations are only meaningful for an arm that HAS a contract.
        # Traces recorded before the controller guarded this can carry
        # RelaxationEvents from arms 1-2, where `active_contract` was None and
        # each "relaxation" weakened an empty contract. Those are zeroed here
        # and counted separately, so historical rows are corrected on re-score
        # rather than needing a re-run that costs API spend.
        "n_relaxations": (
            int(r.get("n_relaxations") or 0)
            if str(r.get("arm") or "") in ("A3_CONTRACT", "A4_PRIOR",
                                           "A5_REFLECTION")
            else 0
        ),
        "relax_without_contract": (
            int(r.get("n_relax_without_contract") or 0)
            + (int(r.get("n_relaxations") or 0)
               if str(r.get("arm") or "") in ("A1_FRESH", "A2_CONDITIONING")
               else 0)
        ),
        "n_declines": int(r.get("n_declines") or 0),
        "n_malformed": int(r.get("n_malformed") or 0) - _n_api_error_steps(r),
        "n_api_errors": int(r.get("n_errors") or 0) or _n_api_error_steps(r),
        "termination": r.get("termination"),
        "llm_calls": int(r.get("total_llm_calls") or 0),
        "cost_usd": float(r.get("total_cost_usd") or 0.0),
        "seconds": float(r.get("total_seconds") or 0.0),
        "seeds_spent": int(r.get("seeds_spent") or 0),
        "n_scope_leaks_caught": int(svc.get("n_scope_leaks_caught") or 0),
        "n_budget_filtered": int(svc.get("n_budget_filtered") or 0),
        "n_overwide_masks": int(svc.get("n_overwide_masks") or 0),
        "n_compile_rejected": int(svc.get("n_compile_rejected") or 0),
        "n_proposed": int(svc.get("n_proposed") or 0),
        "n_escalations": sum(
            1 for it in its
            if (it.get("allocation") or {}).get("action") == "ESCALATE"
        ),
        "n_edit_actions": sum(
            1 for it in its
            if (it.get("allocation") or {}).get("action") == "EDIT"
        ),
        "diagnosis_correct": r.get("_diagnosis_correct"),
        "search_utility": r.get("_search_utility"),
        "heldout_utility": r.get("_heldout_utility"),
        "heldout_qualifies": r.get("_heldout_qualifies"),
        "heldout_contract_holding": r.get("_heldout_contract_holding"),
        "heldout_gap": (
            r["_search_utility"] - r["_heldout_utility"]
            if r.get("_search_utility") is not None
            and r.get("_heldout_utility") is not None else float("nan")
        ),
        "final_strain": card.get("max_strain"),
        "final_bsa": card.get("bsa"),
        "final_iptm": card.get("iptm"),
    }


def enrich(rows: list[dict], parents_full: Path, oracle_path: Path) -> None:
    """Recompute utilities and the held-out protocol from the saved traces.

    Done here rather than in the campaign runner on purpose: the utility
    definition and the held-out protocol are SCORING decisions, and if they
    were baked into the traces at run time then changing either would mean
    re-running every campaign -- including the API ones, which cost money.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lp2cp.backends.oracle import OracleBackend
    from lp2cp.enums import Operator, parse_enum
    from lp2cp.envbuild.build_parents import load_parents
    from lp2cp.envbuild.enumerate_actions import reference_contract, utility
    from lp2cp.qualification import qualify
    from lp2cp.types import ScoreCard

    parents = {p.parent_id: p for p in load_parents(parents_full, with_labels=True)}
    # Index the oracle table by (parent, intervention) so a campaign's chosen
    # cell can be looked up without re-running the surrogate.
    cells: dict[tuple[str, str], dict] = {}
    for line in oracle_path.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            cells[(c["parent_id"], c["intervention_id"])] = c

    for r in rows:
        p = parents.get(r.get("parent_id"))
        if p is None:
            continue
        card_row = r.get("best_card")
        if card_row:
            card = ScoreCard(
                **{**card_row,
                   "strain_locus": tuple(card_row.get("strain_locus") or ()),
                   "hbonds_retained": tuple(card_row.get("hbonds_retained") or ())}
            )
            ref = reference_contract(p)
            ok = r.get("best_outcome") == "CONTRACT_HOLDING_REPAIR"
            r["_achieved_utility"] = utility(card, p, ok, False)
            r["_search_utility"] = r["_achieved_utility"]
        else:
            r["_achieved_utility"] = float("nan")

        # Held-out protocol on the campaign's own final intervention.
        last = None
        for it in reversed(r.get("iterations") or []):
            if it.get("best_cand") and it.get("spec"):
                last = it
                break
        if last is not None:
            spec = last["spec"]
            ops = []
            seg = spec.get("rebuild_segment")
            if seg:
                lo, hi = (seg if isinstance(seg, (list, tuple))
                          else (seg.get("start"), seg.get("end")))
                span = (tuple(range(lo, hi + 1)) if lo <= hi
                        else tuple(list(range(lo, p.n_pos + 1)) + list(range(1, hi + 1))))
                ops.append((Operator.REBUILD_SEGMENT, span))
            if spec.get("open_positions"):
                ops.append((Operator.SUBSTITUTE, tuple(spec["open_positions"])))
            if ops:
                from lp2cp.envbuild import surrogate as S

                iid = "|".join(
                    f"{op.value}:{','.join(map(str, ps))}" for op, ps in ops)
                held = S.evaluate(
                    p, tuple(ops), iid,
                    int((r.get("best_card") or {}).get("edit_distance") or 0),
                    n_seeds=S.CFG.heldout_seeds, heldout=True,
                )
                ref = reference_contract(p)
                hok, _ = ref.holds(held, p.cp_sequence, p.cp_sequence)
                hv = qualify(held, p.native_bsa, p.k_budget)
                r["_heldout_utility"] = utility(held, p, hok and hv.qualifies, False)

                # The LATENT (noise-free) value of the intervention the arm
                # actually committed to. This is what regret must be measured
                # against, and the reason is an inconsistency rather than a
                # preference: every cell in the oracle table is scored on its
                # LATENT card, so comparing an arm's OBSERVED card against it
                # was comparing two different quantities. It let an arm pick a
                # high-variance spec, take the luckiest of six noisy draws, and
                # record a utility above the table's own maximum -- measured,
                # Sonnet "beat" the oracle controller 27.05 to 12.57 at search
                # time and then scored -0.86 on independent re-evaluation.
                #
                # The agent still CHOOSES under noise, which is realistic. It
                # is simply no longer GRADED on its own noisy estimate.
                lat = S.latent_card(
                    p, tuple(ops), iid,
                    int((r.get("best_card") or {}).get("edit_distance") or 0),
                )
                lok, _ = ref.holds(lat, p.cp_sequence, p.cp_sequence)
                lv = qualify(lat, p.native_bsa, p.k_budget)
                r["_latent_utility"] = utility(lat, p, lok and lv.qualifies, False)
                r["_latent_contract_holding"] = bool(lv.qualifies and lok)
                # Stored separately from the gap because the GAP IS CONFOUNDED
                # WITH ACHIEVEMENT: you can only lose at held-out what you
                # gained at search time, so a controller that does nothing
                # scores a perfect gap of ~0. `noop` measured 0.76 against
                # Sonnet's 29.45, which reads backwards. These two say whether
                # the frozen shortlist actually SURVIVED the protocol, which is
                # the question the gap was standing in for.
                r["_heldout_qualifies"] = bool(hv.qualifies)
                r["_heldout_contract_holding"] = bool(hv.qualifies and hok)

        # Per-iteration best utility, so a controller can be compared against
        # another one TRUNCATED TO THE SAME NUMBER OF ITERATIONS.
        #
        # This is needed, not a refinement. An API arm stops when its spend
        # ledger stops it: measured here, 21 of 32 sonnet-5-5 campaigns
        # terminated `budget_cap_reached` after a mean of 1.47 iterations,
        # while every rule controller ran to natural termination. Comparing
        # those directly would score "ran out of money" as "worse controller".
        best = utility(
            ScoreCard(**{**p.scorecard0.__dict__}), p, False, False
        )
        prog = []
        for it in r.get("iterations") or []:
            bc = it.get("best_card")
            if bc:
                c = ScoreCard(
                    **{**bc,
                       "strain_locus": tuple(bc.get("strain_locus") or ()),
                       "hbonds_retained": tuple(bc.get("hbonds_retained") or ())}
                )
                u = utility(
                    c, p,
                    it.get("outcome") == "CONTRACT_HOLDING_REPAIR", False,
                )
                best = max(best, u)
            prog.append(best)
        r["_utility_by_iteration"] = prog

        # Did the agent's diagnosis match the frozen verdict's own class?
        first = next(
            (it for it in (r.get("iterations") or []) if it.get("diagnosis")), None
        )
        if first:
            r["_diagnosis_correct"] = (
                first["diagnosis"]["defect_class"] == p.true_defect_class.value
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(ms: list[dict]) -> dict:
    ok = [m for m in ms if not m["error"] and not m["truncated"]
          and not m["no_llm_response"]]
    reg = [m["regret"] for m in ok if m["regret"] == m["regret"]]
    gaps = [m["heldout_gap"] for m in ok if m["heldout_gap"] == m["heldout_gap"]]
    diag = [m["diagnosis_correct"] for m in ok if m["diagnosis_correct"] is not None]
    lo, hi = boot_ci(reg)
    return {
        "n": len(ok),
        "n_submitted": len(ms),
        "n_errors": sum(1 for m in ms if m["error"]),
        "n_truncated_by_spend_cap": sum(1 for m in ms if m["truncated"]),
        "n_no_llm_response": sum(1 for m in ms if m["no_llm_response"]),
        "contract_holding_rate": mean(m["contract_holding"] for m in ok),
        "qualified_rate": mean(m["qualified"] for m in ok),
        "resolved_broken_rate": mean(m["resolved_broken"] for m in ok),
        "mean_regret": mean(reg),
        "median_regret": median(reg),
        "regret_ci95": [round(lo, 3), round(hi, 3)],
        "oracle_hit_rate": mean(1.0 if m["regret"] < 1e-6 else 0.0 for m in ok
                                if m["regret"] == m["regret"]),
        "beat_noop_rate": mean(
            1.0 if m["achieved_utility"] > m["noop_utility"] + 1e-9 else 0.0
            for m in ok
            if m["achieved_utility"] == m["achieved_utility"]
            and m["noop_utility"] == m["noop_utility"]
        ),
        "mean_heldout_gap": mean(gaps),
        # Survival of the frozen shortlist under the held-out protocol. Unlike
        # the gap, this is not confounded with how much the controller achieved
        # at search time -- a controller that never qualified scores 0 here
        # rather than a flattering gap of ~0.
        "heldout_contract_holding_rate": mean(
            1.0 if m["heldout_contract_holding"] else 0.0 for m in ok
            if m["heldout_contract_holding"] is not None),
        "search_to_heldout_survival": (
            mean(1.0 if m["heldout_contract_holding"] else 0.0
                 for m in ok if m["contract_holding"])
            if any(m["contract_holding"] for m in ok) else float("nan")),
        "diagnosis_accuracy": mean(diag) if diag else None,
        "mean_iterations": mean(m["n_iterations"] for m in ok),
        # What fraction of campaigns were stopped by the SPEND LEDGER rather
        # than by the controller's own termination rule. A high value means the
        # arm's headline is an artefact of budget, and the compute-matched
        # comparison below is the one to read.
        "spend_truncated_rate": mean(m["spend_truncated"] for m in ok),
        "relaxation_rate": mean(1.0 if m["n_relaxations"] else 0.0 for m in ok),
        "mean_relaxations": mean(m["n_relaxations"] for m in ok),
        "relax_without_contract": sum(m["relax_without_contract"] for m in ok),
        "decline_rate": mean(1.0 if m["n_declines"] else 0.0 for m in ok),
        "malformed_rate": mean(1.0 if m["n_malformed"] > 0 else 0.0 for m in ok),
        # Fraction of campaigns with ANY transport/billing failure. An arm
        # above ~0.5 here is an infrastructure record, not a model result, and
        # the table marks it so it cannot be quoted as one.
        "api_error_rate": mean(1.0 if m["n_api_errors"] else 0.0 for m in ok),
        "escalation_rate": mean(1.0 if m["n_escalations"] else 0.0 for m in ok),
        "mean_seeds_spent": mean(m["seeds_spent"] for m in ok),
        "scope_leaks_caught": sum(m["n_scope_leaks_caught"] for m in ok),
        "budget_filtered": sum(m["n_budget_filtered"] for m in ok),
        "overwide_masks": sum(m["n_overwide_masks"] for m in ok),
        "compile_rejected": sum(m["n_compile_rejected"] for m in ok),
        "n_proposed": sum(m["n_proposed"] for m in ok),
        "mean_llm_calls": mean(m["llm_calls"] for m in ok),
        "total_cost_usd": round(sum(m["cost_usd"] for m in ok), 6),
        "mean_seconds": mean(m["seconds"] for m in ok),
        "terminations": dict(sorted(
            ((k, v) for k, v in
             __import__("collections").Counter(
                 m["termination"] for m in ok).items()),
            key=lambda kv: -kv[1])),
    }


def paired(a: list[dict], b: list[dict], field: str = "regret",
           match_iterations: bool = False) -> dict:
    """Paired comparison of two arms over the parents both completed.

    `match_iterations=True` truncates B to the number of iterations A actually
    ran on that parent, which is the only fair comparison when A was stopped by
    a spend cap rather than by its own termination rule. B's regret is then
    recomputed from its own per-iteration progress curve.
    """
    # Truncated campaigns are dropped from BOTH sides before pairing, so the
    # shared set is parents that each controller actually got to finish.
    ia = {m["parent_id"]: m for m in a
          if not m["error"] and not m["truncated"] and not m["no_llm_response"]}
    ib = {m["parent_id"]: m for m in b
          if not m["error"] and not m["truncated"] and not m["no_llm_response"]}
    shared = sorted(set(ia) & set(ib))
    deltas = []
    for pid in shared:
        va, vb = ia[pid][field], ib[pid][field]
        if match_iterations and field == "regret":
            curve = ib[pid].get("utility_by_iteration") or []
            k = max(1, int(ia[pid]["n_iterations"] or 1))
            if curve:
                vb = ib[pid]["oracle_utility"] - curve[min(k, len(curve)) - 1]
        if va == va and vb == vb:
            deltas.append(va - vb)
    if not deltas:
        return {"n_shared": 0}
    p, w, l, t = wilcoxon_signed_rank(deltas)
    med, mu = median(deltas), mean(deltas)
    # Direction, when the comparison is significant. The MEDIAN is often exactly
    # zero here -- most parents tie, because many have no winnable cell and both
    # controllers end on the same no-op -- so reading direction off the median
    # alone reported "tie" for a comparison at p=0.0015 with 13 losses and 0
    # wins. Fall through to the mean, then to the win/loss counts.
    if med < -1e-12 or (abs(med) <= 1e-12 and (mu < -1e-12 or (mu == 0 and w > l))):
        direction = "a_better"
    elif med > 1e-12 or (abs(med) <= 1e-12 and (mu > 1e-12 or (mu == 0 and l > w))):
        direction = "b_better"
    else:
        direction = "tie"
    return {
        "n_shared": len(deltas),
        "median_delta": round(med, 4),
        "mean_delta": round(mu, 4),
        # Named for meaning, not for the sign of the subtraction: a win for A is
        # A having LOWER regret than B on that parent.
        "a_wins": w, "a_losses": l, "ties": t,
        "p_wilcoxon": None if p != p else round(p, 5),
        "verdict": (
            "insufficient_n" if p != p
            else direction if p < 0.05
            else "tie"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaigns", required=True)
    ap.add_argument("--parents", required=True, help="parents_full.json")
    ap.add_argument("--oracle-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-manifest", default="",
                    help="bench/env_manifest.json; scoring refuses to run if "
                         "the live surrogate differs from the frozen one")
    ap.add_argument("--allow-env-drift", action="store_true",
                    help="inspect a drifted tree. NEVER use for reported numbers.")
    ap.add_argument("--baseline", default="rule:rim_first_closure",
                    help="the arm_key every LLM arm is tested against")
    args = ap.parse_args()

    if args.allow_env_drift:
        print("[score] *** --allow-env-drift: numbers below may be measured "
              "against a mismatched response model. NOT REPORTABLE. ***")
    else:
        assert_env_frozen(Path(args.env_manifest) if args.env_manifest else None)

    cdir = Path(args.campaigns)
    rows = load_campaigns(cdir)
    if not rows:
        print(f"[score] GATE FAILED: no campaigns__*.jsonl under {cdir}. Exiting 0.")
        return 0
    print(f"[score] loaded {len(rows)} campaigns from {cdir}")

    # Refuse to pool campaigns from different ENVIRONMENTS.
    #
    # Every campaign row carries the env_hash it ran under. Measured: after the
    # surrogate was upgraded to v2 and the benchmark rebuilt, the resumable
    # runner skipped every parent already present in the campaigns directory,
    # so 480 v1 rows sat alongside 96 v2 rows and were about to be scored
    # against the v2 oracle table. Regret would have been computed by
    # subtracting one environment's achieved utility from another's ceiling.
    #
    # This is the same class of failure assert_frozen() guards for the
    # qualification function, and it gets the same treatment: a hard stop that
    # names the mixture, not a warning in a log nobody reads.
    env_counts = Counter(r.get("env_hash") or "<none>" for r in rows)
    bench_env = ""
    man = Path(args.oracle_table).parent / "env_manifest.json"
    if man.exists():
        bench_env = json.loads(man.read_text()).get("env_hash", "")
    if bench_env and set(env_counts) - {bench_env, "<none>"}:
        lines = "\n".join(f"      {h}: {n} rows" for h, n in env_counts.most_common())
        raise SystemExit(
            f"\nREFUSING TO SCORE: campaigns from more than one environment.\n"
            f"  oracle table / bench env_hash: {bench_env}\n"
            f"  env_hashes found in {cdir}:\n{lines}\n\n"
            f"  Rows produced under different surrogate versions are not\n"
            f"  poolable: regret would subtract one environment's achieved\n"
            f"  utility from another's ceiling. Move the stale rows aside (the\n"
            f"  runner is resumable, so it skips parents already present) and\n"
            f"  re-run, or score each environment's campaigns separately.\n"
        )

    enrich(rows, Path(args.parents), Path(args.oracle_table))
    oracle = load_oracle(Path(args.oracle_table))
    noop_u = oracle_noop(Path(args.oracle_table))
    noop_latent = dict(noop_u)
    ms = [campaign_metrics(r, oracle, noop_u, noop_latent) for r in rows]

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for m in ms:
        by_arm[m["arm_key"]].append(m)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_campaign.jsonl").open("w") as fh:
        for m in ms:
            fh.write(json.dumps(m, default=str) + "\n")

    summary = {k: summarise(v) for k, v in sorted(by_arm.items())}

    # Split consistency: the by-TARGET held-out split, declared before the run.
    by_split: dict[str, dict] = {}
    for sp in ("dev", "test"):
        by_split[sp] = {
            k: summarise([m for m in v if m["split"] == sp])
            for k, v in sorted(by_arm.items())
        }

    comparisons = {}
    base = by_arm.get(args.baseline)
    if base:
        for k, v in sorted(by_arm.items()):
            if k == args.baseline:
                continue
            comparisons[f"{k} vs {args.baseline}"] = paired(v, base, "regret")
            comparisons[f"{k} vs {args.baseline} [compute-matched]"] = paired(
                v, base, "regret", match_iterations=True)
    # The two ablations the doc names, if both arms ran.
    for a, b, label in (
        ("A3_CONTRACT", "A2_CONDITIONING", "2->3 preservation contract"),
        ("A4_PRIOR", "A3_CONTRACT", "3->4 history prior"),
        ("A5_REFLECTION", "A4_PRIOR", "4->5 distilled reflections"),
    ):
        for model in {m["model_id"] for m in ms if m["model_id"]}:
            ka = next((k for k in by_arm if k.startswith(f"{model}|{a}|")), None)
            kb = next((k for k in by_arm if k.startswith(f"{model}|{b}|")), None)
            if ka and kb:
                comparisons[f"[{model}] {label}"] = paired(by_arm[ka], by_arm[kb], "regret")

    result = {
        "n_campaigns": len(rows),
        "n_parents": len({m["parent_id"] for m in ms}),
        "arms": summary,
        "by_split": by_split,
        "paired_comparisons": comparisons,
        "baseline": args.baseline,
        "WARNING": (
            "Regret, qualification and every rate here are measured against a "
            "DECLARED SURROGATE response model (lp2cp/envbuild/surrogate.py) "
            "applied to real measured structures. They quantify CONTROLLER "
            "quality, not biophysical success. No sequence here is a binding "
            "claim."
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, default=str))

    # --- the table ------------------------------------------------------
    hdr = (f"{'arm':42s} {'n':>3s} {'cap':>4s} {'CHR':>6s} {'hoCHR':>6s} {'surv':>5s} "
           f"{'regret':>8s} {'orc':>5s} {'relax':>6s} {'decl':>5s} {'malf':>5s} "
           f"{'err':>5s} {'iters':>6s} {'calls':>6s} {'$':>8s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, s in sorted(summary.items(), key=lambda kv: kv[1]["mean_regret"]):
        surv = s["search_to_heldout_survival"]
        print(f"{k[:42]:42s} {s['n']:3d} "
              f"{s['n_truncated_by_spend_cap']:4d} "
              f"{s['contract_holding_rate']:6.3f} "
              f"{s['heldout_contract_holding_rate']:6.3f} "
              f"{('  -  ' if surv != surv else f'{surv:5.2f}')} "
              f"{s['mean_regret']:8.3f} {s['oracle_hit_rate']:5.2f} "
              f"{s['relaxation_rate']:6.3f} "
              f"{s['decline_rate']:5.2f} {s['malformed_rate']:5.2f} "
              f"{s['api_error_rate']:5.2f} "
              f"{s['mean_iterations']:6.2f} "
              f"{s['mean_llm_calls']:6.1f} {s['total_cost_usd']:8.4f}")
    print("\nn     = campaigns the controller FINISHED (truncated ones excluded)")
    print("cap   = campaigns the spend ledger cut short; these are stubs, not")
    print("        measurements, and are excluded from every column here")
    print("CHR   = contract-holding repair rate at search time")
    print("hoCHR = same after the held-out protocol (fresh seeds, 2nd predictor)")
    print("surv  = of the campaigns that repaired at search time, the fraction")
    print("        still repairing at held-out. The doc's reportable number, in")
    print("        the form that is NOT confounded with doing nothing.")
    print("orc   = oracle-hit rate (regret exactly 0)")
    print("err   = campaigns hit by a TRANSPORT/BILLING failure. An arm with a")
    print("        high err is an infrastructure record, NOT a model result.")
    print("cap   = campaigns stopped by the SPEND LEDGER, not by the controller.")
    print("        Where cap is large the arm's mean is an artefact of budget;")
    print("        read the [compute-matched] comparison instead, which truncates")
    print("        the baseline to the same iteration count on each parent.")

    if comparisons:
        print(f"\n{'comparison  (A vs B; W/L/T is for A, lower regret wins)':62s} "
              f"{'n':>3s} {'medD':>7s} {'meanD':>7s} {'W/L/T':>10s} "
              f"{'p':>8s}  verdict")
        print("-" * 116)
        for k, c in comparisons.items():
            if not c.get("n_shared"):
                continue
            print(f"{k[:62]:62s} {c['n_shared']:3d} {c['median_delta']:7.3f} "
                  f"{c['mean_delta']:7.3f} "
                  f"{c['a_wins']:3d}/{c['a_losses']:3d}/{c['ties']:3d} "
                  f"{str(c['p_wilcoxon']):>8s}  {c['verdict']}")

    print(f"\n[score] wrote {out_dir}/results.json and per_campaign.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
