"""The control loop. Eight steps per iteration, three of them LLM-owned.

    1 OBSERVE   assemble the diagnostics for the current candidate   service
    2 DIAGNOSE  numeric diagnostics -> a typed DiagnosisRecord       LLM
    3 SPECIFY   the diagnosis -> a DesignSpec (a constraint set)     LLM
    4 COMPILE   validate; enforce the budget at the filter stage     service
    5 ACT       the backend proposes inside the mask                 backend
    6 VERIFY    diff every candidate vs the LP and vs the mask       service
    7 SCORE     the frozen qualification function                    service
    8 ALLOCATE  edit again, escalate evidence, relax, or stop        LLM

Loop skeleton after PDAgent, which the design doc picks because "that is what
reviewers will compare against" -- its five-phase ReAct shape (THINK, PLAN,
ACT, OBSERVE, REFLECT) maps onto steps 2-3, 4, 5, 6-7 and 8. What is NOT taken
from PDAgent is its action space: there the LLM emits point mutations directly,
and that is the line this design forbids crossing.

Termination, from the doc: budget exhaustion, two consecutive iterations with
no contract-holding improvement, or n qualifying designs. The shortlist is then
frozen BY THE QUALIFICATION FUNCTION with no agent input into ranking -- which
is why `best_card` is chosen on the frozen utility and never on anything the
agent said.

Every LLM step can decline or come back malformed. Both are recorded as step
outcomes with a DECLARED DETERMINISTIC FALLBACK, never as an exception and
never as a silent default:

  DIAGNOSE declines  -> fall back to the frozen verdict's own defect class.
                        This is the honest fallback: the qualification function
                        already knows which clause failed, so the campaign can
                        continue, and the fallback is COUNTED so that an arm
                        which declined its way to a result is visible as such.
  SPECIFY declines   -> the iteration is abandoned, not guessed. Specifying is
                        the step under test; substituting a rule here would
                        make the arm a measurement of the rule.
  ALLOCATE declines  -> EDIT while budget remains, else STOP. The least
                        informative continuation.
"""

from __future__ import annotations

import time

from . import prompts
from .backends.oracle import OracleBackend
from .enums import (
    Arm,
    DefectClass,
    Operator,
    RepairOutcome,
    StepStatus,
    arm_flags,
)
from .envbuild.enumerate_actions import edit_cost, reference_contract, utility
from .llm.base import LLMClient, Reply, Usage
from .memory import Archive, episode_from
from .qualification import QUAL_HASH
from .service import ExecutionService
from .types import (
    AllocationDecision,
    CampaignRecord,
    Candidate,
    DesignSpec,
    DiagnosisRecord,
    IterationRecord,
    ParentProblem,
    PreservationContract,
    RelaxationEvent,
    ScoreCard,
    StepRecord,
    replace,
)

# Operators the agent may choose at step 3. STOP, RELAX_CONTRACT and
# EVIDENCE_ESCALATION are step-8 actions, not design-problem operators, so they
# are not offered here -- offering them twice would let an agent "specify" a
# stop and make the two steps' counts incomparable.
SPEC_OPERATORS = [
    Operator.SUBSTITUTE,
    Operator.REBUILD_SEGMENT,
    Operator.NARROW_PIVOTS,
    Operator.ANCHOR_SHIFT,
    Operator.RESTRAIN_AND_REPACK,
]


def ask_step(
    llm, step: str, system: str, brief: str, parse, schema: str, usage: Usage,
    max_retries: int = 1, save_raw: bool = False,
) -> tuple[object | None, StepRecord]:
    """One LLM step, with the error-correction re-prompt and strict parsing.

    `parse` returns (value, status-string); a status of "ok"/"ok_loose"
    accepts. Anything else triggers one re-prompt that restates the schema
    and nothing else, which is Agent Rosetta's explicit error-correction
    call. After that the step is MALFORMED -- not defaulted.
    """
    messages = [{"role": "user", "content": brief}]
    transcript: list[str] = []
    n_calls = 0
    t0 = time.time()
    last = "no_tag"
    for attempt in range(max_retries + 1):
        reply: Reply = llm.complete(system, messages)
        n_calls += 1
        usage.add(step, reply)
        if reply.status in (StepStatus.DECLINED, StepStatus.ERROR,
                           StepStatus.BUDGET_CAP):
            return None, StepRecord(
                step=step, status=reply.status, n_calls=n_calls,
                n_retries=attempt, seconds=round(time.time() - t0, 2),
                cost_usd=reply.cost_usd, input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens, detail=reply.detail,
                raw=reply.text[:4000] if save_raw else "",
            )
        transcript.append(reply.text)
        value, last = parse(reply.text)
        if value is not None:
            return value, StepRecord(
                step=step, status=StepStatus.OK, n_calls=n_calls,
                n_retries=attempt, seconds=round(time.time() - t0, 2),
                cost_usd=reply.cost_usd, input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens, detail=last,
                raw="\n--RETRY--\n".join(transcript)[:4000] if save_raw else "",
            )
        if attempt < max_retries:
            messages = messages + [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": prompts.RETRY_TEMPLATE.format(
                    reason=prompts.RETRY_REASON.get(last, last), schema=schema)},
            ]
    return None, StepRecord(
        step=step, status=StepStatus.MALFORMED, n_calls=n_calls,
        n_retries=max_retries, seconds=round(time.time() - t0, 2),
        detail=last,
        raw="\n--RETRY--\n".join(transcript)[:4000] if save_raw else "",
    )


class Controller:
    """One campaign: one parent, one arm, one model."""

    def __init__(
        self,
        llm: LLMClient,
        arm: Arm,
        archive: Archive | None = None,
        max_iters: int = 5,
        n_qualifying_target: int = 1,
        patience: int = 2,
        max_retries: int = 1,
        two_call_specify: bool = True,
        default_seeds: int = 5,
        seed_budget: int = 60,
        max_eval_seeds: int = 20,
        backend_fidelity: str = "declared",
        save_raw: bool = False,
        inject_contract: bool = False,
    ):
        self.llm = llm
        self.arm = arm
        self.flags = arm_flags(arm)
        self.archive = archive or Archive()
        self.max_iters = max_iters
        self.n_qualifying_target = n_qualifying_target
        self.patience = patience
        self.max_retries = max_retries
        # Agent Rosetta's two-call action pattern: choose the action, then
        # re-prompt with that action's documentation appended to fill in its
        # parameters. Switchable because it doubles the call count on the
        # SPECIFY step and an API arm is cost-bound.
        self.two_call_specify = two_call_specify
        self.default_seeds = default_seeds
        self.seed_budget = seed_budget
        self.max_eval_seeds = max_eval_seeds
        self.backend_fidelity = backend_fidelity
        self.save_raw = save_raw
        # Restate the contract's protected positions inside the SPECIFY brief,
        # where the mask is actually chosen. Off by default so that runs made
        # before this existed stay reproducible; the scaffold tag records it.
        self.inject_contract = inject_contract
        self.scaffold_tag = (
            ('twocall' if two_call_specify else 'onecall')
            + ('+inject' if inject_contract else '')
        )

    # -- LLM plumbing -----------------------------------------------------
    def _ask(
        self, step: str, system: str, brief: str, parse, schema: str, usage: Usage,
    ) -> tuple[object | None, StepRecord]:
        """One LLM step. Delegates to the module-level `ask_step`.

        Extracted so that the tool-selection controller in lp2cp/toolselect.py
        runs the IDENTICAL retry and decline policy. Two copies of this policy
        would drift, and a difference in how a malformed reply is handled would
        show up as a difference in allocation quality between the two action
        spaces -- which is exactly the comparison those arms exist to make.
        """
        return ask_step(
            llm=self.llm, step=step, system=system, brief=brief, parse=parse,
            schema=schema, usage=usage, max_retries=self.max_retries,
            save_raw=self.save_raw,
        )

    # -- the loop ---------------------------------------------------------
    def run(self, parent: ParentProblem, env_hash: str = "") -> CampaignRecord:
        backend = OracleBackend(parent, fidelity=self.backend_fidelity)
        service = ExecutionService(
            parent=parent,
            backend=backend,
            enforce_contract=self.flags["contract"],
            max_eval_seeds=self.max_eval_seeds,
            seed_budget=self.seed_budget,
        )
        ref_contract = reference_contract(parent)
        usage = Usage()
        rec = CampaignRecord(
            # The scaffold is part of a run's identity: without it, two
            # scaffolds of the same (parent, arm, model) produce identical
            # campaign_ids and anything keyed on that id silently merges them.
            campaign_id=f"{parent.parent_id}__{self.arm.value}__"
                        f"{self.llm.model_id}__{self.scaffold_tag}",
            parent_id=parent.parent_id,
            cluster_id=parent.cluster_id,
            split=parent.split,
            arm=self.arm,
            controller=f"llm:{self.llm.model_id}",
            model_id=self.llm.model_id,
            qualification_hash=QUAL_HASH,
            env_hash=env_hash,
            k_budget=parent.k_budget,
        )

        cur_seq = parent.cp_sequence
        cur_card: ScoreCard = parent.scorecard0
        cur_cand = "C-000"
        best_u = utility(cur_card, parent, False, False)
        best_cand, best_card, best_outcome = cur_cand, cur_card, RepairOutcome.NO_CHANGE
        history: list[dict] = []
        no_improve = 0
        t_start = time.time()
        last_seeds = cur_card.n_seeds
        # The state the service needs to re-score a candidate on escalation.
        last_compiled = None
        last_candidate: Candidate | None = None

        for it in range(1, self.max_iters + 1):
            ir = IterationRecord(iteration=it, candidate_in=cur_cand)
            # --- step 1 OBSERVE (deterministic) --------------------------
            # The parent object the agent reads always carries the CURRENT
            # candidate's sequence and scorecard, so the diagnostics it reasons
            # over are the ones belonging to what it is about to repair.
            view = replace(parent, cp_sequence=cur_seq, scorecard0=cur_card)

            # --- step 2 DIAGNOSE (LLM) -----------------------------------
            system, brief = prompts.build_diagnose(
                view, cur_card, cur_cand, service, self.arm, history
            )
            schema = prompts.DIAGNOSE_ACTION.format(
                classes=" | ".join(
                    d.value for d in DefectClass if d != DefectClass.NO_DEFECT),
                n_pos=parent.n_pos,
            )

            def parse_diag(text, _view=view, _cand=cur_cand):
                obj, st = prompts.extract_json(text, "diagnosis")
                if obj is None:
                    return None, st
                try:
                    return DiagnosisRecord.parse(obj, _view.n_pos, _cand), "ok"
                except ValueError as exc:
                    msg = str(exc)
                    return None, (
                        "unknown_defect" if "DefectClass" in msg else "out_of_range"
                    )

            diagnosis, step = self._ask(
                "DIAGNOSE", system, brief, parse_diag, schema, usage
            )
            ir.steps.append(step)
            if diagnosis is None:
                # Declared fallback: the frozen verdict already names the
                # failing clause, so the campaign continues on it and the
                # fallback is counted. An arm that declined its way through is
                # visible in n_declines, not hidden in a result.
                v = service.verdict(cur_card)
                diagnosis = DiagnosisRecord(
                    candidate=cur_cand,
                    defect_class=v.defect_class,
                    defect_locus=v.defect_locus or tuple(
                        r.pos for r in parent.residues if r.ring_dist <= 1
                    ),
                    preserved_features=(),
                    evidence_refs=("fallback:frozen_verdict",),
                    confidence=0.0,
                )
                ir.steps[-1] = replace(step, fallback_used=True)
                if step.status == StepStatus.DECLINED:
                    rec.n_declines += 1
                elif step.status == StepStatus.MALFORMED:
                    rec.n_malformed += 1
                elif step.status == StepStatus.ERROR:
                    # Transport / rate-limit / billing failure, not a bad
                    # action by the model. See the SPECIFY branch below.
                    rec.n_errors += 1
                elif step.status == StepStatus.BUDGET_CAP:
                    rec.termination = "budget_cap_reached"
                    rec.iterations.append(ir)
                    break
            ir.diagnosis = diagnosis

            # Arm 3+ installs a preservation contract derived from what the
            # agent said it would preserve. Arms 1-2 never get one, which is
            # the 2->3 mechanism: not a different prompt, a different
            # guarantee at the service.
            if self.flags["contract"] and service.active_contract is None:
                service.active_contract = self._contract_from(diagnosis, view)

            # --- step 3 SPECIFY (LLM, two calls) ------------------------
            spec, sstep = self._specify(
                view, cur_card, diagnosis, service, usage,
                prior=self._prior_for(diagnosis, view),
                reflections=self._reflections_for(diagnosis, view),
            )
            ir.steps.extend(sstep)
            if spec is None:
                # No fallback here, deliberately: specifying the design problem
                # is the step under test, and substituting a rule would turn
                # the arm into a measurement of that rule.
                worst = [s for s in sstep if s.status != StepStatus.OK]
                if any(s.status == StepStatus.DECLINED for s in worst):
                    rec.n_declines += 1
                elif any(s.status == StepStatus.BUDGET_CAP for s in worst):
                    rec.termination = "budget_cap_reached"
                    rec.iterations.append(ir)
                    break
                elif any(s.status == StepStatus.ERROR for s in worst):
                    # A transport, rate-limit or BILLING failure is not the
                    # model emitting a bad action. Measured: when the API
                    # credit balance ran out, every call returned HTTP 400 and
                    # two whole arms were reported as 100% "malformed", which
                    # reads as the model being incapable of the task rather
                    # than as the account being empty. Errors get their own
                    # counter and an arm dominated by them is not a result.
                    rec.n_errors += 1
                    ir.notes = (ir.notes + ";" if ir.notes else "") + "api_error"
                else:
                    rec.n_malformed += 1
                ir.outcome = RepairOutcome.NO_CHANGE
                ir.notes = "specify_failed:iteration_abandoned"
                ir.k_spent_after = service.k_spent
                rec.iterations.append(ir)
                no_improve += 1
                if no_improve >= self.patience:
                    rec.termination = "no_improvement"
                    break
                continue
            ir.spec = spec

            # --- step 4 COMPILE + ENFORCE (deterministic) ----------------
            compiled, reason = service.compile_spec(spec)
            ir.compile_ok = compiled is not None
            ir.compile_reason = reason
            if compiled is None:
                ir.outcome = RepairOutcome.INFEASIBLE
                ir.notes = f"compile_rejected:{reason}"
                service.ledger.reject(reason.split(":")[0])
                ir.k_spent_after = service.k_spent
                rec.iterations.append(ir)
                no_improve += 1
                if no_improve >= self.patience:
                    rec.termination = "no_improvement"
                    break
                continue
            ir.overwide_mask = compiled.overwide_mask

            # --- steps 5, 6, 7 ------------------------------------------
            survivors, info = service.act_and_score(
                compiled, spec.n_eval_seeds, cur_seq
            )
            ir.n_proposed = info.get("n_proposed", 0)
            ir.n_budget_filtered = info.get("n_budget_filtered", 0)
            ir.n_scope_leaks = info.get("n_scope_leaks", 0)
            ir.n_scored = info.get("n_scored", 0)
            ir.seeds_spent = info.get("seeds", 0)
            last_seeds = max(1, info.get("seeds", last_seeds))
            if info.get("stop"):
                ir.notes = info["stop"]
                rec.termination = info["stop"]
                rec.iterations.append(ir)
                break

            # The shortlist is frozen by the qualification function with no
            # agent input into ranking: the pick is argmax of the frozen
            # utility, which the agent cannot see or influence.
            accepted: list[tuple[Candidate, ScoreCard, float, RepairOutcome, tuple]] = []
            for cand, card in survivors:
                outcome, violations = service.classify(card, cand.sequence, ref_contract)
                ok_local, local_viol = service.accepts(card, cand.sequence)
                if not ok_local:
                    # Arm 3+ refuses a candidate that breaks its own contract
                    # even when the defect was resolved. This single branch is
                    # the 2->3 ablation.
                    continue
                u = utility(card, parent, outcome == RepairOutcome.CONTRACT_HOLDING_REPAIR,
                            False)
                accepted.append((cand, card, u, outcome, violations + local_viol))

            if not accepted:
                ir.outcome = RepairOutcome.NO_CHANGE
                ir.notes = (
                    "all_candidates_rejected:"
                    + (f"scope_leaks={ir.n_scope_leaks},"
                       f"budget={ir.n_budget_filtered},"
                       f"contract={len(survivors)}")
                )
                ir.k_spent_after = service.k_spent
                rec.iterations.append(ir)
                no_improve += 1
                if no_improve >= self.patience:
                    rec.termination = "no_improvement"
                    break
                continue

            cand, card, u, outcome, violations = max(accepted, key=lambda t: t[2])
            ir.best_cand = cand.cand_id
            ir.best_card = card
            ir.outcome = outcome
            ir.contract_violations = tuple(violations)
            last_compiled, last_candidate = compiled, cand

            # Archive in every arm. Storing alone conditions nothing.
            self.archive.write(
                episode_from(
                    parent, it, diagnosis, compiled.groups, outcome,
                    cur_card, card,
                    sum(edit_cost(op, ps) for op, ps in compiled.groups),
                    bool(violations),
                )
            )

            improved = u > best_u + 1e-9
            if improved:
                best_u, best_cand, best_card, best_outcome = u, cand.cand_id, card, outcome
                service.commit(cand, card)
                cur_seq, cur_card, cur_cand = cand.sequence, card, cand.cand_id
                no_improve = 0
            else:
                no_improve += 1
            ir.k_spent_after = service.k_spent

            if service.verdict(card).qualifies:
                rec.qualified.append(cand.cand_id)

            history.append(
                {
                    "iteration": it,
                    "operator": "+".join(op.value for op, _ in compiled.groups),
                    "locus": ",".join(map(str, compiled.touched)),
                    "cost": str(sum(edit_cost(op, ps) for op, ps in compiled.groups)),
                    # Deltas are against the ORIGINAL parent, not the previous
                    # iteration: the agent needs to see cumulative drift from
                    # what it started with, since the contract and the budget
                    # are both defined against that.
                    "d_strain": f"{card.max_strain - parent.scorecard0.max_strain:+.2f}",
                    "d_bsa": f"{card.bsa - parent.scorecard0.bsa:+.0f}",
                    "d_ddg": f"{card.ddg_bind - parent.scorecard0.ddg_bind:+.1f}",
                    "outcome": outcome.value,
                }
            )

            # --- step 8 ALLOCATE (LLM) ----------------------------------
            if len(rec.qualified) >= self.n_qualifying_target:
                rec.termination = "n_qualifying_reached"
                rec.iterations.append(ir)
                break
            if no_improve >= self.patience:
                rec.termination = "no_improvement"
                rec.iterations.append(ir)
                break
            if service.k_spent >= parent.k_budget:
                rec.termination = "budget_exhausted"
                rec.iterations.append(ir)
                break

            alloc, astep = self._allocate(
                view, card, best_cand, service, history, it, last_seeds, diagnosis, usage
            )
            ir.steps.append(astep)
            ir.allocation = alloc
            if astep.status == StepStatus.DECLINED:
                rec.n_declines += 1
            elif astep.status == StepStatus.MALFORMED:
                rec.n_malformed += 1
            elif astep.status == StepStatus.ERROR:
                rec.n_errors += 1
            elif astep.status == StepStatus.BUDGET_CAP:
                rec.termination = "budget_cap_reached"
                rec.iterations.append(ir)
                break

            if alloc.action == "STOP":
                rec.termination = "agent_stopped"
                rec.iterations.append(ir)
                break
            if alloc.action == "ESCALATE" and last_candidate is not None:
                # The doc's iteration 3: spend evidence, not mutations. The
                # re-evaluation can DISQUALIFY the agent's own best candidate,
                # which is the non-greedy decision the capability survey finds
                # least supported -- and the reason it can is that observation
                # noise here falls as 1/sqrt(seeds).
                deeper = max(alloc.n_eval_seeds, last_seeds + 2)
                newcard, einfo = service.reevaluate(last_candidate, last_compiled, deeper)
                if newcard is not None:
                    ir.seeds_spent += einfo.get("seeds", 0)
                    last_seeds = einfo.get("seeds", last_seeds)
                    was = service.verdict(card).qualifies
                    now = service.verdict(newcard).qualifies
                    ir.notes = (
                        f"escalated_{last_seeds}_seeds:"
                        f"qualified_{was}->{now}"
                    )
                    if was and not now and cand.cand_id in rec.qualified:
                        rec.qualified.remove(cand.cand_id)
                    cur_card = newcard
                    if cand.cand_id == best_cand:
                        best_card = newcard
                        best_u = utility(
                            newcard, parent,
                            outcome == RepairOutcome.CONTRACT_HOLDING_REPAIR, False)
            if alloc.action == "RELAX":
                # A RelaxationEvent requires a contract to relax. Arms 1 and 2
                # have none, so a RELAX action there is not a non-greedy
                # decision to weaken a commitment -- there is no commitment --
                # and recording one would put phantom events into the
                # relaxation rate, which the design doc calls "probably the
                # cleanest quantitative support" for the closure/interface
                # conflict. Measured before this guard: arm 2 logged 8
                # relaxations against arm 3's 2, which is backwards and was
                # entirely an artefact of relaxing an empty contract.
                if not (self.flags["contract"] and service.active_contract):
                    ir.notes = (ir.notes + ";" if ir.notes else "") + \
                        "relax_requested_without_contract:ignored"
                    rec.n_relax_without_contract += 1
                else:
                    ev, rstep = self._relax(view, cur_card, service, usage)
                    ir.steps.append(rstep)
                    if ev is not None:
                        ir.relaxation = ev
                        rec.relaxations.append(ev)

            rec.iterations.append(ir)
        else:
            rec.termination = rec.termination or "max_iterations"

        if not rec.termination:
            rec.termination = "max_iterations"
        rec.best_cand = best_cand
        rec.best_card = best_card
        rec.best_outcome = best_outcome
        rec.total_cost_usd = usage.cost_usd
        rec.total_llm_calls = usage.calls
        rec.total_seconds = round(time.time() - t_start, 2)
        rec.seeds_spent = service.seeds_spent
        rec.service_ledger = service.ledger.to_row()      # type: ignore[attr-defined]
        rec.usage = usage.to_row()                         # type: ignore[attr-defined]
        return rec

    # -- step 3 helper ----------------------------------------------------
    def _prior_for(self, diagnosis, view):
        return (self.archive.prior(diagnosis.defect_class,
                                   exclude_cluster=view.cluster_id)
                if self.flags['history_prior'] else None)

    def _reflections_for(self, diagnosis, view):
        return (self.archive.reflections(diagnosis.defect_class,
                                         exclude_cluster=view.cluster_id)
                if self.flags['reflections'] else None)

    def _specify(self, view, card, diagnosis, service, usage,
                 prior=None, reflections=None):
        steps: list[StepRecord] = []
        operator = None
        if self.two_call_specify:
            system, brief = prompts.build_specify_select(
                view, diagnosis, service, self.arm, SPEC_OPERATORS
            )
            op, st = self._ask(
                "SPECIFY_SELECT", system, brief,
                lambda t: prompts.parse_operator(t, SPEC_OPERATORS),
                "<operator>NAME</operator>", usage,
            )
            steps.append(st)
            if op is None:
                return None, steps
            operator = op

        # `operator` is None in the one-call scaffold, and build_specify_fill
        # then shows every operator's documentation with an open operator
        # field. Passing a placeholder here instead would silently narrow the
        # one-call arm's action space to that one operator.
        system, brief = prompts.build_specify_fill(
            view, card, diagnosis, operator, service, self.arm, SPEC_OPERATORS,
            inject_contract=self.inject_contract,
            prior=prior, reflections=reflections,
        )
        schema = prompts.SPECIFY_ACTION.format(
            operator=(
                operator.value if operator is not None
                else "<ONE OF: " + " | ".join(o.value for o in SPEC_OPERATORS) + ">"
            ),
            max_seeds=service.max_eval_seeds,
            n_pos=view.n_pos,
            seeds_left=max(0, service.seed_budget - service.seeds_spent),
            contract_field=prompts.CONTRACT_FIELD if self.flags["contract"] else "",
        )

        def parse_spec(text):
            obj, st = prompts.extract_json(text, "design_spec")
            if obj is None:
                return None, st
            if operator is not None:
                # The two-call pattern already fixed the operator; an object
                # that names a different one is taken as the committed choice
                # rather than silently overridden, so the trace says what the
                # model actually asked for.
                obj.setdefault("operator", operator.value)
            try:
                return DesignSpec.parse(obj, view.n_pos, service.max_eval_seeds), "ok"
            except ValueError:
                return None, "out_of_range"

        spec, st = self._ask("SPECIFY", system, brief, parse_spec, schema, usage)
        steps.append(st)
        return spec, steps

    def _contract_from(self, diagnosis, view) -> PreservationContract:
        """Turn what the agent said it would preserve into an enforced contract.

        Parsed from `preserved_features` -- "IDENTITY:4", "HBOND:7",
        "BSA_FLOOR" -- so the contract is the agent's, not the reference one.
        An agent that named nothing gets an EMPTY contract, which is the honest
        reading of arm 3: the mechanism is "the agent may protect a partial
        solution", not "a contract is imposed on it".
        """
        fixed: list[int] = []
        hbonds: list[int] = []
        floor = 0.0
        for feat in diagnosis.preserved_features:
            f = str(feat).strip().upper()
            if f.startswith("IDENTITY:"):
                try:
                    p = int(f.split(":", 1)[1].strip().lstrip("ACDEFGHIKLMNPQRSTVWY"))
                    if 1 <= p <= view.n_pos:
                        fixed.append(p)
                except (ValueError, IndexError):
                    continue
            elif f.startswith("HBOND:"):
                try:
                    p = int("".join(ch for ch in f.split(":", 1)[1] if ch.isdigit()))
                    if 1 <= p <= view.n_pos:
                        hbonds.append(p)
                except (ValueError, IndexError):
                    continue
            elif "BSA" in f:
                # The doc's worked example: "BSA floor 527 (0.85 x parent)".
                floor = round(0.85 * view.scorecard0.bsa, 1)
        return PreservationContract(
            fixed_identities=tuple(sorted(set(fixed))),
            required_hbonds=tuple(sorted(set(hbonds))),
            bsa_floor=floor,
        )

    # -- step 8 helper ----------------------------------------------------
    def _allocate(self, view, card, best_cand, service, history, it, last_seeds,
                  diagnosis, usage):
        prior = (
            self.archive.prior(diagnosis.defect_class, exclude_cluster=view.cluster_id)
            if self.flags["history_prior"] else None
        )
        reflections = (
            self.archive.reflections(
                diagnosis.defect_class, exclude_cluster=view.cluster_id)
            if self.flags["reflections"] else None
        )
        system, brief = prompts.build_allocate(
            view, card, best_cand, service, self.arm, history, it,
            self.max_iters, last_seeds, prior, reflections,
        )
        # Must mirror build_allocate's action list exactly: this is the schema
        # the error-correction reprompt restates, and offering RELAX here to an
        # arm that was not offered it in the brief would reintroduce the
        # phantom-relaxation bug through the retry path.
        can_relax = bool(self.flags["contract"] and service.active_contract)
        schema = prompts.ALLOCATE_ACTION.format(
            max_seeds=service.max_eval_seeds,
            actions=("EDIT | ESCALATE | RELAX | STOP" if can_relax
                     else "EDIT | ESCALATE | STOP"),
            relax_line=(prompts.RELAX_LINE if can_relax else ""),
        )

        def parse_alloc(text):
            obj, st = prompts.extract_json(text, "allocation")
            if obj is None:
                return None, st
            try:
                return AllocationDecision.parse(obj, service.max_eval_seeds), "ok"
            except ValueError:
                return None, "out_of_range"

        alloc, st = self._ask("ALLOCATE", system, brief, parse_alloc, schema, usage)
        if alloc is None:
            # Least informative continuation: keep editing while there is
            # budget, else stop.
            alloc = AllocationDecision(
                action="EDIT" if service.k_spent < view.k_budget else "STOP",
                rationale="fallback:allocate_unavailable",
                n_eval_seeds=self.default_seeds,
            )
            st = replace(st, fallback_used=True)
        return alloc, st

    def _relax(self, view, card, service, usage):
        """An explicit, justified, recorded weakening of the contract."""
        system, brief = prompts.build_relax(view, card, service)

        def parse_relax(text):
            obj, st = prompts.extract_json(text, "relaxation")
            if obj is None:
                return None, st
            clause = str(obj.get("clause", "")).strip().lower()
            if clause not in ("bsa_floor", "required_hbonds", "fixed_identities"):
                return None, "out_of_range"
            return obj, "ok"

        obj, st = self._ask(
            "RELAX", system, brief, parse_relax,
            prompts.RELAX_BRIEF.split("Emit exactly one block:")[-1], usage,
        )
        if obj is None:
            return None, st
        c = service.active_contract or PreservationContract()
        clause = str(obj["clause"]).strip().lower()
        ev = None
        if clause == "bsa_floor":
            new = float(obj.get("relax_to") or 0.0)
            ev = RelaxationEvent(
                iteration=-1, clause="bsa_floor", from_value=f"{c.bsa_floor:.0f}",
                to_value=f"{new:.0f}",
                justification=str(obj.get("justification", ""))[:400],
                evidence_refs=tuple(str(x) for x in (obj.get("evidence_refs") or [])),
            )
            # The lineage BRANCHES: the contract is replaced, not edited in
            # place, so a design that qualifies afterwards names the relaxed
            # contract it qualified under.
            service.active_contract = replace(c, bsa_floor=new)
        elif clause == "required_hbonds":
            drop = obj.get("relax_to")
            keep = tuple(p for p in c.required_hbonds if str(p) != str(drop))
            ev = RelaxationEvent(
                iteration=-1, clause="required_hbonds",
                from_value=",".join(map(str, c.required_hbonds)),
                to_value=",".join(map(str, keep)),
                justification=str(obj.get("justification", ""))[:400],
                evidence_refs=tuple(str(x) for x in (obj.get("evidence_refs") or [])),
            )
            service.active_contract = replace(c, required_hbonds=keep)
        else:
            drop = obj.get("relax_to")
            keep = tuple(p for p in c.fixed_identities if str(p) != str(drop))
            ev = RelaxationEvent(
                iteration=-1, clause="fixed_identities",
                from_value=",".join(map(str, c.fixed_identities)),
                to_value=",".join(map(str, keep)),
                justification=str(obj.get("justification", ""))[:400],
                evidence_refs=tuple(str(x) for x in (obj.get("evidence_refs") or [])),
            )
            service.active_contract = replace(c, fixed_identities=keep)
        return ev, st
