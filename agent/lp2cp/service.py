"""The deterministic execution service. Steps 4, 6 and 7 of the control loop.

The design doc's division of labour gives this layer "mask verification, budget
enforcement, caching, provenance" and one word of justification: "A guarantee,
not a request." Agent Rosetta's headline finding is the same point from the
other side -- prompt engineering alone often fails to generate valid actions,
and tool and environment design are what make the integration work, so a
meaningful share of the engineering is this file rather than the prompt.

Three guarantees, each answering a measured failure mode:

STEP 4, budget enforcement at the FILTER stage. The doc is specific that the
service "does not ask, it enforces the predicate at the filter stage" -- its
worked example has the LLM open three positions against a two-substitution
budget, and the right response is not to reject the design problem but to let
the backend propose over the wider mask and then discard every returned
candidate that spent more than the budget. An over-wide mask is recorded, not
refused, because whether agents over-open masks is a result.

STEP 6, mask verification. PROTEAN identifies backend SCOPE LEAKAGE as one of
its two dominant bottlenecks: a solver silently moves a position that was
declared fixed. Invariant 3 of the doc says every returned sequence is diffed
against the parent LP AND against the lineage's declared mask before anything
is scored, because "soft conditioning is a request, not a contract". Caught
leaks are counted per arm -- a backend that leaks and a service that catches it
is the system working, and the count is reportable.

STEP 7, scoring through the frozen predicate only. The service holds the only
reference to the qualification function, so no arm can score itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends.base import Backend
from .enums import Operator, RepairOutcome
from .envbuild.enumerate_actions import edit_cost
from .qualification import Verdict, qualify
from .types import (
    Candidate,
    DesignSpec,
    ParentProblem,
    PreservationContract,
    ScoreCard,
    sha,
)


@dataclass(frozen=True)
class CompiledSpec:
    """A design problem the backend may be handed, plus what to filter on."""

    spec: DesignSpec
    parent: ParentProblem
    touched: tuple[int, ...]
    declared_fixed: tuple[int, ...]
    k_remaining: int
    max_edit_cost: int
    overwide_mask: bool
    notes: tuple[str, ...] = ()

    @property
    def groups(self) -> tuple[tuple[Operator, tuple[int, ...]], ...]:
        """(operator, positions) groups, matching the surrogate's intervention form.

        A spec that names BOTH a rebuild segment and open positions is a
        coordinated move and compiles to two groups -- that is how "open
        position 6 and rebuild 10-12" reaches the backend as one intervention
        rather than two sequential ones that would each pay their own noise.
        """
        out: list[tuple[Operator, tuple[int, ...]]] = []
        seg = self.spec.rebuild_segment
        if seg:
            lo, hi = seg
            n = self.parent.n_pos
            # A segment given as (hi, lo) with hi > lo wraps the ring: a cyclic
            # peptide has no last residue, so 15->2 of a 16-mer is contiguous.
            span = (
                tuple(range(lo, hi + 1)) if lo <= hi
                else tuple(list(range(lo, n + 1)) + list(range(1, hi + 1)))
            )
            out.append((self.spec.operator if self.spec.operator in (
                Operator.REBUILD_SEGMENT, Operator.NARROW_PIVOTS,
                Operator.ANCHOR_SHIFT) else Operator.REBUILD_SEGMENT, span))
        open_only = tuple(
            p for p in self.spec.open_positions
            if not seg or p not in (out[0][1] if out else ())
        )
        if open_only:
            op = (
                self.spec.operator
                if self.spec.operator in (Operator.SUBSTITUTE,
                                          Operator.RESTRAIN_AND_REPACK)
                else Operator.SUBSTITUTE
            )
            out.append((op, open_only))
        if not out and self.spec.operator == Operator.RESTRAIN_AND_REPACK:
            # Restrain-and-repack with no explicit mask acts on what the
            # contract protects; with no contract it acts on the parent's own
            # polar contacts. Either way it is a real, locatable intervention
            # rather than a no-op.
            tgt = (
                self.spec.contract.required_hbonds if self.spec.contract
                and self.spec.contract.required_hbonds
                else self.parent.scorecard0.hbonds_retained
            )
            if tgt:
                out.append((Operator.RESTRAIN_AND_REPACK, tuple(tgt)))
        return tuple(out)

    @property
    def intervention_id(self) -> str:
        g = self.groups
        if not g:
            return "noop"
        return "|".join(f"{op.value}:{','.join(map(str, ps))}" for op, ps in g)


@dataclass
class ServiceLedger:
    """Counts the service's own catches. Reported per arm, per campaign."""

    n_compiled: int = 0
    n_compile_rejected: int = 0
    n_overwide_masks: int = 0
    n_proposed: int = 0
    n_budget_filtered: int = 0
    n_scope_leaks_caught: int = 0
    n_refold_failures: int = 0
    n_scored: int = 0
    n_cache_hits: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.n_compile_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def to_row(self) -> dict:
        return {
            "n_compiled": self.n_compiled,
            "n_compile_rejected": self.n_compile_rejected,
            "n_overwide_masks": self.n_overwide_masks,
            "n_proposed": self.n_proposed,
            "n_budget_filtered": self.n_budget_filtered,
            "n_scope_leaks_caught": self.n_scope_leaks_caught,
            "n_refold_failures": self.n_refold_failures,
            "n_scored": self.n_scored,
            "n_cache_hits": self.n_cache_hits,
            "reject_reasons": dict(self.reject_reasons),
        }


class ExecutionService:
    """Owns the frozen predicate, the budget, the mask and the cache.

    Constructed once per campaign. The controller may call it; it may not reach
    past it to the backend or to `qualify`.
    """

    def __init__(
        self,
        parent: ParentProblem,
        backend: Backend,
        enforce_contract: bool,
        max_eval_seeds: int = 20,
        seed_budget: int = 60,
    ):
        self.parent = parent
        self.backend = backend
        # Arm 1 and 2 have no preservation contract. The flag is held HERE and
        # not in the prompt layer: arm 2's whole claim is that it conditions on
        # a mask without protecting a partial solution, and if the contract
        # were merely absent from the prompt while still being enforced at
        # scoring, 2 and 3 would differ only in wording.
        self.enforce_contract = enforce_contract
        self.max_eval_seeds = max_eval_seeds
        # The evidence budget, fixed across arms. The doc: "Fix the evidence
        # budget per decision across arms here. That isolates diagnosis quality
        # from evaluation depth and removes the BioDesignBench confound by
        # construction."
        self.seed_budget = seed_budget
        self.seeds_spent = 0
        self.ledger = ServiceLedger()
        self._cache: dict[str, list[tuple[Candidate, ScoreCard]]] = {}
        self.k_spent = parent.k_spent
        self.active_contract: PreservationContract | None = None

    # -- step 4 -----------------------------------------------------------
    def compile_spec(self, spec: DesignSpec) -> tuple[CompiledSpec | None, str]:
        """Validate and compile. Returns (compiled, reason-if-rejected).

        Rejections here are STRUCTURAL only -- a mask that cannot be acted on
        at all. Budget is NOT a rejection reason: an over-wide mask compiles,
        is flagged, and is enforced when candidates come back.
        """
        n = self.parent.n_pos
        notes: list[str] = []
        k_remaining = max(0, self.parent.k_budget - self.k_spent)

        bad = [p for p in spec.open_positions + spec.fixed_positions if not 1 <= p <= n]
        if bad:
            return None, f"position_out_of_range:{bad}"
        clash = sorted(set(spec.open_positions) & set(spec.fixed_positions))
        if clash:
            # Declaring a position both open and fixed is not a preference to
            # be resolved, it is a contradiction: whichever way the service
            # resolved it, the resulting trace would not say what the agent
            # asked for.
            return None, f"position_both_open_and_fixed:{clash}"

        if spec.operator in (Operator.EVIDENCE_ESCALATION, Operator.STOP,
                             Operator.RELAX_CONTRACT):
            return None, f"not_an_edit_operator:{spec.operator.value}"

        if spec.operator in (Operator.REBUILD_SEGMENT, Operator.NARROW_PIVOTS):
            if not spec.rebuild_segment:
                return None, f"{spec.operator.value}_without_rebuild_segment"
        if spec.operator == Operator.SUBSTITUTE and not spec.open_positions:
            return None, "SUBSTITUTE_without_open_positions"

        probe = CompiledSpec(
            spec=spec, parent=self.parent, touched=(), declared_fixed=(),
            k_remaining=k_remaining, max_edit_cost=k_remaining,
            overwide_mask=False,
        )
        groups = probe.groups
        if not groups:
            return None, "empty_mask"
        touched = tuple(sorted({p for _, ps in groups for p in ps}))

        # The contract's protected identities are added to the declared-fixed
        # set so that step 6 verifies against them too. An agent that declares
        # a contract and then opens a protected position is caught at compile:
        # the contract is the thing it said it would not spend.
        declared_fixed = set(spec.fixed_positions)
        if self.enforce_contract and self.active_contract:
            declared_fixed |= set(self.active_contract.fixed_identities)
        overlap = sorted(declared_fixed & set(touched))
        if overlap:
            return None, f"mask_touches_contract_protected:{overlap}"

        cost = sum(edit_cost(op, ps) for op, ps in groups)
        overwide = cost > k_remaining
        if overwide:
            notes.append(
                f"mask_costs_{cost}_against_{k_remaining}_remaining:"
                f"enforced_at_filter_stage"
            )
            self.ledger.n_overwide_masks += 1
        if k_remaining <= 0:
            return None, "no_budget_remaining"

        self.ledger.n_compiled += 1
        return (
            CompiledSpec(
                spec=spec,
                parent=self.parent,
                touched=touched,
                declared_fixed=tuple(sorted(declared_fixed)),
                k_remaining=k_remaining,
                max_edit_cost=k_remaining,
                overwide_mask=overwide,
                notes=tuple(notes),
            ),
            "",
        )

    # -- steps 5, 6, 7 ----------------------------------------------------
    def act_and_score(
        self, compiled: CompiledSpec, n_eval_seeds: int, current_seq: str
    ) -> tuple[list[tuple[Candidate, ScoreCard]], dict]:
        """Propose inside the mask, verify, filter, score. Returns survivors.

        The order is load-bearing. Verification happens BEFORE scoring, so a
        scope-leaked candidate never reaches the qualification function and
        cannot become a reported result. Budget filtering also happens before
        scoring, for the same reason.
        """
        seeds = max(1, min(int(n_eval_seeds), self.max_eval_seeds))
        remaining_seeds = max(0, self.seed_budget - self.seeds_spent)
        if remaining_seeds <= 0:
            return [], {"stop": "evidence_budget_exhausted", "seeds": 0}
        seeds = min(seeds, remaining_seeds)

        key = sha((compiled.intervention_id, seeds, current_seq,
                   self.k_spent, self.enforce_contract))
        if key in self._cache:
            self.ledger.n_cache_hits += 1
            # A cache hit costs no seeds: the doc gives the service "caching"
            # as a guarantee, and charging twice for an identical query would
            # make the evidence budget depend on how often the agent repeated
            # itself rather than on how deeply it chose to look.
            return self._cache[key], {"cache_hit": True, "seeds": 0}

        proposed = self.backend.propose(compiled, current_seq)
        self.ledger.n_proposed += len(proposed)

        survivors: list[tuple[Candidate, ScoreCard]] = []
        n_leak = n_budget = n_refold = 0
        allowed = set(compiled.touched)
        for cand in proposed:
            if cand.sequence is None:
                n_refold += 1
                continue
            # --- step 6: diff against the parent LP and the declared mask ---
            edited = {
                i + 1 for i, (a, b) in enumerate(zip(current_seq, cand.sequence))
                if a != b
            }
            leaked = sorted(edited - allowed)
            if leaked:
                n_leak += 1
                continue
            # --- step 4's predicate, at the filter stage --------------------
            lp_edits = sum(
                1 for a, b in zip(self.parent.lp_sequence, cand.sequence) if a != b
            )
            if lp_edits > self.parent.k_budget:
                n_budget += 1
                continue
            if self.k_spent + edit_cost(cand.operator, tuple(sorted(edited))) \
                    > self.parent.k_budget and edited:
                n_budget += 1
                continue
            # --- step 7: the frozen predicate, through the service only ------
            card = self.backend.score(compiled, cand, seeds)
            survivors.append((cand, card))

        self.ledger.n_scope_leaks_caught += n_leak
        self.ledger.n_budget_filtered += n_budget
        self.ledger.n_refold_failures += n_refold
        self.ledger.n_scored += len(survivors)
        self.seeds_spent += seeds
        self._cache[key] = survivors
        return survivors, {
            "seeds": seeds,
            "n_proposed": len(proposed),
            "n_scope_leaks": n_leak,
            "n_budget_filtered": n_budget,
            "n_refold_failures": n_refold,
            "n_scored": len(survivors),
        }

    def reevaluate(
        self, cand: Candidate, compiled: CompiledSpec, n_eval_seeds: int
    ) -> tuple[ScoreCard | None, dict]:
        """EVIDENCE_ESCALATION: re-score an existing candidate more deeply.

        No edit, no budget spent, more seeds. This is the doc's iteration 3 --
        "the evidence plan fires: 10 seeds rather than a new edit" -- and the
        reason it can change the answer is that the environment's observation
        noise falls as 1/sqrt(seeds), so a candidate that qualified at 5 seeds
        can fail at 10. An agent that never escalates keeps false qualifies.
        """
        seeds = max(1, min(int(n_eval_seeds), self.max_eval_seeds))
        remaining = max(0, self.seed_budget - self.seeds_spent)
        if remaining <= 0:
            return None, {"stop": "evidence_budget_exhausted", "seeds": 0}
        seeds = min(seeds, remaining)
        card = self.backend.score(compiled, cand, seeds)
        self.seeds_spent += seeds
        return card, {"seeds": seeds, "escalated_to": seeds}

    # -- verdicts ---------------------------------------------------------
    def verdict(self, card: ScoreCard) -> Verdict:
        return qualify(card, self.parent.native_bsa, self.parent.k_budget)

    def classify(
        self, card: ScoreCard, seq: str, reference_contract: PreservationContract
    ) -> tuple[RepairOutcome, tuple[str, ...]]:
        """Label one scored candidate. The 2->3 ablation, in one method.

        The outcome is ALWAYS computed against the REFERENCE contract, whatever
        the arm is, so a repair means the same thing in every arm. What the arm
        changes is whether `DEFECT_RESOLVED_CONTRACT_BROKEN` is accepted as
        progress: arm 2 has no contract and will take it; arm 3 enforces one
        and will not. If the label itself moved with the arm, arm 2's repair
        rate would be measuring its own permissiveness.
        """
        v = self.verdict(card)
        ok, violations = reference_contract.holds(card, seq, self.parent.cp_sequence)
        if v.qualifies and ok:
            return RepairOutcome.CONTRACT_HOLDING_REPAIR, violations
        if v.qualifies:
            return RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN, violations
        return RepairOutcome.NO_CHANGE, violations

    def commit(self, cand: Candidate, card: ScoreCard) -> None:
        """Accept a candidate as the new lineage head and charge its budget."""
        self.k_spent = max(
            self.k_spent,
            sum(1 for a, b in zip(self.parent.lp_sequence, cand.sequence) if a != b),
        )

    def accepts(
        self, card: ScoreCard, seq: str
    ) -> tuple[bool, tuple[str, ...]]:
        """Would this arm accept the candidate locally?

        Arms without a contract accept anything that improves; arms with one
        refuse a candidate that breaks it even when the defect was resolved.
        This is the only place the arm flag changes behaviour rather than
        prompt text, which is what makes 2->3 a mechanism ablation.
        """
        if not (self.enforce_contract and self.active_contract):
            return True, ()
        ok, violations = self.active_contract.holds(
            card, seq, self.parent.cp_sequence
        )
        return ok, violations
