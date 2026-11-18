"""Rule-based controllers. The bar, the ceiling, and the two needed controls.

Every one of these runs through the SAME execution service, the same backend,
the same frozen qualification function and the same budget as an LLM arm, and
returns the same `CampaignRecord`. That is what makes the comparison paired:
the only thing that varies is who decides.

Stage 0's result is why this file is long. On the SKEMPI allocation slice a
local 8B LOST to the best simple rule and Opus/Sonnet only TIED it -- and tied
"largely by rediscovering the same rule" (74% agreement with min_contacts). So
the rule-based bar here is not a formality, and it is deliberately several
rules with the headline taken against the STRONGEST of them. Taking the max
over several heuristics inflates the bar by multiple comparisons, which is the
conservative direction: it makes an LLM win harder to claim, not easier.

Two entries are controls rather than baselines:

  oracle_greedy   Reads the enumerated table -- the answer key. It is the
                  ceiling, and the denominator for regret. If an LLM arm ever
                  matched it, the benchmark would be too easy to be measuring
                  anything.
  uniform_depth   Spends the WHOLE evidence budget at uniform depth and never
                  escalates. This is the control the design doc insists on:
                  BioDesignBench found that a plain forced-depth instruction
                  recovers most of the apparent evidence-allocation gap, and
                  that the same intervention applied to a tuned deterministic
                  pipeline produced no improvement (delta +1.6, n.s.). So an
                  escalation rule must beat UNIFORM forced depth at matched
                  compute, not a shallow default, or it has shown nothing.
"""

from __future__ import annotations

import time

from .backends.oracle import OracleBackend
from .enums import Arm, Operator, RepairOutcome
from .envbuild import surrogate
from .envbuild.enumerate_actions import (
    edit_cost,
    reference_contract,
    ring_span,
    utility,
)
from .memory import Archive
from .qualification import QUAL_HASH
from .qualification import THRESHOLDS as _T_THRESHOLDS
from .service import ExecutionService
from .types import (
    CampaignRecord,
    DesignSpec,
    IterationRecord,
    ParentProblem,
    ScoreCard,
    replace,
)

BASELINES = (
    "oracle_greedy",
    "fixed_pipeline",
    "uniform_depth",
    "greedy_interface",
    "rim_first_closure",
    "widest_rebuild",
    "random_controller",
    "noop",
    # Added after the Stage 2 verdict. The three above the line all pick a
    # position by how much interface it CARRIES; none of them uses the
    # headroom columns (`burf`, `mind`) at all, which is the lever the Stage 2
    # failure analysis points at. These three separate the parts of that idea:
    # the free operator, the predicate, and the routing.
    "headroom_repack",      # the free operator alone (RESTRAIN_AND_REPACK)
    "headroom_first",       # the predicate alone, as a rim_first counterpart
    "tool_router",          # deterministic selection over the whole tool menu
    "rebuild_cheap2",       # the bar's position rule with the oracle's operator
    "tool_router2",         # the deterministic selector that does not quit early
)


class RuleController:
    """A deterministic controller. Same interface as the LLM Controller."""

    def __init__(
        self,
        rule: str,
        max_iters: int = 5,
        patience: int = 2,
        default_seeds: int = 5,
        seed_budget: int = 60,
        max_eval_seeds: int = 20,
        backend_fidelity: str = "declared",
        enforce_contract: bool = False,
        oracle_rows: dict[str, list[dict]] | None = None,
    ):
        if rule not in BASELINES:
            raise ValueError(f"unknown baseline {rule!r}; one of {BASELINES}")
        self.rule = rule
        self.model_id = f"rule:{rule}"
        self.max_iters = max_iters
        self.patience = patience
        self.default_seeds = default_seeds
        self.seed_budget = seed_budget
        self.max_eval_seeds = max_eval_seeds
        self.backend_fidelity = backend_fidelity
        # Baselines run WITHOUT a preservation contract by default, which is
        # the honest setting: a rule has no way to decide what to preserve, so
        # giving it the reference contract would hand it arm 3's mechanism for
        # free and make the 2->3 comparison meaningless.
        self.enforce_contract = enforce_contract
        self._oracle_rows = oracle_rows or {}
        self.arm = Arm.A2_CONDITIONING
        # Which tools this campaign has already committed. Reset per parent in
        # run(); a tool that gained nothing once will gain nothing again on an
        # unchanged residue table, so repeating it just burns an iteration.
        self._tools_tried: set[str] = set()
        self._tool_choices: list[str] = []
        self._tool_attempts: dict[str, int] = {}

    # -- the rules --------------------------------------------------------
    def _choose(self, parent: ParentProblem, card: ScoreCard, service, it: int
                ) -> DesignSpec | None:
        n = parent.n_pos
        res = {r.pos: r for r in parent.residues}
        k_remaining = max(0, parent.k_budget - service.k_spent)
        if k_remaining <= 0:
            return None

        if self.rule == "noop":
            return None

        if self.rule == "random_controller":
            # Seeded on (parent, iteration), so the floor is reproducible and
            # paired across arms rather than being a different floor each run.
            u = surrogate._unit("baseline_random", parent.parent_id, it)
            lo = int(u * n) + 1
            return DesignSpec(
                operator=Operator.REBUILD_SEGMENT,
                rebuild_segment=(lo, min(lo + 1, n)),
                open_positions=(),
                n_eval_seeds=self.default_seeds,
                rationale="random",
            )

        if self.rule in ("fixed_pipeline", "uniform_depth"):
            # The tuned deterministic workflow: rebuild the span centred on the
            # worst measured strain. This is the right pipeline for the defect
            # the LP->CP premise is about, and it ignores the interface
            # entirely -- which is exactly why it is the thing to beat.
            locus = card.strain_locus or tuple(
                sorted(res, key=lambda p: -res[p].strain_rel)[:1]
            )
            centre = locus[0]
            span = ring_span(((centre - 2) % n) + 1, 3, n)
            cost = edit_cost(Operator.REBUILD_SEGMENT, span)
            if cost > k_remaining:
                span = ring_span(centre, 2, n)
            seeds = (
                # uniform_depth spends its whole evidence budget evenly over
                # the iterations it will run, never escalating on a signal.
                max(1, min(self.max_eval_seeds, self.seed_budget // self.max_iters))
                if self.rule == "uniform_depth" else self.default_seeds
            )
            return DesignSpec(
                operator=Operator.REBUILD_SEGMENT,
                rebuild_segment=(min(span), max(span)) if max(span) - min(span) < n - 1
                else (span[-1], span[0]),
                n_eval_seeds=seeds,
                rationale=f"{self.rule}:rebuild_at_max_strain",
            )

        if self.rule == "greedy_interface":
            # Stage 0's winning idea, transplanted: spend where the fewest
            # cross-interface contacts are. There it beat every other rule and
            # tied the frontier models.
            order = sorted(res, key=lambda p: (-surrogate.interface_mass(res[p]), p))
            pick = tuple(order[: max(1, min(k_remaining, 1))])
            return DesignSpec(
                operator=Operator.SUBSTITUTE,
                open_positions=pick,
                n_eval_seeds=self.default_seeds,
                rationale="greedy_interface:fewest_contacts",
            )

        if self.rule == "rim_first_closure":
            # The closure-aware version of the same idea: among the positions
            # next to the cyclisation bond, take the one carrying the least
            # interface. This is the strongest rule available that uses BOTH
            # signals, and is the honest bar for "an LLM reads two columns".
            near = [p for p in res if res[p].ring_dist <= 2]
            if not near:
                near = list(res)
            centre = min(near, key=lambda p: (-surrogate.interface_mass(res[p]), p))
            span = ring_span(((centre - 1) % n) + 1, 2, n)
            return DesignSpec(
                operator=Operator.NARROW_PIVOTS,
                rebuild_segment=(min(span), max(span)) if max(span) - min(span) < n - 1
                else (span[-1], span[0]),
                n_eval_seeds=self.default_seeds,
                rationale="rim_first_closure:cheapest_near_closure",
            )

        if self.rule == "widest_rebuild":
            # Deliberate anti-baseline: rebuild as much as the budget allows.
            # Should be clearly WORSE than the targeted rules; if it is not,
            # the environment is not charging for interface damage and the
            # whole conflict is not in it.
            L = 4 if k_remaining >= 2 else 2
            span = ring_span(1, L, n)
            return DesignSpec(
                operator=Operator.REBUILD_SEGMENT,
                rebuild_segment=(min(span), max(span)),
                n_eval_seeds=self.default_seeds,
                rationale="widest_rebuild:anti_baseline",
            )

        if self.rule in ("headroom_repack", "headroom_first", "tool_router",
                         "tool_router2", "rebuild_cheap2"):
            # All three draw their moves from lp2cp.tools, the SAME registry the
            # LLM tool-selection arm selects from, so a difference between this
            # row and that one is a difference in the selector and nothing
            # else. Stage 2's one-call scaffold bug was exactly this mistake
            # made in the other direction.
            from . import tools as T  # noqa: PLC0415

            ctx0 = T.ToolCtx(
                parent=parent, card=card, k_remaining=k_remaining,
                seeds=self.default_seeds, iteration=it,
                tried=tuple(sorted(self._tools_tried)),
            )
            if self.rule == "tool_router":
                name = T.route(ctx0)
            elif self.rule == "tool_router2":
                name = T.route_v2(ctx0)
            elif self.rule == "rebuild_cheap2":
                # Needs the v2 menu by construction: its whole content is the
                # tool that only v2 carries. Under v1 it would silently become
                # `headroom_repack`, so it says so instead.
                if "rebuild_cheapest_2span" not in T.MENUS[T.ACTIVE_MENU]:
                    raise ValueError(
                        "rule 'rebuild_cheap2' needs tool menu v2; active menu "
                        f"is {T.ACTIVE_MENU}. Pass --tool-menu v2.")
                name = ("rebuild_cheapest_2span"
                        if "rebuild_cheapest_2span" not in self._tools_tried
                        else "repack_headroom"
                        if "repack_headroom" not in self._tools_tried
                        else "stop")
            elif self.rule == "headroom_repack":
                name = ("repack_headroom"
                        if "repack_headroom" not in self._tools_tried
                        else "narrow_near_closure")
            else:
                # headroom_first: the direct counterpart to rim_first_closure.
                # Same shape -- read two columns, pick one position -- but the
                # columns are the headroom pair rather than the contact pair.
                t = _T_THRESHOLDS
                floor = t["bsa_frac_of_native_min"] * max(parent.native_bsa, 1.0)
                binding = (card.ddg_bind > t["ddg_bind_max"]) or (card.bsa < floor)
                name = ("substitute_headroom" if binding
                        else "narrow_near_closure"
                        if card.max_strain > t["max_strain_max"] else "stop")
            self._tool_choices.append(name)
            if name == "stop":
                return None
            attempt = self._tool_attempts.get(name, 0)
            self._tool_attempts[name] = attempt + 1
            ctx = replace(ctx0, attempt=attempt)
            spec = T.build(name, ctx)
            if spec is None:
                # Ranked candidates exhausted: retire the tool so the next
                # iteration routes elsewhere rather than asking again.
                self._tools_tried.add(name)
                return None
            return spec

        if self.rule == "oracle_greedy":
            rows = self._oracle_rows.get(parent.parent_id, [])
            if not rows:
                return None
            feasible = [
                r for r in rows
                if not r["over_budget"] and r["groups"]
                and r["edit_cost"] <= k_remaining
            ]
            if not feasible:
                return None
            best = max(feasible, key=lambda r: r["utility"])
            ops = {g[0] for g in best["groups"]}
            seg = None
            opens: list[int] = []
            for opname, ps in best["groups"]:
                if opname in (Operator.REBUILD_SEGMENT.value,
                              Operator.NARROW_PIVOTS.value,
                              Operator.ANCHOR_SHIFT.value):
                    seg = (min(ps), max(ps)) if max(ps) - min(ps) < n - 1 \
                        else (max(ps), min(ps))
                else:
                    opens.extend(ps)
            primary = (
                Operator.REBUILD_SEGMENT if Operator.REBUILD_SEGMENT.value in ops
                else Operator.NARROW_PIVOTS if Operator.NARROW_PIVOTS.value in ops
                else Operator.ANCHOR_SHIFT if Operator.ANCHOR_SHIFT.value in ops
                else Operator.RESTRAIN_AND_REPACK
                if Operator.RESTRAIN_AND_REPACK.value in ops
                else Operator.SUBSTITUTE
            )
            return DesignSpec(
                operator=primary,
                open_positions=tuple(sorted(set(opens))),
                rebuild_segment=seg,
                n_eval_seeds=self.max_eval_seeds,
                rationale=f"oracle:{best['intervention_id']}",
            )
        return None

    # -- the loop ---------------------------------------------------------
    def run(self, parent: ParentProblem, env_hash: str = "") -> CampaignRecord:
        backend = OracleBackend(parent, fidelity=self.backend_fidelity)
        service = ExecutionService(
            parent=parent, backend=backend,
            enforce_contract=self.enforce_contract,
            max_eval_seeds=self.max_eval_seeds, seed_budget=self.seed_budget,
        )
        ref_contract = reference_contract(parent)
        if self.enforce_contract:
            service.active_contract = ref_contract
        rec = CampaignRecord(
            campaign_id=f"{parent.parent_id}__{self.rule}",
            parent_id=parent.parent_id, cluster_id=parent.cluster_id,
            split=parent.split, arm=self.arm, controller=self.model_id,
            model_id=self.model_id, qualification_hash=QUAL_HASH,
            env_hash=env_hash, k_budget=parent.k_budget,
        )
        self._tools_tried = set()
        self._tool_choices = []
        self._tool_attempts = {}
        cur_seq, cur_card, cur_cand = parent.cp_sequence, parent.scorecard0, "C-000"
        best_u = utility(cur_card, parent, False, False)
        best_cand, best_card = cur_cand, cur_card
        best_outcome = RepairOutcome.NO_CHANGE
        no_improve = 0
        t0 = time.time()

        for it in range(1, self.max_iters + 1):
            ir = IterationRecord(iteration=it, candidate_in=cur_cand)
            view = replace(parent, cp_sequence=cur_seq, scorecard0=cur_card)
            spec = self._choose(view, cur_card, service, it)
            if spec is None:
                ir.notes = "rule_declined_to_act"
                rec.iterations.append(ir)
                rec.termination = "rule_stopped"
                break
            ir.spec = spec
            compiled, reason = service.compile_spec(spec)
            ir.compile_ok = compiled is not None
            ir.compile_reason = reason
            if compiled is None:
                ir.outcome = RepairOutcome.INFEASIBLE
                ir.notes = f"compile_rejected:{reason}"
                service.ledger.reject(reason.split(":")[0])
                rec.iterations.append(ir)
                no_improve += 1
                if no_improve >= self.patience:
                    rec.termination = "no_improvement"
                    break
                continue
            ir.overwide_mask = compiled.overwide_mask

            survivors, info = service.act_and_score(compiled, spec.n_eval_seeds, cur_seq)
            ir.n_proposed = info.get("n_proposed", 0)
            ir.n_budget_filtered = info.get("n_budget_filtered", 0)
            ir.n_scope_leaks = info.get("n_scope_leaks", 0)
            ir.n_scored = info.get("n_scored", 0)
            ir.seeds_spent = info.get("seeds", 0)
            if info.get("stop"):
                ir.notes = info["stop"]
                rec.termination = info["stop"]
                rec.iterations.append(ir)
                break

            accepted = []
            for cand, card in survivors:
                outcome, violations = service.classify(card, cand.sequence, ref_contract)
                ok_local, extra = service.accepts(card, cand.sequence)
                if not ok_local:
                    continue
                u = utility(
                    card, parent,
                    outcome == RepairOutcome.CONTRACT_HOLDING_REPAIR, False)
                accepted.append((cand, card, u, outcome, violations + extra))
            if not accepted:
                ir.outcome = RepairOutcome.NO_CHANGE
                ir.notes = "all_candidates_rejected"
                rec.iterations.append(ir)
                no_improve += 1
                if no_improve >= self.patience:
                    rec.termination = "no_improvement"
                    break
                continue

            cand, card, u, outcome, violations = max(accepted, key=lambda t: t[2])
            ir.best_cand, ir.best_card = cand.cand_id, card
            ir.outcome, ir.contract_violations = outcome, tuple(violations)
            if u > best_u + 1e-9:
                best_u, best_cand, best_card, best_outcome = (
                    u, cand.cand_id, card, outcome)
                service.commit(cand, card)
                cur_seq, cur_card, cur_cand = cand.sequence, card, cand.cand_id
                no_improve = 0
            else:
                no_improve += 1
            ir.k_spent_after = service.k_spent
            if service.verdict(card).qualifies:
                rec.qualified.append(cand.cand_id)
            rec.iterations.append(ir)

            if rec.qualified:
                rec.termination = "n_qualifying_reached"
                break
            if no_improve >= self.patience:
                rec.termination = "no_improvement"
                break
            if service.k_spent >= parent.k_budget:
                rec.termination = "budget_exhausted"
                break
        if not rec.termination:
            rec.termination = "max_iterations"

        rec.best_cand, rec.best_card, rec.best_outcome = best_cand, best_card, best_outcome
        rec.total_seconds = round(time.time() - t0, 2)
        rec.seeds_spent = service.seeds_spent
        rec.service_ledger = service.ledger.to_row()
        # The tool rules select from the same registry as the LLM arm, so their
        # choices have to survive into the trace or the two selectors cannot be
        # compared on what they actually picked.
        rec.usage = ({"tool_choices": list(self._tool_choices)}
                     if self._tool_choices else {})
        return rec


def load_oracle_rows(path) -> dict[str, list[dict]]:
    """Group the enumerated table by parent, for the oracle control arm only."""
    import json
    from collections import defaultdict
    from pathlib import Path

    out: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["parent_id"]].append(r)
    return dict(out)
