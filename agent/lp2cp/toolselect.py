"""The LLM selecting from the TOOL action space, PDAgent's architecture.

What this changes, and why
-------------------------
In every Stage 2 arm the LLM authored the mask: it emitted `open_positions` and
`rebuild_segment` directly, and that is the step the results localise the
failure to. Five models, two environment versions, the whole arm ladder, and
the leading alternative explanation (the agent contradicting its own contract)
experimentally excluded -- nothing beat a two-column rule.

Here the LLM never names a position. It names one of the tools in
lp2cp/tools.py, each of which carries its own position rule. This is PDAgent's
division of labour (Ouyang et al., ICML 2026): their LLM selects among seven
mutation tools whose predicates are hard-coded, and the lever-versus-symptom
judgement lives in those predicates rather than in the model's reasoning.

The arm exists to answer one question that the Stage 2 data cannot: is the
agent bad at ALLOCATION, or bad at AUTHORING A MASK? Those are different
claims, and only the first is the one the design doc says has no prior-art
answer. Measured against the same parents, the same service, the same budget
and the same scorer, with `tool_router` as the deterministic control over the
identical action space -- which is PDAgent's Table 7 ablation, reproduced here
because their +12.92 CSR on conflicting objectives is a claim about exactly
the regime this benchmark is made of.

Policy choices, each matching an existing project rule rather than inventing one
--------------------------------------------------------------------------------
* A DECLINE OR A MALFORMED REPLY ABANDONS THE ITERATION. It does not fall back
  to a rule. `agent/README.md` sets that policy for SPECIFY -- "specifying is
  the step under test, and a fallback would make the arm a mixture" -- and
  selection is the step under test here, so the same rule applies. The retry
  policy is `ask_step`, shared with the mask arms, not a second copy.
* `stop` IS A REAL CHOICE, not a failure. It terminates the campaign with
  `termination="tool_stop"`, kept distinct from `tool_declined` and
  `tool_malformed` so the trace says which happened.
* A TOOL IS OFFERED UNTIL ITS CANDIDATES RUN OUT. Each tool ranks its own
  candidates and returns the nth on its nth selection, so re-selecting a tool
  steps down its ranking instead of repeating one intervention; a tool is
  retired only when it returns nothing. Without this a campaign could reach at
  most ~6 of the ~150 enumerated interventions for its parent and a tool whose
  single guess was wrong lost the parent outright. The deterministic router
  draws on the identical ranked registry, so the two selectors still face the
  same action space.
"""

from __future__ import annotations

import time

from . import prompts
from . import tools as T
from .baselines import RuleController
from .controller import ask_step
from .envbuild.enumerate_actions import edit_cost
from .enums import Arm, Operator, RepairOutcome, StepStatus
from .llm.base import Usage
from .types import DesignSpec, ParentProblem, ScoreCard


class ToolSelectController(RuleController):
    """One LLM call per iteration: which tool. No mask, no identities."""

    def __init__(
        self,
        llm,
        max_iters: int = 5,
        patience: int = 2,
        default_seeds: int = 5,
        seed_budget: int = 60,
        max_eval_seeds: int = 20,
        backend_fidelity: str = "declared",
        enforce_contract: bool = False,
        max_retries: int = 1,
        save_raw: bool = False,
        arm: Arm = Arm.A2_CONDITIONING,
    ):
        # The parent class validates `rule` against BASELINES; `tool_router` is
        # a real member of it and is the right label for the shape of this
        # controller. `model_id` is then overwritten with the model's own id so
        # a row is never attributed to the rule.
        super().__init__(
            rule="tool_router", max_iters=max_iters, patience=patience,
            default_seeds=default_seeds, seed_budget=seed_budget,
            max_eval_seeds=max_eval_seeds, backend_fidelity=backend_fidelity,
            enforce_contract=enforce_contract,
        )
        self.llm = llm
        self.model_id = llm.model_id
        self.max_retries = max_retries
        self.save_raw = save_raw
        self.arm = arm
        self.usage = Usage()
        self._steps: list = []
        self._history: list[dict] = []
        self._status: str = ""
        self._pending: tuple | None = None
        self._attempts: dict[str, int] = {}
        self._exhausted: set[str] = set()

    # -- the one decision -------------------------------------------------
    def _choose(self, parent: ParentProblem, card: ScoreCard, service, it: int
                ) -> DesignSpec | None:
        self._status = ""
        self._record_previous(parent, card)
        k_remaining = max(0, parent.k_budget - service.k_spent)
        if k_remaining <= 0 and "repack_headroom" in self._exhausted:
            self._status = "budget_exhausted"
            return None

        offered = [t for t in T.active_tools() if t.name not in self._exhausted]
        if not offered or all(t.name == "stop" for t in offered):
            self._status = "tools_exhausted"
            return None
        menu = "\n\n".join(f"  {t.name}  [costs {t.cost}]\n    {t.doc}"
                           for t in offered)
        names = tuple(t.name for t in offered)

        system, brief = prompts.build_tool_select(
            parent, card, service, menu, self._history)
        schema = "<tool>" + " | ".join(names) + "</tool>"
        name, step = ask_step(
            llm=self.llm, step="SELECT_TOOL", system=system, brief=brief,
            parse=lambda t: prompts.parse_tool(t, names),
            schema=schema, usage=self.usage,
            max_retries=self.max_retries, save_raw=self.save_raw,
        )
        self._steps.append(step)
        if name is None:
            self._status = ("tool_declined"
                            if step.status == StepStatus.DECLINED
                            else "tool_budget_cap"
                            if step.status == StepStatus.BUDGET_CAP
                            else "tool_error" if step.status == StepStatus.ERROR
                            else "tool_malformed")
            return None

        self._tool_choices.append(name)
        if name == "stop":
            self._status = "tool_stop"
            return None
        attempt = self._attempts.get(name, 0)
        self._attempts[name] = attempt + 1
        ctx = T.ToolCtx(
            parent=parent, card=card, k_remaining=k_remaining,
            seeds=self.default_seeds, iteration=it,
            tried=tuple(sorted(self._exhausted)), attempt=attempt,
        )
        spec = T.build(name, ctx)
        if spec is None:
            # The tool's ranked candidates are exhausted (or it was never
            # applicable here). Not a model failure and not a decline: the tool
            # is retired and the iteration ends without an edit.
            self._exhausted.add(name)
            self._status = f"tool_exhausted:{name}"
            return None
        self._tools_tried.add(name)
        self._pending = (name, spec, card)
        return spec

    # -- the loop ---------------------------------------------------------
    def run(self, parent: ParentProblem, env_hash: str = ""):
        self.usage = Usage()
        self._steps = []
        self._history = []
        self._pending = None
        self._attempts = {}
        self._exhausted = set()
        rec = super().run(parent, env_hash=env_hash)
        # The parent loop labels a None spec "rule_stopped"; say which of the
        # several reasons it actually was.
        if self._status:
            rec.termination = self._status
        rec.campaign_id = f"{parent.parent_id}__tools__{self.model_id}"
        rec.controller = f"tools:{self.model_id}"
        rec.model_id = self.model_id
        rec.n_declines = sum(
            1 for s in self._steps if s.status == StepStatus.DECLINED)
        rec.n_malformed = sum(
            1 for s in self._steps if s.status == StepStatus.MALFORMED)
        rec.n_errors = sum(
            1 for s in self._steps if s.status == StepStatus.ERROR)
        rec.total_cost_usd = round(
            sum(s.cost_usd for s in self._steps), 6)
        # The scorer reads `total_llm_calls` off the row; the parent class is a
        # rule controller and never sets it, so a tool arm would report zero
        # calls and look free.
        rec.total_llm_calls = sum(s.n_calls for s in self._steps)
        rec.usage = self.usage.to_row()
        rec.usage["tool_choices"] = list(self._tool_choices)
        for ir, step in zip(rec.iterations, self._steps):
            ir.steps = list(getattr(ir, "steps", [])) + [step]
        return rec

    # -- history ----------------------------------------------------------
    def _record_previous(self, parent: ParentProblem, card: ScoreCard) -> None:
        """Close out the previous iteration's row from the card handed in now.

        The parent run loop owns the iteration records and does not pass them
        back into `_choose`, so the outcome of the last tool is read off the
        card the loop hands over on the next call. Deltas are against the
        ORIGINAL parent, matching render_history's contract and the mask arms'
        rows byte for byte -- the budget and the contract are both defined
        against the parent, so cumulative drift is the quantity that matters.
        """
        if not self._pending:
            return
        name, spec, card0 = self._pending
        self._pending = None
        improved = (
            card.ddg_bind < card0.ddg_bind - 1e-9
            or card.bsa > card0.bsa + 1e-9
            or card.max_strain < card0.max_strain - 1e-9
        )
        positions = tuple(spec.open_positions) or (
            tuple(spec.rebuild_segment) if spec.rebuild_segment else ())
        self._history.append({
            "iteration": len(self._history) + 1,
            "operator": name,
            "locus": ",".join(map(str, positions)),
            "cost": str(edit_cost(spec.operator, positions)),
            "d_strain": f"{card.max_strain - parent.scorecard0.max_strain:+.2f}",
            "d_bsa": f"{card.bsa - parent.scorecard0.bsa:+.0f}",
            "d_ddg": f"{card.ddg_bind - parent.scorecard0.ddg_bind:+.1f}",
            "outcome": "committed" if improved else "no improvement",
        })
