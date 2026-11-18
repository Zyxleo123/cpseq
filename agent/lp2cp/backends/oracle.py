"""The enumerated-table backend: the campaign simulator.

Design doc, on why the enumerated table is the right instrument: "the table
answers every query a controller would have made, so a whole campaign replays
offline at any budget T. End-to-end success becomes a deterministic function of
measured per-decision regret, and hundreds of prompt variants can run against
the simulator in an afternoon."

This backend deliberately misbehaves in three declared ways, because a backend
that never misbehaves makes the execution service's guarantees untestable and
PROTEAN's measured finding is that backend behaviour, not constraint
compilation, is the bottleneck:

  SCOPE LEAKAGE   a fraction of returned candidates silently edit a position
                  that was declared fixed. PROTEAN names this as one of its two
                  dominant bottlenecks. Step 6 of the loop exists for it, and
                  the service's catch rate is a reportable number.
  BUDGET OVERRUN  a fraction spend more substitutions than remain, which is
                  what step 4's filter stage is for.
  REFOLD FAILURE  a fraction return nothing at all. PROTEAN's other dominant
                  bottleneck. A controller that cannot make progress because
                  the backend failed must not record that as a design verdict.

All three are seeded off the candidate's own identity, so a campaign replays
identically and two arms asking the same question get the same misbehaviour --
without which a difference between arms could be the backend's luck.

Residue identities come from a small declared substitution table rather than
from the LLM. That is invariant 1: the LLM never emits a sequence, so a win
cannot be attributed to luck on chemistry. Which identity is chosen does not
enter the surrogate's response at all -- only WHICH POSITIONS were touched does
-- so the table exists to produce a concrete, diffable sequence for the
service to verify, not to model chemistry.
"""

from __future__ import annotations

from ..enums import Operator
from ..envbuild import surrogate
from ..envbuild.surrogate import SurrogateConfig, _unit
from ..types import Candidate, ScoreCard, sha

# What a side-chain solver would reach for, per wild-type residue: a
# conservative swap and a smaller one. Declared, not searched.
SOLVER_CHOICES: dict[str, str] = {
    "A": "G", "C": "S", "D": "N", "E": "Q", "F": "L", "G": "A", "H": "N",
    "I": "V", "K": "R", "L": "A", "M": "L", "N": "D", "P": "A", "Q": "E",
    "R": "K", "S": "A", "T": "S", "V": "A", "W": "F", "Y": "F",
}


class OracleBackend:
    """Deterministic simulator over the enumerated action space."""

    name = "oracle_table"

    def __init__(
        self,
        parent,
        cfg: SurrogateConfig = surrogate.CFG,
        n_proposals: int = 6,
        fidelity: str = "declared",
    ):
        self.parent = parent
        self.cfg = cfg
        self.n_proposals = n_proposals
        # `fidelity="perfect"` turns the three failure modes off. Used only by
        # the unit tests and by a control arm that measures how much of each
        # arm's shortfall is the backend rather than the controller.
        self.fidelity = fidelity
        self.version = f"{surrogate.provenance(cfg)}|fid={fidelity}|n={n_proposals}"

    # -- step 5 -----------------------------------------------------------
    def propose(self, compiled, current_seq: str) -> list[Candidate]:
        """Sequences inside the mask, plus the declared misbehaviour."""
        groups = compiled.groups
        if not groups:
            return []
        touched = compiled.touched
        iid = compiled.intervention_id
        out: list[Candidate] = []
        leak_rate = 0.0 if self.fidelity == "perfect" else self.cfg.scope_leak_rate
        over_rate = 0.0 if self.fidelity == "perfect" else self.cfg.budget_overrun_rate
        fail_rate = 0.0 if self.fidelity == "perfect" else self.cfg.refold_failure_rate

        for i in range(self.n_proposals):
            key = (self.parent.parent_id, iid, i)
            if _unit("refold", *key) < fail_rate:
                out.append(
                    Candidate(
                        cand_id=f"{self.parent.parent_id}#{sha(key, 8)}",
                        parent_id=self.parent.parent_id,
                        sequence=None,          # the service counts this as a
                        edits=(),               # refold failure and drops it
                        operator=groups[0][0],
                        locus=touched,
                        backend=self.name,
                        intervention_id=iid,
                    )
                )
                continue

            seq = list(current_seq)
            edited: list[int] = []
            # RESTRAIN_AND_REPACK changes no identity by construction, so it
            # produces a candidate identical in sequence to its parent. That is
            # correct and is why it costs no budget.
            identity_changing = [
                p for op, ps in groups for p in ps
                if op in (Operator.SUBSTITUTE, Operator.REBUILD_SEGMENT,
                          Operator.ANCHOR_SHIFT, Operator.NARROW_PIVOTS)
            ]
            # A rebuild only changes identities at some of its span -- the
            # budget model in edit_cost() charges ceil(L/2), so the sequence it
            # produces has to match that or the service's filter and the
            # oracle's cost model would disagree.
            budget_here = min(
                compiled.max_edit_cost,
                max(1, (len(identity_changing) + 1) // 2)
                if any(op == Operator.REBUILD_SEGMENT for op, _ in groups)
                else len(identity_changing),
            )
            ranked = sorted(
                set(identity_changing),
                key=lambda p: _unit("pick", *key, p),
            )
            for p in ranked[:budget_here]:
                wt = current_seq[p - 1]
                new = SOLVER_CHOICES.get(wt, "A")
                if new == wt:
                    continue
                seq[p - 1] = new
                edited.append(p)

            if _unit("leak", *key) < leak_rate:
                # Scope leakage: move a position OUTSIDE the mask. Chosen from
                # the declared-fixed set when there is one, because that is the
                # case the verification step is for -- a solver that quietly
                # moved a position the lineage promised to keep.
                pool = [
                    p for p in (compiled.declared_fixed or
                                tuple(range(1, self.parent.n_pos + 1)))
                    if p not in touched
                ]
                if pool:
                    p = pool[int(_unit("leakpos", *key) * len(pool)) % len(pool)]
                    wt = current_seq[p - 1]
                    seq[p - 1] = SOLVER_CHOICES.get(wt, "A")
                    edited.append(p)

            if _unit("over", *key) < over_rate:
                # Budget overrun: spend one more substitution than allowed,
                # inside the mask. Caught at the filter stage, not refused at
                # compile, which is the doc's instruction.
                pool = [p for p in touched if p not in edited]
                if pool:
                    p = pool[int(_unit("overpos", *key) * len(pool)) % len(pool)]
                    wt = current_seq[p - 1]
                    seq[p - 1] = SOLVER_CHOICES.get(wt, "A")
                    edited.append(p)

            new_seq = "".join(seq)
            out.append(
                Candidate(
                    cand_id=f"{self.parent.parent_id}#{sha((iid, i, new_seq), 8)}",
                    parent_id=self.parent.parent_id,
                    sequence=new_seq,
                    edits=tuple(
                        (p, current_seq[p - 1], new_seq[p - 1])
                        for p in sorted(set(edited))
                    ),
                    operator=groups[0][0],
                    locus=touched,
                    backend=self.name,
                    intervention_id=iid,
                )
            )
        return out

    # -- step 7 -----------------------------------------------------------
    def score(self, compiled, cand: Candidate, n_seeds: int) -> ScoreCard:
        """Evaluate at the given depth. LLM-independent by construction.

        The surrogate's response depends on which POSITIONS the intervention
        touched, never on which identities the solver picked -- so this cannot
        reward an agent for guessing chemistry, which is invariant 1 enforced
        at the scoring layer rather than only at the prompt.
        """
        groups = compiled.groups
        edit_distance = sum(
            1 for a, b in zip(self.parent.lp_sequence, cand.sequence or "") if a != b
        )
        if not groups:
            return self.parent.scorecard0
        return surrogate.evaluate(
            self.parent, groups, cand.intervention_id or compiled.intervention_id,
            edit_distance, n_seeds=n_seeds, cfg=self.cfg,
        )

    def heldout(self, compiled, cand: Candidate) -> ScoreCard:
        """The design doc's held-out protocol: fresh seeds, second predictor.

        "every method's frozen shortlist goes through the same held-out
        protocol: fresh folding seeds, a second predictor, chemistry-aware
        refinement, extended free-peptide sampling. The reportable number is
        the search-time-to-held-out gap per method, not the win."
        """
        groups = compiled.groups
        edit_distance = sum(
            1 for a, b in zip(self.parent.lp_sequence, cand.sequence or "") if a != b
        )
        if not groups:
            return self.parent.scorecard0
        return surrogate.evaluate(
            self.parent, groups, cand.intervention_id or compiled.intervention_id,
            edit_distance, n_seeds=self.cfg.heldout_seeds, heldout=True, cfg=self.cfg,
        )
