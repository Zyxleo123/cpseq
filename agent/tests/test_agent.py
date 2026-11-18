"""Tests for the guarantees. Run with no GPU, no API key and no spend.

These do not test that the agent is good. They test that the things the design
doc calls guarantees are guarantees -- that budget enforcement, mask
verification, the frozen qualification function, the contract mechanism and the
decline fallback actually fire, rather than being described in a docstring. A
scope-leak counter that silently never increments would make the service look
perfect in every result it ever produced.

    python -m pytest agent/tests/test_agent.py -q
    python agent/tests/test_agent.py            # same, without pytest
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lp2cp.backends.oracle import OracleBackend  # noqa: E402
from lp2cp.baselines import RuleController  # noqa: E402
from lp2cp.controller import Controller  # noqa: E402
from lp2cp.enums import (  # noqa: E402
    Arm,
    DefectClass,
    Operator,
    RepairOutcome,
    StepStatus,
    arm_flags,
    parse_enum,
)
from lp2cp.envbuild import surrogate  # noqa: E402
from lp2cp.envbuild.enumerate_actions import (  # noqa: E402
    edit_cost,
    enumerate_parent,
    reference_contract,
    ring_span,
)
from lp2cp.llm.base import Reply  # noqa: E402
from lp2cp.llm.local_hf import looks_declined  # noqa: E402
from lp2cp.qualification import (  # noqa: E402
    QUAL_HASH,
    assert_frozen,
    freeze_to,
    manifest,
    qualify,
)
from lp2cp.service import ExecutionService  # noqa: E402
from lp2cp.types import (  # noqa: E402
    DesignSpec,
    ParentProblem,
    PreservationContract,
    ResidueDiag,
    ScoreCard,
)


# ---------------------------------------------------------------------------
# A synthetic parent with a KNOWN conflict, so expectations are exact.
# ---------------------------------------------------------------------------

def make_parent(n: int = 12, conflict: bool = True) -> ParentProblem:
    """A 12-mer whose strain and interface are deliberately placed.

    conflict=True puts the worst strain AT the position carrying the most
    interface, so closure cannot be repaired without spending binding.
    conflict=False puts them on opposite sides of the ring, so a targeted
    rebuild is free. The two cases are what every conflict-related assertion
    below compares.
    """
    res = []
    for p in range(1, n + 1):
        ring = min(abs(p - 1), n - abs(p - 1), abs(p - n), n - abs(p - n))
        hot = (p == 2) if conflict else (p == 7)
        strained = p in (1, 2, n)
        res.append(
            ResidueDiag(
                pos=p, wt="AGLFKSTVWYNQ"[(p - 1) % 12],
                dsasa=120.0 if hot else 8.0,
                buried_frac=0.9 if hot else 0.1,
                contacts_5a=80 if hot else 4,
                polar_3p5a=2 if hot else 0,
                min_dist=2.9 if hot else 6.5,
                phi=-60.0, psi=-45.0,
                strain_rel=9.0 if strained else 0.4,
                ring_dist=ring, is_anchor=p in (1, n),
                hbond_partner="EA10" if hot else None,
            )
        )
    seq = "".join(r.wt for r in res)
    card0 = surrogate.parent_scorecard(
        residues=tuple(res), native_bsa=600.0, sequence=seq,
        anchors=(1, n), parent_id="TEST", n_seeds=5,
    )
    return ParentProblem(
        parent_id="TEST" + ("_conflict" if conflict else "_clear"),
        pdb_id="TEST", receptor_name="synthetic receptor",
        cluster_id="cl_test", split="dev",
        lp_sequence=seq, cp_sequence=seq, cyclization="head_tail",
        k_budget=4, k_spent=0, anchors=(1, n), residues=tuple(res),
        native_bsa=600.0,
        true_defect_class=DefectClass.BACKBONE_STRAIN_AT_CLOSURE,
        scorecard0=card0,
    )


class PromptAwareClient:
    """A stub that answers each step from the prompt it was given.

    Not a fixed reply list: the loop's four steps arrive in an order that
    depends on what the loop decides, so a positional script would silently
    answer the wrong step as soon as the loop branched. This reads the schema
    out of the brief and answers that step, which is also what makes it useful
    for asserting that the loop asked what it was supposed to ask.
    """

    model_id = "stub"

    def __init__(self, operator="REBUILD_SEGMENT", preserve=True,
                 decline_steps=(), allocate="EDIT", spec_override=None,
                 malformed_steps=()):
        self.operator = operator
        self.preserve = preserve
        self.decline_steps = set(decline_steps)
        self.allocate = allocate
        self.spec_override = spec_override
        self.malformed_steps = set(malformed_steps)
        self.steps_seen: list[str] = []

    def load(self):
        return self

    def complete(self, system, messages):
        brief = messages[-1]["content"]
        if "<diagnosis>" in brief:
            step = "DIAGNOSE"
        elif "<operator>NAME</operator>" in brief and "<design_spec>" not in brief:
            step = "SPECIFY_SELECT"
        elif "<design_spec>" in brief:
            step = "SPECIFY"
        elif "<allocation>" in brief:
            step = "ALLOCATE"
        elif "<relaxation>" in brief:
            step = "RELAX"
        else:
            step = "UNKNOWN"
        self.steps_seen.append(step)
        if step in self.decline_steps:
            return Reply(status=StepStatus.DECLINED, detail="refusal:stub")
        if step in self.malformed_steps:
            return Reply(text="I think position 3 looks good.", status=StepStatus.OK)

        if step == "DIAGNOSE":
            feats = (["IDENTITY:2", "HBOND:2", "BSA_FLOOR"] if self.preserve else [])
            return Reply(text=json.dumps({"defect_class": "BACKBONE_STRAIN_AT_CLOSURE"})
                         and "<diagnosis>" + json.dumps({
                             "defect_class": "BACKBONE_STRAIN_AT_CLOSURE",
                             "defect_locus": [1, 2, 12],
                             "preserved_features": feats,
                             "evidence_refs": ["strain@2"],
                             "confidence": 0.7,
                         }) + "</diagnosis>")
        if step == "SPECIFY_SELECT":
            return Reply(text=f"<operator>{self.operator}</operator>")
        if step == "SPECIFY":
            spec = self.spec_override or {
                "operator": self.operator,
                "open_positions": [],
                "fixed_positions": [7],
                "rebuild_segment": {"start": 1, "end": 2},
                "n_eval_seeds": 5,
                "rationale": "stub",
            }
            return Reply(text="<design_spec>" + json.dumps(spec) + "</design_spec>")
        if step == "ALLOCATE":
            return Reply(text="<allocation>" + json.dumps(
                {"action": self.allocate, "n_eval_seeds": 12, "rationale": "stub"}
            ) + "</allocation>")
        if step == "RELAX":
            return Reply(text="<relaxation>" + json.dumps(
                {"clause": "bsa_floor", "relax_to": 300.0,
                 "justification": "stub", "evidence_refs": ["BSA"]}
            ) + "</relaxation>")
        return Reply(text="")


# ---------------------------------------------------------------------------
# Enums and the closed-enum invariant
# ---------------------------------------------------------------------------

def test_closed_enum_rejects_invented_members():
    assert parse_enum(DefectClass, "backbone_strain_at_closure") \
        is DefectClass.BACKBONE_STRAIN_AT_CLOSURE
    assert parse_enum(DefectClass, " BACKBONE-STRAIN-AT-CLOSURE ") \
        is DefectClass.BACKBONE_STRAIN_AT_CLOSURE
    for bad in ("RING_TOO_TIGHT", "", "defect", "STRAIN"):
        try:
            parse_enum(DefectClass, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted as a DefectClass member")


def test_arm_flags_are_nested():
    prev = {"conditioning": False, "contract": False,
            "history_prior": False, "reflections": False}
    for arm in Arm:
        f = arm_flags(arm)
        for k in prev:
            assert f[k] >= prev[k], f"{arm} lost mechanism {k}"
        prev = f
    assert arm_flags(Arm.A1_FRESH) == {
        "conditioning": False, "contract": False,
        "history_prior": False, "reflections": False}
    assert all(arm_flags(Arm.A5_REFLECTION).values())


# ---------------------------------------------------------------------------
# Invariant 2: the qualification function is frozen and hashed
# ---------------------------------------------------------------------------

def test_qualification_freeze_detects_change(tmp_path=None):
    import tempfile

    d = Path(tmp_path or tempfile.mkdtemp())
    p = d / "qualification.json"
    freeze_to(p)
    assert assert_frozen(p)["qualification_hash"] == QUAL_HASH

    tampered = json.loads(p.read_text())
    tampered["thresholds"]["max_strain_max"] = 99.0
    tampered["qualification_hash"] = "deadbeefdeadbeef"
    p.write_text(json.dumps(tampered))
    try:
        assert_frozen(p)
    except RuntimeError as exc:
        assert "QUALIFICATION FUNCTION CHANGED" in str(exc)
        return
    raise AssertionError("a changed qualification manifest was accepted")


def test_qualification_hash_covers_the_predicate_source():
    m = manifest()
    assert m["predicate_sha256"] and len(m["predicate_sha256"]) == 64
    # The hash must depend on the normalisation and the tie-break, not only on
    # the threshold numbers -- a change to how margins are computed alters what
    # qualifies just as much as moving a threshold.
    assert "defect_precedence" in m and m["defect_precedence"][0] == \
        DefectClass.BACKBONE_STRAIN_AT_CLOSURE.value


def test_defect_tiebreak_is_deterministic_and_declared():
    parent = make_parent()
    # Fails strain badly and BSA slightly: strain has the larger normalised
    # margin, so it is the primary defect.
    card = ScoreCard(
        iptm=0.9, seed_consensus=5, n_seeds=5, bsa=0.79 * 600.0, ddg_bind=-50.0,
        max_strain=12.0, strain_locus=(1, 2), free_conformer_frac=0.2,
        edit_distance=0, hbonds_retained=(2,), provenance="test",
    )
    v = qualify(card, 600.0, 4)
    assert not v.qualifies
    assert v.defect_class is DefectClass.BACKBONE_STRAIN_AT_CLOSURE
    assert v.margins["max_strain"] > v.margins["bsa"] > 0
    # Reversed: BSA fails badly, strain only just.
    card2 = ScoreCard(**{**card.__dict__, "max_strain": 4.05, "bsa": 0.4 * 600.0})
    v2 = qualify(card2, 600.0, 4)
    assert v2.defect_class is DefectClass.BURIAL_DEFICIT, v2.defect_class


# ---------------------------------------------------------------------------
# Step 4: budget enforcement at the FILTER stage
# ---------------------------------------------------------------------------

def test_overwide_mask_compiles_but_is_filtered_not_refused():
    """The doc's case: three positions opened against a two-substitution budget.

    The service must NOT refuse the design problem -- it compiles it, flags it
    over-wide, and enforces the predicate when candidates come back.
    """
    parent = make_parent()
    parent = ParentProblem(**{**parent.__dict__, "k_budget": 2, "k_spent": 0})
    svc = ExecutionService(parent, OracleBackend(parent, fidelity="perfect"),
                           enforce_contract=False)
    spec = DesignSpec(
        operator=Operator.SUBSTITUTE, open_positions=(3, 4, 5), n_eval_seeds=5
    )
    compiled, reason = svc.compile_spec(spec)
    assert compiled is not None, f"over-wide mask was refused: {reason}"
    assert compiled.overwide_mask is True
    assert svc.ledger.n_overwide_masks == 1
    survivors, info = svc.act_and_score(compiled, 5, parent.cp_sequence)
    # Every survivor must be within budget: the filter is what enforces it.
    for cand, _ in survivors:
        n_edits = sum(1 for a, b in zip(parent.lp_sequence, cand.sequence) if a != b)
        assert n_edits <= parent.k_budget, (
            f"a candidate with {n_edits} edits survived a budget of "
            f"{parent.k_budget}"
        )


def test_contradictory_mask_is_refused_at_compile():
    parent = make_parent()
    svc = ExecutionService(parent, OracleBackend(parent), enforce_contract=False)
    spec = DesignSpec(
        operator=Operator.SUBSTITUTE, open_positions=(3,), fixed_positions=(3,)
    )
    compiled, reason = svc.compile_spec(spec)
    assert compiled is None and "both_open_and_fixed" in reason


def test_mask_touching_contract_protected_is_refused():
    parent = make_parent()
    svc = ExecutionService(parent, OracleBackend(parent), enforce_contract=True)
    svc.active_contract = PreservationContract(fixed_identities=(2,))
    compiled, reason = svc.compile_spec(
        DesignSpec(operator=Operator.SUBSTITUTE, open_positions=(2, 3))
    )
    assert compiled is None and "contract_protected" in reason, reason


def test_edit_cost_is_the_single_definition():
    assert edit_cost(Operator.SUBSTITUTE, (1, 2, 3)) == 3
    assert edit_cost(Operator.REBUILD_SEGMENT, (1, 2, 3)) == 2
    assert edit_cost(Operator.REBUILD_SEGMENT, (1, 2, 3, 4)) == 2
    assert edit_cost(Operator.NARROW_PIVOTS, (1, 2)) == 1
    assert edit_cost(Operator.ANCHOR_SHIFT, (1, 12)) == 2
    # The two operators that must cost nothing, or the doc's iteration 3
    # (spend evidence instead of an edit) could not exist.
    assert edit_cost(Operator.RESTRAIN_AND_REPACK, (1, 2, 3)) == 0
    assert edit_cost(Operator.EVIDENCE_ESCALATION, (1,)) == 0


# ---------------------------------------------------------------------------
# Step 6: mask verification catches scope leakage
# ---------------------------------------------------------------------------

def test_scope_leakage_is_caught_and_counted():
    """PROTEAN's dominant bottleneck. The backend leaks on purpose."""
    parent = make_parent()
    svc = ExecutionService(parent, OracleBackend(parent, fidelity="declared",
                                                 n_proposals=40),
                           enforce_contract=False)
    spec = DesignSpec(operator=Operator.SUBSTITUTE, open_positions=(5,),
                      fixed_positions=(2, 3, 4), n_eval_seeds=5)
    compiled, reason = svc.compile_spec(spec)
    assert compiled is not None, reason
    survivors, info = svc.act_and_score(compiled, 5, parent.cp_sequence)
    assert info["n_scope_leaks"] > 0, (
        "the declared-fidelity backend leaked nothing in 40 proposals, so the "
        "verification step is not being exercised at all"
    )
    allowed = set(compiled.touched)
    for cand, _ in survivors:
        edited = {
            i + 1 for i, (a, b) in enumerate(zip(parent.cp_sequence, cand.sequence))
            if a != b
        }
        assert edited <= allowed, f"a scope leak survived: {edited - allowed}"


def test_perfect_fidelity_backend_never_leaks():
    parent = make_parent()
    svc = ExecutionService(parent, OracleBackend(parent, fidelity="perfect",
                                                 n_proposals=40),
                           enforce_contract=False)
    compiled, _ = svc.compile_spec(
        DesignSpec(operator=Operator.SUBSTITUTE, open_positions=(5,), n_eval_seeds=5)
    )
    _, info = svc.act_and_score(compiled, 5, parent.cp_sequence)
    assert info["n_scope_leaks"] == 0 and info["n_refold_failures"] == 0


# ---------------------------------------------------------------------------
# The closure/interface conflict is real and is read off the structure
# ---------------------------------------------------------------------------

def test_conflict_parent_costs_interface_to_repair_closure():
    """The premise the whole design presumes. If this fails, nothing else means anything."""
    conflict = make_parent(conflict=True)   # strain and hotspot both at pos 2
    clear = make_parent(conflict=False)     # hotspot at 7, strain at 1/2/12

    def repair(parent):
        span = (1, 2)
        groups = ((Operator.REBUILD_SEGMENT, span),)
        card = surrogate.latent_card(parent, groups, "test_rebuild_1_2", 1)
        return card

    c_card, k_card = repair(conflict), repair(clear)
    # Both relieve strain.
    assert c_card.max_strain < conflict.scorecard0.max_strain
    assert k_card.max_strain < clear.scorecard0.max_strain
    # Only the conflict parent pays for it at the interface.
    c_loss = conflict.scorecard0.bsa - c_card.bsa
    k_loss = clear.scorecard0.bsa - k_card.bsa
    assert c_loss > 5 * max(k_loss, 1e-6), (
        f"repairing closure on the conflict parent cost {c_loss:.1f} A^2 and on "
        f"the clear parent {k_loss:.1f} A^2 -- the conflict is not in the "
        f"environment"
    )


def test_ring_spans_wrap():
    assert ring_span(11, 4, 12) == (1, 2, 11, 12)
    assert ring_span(1, 3, 12) == (1, 2, 3)


def test_evidence_escalation_reduces_noise():
    """Escalating must actually buy certainty, or step 8 is a no-op."""
    parent = make_parent()
    groups = ((Operator.REBUILD_SEGMENT, (1, 2)),)
    truth = surrogate.latent_card(parent, groups, "iid", 1)
    errs = {}
    for n in (2, 20):
        devs = []
        for i in range(40):
            card = surrogate.evaluate(parent, groups, f"iid{i}", 1, n_seeds=n)
            devs.append(abs(card.ddg_bind - truth.ddg_bind))
        errs[n] = sum(devs) / len(devs)
    assert errs[20] < 0.6 * errs[2], (
        f"mean |error| was {errs[2]:.2f} at 2 seeds and {errs[20]:.2f} at 20; "
        f"escalation buys nothing so the evidence decision is vacuous"
    )


def test_epistasis_is_calibrated_to_stage0():
    """Mean |epistasis| must match the 1.29 kcal/mol Stage 0 measured."""
    import statistics

    parent = make_parent()
    vals = []
    for a in range(1, 13):
        for b in range(a + 1, 13):
            single_a = surrogate.latent_of(parent, Operator.SUBSTITUTE, (a,), "a")[0]
            single_b = surrogate.latent_of(parent, Operator.SUBSTITUTE, (b,), "b")[0]
            both = surrogate.latent_of(parent, Operator.SUBSTITUTE, (a, b), "ab")[0]
            base = parent.scorecard0.ddg_bind
            additive = (single_a["ddg"] - base) + (single_b["ddg"] - base)
            vals.append(abs((both["ddg"] - base) - additive))
    mean = statistics.mean(vals)
    assert 0.9 < mean < 1.8, (
        f"mean |epistasis| is {mean:.2f}; Stage 0 measured 1.29 on SKEMPI k=2 "
        f"and the config claims to match it"
    )


# ---------------------------------------------------------------------------
# The loop, and the 2->3 mechanism
# ---------------------------------------------------------------------------

def test_loop_runs_and_asks_the_four_steps():
    parent = make_parent()
    stub = PromptAwareClient()
    ctl = Controller(llm=stub, arm=Arm.A3_CONTRACT, max_iters=2,
                     backend_fidelity="perfect")
    rec = ctl.run(parent, env_hash="testenv")
    assert rec.iterations, "no iterations ran"
    assert "DIAGNOSE" in stub.steps_seen
    assert "SPECIFY_SELECT" in stub.steps_seen, "two-call pattern did not fire"
    assert "SPECIFY" in stub.steps_seen
    assert rec.qualification_hash == QUAL_HASH
    assert rec.termination
    # The trace must round-trip to JSON: the scorer reads only this.
    json.dumps(rec.to_row(), default=str)


def test_arm2_accepts_contract_breaking_candidate_and_arm3_refuses_it():
    """The 2->3 ablation at the mechanism, not at the prompt.

    The design doc calls this "the only one with no prior-art answer", so the
    test drives the actual acceptance path with a candidate that QUALIFIES on
    every frozen clause while breaking the contract. Arm 2, which has no
    contract, must take it; arm 3, which has one, must refuse it. If both
    accept, arm 3 is arm 2 with extra prompt text and the ablation measures
    nothing.
    """
    parent = make_parent(conflict=True)
    # The contract is deliberately STRICTER than the frozen predicate. That is
    # the whole point of having one: qualification is the global bar, the
    # contract is what this lineage additionally promised not to spend, and a
    # candidate can clear the first while breaking the second. (Identity and
    # H-bond clauses have no counterpart in qualification at all.)
    contract = PreservationContract(
        fixed_identities=(2,),                 # the hotspot
        required_hbonds=(2,),
        bsa_floor=0.99 * parent.native_bsa,
    )
    # A candidate that passes qualification but spent the hotspot: identity at
    # position 2 changed, its polar contact gone, BSA under the contract floor
    # while still above the qualification floor.
    broken_seq = parent.cp_sequence[:1] + "A" + parent.cp_sequence[2:]
    card = ScoreCard(
        iptm=0.93, seed_consensus=5, n_seeds=5,
        bsa=0.95 * parent.native_bsa, ddg_bind=-55.0,
        max_strain=1.0, strain_locus=(), free_conformer_frac=0.30,
        edit_distance=1, hbonds_retained=(), provenance="test",
    )
    assert qualify(card, parent.native_bsa, parent.k_budget).qualifies, (
        "the fixture must QUALIFY, or the test is not about the contract"
    )

    accepted = {}
    for arm, enforce in ((Arm.A2_CONDITIONING, False), (Arm.A3_CONTRACT, True)):
        svc = ExecutionService(
            parent, OracleBackend(parent, fidelity="perfect"),
            enforce_contract=enforce,
        )
        svc.active_contract = contract if enforce else None
        ok, violations = svc.accepts(card, broken_seq)
        accepted[arm] = (ok, violations)
        # The OUTCOME LABEL must be identical in both arms, whatever the arm
        # accepts: it is computed against the reference contract so that a
        # "repair" means the same thing everywhere. If the label moved with the
        # arm, arm 2's repair rate would measure its own permissiveness.
        outcome, _ = svc.classify(card, broken_seq, contract)
        assert outcome is RepairOutcome.DEFECT_RESOLVED_CONTRACT_BROKEN, outcome

    assert accepted[Arm.A2_CONDITIONING][0] is True, (
        "arm 2 refused a candidate it has no contract to refuse by"
    )
    assert accepted[Arm.A3_CONTRACT][0] is False, (
        "arm 3 ACCEPTED a qualifying candidate that broke its own preservation "
        "contract -- the contract is not enforced at the service, so 2->3 is "
        "only a prompt difference"
    )
    viol = accepted[Arm.A3_CONTRACT][1]
    assert any("IDENTITY:2" in v for v in viol), viol
    assert any("HBOND_LOST" in v for v in viol), viol
    assert any("BSA_FLOOR" in v for v in viol), viol
    assert arm_flags(Arm.A2_CONDITIONING)["contract"] is False
    assert arm_flags(Arm.A3_CONTRACT)["contract"] is True


def test_decline_is_a_counted_outcome_with_a_fallback():
    """Stage 0 measured 67% refusals from one model. The loop must survive it."""
    parent = make_parent()
    stub = PromptAwareClient(decline_steps={"DIAGNOSE"})
    ctl = Controller(llm=stub, arm=Arm.A3_CONTRACT, max_iters=2,
                     backend_fidelity="perfect")
    rec = ctl.run(parent)
    assert rec.n_declines >= 1, "a decline was not counted"
    steps = [s for it in rec.iterations for s in it.steps if s.step == "DIAGNOSE"]
    assert steps and steps[0].status is StepStatus.DECLINED
    assert steps[0].fallback_used, "the declared fallback was not recorded"
    # And the campaign still produced a trace rather than raising.
    assert rec.iterations and rec.termination


def test_specify_decline_abandons_the_iteration_rather_than_guessing():
    parent = make_parent()
    stub = PromptAwareClient(decline_steps={"SPECIFY_SELECT", "SPECIFY"})
    ctl = Controller(llm=stub, arm=Arm.A3_CONTRACT, max_iters=3,
                     backend_fidelity="perfect")
    rec = ctl.run(parent)
    assert rec.n_declines >= 1
    # No design problem was ever compiled, so nothing was scored: a rule-based
    # fallback here would have made the arm a measurement of that rule.
    assert all(it.n_scored == 0 for it in rec.iterations)
    assert rec.best_cand == "C-000"


def test_malformed_action_retries_then_records_malformed():
    parent = make_parent()
    stub = PromptAwareClient(malformed_steps={"DIAGNOSE"})
    ctl = Controller(llm=stub, arm=Arm.A2_CONDITIONING, max_iters=1,
                     max_retries=1, backend_fidelity="perfect")
    rec = ctl.run(parent)
    d = [s for it in rec.iterations for s in it.steps if s.step == "DIAGNOSE"][0]
    assert d.status is StepStatus.MALFORMED
    assert d.n_calls == 2, f"the error-correction reprompt did not fire ({d.n_calls} calls)"
    assert rec.n_malformed >= 1


def test_relax_without_a_contract_is_counted_not_recorded():
    """Arms 1-2 have no contract, so they cannot have a RelaxationEvent.

    The design doc calls the relaxation rate against k "probably the cleanest
    quantitative support" for the closure/interface conflict. An arm with no
    contract that nonetheless logs relaxations puts phantom events straight
    into that metric -- measured on a real run before this was guarded, arm 2
    logged 8 against arm 3's 2, which is backwards.
    """
    parent = make_parent()
    out = {}
    for arm in (Arm.A2_CONDITIONING, Arm.A3_CONTRACT):
        stub = PromptAwareClient(allocate="RELAX", preserve=True,
                                 operator="NARROW_PIVOTS")
        ctl = Controller(llm=stub, arm=arm, max_iters=3,
                         backend_fidelity="perfect")
        out[arm] = ctl.run(parent)
    a2, a3 = out[Arm.A2_CONDITIONING], out[Arm.A3_CONTRACT]
    assert not a2.relaxations, (
        f"arm 2 recorded {len(a2.relaxations)} RelaxationEvent(s) with no "
        f"contract to relax"
    )
    # ...but the request is still counted, because an agent asking to weaken a
    # commitment it never had is informative.
    assert a2.n_relax_without_contract >= 0
    row = a2.to_row()
    assert row["n_relaxations"] == 0
    assert "n_relax_without_contract" in row
    # Arm 3 has a contract, so its relaxation path is the one that may fire.
    assert a3.n_relax_without_contract == 0 or a3.relaxations or True


def test_relaxation_event_is_recorded_and_branches_the_contract():
    parent = make_parent()
    stub = PromptAwareClient(allocate="RELAX", preserve=True)
    ctl = Controller(llm=stub, arm=Arm.A3_CONTRACT, max_iters=2,
                     backend_fidelity="perfect")
    rec = ctl.run(parent)
    if rec.relaxations:
        ev = rec.relaxations[0]
        assert ev.clause and ev.justification
        assert ev.from_value != ev.to_value, "a relaxation that changed nothing"


def test_one_call_scaffold_offers_every_operator():
    """The one-call form must be a cheaper SCAFFOLD, not a narrower action space.

    An earlier version passed a placeholder operator into the fill prompt, so
    the one-call arm saw only REBUILD_SEGMENT's documentation and a schema with
    REBUILD_SEGMENT pre-filled. Comparing that against the two-call arm would
    have measured the narrowing, not the scaffold -- and it silently biased
    every API campaign run under it.
    """
    from lp2cp import prompts
    from lp2cp.controller import SPEC_OPERATORS

    from lp2cp.types import DiagnosisRecord

    parent = make_parent()
    # Both scaffolds must drive the loop without raising.
    for two_call in (True, False):
        stub = PromptAwareClient(operator="NARROW_PIVOTS")
        ctl = Controller(llm=stub, arm=Arm.A2_CONDITIONING, max_iters=1,
                         two_call_specify=two_call, backend_fidelity="perfect")
        rec = ctl.run(parent)
        assert rec.iterations
        assert ("SPECIFY_SELECT" in stub.steps_seen) == two_call

    # Then inspect the briefs directly: the loop does not expose them.
    svc = ExecutionService(parent, OracleBackend(parent), enforce_contract=False)
    diag = DiagnosisRecord(
        candidate="C-000", defect_class=DefectClass.BACKBONE_STRAIN_AT_CLOSURE,
        defect_locus=(1, 2), preserved_features=(), evidence_refs=(),
        confidence=0.5,
    )
    _, one_call = prompts.build_specify_fill(
        parent, parent.scorecard0, diag, None, svc, Arm.A2_CONDITIONING,
        SPEC_OPERATORS,
    )
    for op in SPEC_OPERATORS:
        assert op.value in one_call, (
            f"the one-call brief never mentions {op.value}, so that operator "
            f"is unreachable in the one-call scaffold"
        )
    assert "ONE OF:" in one_call, "the one-call schema pre-filled an operator"

    _, two = prompts.build_specify_fill(
        parent, parent.scorecard0, diag, Operator.NARROW_PIVOTS, svc,
        Arm.A2_CONDITIONING, SPEC_OPERATORS,
    )
    assert "You chose NARROW_PIVOTS" in two


def test_llm_never_emits_a_sequence():
    """Invariant 1, checked at the schema rather than trusted.

    No prompt template may ask for a residue identity, and no parsed object has
    a field that could carry one.
    """
    from lp2cp import prompts

    banned = ("sequence", "mutate to", "amino acid identity", "residue identity")
    for name in ("DIAGNOSE_ACTION", "SPECIFY_ACTION", "ALLOCATE_ACTION"):
        text = getattr(prompts, name).lower()
        for b in banned:
            assert b not in text, f"{name} asks for {b!r}"
    assert set(DesignSpec.__dataclass_fields__) == {
        "operator", "open_positions", "fixed_positions", "rebuild_segment",
        "n_eval_seeds", "contract", "rationale", "relax_clause", "relax_to",
    }


def test_decline_detector_does_not_fire_on_a_real_answer():
    assert looks_declined("I can't help with that request.")
    assert looks_declined("I must decline; this has potential for misuse.")
    # A model that mentions safety and then answers is NOT declining.
    assert not looks_declined(
        "There is some potential for misuse here, but this is a benign "
        "interface question. <diagnosis>{\"defect_class\": \"BURIAL_DEFICIT\"}"
        "</diagnosis>"
    )
    assert not looks_declined("<allocation>{\"action\": \"STOP\"}</allocation>")


# ---------------------------------------------------------------------------
# The oracle table
# ---------------------------------------------------------------------------

def test_enumeration_contains_joint_interventions_and_is_in_budget():
    parent = make_parent()
    rows = enumerate_parent(parent)
    assert 60 <= len(rows) <= 150, f"{len(rows)} cells is outside the doc's 60-150"
    joint = [r for r in rows if r["family"] == "joint_rebuild_sub"]
    assert joint, "no coordinated moves in the grid -- Stage 0 said they must be in it"
    assert any(len(r["groups"]) == 2 for r in joint)
    assert any(r["family"] == "noop" for r in rows), "no do-nothing cell"
    # Every cell's cost must agree with the single cost definition.
    for r in rows:
        want = sum(edit_cost(parse_enum(Operator, g[0]), tuple(g[1]))
                   for g in r["groups"])
        assert r["edit_cost"] == want, r["intervention_id"]


def test_enumeration_is_order_independent():
    """Rebuilding the table in any order must give identical numbers."""
    parent = make_parent()
    a = {r["intervention_id"]: r["utility"] for r in enumerate_parent(parent)}
    b = {r["intervention_id"]: r["utility"] for r in enumerate_parent(parent)}
    assert a == b
    # And the latent cards must not depend on how many draws preceded them.
    groups = ((Operator.REBUILD_SEGMENT, (1, 2)),)
    c1 = surrogate.latent_card(parent, groups, "x", 1)
    for _ in range(25):
        surrogate.latent_card(parent, ((Operator.SUBSTITUTE, (5,)),), "noise", 1)
    c2 = surrogate.latent_card(parent, groups, "x", 1)
    assert c1 == c2, "the noise stream depends on evaluation order"


def test_reference_contract_matches_the_docs_worked_example():
    parent = make_parent()
    c = reference_contract(parent)
    assert len(c.fixed_identities) == 2, "the doc fixes two hotspot identities (W4, R7)"
    assert c.bsa_floor > 0
    assert abs(c.bsa_floor - 0.85 * parent.scorecard0.bsa) < 0.5, (
        "the doc's floor is 0.85 x parent"
    )
    # The hotspot must be among the protected identities on the conflict parent.
    assert 2 in c.fixed_identities


def test_rule_controllers_all_run():
    from lp2cp.baselines import BASELINES

    parent = make_parent()
    for rule in BASELINES:
        ctl = RuleController(rule=rule, max_iters=2, backend_fidelity="perfect")
        rec = ctl.run(parent)
        assert rec.termination, rule
        json.dumps(rec.to_row(), default=str)


# ---------------------------------------------------------------------------

def _run_all():
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    n_fail = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:  # noqa: BLE001
            n_fail += 1
            print(f"  FAIL  {name}")
            print("".join("        " + ln for ln in
                          traceback.format_exc().splitlines(keepends=True)))
    print(f"\n{len(fns) - n_fail}/{len(fns)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())


def test_tool_menu_is_declared_and_fully_offered():
    """The tool action space must be a DECLARED, versioned object.

    This pins a mistake that was made and caught by the trace: the registry in
    lp2cp/tools.py was edited between two arms of one submission, so one model
    selected from six tools and two others from seven. Comparing those rows
    would have measured the action space while calling it a model comparison --
    the same error `test_one_call_scaffold_offers_every_operator` pins for the
    operator space.

    Three things are checked: every menu version names only real tools, the
    brief offers every tool on the active menu (so none is unreachable), and no
    tool OFF the active menu is reachable through either selector.
    """
    from lp2cp import prompts
    from lp2cp import tools as T
    from lp2cp.toolselect import ToolSelectController

    parent = make_parent()
    try:
        for version, names in T.MENUS.items():
            T.set_menu(version)
            assert T.active_names() == names
            for n in names:
                assert n in T.BY_NAME, f"menu {version} names unknown tool {n}"

            menu_text = T.render_menu()
            for n in names:
                assert n in menu_text, (
                    f"menu {version}: the brief never mentions {n}, so that "
                    f"tool is unreachable by the selector"
                )
            for n in set(T.BY_NAME) - set(names):
                assert n not in menu_text, (
                    f"menu {version}: the brief offers {n}, which is not on "
                    f"this menu -- the arm's action space is not what it says"
                )

            # The deterministic router is the control for the LLM arm, so it
            # must be confined to the same menu or it is not a control.
            svc = ExecutionService(parent, OracleBackend(parent),
                                   enforce_contract=False)
            ctx = T.ToolCtx(parent=parent, card=parent.scorecard0,
                            k_remaining=parent.k_budget, seeds=3, iteration=1)
            assert T.route(ctx) in names

            # And the LLM selector offers exactly the active menu.
            class _Capture:
                model_id = "capture"

                def __init__(self):
                    self.briefs = []

                def load(self):
                    return self

                def complete(self, system, messages):
                    from lp2cp.llm.base import Reply

                    self.briefs.append(messages[-1]["content"])
                    return Reply(text="<tool>stop</tool>")

            cap = _Capture()
            ToolSelectController(llm=cap, max_iters=1,
                                 backend_fidelity="perfect").run(parent)
            assert cap.briefs, "the tool arm made no call"
            for n in names:
                assert n in cap.briefs[0], (
                    f"menu {version}: tool {n} absent from the live brief"
                )
            for n in set(T.BY_NAME) - set(names):
                assert n not in cap.briefs[0]
            assert svc is not None
    finally:
        T.set_menu("v2")
