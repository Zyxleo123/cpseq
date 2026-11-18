"""Allocators: the rule-based bar, the controls, and the LLM.

Every proposer answers the same question -- given a decision and its evidence,
which single candidate gets the budget -- so that regret is paired across arms
on identical decisions.

The bar the doc's kill criterion names is a "hotspot/conservation heuristic".
That is implemented three ways (burial, contact count, ESM tolerance) plus a
`composite` that z-scores and sums them, and the headline comparison is against
the STRONGEST of them. Taking the max over several heuristics inflates the bar
by multiple comparisons, which is the conservative direction: it makes an LLM
win harder to claim, not easier.

Two entries here are controls rather than baselines:

  first_listed   Always takes the first candidate in the presented order. The
                 order is a seeded shuffle, so this is a floor equal to random
                 -- but run against an LLM it detects position bias, which is
                 otherwise indistinguishable from competence.
  most_measured  Reads n_records, which no legitimate arm may see. If this wins,
                 the benchmark is measuring how often someone assayed a
                 position, not where an edit belongs.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

from prompts import (
    RETRY_REASON,
    RETRY_TEMPLATE,
    build_prompt,
    parse_choice,
)


class Choice(dict):
    """{cand_id, status, n_retries, raw, scores}"""


class Proposer(Protocol):
    name: str
    needs_llm: bool

    def choose(
        self, decision: dict, features: dict, conservation: dict, order: list[str]
    ) -> Choice: ...


# ---------------------------------------------------------------------------
# Feature aggregation for a candidate (a position SET of size k)
# ---------------------------------------------------------------------------

def candidate_stats(cand: dict, features: dict, conservation: dict) -> dict:
    """Aggregate per-position evidence over the positions in one candidate.

    Burial and contacts SUM: spending two edits on two buried positions costs
    both. Tolerance takes the MINIMUM entropy, because a set is only as
    affordable as its least tolerant member -- averaging would let one very
    tolerant position hide an intolerant one.
    """
    dsasa = contacts = polar = 0.0
    burfrac: list[float] = []
    mindist: list[float] = []
    entropy: list[float] = []
    wt_prob: list[float] = []
    rim_rank: list[int] = []
    n_missing = 0
    class_rank = {"RIM": 0, "SUP": 1, "COR": 2}

    for (chain, pos), loc in zip(cand["pdb_sites"], cand["locations"]):
        f = features.get(f"{chain}:{pos}")
        c = conservation.get(f"{chain}:{pos}", {})
        rim_rank.append(class_rank.get(loc, 1))
        if not f:
            n_missing += 1
            continue
        dsasa += f.get("dsasa") or 0.0
        contacts += f.get("n_contacts_5A") or 0
        polar += f.get("n_polar_pairs_3p5A") or 0
        if f.get("buried_frac_on_binding") is not None:
            burfrac.append(f["buried_frac_on_binding"])
        if f.get("min_dist_to_partner") is not None:
            mindist.append(f["min_dist_to_partner"])
        if c.get("entropy") is not None:
            entropy.append(c["entropy"])
        if c.get("wt_prob") is not None:
            wt_prob.append(c["wt_prob"])

    return {
        "dsasa": dsasa,
        "contacts": contacts,
        "polar": polar,
        "burfrac": max(burfrac) if burfrac else None,
        "mindist": min(mindist) if mindist else None,
        "entropy": min(entropy) if entropy else None,
        "wt_prob": max(wt_prob) if wt_prob else None,
        "class_rank": max(rim_rank) if rim_rank else 1,
        "n_missing": n_missing,
    }


def _zscores(vals: list[float | None]) -> list[float]:
    got = [v for v in vals if v is not None]
    if len(got) < 2:
        return [0.0 for _ in vals]
    mu = sum(got) / len(got)
    sd = math.sqrt(sum((v - mu) ** 2 for v in got) / (len(got) - 1)) or 1.0
    return [0.0 if v is None else (v - mu) / sd for v in vals]


class HeuristicProposer:
    """Scores every candidate with a closed-form rule; lowest score wins."""

    needs_llm = False

    def __init__(self, name: str, oracle_lookup=None):
        self.name = name
        self._oracle_lookup = oracle_lookup

    def choose(self, decision, features, conservation, order) -> Choice:
        by_id = {c["cand_id"]: c for c in decision["candidates"]}
        cands = [by_id[i] for i in order]
        stats = [candidate_stats(c, features, conservation) for c in cands]
        name = self.name

        if name == "first_listed":
            scores = [float(i) for i in range(len(cands))]
        elif name == "random":
            rng = random.Random(f"random|{decision['decision_id']}")
            scores = [rng.random() for _ in cands]
        elif name == "min_dsasa":
            scores = [s["dsasa"] for s in stats]
        elif name == "min_contacts":
            scores = [s["contacts"] for s in stats]
        elif name == "rim_first":
            # Interface periphery first, burial as the tie-break.
            scores = [s["class_rank"] * 1e6 + s["dsasa"] for s in stats]
        elif name == "max_entropy":
            scores = [-(s["entropy"] if s["entropy"] is not None else 0.0) for s in stats]
        elif name == "min_wt_prob":
            scores = [s["wt_prob"] if s["wt_prob"] is not None else 1.0 for s in stats]
        elif name == "composite":
            # The doc's named bar: hotspot (burial + contacts) AND conservation.
            zb = _zscores([s["dsasa"] for s in stats])
            zc = _zscores([s["contacts"] for s in stats])
            ze = _zscores([s["entropy"] for s in stats])
            scores = [zb[i] + zc[i] - ze[i] for i in range(len(cands))]
        elif name == "max_dsasa":
            # Deliberate anti-baseline: spend the edit at the most buried
            # position. Should be clearly WORSE than random; if it is not, the
            # value function is not measuring interface damage.
            scores = [-s["dsasa"] for s in stats]
        elif name == "most_measured":
            lookup = (self._oracle_lookup or {}).get(decision["decision_id"], {})
            scores = [
                -(lookup.get(c["cand_id"], {}).get("n_records", 0)) for c in cands
            ]
        elif name == "greedy_singles":
            # What a greedy per-position controller would pick at k>=2: the
            # position set whose summed SINGLE-mutation cost is lowest.
            #
            # This is a CONTROL, not a baseline -- it reads measured single
            # ddG values, which no agent-visible arm may see. It exists to
            # answer the design doc's open question about joint interventions:
            # an arm that beats greedy_singles found a coordinated move, one
            # whose pair is cheaper together than its parts predict. Candidates
            # whose singles were never measured sort last rather than being
            # treated as free.
            lookup = (self._oracle_lookup or {}).get(decision["decision_id"], {})
            scores = [
                (lookup.get(c["cand_id"], {}).get("additive")
                 if lookup.get(c["cand_id"], {}).get("additive") is not None
                 else float("inf"))
                for c in cands
            ]
        else:
            raise ValueError(f"unknown heuristic {name!r}")

        best = min(range(len(cands)), key=lambda i: (scores[i], cands[i]["cand_id"]))
        return Choice(
            cand_id=cands[best]["cand_id"],
            status="ok",
            n_retries=0,
            raw="",
            scores={cands[i]["cand_id"]: scores[i] for i in range(len(cands))},
        )


HEURISTIC_ARMS = [
    "random", "first_listed", "min_dsasa", "min_contacts", "rim_first",
    "max_entropy", "min_wt_prob", "composite", "max_dsasa",
]
CONTROL_ARMS = ["most_measured", "greedy_singles"]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LocalHFProposer:
    """A local instruct model, loaded once and reused across decisions.

    Kept behind the same `choose` interface as the heuristics so an API-backed
    proposer can be dropped in later without touching the runner or the scorer.
    """

    needs_llm = True

    def __init__(
        self,
        model_path: str,
        evidence_arm: str = "full",
        name: str | None = None,
        max_new_tokens: int = 420,
        temperature: float = 0.0,
        max_retries: int = 1,
        device_map: str | None = None,
        seed: int = 0,
        load_4bit: bool = False,
    ):
        self.model_path = model_path
        self.evidence_arm = evidence_arm
        self.name = name or f"llm_{evidence_arm}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.seed = seed
        self._tok = None
        self._net = None
        self._device_map = device_map
        self.load_4bit = load_4bit

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        kwargs = {"torch_dtype": torch.bfloat16}
        if self.load_4bit:
            # NF4 double-quantised weights with bf16 compute. This is what makes
            # a 70B fit one 48 GB a6000 (~40 GB) instead of needing four cards
            # for bf16 (~140 GB) -- on this cluster four free a6000s are rarely
            # available at once, so 4-bit is the difference between running and
            # queueing indefinitely.
            #
            # It is a DIFFERENT MODEL from the bf16 weights, not a cheaper way
            # of running the same one, so `run_arm.py` stamps "-4bit" into the
            # model_id and the rows are never pooled with a bf16 run.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            kwargs.setdefault("device_map", self._device_map or "auto")
        elif self._device_map:
            # Needs `accelerate`; only used for the multi-card rungs.
            kwargs["device_map"] = self._device_map
        self._net = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if not kwargs.get("device_map"):
            self._net = self._net.to("cuda" if torch.cuda.is_available() else "cpu")
        self._net.eval()
        return self

    def _generate(self, messages: list[dict]) -> str:
        import torch

        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tok(text, return_tensors="pt").to(self._net.device)
        gen = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tok.pad_token_id,
        }
        if self.temperature and self.temperature > 0:
            gen.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            gen.update(do_sample=False)
        with torch.no_grad():
            out = self._net.generate(**enc, **gen)
        return self._tok.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def choose(self, decision, features, conservation, order) -> Choice:
        valid = {c["cand_id"] for c in decision["candidates"]}
        system, brief = build_prompt(
            decision, features, conservation, self.evidence_arm, order
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        transcript: list[str] = []
        status = "no_tag"
        cand = None
        for attempt in range(self.max_retries + 1):
            reply = self._generate(messages)
            transcript.append(reply)
            cand, status = parse_choice(reply, valid)
            if cand is not None:
                return Choice(
                    cand_id=cand,
                    status=status,
                    n_retries=attempt,
                    raw="\n---RETRY---\n".join(transcript),
                    scores={},
                )
            if attempt < self.max_retries:
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": RETRY_TEMPLATE.format(
                            reason=RETRY_REASON.get(status, status)
                        ),
                    },
                ]
        # Exhausted retries. Recorded as INVALID, with no fallback choice: a
        # defaulted pick would be scored as if the model had made it.
        return Choice(
            cand_id=None,
            status=f"invalid_{status}",
            n_retries=self.max_retries,
            raw="\n---RETRY---\n".join(transcript),
            scores={},
        )


class AnthropicProposer:
    """A Claude model behind the same `choose` interface as every other arm.

    The prompt, the evidence table, the candidate order and the `<choice>`
    parsing are IDENTICAL to the local-weights arm. That is deliberate: the
    point of this arm is to vary the model and nothing else, so no structured
    output, no tool calling and no model-specific prompt tuning is used here
    even though they would all make parsing more reliable. A scaffold change
    bundled with a model change produces a number that cannot be attributed to
    either.

    Two further deliberate choices:

    `fallbacks` is NOT enabled. The API's server-side refusal fallback would
    silently re-run a refused request on a different model and return it under
    the same call -- which is correct for production and wrong for an
    experiment, where every row must name the model that actually produced it.
    A refusal is recorded as an outcome instead.

    Spend is metered per call and capped. The ledger is checked BEFORE each
    request against a conservative estimate, so a run stops short of its budget
    rather than discovering afterwards that it passed it.
    """

    needs_llm = True

    def __init__(
        self,
        model: str,
        evidence_arm: str = "full",
        name: str | None = None,
        effort: str = "low",
        max_tokens: int = 2000,
        max_retries: int = 1,
        max_spend_usd: float = 0.0,
        api_max_retries: int = 4,
        timeout_s: float = 180.0,
    ):
        from pricing import SpendLedger

        self.model = model
        self.evidence_arm = evidence_arm
        self.name = name or f"api_{model}_{evidence_arm}"
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.ledger = SpendLedger(model=model, max_spend_usd=max_spend_usd)
        self._api_max_retries = api_max_retries
        self._timeout_s = timeout_s
        self._client = None
        # Rolling estimate used for the pre-call budget check; seeded from a
        # typical Stage 0 prompt and replaced by measured means once calls land.
        self._est_in, self._est_out = 1600, 400

    def load(self):
        import anthropic

        # Key comes from ANTHROPIC_API_KEY in the environment; never passed on a
        # command line or written to a log.
        self._client = anthropic.Anthropic(
            max_retries=self._api_max_retries, timeout=self._timeout_s
        )
        return self

    def count_input_tokens(self, system: str, brief: str) -> int:
        """Exact input token count without running inference."""
        resp = self._client.messages.count_tokens(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": brief}],
        )
        return int(resp.input_tokens)

    def _call(self, system: str, messages: list[dict]):
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
            # Thinking is always on for the current Opus/Sonnet generation and
            # cannot be disabled; effort is the only depth control, and thinking
            # tokens bill as output. `low` keeps a 73-decision arm affordable --
            # raising it is a cost decision, so it is a parameter, not a
            # constant.
            "output_config": {"effort": self.effort},
        }
        return self._client.messages.create(**kwargs)

    def choose(self, decision, features, conservation, order) -> Choice:
        valid = {c["cand_id"] for c in decision["candidates"]}
        system, brief = build_prompt(
            decision, features, conservation, self.evidence_arm, order
        )
        messages = [{"role": "user", "content": brief}]
        transcript: list[str] = []
        spend_here = 0.0
        status = "no_tag"
        usage_total = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

        for attempt in range(self.max_retries + 1):
            if self.ledger.would_exceed(self._est_in, self._est_out):
                self.ledger.stopped_reason = "budget_cap_reached"
                return Choice(
                    cand_id=None, status="budget_cap_reached",
                    n_retries=attempt, raw="", scores={},
                    cost_usd=spend_here, usage=usage_total,
                )

            response = self._call(system, messages)
            spend_here += self.ledger.record(response.usage)
            for k in usage_total:
                usage_total[k] += int(getattr(response.usage, k, 0) or 0)
            # Refresh the pre-call estimate from what calls actually cost.
            self._est_in = int(self.ledger.input_tokens / max(self.ledger.calls, 1))
            self._est_out = int(self.ledger.output_tokens / max(self.ledger.calls, 1))

            # stop_reason must be checked before reading content: a safety
            # decline returns HTTP 200 with no usable answer.
            if getattr(response, "stop_reason", None) == "refusal":
                details = getattr(response, "stop_details", None)
                cat = getattr(details, "category", None) if details else None
                return Choice(
                    cand_id=None, status=f"refusal_{cat or 'unknown'}",
                    n_retries=attempt, raw="", scores={},
                    cost_usd=spend_here, usage=usage_total,
                )

            reply = "".join(
                b.text for b in response.content if getattr(b, "type", "") == "text"
            )
            transcript.append(reply)
            cand, status = parse_choice(reply, valid)
            if cand is not None:
                return Choice(
                    cand_id=cand, status=status, n_retries=attempt,
                    raw="\n---RETRY---\n".join(transcript), scores={},
                    cost_usd=spend_here, usage=usage_total,
                )
            if attempt < self.max_retries:
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": RETRY_TEMPLATE.format(
                        reason=RETRY_REASON.get(status, status))},
                ]

        return Choice(
            cand_id=None, status=f"invalid_{status}", n_retries=self.max_retries,
            raw="\n---RETRY---\n".join(transcript), scores={},
            cost_usd=spend_here, usage=usage_total,
        )
