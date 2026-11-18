"""A Claude model behind the shared `complete` interface, spend-metered.

Three deliberate choices, carried over from the Stage 0 API path because each
one was needed there:

NO SERVER-SIDE REFUSAL FALLBACK. The API can silently re-run a refused request
on a different model and return it under the same call. That is correct for
production and wrong for an experiment, where every row must name the model
that actually produced it. A refusal is recorded as an outcome instead.

SPEND IS METERED PER CALL AND CAPPED, and the ledger is checked BEFORE each
request against a rolling estimate, so a run stops short of its budget rather
than discovering afterwards that it passed it. An end-to-end campaign makes
several calls per iteration per parent, so the call count is an order of
magnitude above Stage 0's and an unmetered run could spend the remaining
balance in one job.

PROMPT CACHING IS ON for the system prompt. Unlike Stage 0's one-shot
prompts, a campaign sends the same long system prompt many times per parent,
which is exactly the case caching is for; cache reads bill at a fraction of
input rate and the ledger accounts for reads and writes separately so the
saving is visible rather than assumed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..enums import StepStatus
from .base import Reply

# The Stage 0 pricing table and dotenv loader are reused rather than copied: a
# second price table would silently diverge from the first and mis-state cost.
_STAGE0 = Path(__file__).resolve().parents[3] / "pilot" / "stage0"
if str(_STAGE0) not in sys.path:
    sys.path.insert(0, str(_STAGE0))


class AnthropicClient:
    """One Claude model, one spend ledger."""

    def __init__(
        self,
        model: str,
        effort: str = "low",
        max_tokens: int = 12000,
        max_spend_usd: float = 0.0,
        api_max_retries: int = 4,
        timeout_s: float = 240.0,
        cache_system: bool = True,
    ):
        from pricing import SpendLedger  # noqa: PLC0415

        self.model = model
        self.model_id = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.ledger = SpendLedger(model=model, max_spend_usd=max_spend_usd)
        self._api_max_retries = api_max_retries
        self._timeout_s = timeout_s
        self._cache_system = cache_system
        self._client = None
        # Seeded from Stage 0's measured per-call means and replaced by this
        # run's own means once calls land.
        self._est_in, self._est_out = 2600, 700

    def load(self):
        import anthropic  # noqa: PLC0415
        from envload import require_api_key  # noqa: PLC0415

        if not require_api_key():
            raise RuntimeError(
                "no ANTHROPIC_API_KEY available. Put it in the dotenv file that "
                "envload.py reads, or export it; it is never passed on a "
                "command line."
            )
        self._client = anthropic.Anthropic(
            max_retries=self._api_max_retries, timeout=self._timeout_s
        )
        return self

    def complete(self, system: str, messages: list[dict]) -> Reply:
        t0 = time.time()
        if self.ledger.would_exceed(self._est_in, self._est_out):
            self.ledger.stopped_reason = "budget_cap_reached"
            return Reply(
                status=StepStatus.BUDGET_CAP,
                detail=f"spend cap {self.ledger.max_spend_usd} would be exceeded "
                       f"(spent {self.ledger.spent_usd:.4f})",
                seconds=round(time.time() - t0, 2),
            )

        sys_block = (
            [{"type": "text", "text": system,
              "cache_control": {"type": "ephemeral"}}]
            if self._cache_system else system
        )
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=sys_block,
                messages=messages,
                # Thinking is always on for the current Opus/Sonnet generation
                # and cannot be disabled; effort is the only depth control and
                # thinking tokens bill as output. `low` is what keeps a
                # multi-iteration campaign affordable, so it is a parameter.
                output_config={"effort": self.effort},
            )
        except Exception as exc:  # noqa: BLE001
            return Reply(
                status=StepStatus.ERROR,
                detail=f"{type(exc).__name__}: {exc}"[:400],
                seconds=round(time.time() - t0, 2),
            )

        cost = self.ledger.record(resp.usage)
        self._est_in = int(self.ledger.input_tokens / max(self.ledger.calls, 1))
        self._est_out = int(self.ledger.output_tokens / max(self.ledger.calls, 1))
        usage = {
            k: int(getattr(resp.usage, k, 0) or 0)
            for k in ("input_tokens", "output_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens")
        }

        # stop_reason must be checked before reading content: a safety decline
        # returns HTTP 200 with no usable answer in it.
        #
        # The CATEGORY matters and must not be flattened. Measured on this
        # benchmark: opus-5-5 returned category `bio` -- a genuine safety
        # decline -- while sonnet-5-5 returned `reasoning_extraction`, which is
        # an operational failure to extract the thinking block and says nothing
        # about safety. Reporting both as "refusal rate" would have attributed
        # a transport problem to a model's safety behaviour, and the Stage 0
        # refusal finding is one of the results this project carries forward.
        if getattr(resp, "stop_reason", None) == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            cat = str(cat or "unknown")
            operational = cat in ("reasoning_extraction", "max_tokens", "unknown")
            return Reply(
                status=StepStatus.ERROR if operational else StepStatus.DECLINED,
                detail=f"refusal:{cat}",
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cost_usd=cost,
                seconds=round(time.time() - t0, 2),
            )

        text = "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        )
        return Reply(
            text=text,
            status=StepStatus.OK,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            cost_usd=cost,
            seconds=round(time.time() - t0, 2),
            detail=(
                f"cache_read={usage['cache_read_input_tokens']} "
                f"cache_write={usage['cache_creation_input_tokens']}"
            ),
        )

    def summary(self) -> dict:
        return self.ledger.summary()


class ScriptedClient:
    """A deterministic stand-in. Used by the unit tests and the smoke path.

    It exists so that the control loop, the execution service, the budget
    enforcement and the scorer can all be exercised with no GPU, no API key and
    no spend -- which is what lets the loop be verified before any model is
    pointed at it, and lets a CI run catch a scaffold regression for free.
    """

    def __init__(self, replies: list[str] | None = None, model_id: str = "scripted"):
        self.replies = list(replies or [])
        self.model_id = model_id
        self.i = 0
        self.seen: list[tuple[str, str]] = []

    def load(self):
        return self

    def complete(self, system: str, messages: list[dict]) -> Reply:
        self.seen.append((system, messages[-1]["content"] if messages else ""))
        if self.i < len(self.replies):
            text = self.replies[self.i]
        else:
            text = self.replies[-1] if self.replies else ""
        self.i += 1
        return Reply(text=text, status=StepStatus.OK, input_tokens=10,
                     output_tokens=10, cost_usd=0.0, seconds=0.0)
