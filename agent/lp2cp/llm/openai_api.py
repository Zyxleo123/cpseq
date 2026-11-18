"""An OpenAI model behind the same `complete` interface, spend-metered.

Written against the raw HTTP API through `urllib` rather than the `openai`
SDK, for one practical reason: the campaign runner's venv on this cluster has
`anthropic` and nothing else, installing into it needs privileges this session
does not have, and a backend that cannot be installed cannot be run. The REST
surface used here is one POST to /v1/chat/completions, which is stable and
needs no dependency at all.

Three things differ from the Anthropic path and all three would corrupt a
number if copied carelessly:

PROMPT TOKENS INCLUDE CACHED TOKENS. OpenAI reports `prompt_tokens` as the
total and `prompt_tokens_details.cached_tokens` as the part that was a cache
hit; Anthropic reports `input_tokens` with cache reads held separately. The
shared ledger follows Anthropic's convention, so the cached part is subtracted
here before the ledger sees it. Forgetting that bills every cache hit twice --
once at the input rate and once at the cache rate.

REASONING TOKENS BILL AS OUTPUT and are already inside `completion_tokens`, so
`completion_tokens` is passed through unmodified. They are recorded separately
in the reply detail because at `reasoning_effort` above `low` they dominate the
bill, and a cost that is 90% invisible deliberation should be visible in the
ledger.

A REFUSAL IS NOT A FIELD HERE, usually. Anthropic returns
`stop_reason=refusal` with a category, which is what let Stage 2 separate a
genuine `bio` decline from a `reasoning_extraction` transport failure. OpenAI
returns an ordinary completion whose text declines in prose, and only sets
`message.refusal` for structured-output refusals or `finish_reason` to
`content_filter` for a filter hit. So DECLINED is set ONLY on those two
explicit signals. A prose decline parses as MALFORMED and carries
`refusal_phrase=1` in the reply detail instead -- deliberately not promoted to
DECLINED, because a heuristic over short unparseable replies would inflate a
refusal rate, and the refusal rates measured in Stage 0 and Stage 2 are among
the results this project carries forward.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..enums import StepStatus
from .base import Reply

_STAGE0 = Path(__file__).resolve().parents[3] / "pilot" / "stage0"
if str(_STAGE0) not in sys.path:
    sys.path.insert(0, str(_STAGE0))

ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Only used to annotate an unparseable reply, never to set DECLINED.
_REFUSAL_RE = re.compile(
    r"\b(i can'?t help|i cannot help|i can'?t assist|i cannot assist|"
    r"i'?m not able to (help|assist)|i won'?t (help|assist)|"
    r"can'?t provide|cannot provide that)\b", re.I)


class OpenAIClient:
    """One OpenAI model, one spend ledger."""

    def __init__(
        self,
        model: str,
        effort: str = "low",
        max_tokens: int = 12000,
        max_spend_usd: float = 0.0,
        api_max_retries: int = 4,
        timeout_s: float = 240.0,
        cache_system: bool = True,     # accepted for interface parity; OpenAI
                                       # caches automatically above ~1024 tok
    ):
        from pricing import SpendLedger  # noqa: PLC0415

        self.model = model
        self.model_id = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.ledger = SpendLedger(model=model, max_spend_usd=max_spend_usd)
        self._api_max_retries = api_max_retries
        self._timeout_s = timeout_s
        self._key = ""
        # Seeded from Stage 2's measured per-call means, replaced by this run's
        # own means once calls land, so the pre-call cap check is calibrated.
        self._est_in, self._est_out = 2600, 700
        # Set to False the first time the API rejects the parameter, so an
        # unsupported knob costs one 400 for the whole run rather than one per
        # call. A 400 is not billed, but it is also not free in wall time.
        self._send_effort = True
        self.reasoning_tokens = 0

    def load(self):
        from envload import require_api_key  # noqa: PLC0415
        import os  # noqa: PLC0415

        if not require_api_key(key="OPENAI_API_KEY"):
            raise RuntimeError(
                "no OPENAI_API_KEY available. Put it in the dotenv file that "
                "envload.py reads, or in the raw key file it checks; it is "
                "never passed on a command line."
            )
        self._key = os.environ["OPENAI_API_KEY"]
        return self

    # -- transport --------------------------------------------------------
    def _post(self, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw or "{}")
            except json.JSONDecodeError:
                return e.code, {"error": {"message": raw[:300]}}

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

        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + list(messages),
            "max_completion_tokens": self.max_tokens,
        }
        if self._send_effort and self.effort:
            body["reasoning_effort"] = self.effort

        status, data, detail = 0, {}, ""
        for attempt in range(self._api_max_retries + 1):
            status, data = self._post(body)
            if status == 200:
                break
            msg = str((data.get("error") or {}).get("message", ""))[:300]
            # An unsupported-parameter 400 is a fact about the model, not a
            # transient failure: drop the knob and retry once, then remember.
            if status == 400 and "reasoning_effort" in msg and self._send_effort:
                self._send_effort = False
                body.pop("reasoning_effort", None)
                continue
            if status in (408, 409, 429, 500, 502, 503, 504, 529) and \
                    attempt < self._api_max_retries:
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
                continue
            return Reply(
                status=StepStatus.ERROR,
                detail=f"http_{status}: {msg}",
                seconds=round(time.time() - t0, 2),
            )

        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        pd = usage.get("prompt_tokens_details") or {}
        cd = usage.get("completion_tokens_details") or {}
        cached = int(pd.get("cached_tokens") or 0)
        reasoning = int(cd.get("reasoning_tokens") or 0)
        self.reasoning_tokens += reasoning
        # Anthropic convention for the shared ledger: input EXCLUDES cache
        # reads. OpenAI's prompt_tokens includes them, so subtract.
        cost = self.ledger.record({
            "input_tokens": max(0, pt - cached),
            "output_tokens": ct,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        })
        self._est_in = int(self.ledger.input_tokens / max(self.ledger.calls, 1))
        self._est_out = int(self.ledger.output_tokens / max(self.ledger.calls, 1))

        choices = data.get("choices") or [{}]
        ch = choices[0] or {}
        finish = str(ch.get("finish_reason") or "")
        msg_obj = ch.get("message") or {}
        refusal = msg_obj.get("refusal")
        text = msg_obj.get("content") or ""

        common = dict(
            input_tokens=max(0, pt - cached),
            output_tokens=ct,
            cost_usd=cost,
            seconds=round(time.time() - t0, 2),
        )
        if refusal or finish == "content_filter":
            return Reply(
                status=StepStatus.DECLINED,
                detail=f"refusal:{'explicit' if refusal else 'content_filter'}",
                **common,
            )
        if finish == "length":
            # Ran out mid-reasoning. An operational failure, never a decline --
            # the Stage 2 split of `max_tokens` away from `bio` is the same
            # distinction.
            return Reply(
                status=StepStatus.ERROR,
                detail=f"truncated:max_completion_tokens reasoning={reasoning}",
                **common,
            )
        detail = f"cache_read={cached} reasoning={reasoning} finish={finish}"
        if _REFUSAL_RE.search(text or ""):
            detail += " refusal_phrase=1"
        return Reply(text=text, status=StepStatus.OK, detail=detail, **common)

    def summary(self) -> dict:
        d = self.ledger.summary()
        d["reasoning_tokens"] = self.reasoning_tokens
        return d
