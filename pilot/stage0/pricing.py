"""Token prices and spend accounting for the API arms.

Prices are USD per million tokens, first-party Anthropic API rates, cached
from the claude-api skill's model table on 2026-11-03. They are written here
rather than fetched at run time so that a cost recorded next to a result is the
price that was actually in force when the row was produced -- a figure that
silently re-prices itself later is not an experimental record.

`cache_read` is the discounted rate for tokens served from a prompt cache.
Stage 0 prompts do not currently reach the minimum cacheable prefix, so that
column is carried for completeness and reports zero.

If a run needs a model not in this table, add it from the live pricing page
rather than guessing -- a wrong constant here silently mis-states every cost
downstream, and nothing else in the pipeline would catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    # model id:            input,  output, cache_read
    "claude-opus-5-5":   {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-opus-5":     {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.10},
    "claude-sonnet-5":   {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5":  {"input": 0.10, "output": 0.50,  "cache_read": 0.01},
    "claude-fable-5-2":  {"input": 8.00, "output": 40.00, "cache_read": 0.20},

    # OpenAI, Standard tier, SHORT context (<= 272K input tokens), read from
    # developers.openai.com/api/docs/pricing on 2026-11-15. Two caveats that
    # matter for a reported cost rather than for a budget cap:
    #   * requests over 272K input tokens bill at 2x input / 1.5x output for
    #     the WHOLE request. No prompt in this benchmark comes near that, and
    #     nothing here checks it.
    #   * gpt-5.6-sol is on promotional pricing until 2026-11-30, after which
    #     these numbers are wrong rather than stale.
    # OpenAI's `prompt_tokens` INCLUDES cached tokens, unlike Anthropic's
    # `input_tokens`, so OpenAIClient subtracts before calling this table. A
    # client that forgets would double-bill every cache hit.
    "gpt-6-astra":       {"input": 10.00, "output": 50.00, "cache_read": 1.00},
    "gpt-6-sol":         {"input": 2.00,  "output": 10.00, "cache_read": 0.20},
    "gpt-6-luna":        {"input": 0.10,  "output": 0.50,  "cache_read": 0.01},
    "gpt-6.1-sol":       {"input": 2.00,  "output": 10.00, "cache_read": 0.10},
    "gpt-6.1-terra":     {"input": 1.00,  "output": 5.00,  "cache_read": 0.05},
    "gpt-5.6-sol":       {"input": 4.00,  "output": 20.00, "cache_read": 0.40},
    "gpt-5.6-terra":     {"input": 2.00,  "output": 12.00, "cache_read": 0.20},
    "gpt-5.6-luna":      {"input": 0.20,  "output": 1.20,  "cache_read": 0.02},
    "gpt-5.5":           {"input": 5.00,  "output": 30.00, "cache_read": 0.50},
    "gpt-5.4":           {"input": 2.50,  "output": 15.00, "cache_read": 0.25},
    "gpt-5.4-mini":      {"input": 0.75,  "output": 4.50,  "cache_read": 0.075},
    "gpt-5.4-nano":      {"input": 0.20,  "output": 1.25,  "cache_read": 0.02},
    "gpt-5.1":           {"input": 1.25,  "output": 10.00, "cache_read": 0.125},
    "gpt-5-mini":        {"input": 0.25,  "output": 2.00,  "cache_read": 0.025},
    "gpt-5-nano":        {"input": 0.05,  "output": 0.40,  "cache_read": 0.005},
    "gpt-4.1-mini":      {"input": 0.40,  "output": 1.60,  "cache_read": 0.10},
}
PRICES_AS_OF = "2026-11-03"


def cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    """Dollar cost of one call. Unknown model -> raises, never silently zero."""
    if model not in PRICES_PER_MTOK:
        raise KeyError(
            f"no price for model {model!r}; add it to PRICES_PER_MTOK rather "
            f"than letting the run record a cost of zero"
        )
    p = PRICES_PER_MTOK[model]
    # Cache WRITES bill at roughly 1.25x the input rate; reads at the discounted
    # rate. Both are zero on Stage 0 prompts today but are accounted for so that
    # enabling caching later does not quietly under-report.
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_read_input_tokens * p["cache_read"]
        + cache_creation_input_tokens * p["input"] * 1.25
    ) / 1_000_000.0


@dataclass
class SpendLedger:
    """Running spend for one arm, with a hard cap.

    The cap is checked BEFORE each call using a conservative estimate of what
    that call will cost, so the ledger stops short of the limit rather than
    discovering it has passed it. A pilot on a fixed budget cannot rely on
    noticing an overrun afterwards.
    """

    model: str
    max_spend_usd: float = 0.0           # 0 = no cap
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stopped_reason: str | None = None
    per_call: list[float] = field(default_factory=list)

    def would_exceed(self, est_input: int, est_output: int) -> bool:
        if not self.max_spend_usd:
            return False
        est = cost_usd(self.model, est_input, est_output)
        return (self.spent_usd + est) > self.max_spend_usd

    def record(self, usage) -> float:
        """Add one call's usage. Accepts an SDK usage object or a dict."""
        def g(name: str) -> int:
            if isinstance(usage, dict):
                return int(usage.get(name) or 0)
            return int(getattr(usage, name, 0) or 0)

        it = g("input_tokens")
        ot = g("output_tokens")
        cr = g("cache_read_input_tokens")
        cw = g("cache_creation_input_tokens")
        c = cost_usd(self.model, it, ot, cr, cw)
        self.spent_usd += c
        self.calls += 1
        self.input_tokens += it
        self.output_tokens += ot
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.per_call.append(c)
        return c

    def summary(self) -> dict:
        n = max(self.calls, 1)
        return {
            "model": self.model,
            "prices_as_of": PRICES_AS_OF,
            "price_per_mtok": PRICES_PER_MTOK.get(self.model),
            "calls": self.calls,
            "spent_usd": round(self.spent_usd, 6),
            "mean_usd_per_call": round(self.spent_usd / n, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "mean_input_tokens": round(self.input_tokens / n, 1),
            "mean_output_tokens": round(self.output_tokens / n, 1),
            "max_spend_usd": self.max_spend_usd,
            "stopped_reason": self.stopped_reason,
        }
