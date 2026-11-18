"""One interface for every model the campaign can run on.

The controller is written against `LLMClient.complete` and nothing else, so a
local 8B in bf16, a local 70B in NF4 and a frontier API model are the same
object to it. That is what makes an arm comparison a model comparison: the
prompt, the schema, the retry policy and the parsing are byte-identical across
backends, and no backend gets structured output, tool calling or a
model-specific prompt tweak even where those would make parsing more reliable.
A scaffold change bundled with a model change produces a number that cannot be
attributed to either.

`Reply.status` carries DECLINED as a first-class outcome rather than an
exception. Stage 0 measured why: Haiku 5.5 refused 67% of benign
protein-interface allocation questions as biosecurity-sensitive, Opus and
Sonnet 3-6 of 73. An agent making hundreds of allocation calls will hit it, and
the design doc's control loop has no provision for a step that declines -- so
the loop here has one, and it is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..enums import StepStatus


@dataclass
class Reply:
    text: str = ""
    status: StepStatus = StepStatus.OK
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    detail: str = ""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    declines: int = 0
    errors: int = 0
    per_step: dict[str, int] = field(default_factory=dict)

    def add(self, step: str, r: Reply) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cost_usd += r.cost_usd
        if r.status == StepStatus.DECLINED:
            self.declines += 1
        elif r.status == StepStatus.ERROR:
            self.errors += 1
        self.per_step[step] = self.per_step.get(step, 0) + 1

    def to_row(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "declines": self.declines,
            "errors": self.errors,
            "per_step": dict(self.per_step),
        }


class LLMClient(Protocol):
    model_id: str

    def load(self) -> "LLMClient": ...

    def complete(self, system: str, messages: list[dict]) -> Reply: ...
