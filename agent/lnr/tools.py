"""The LNR action space: rules that turn a RELEASE BUDGET into a held window.

The Stage 3 result on the surrogate was that an LLM authoring a residue mask
free-hand loses to a rule, but an LLM SELECTING among tools whose position rule
is hard-coded gets past the bar. This module ports that comparison onto the
real closure environment, so the question is asked where the objective is
measured rather than modelled.

A tool here answers the same shape of question a Stage 3 tool did: the LLM says
HOW aggressive to be (a release budget k -- how many residues to free for
closure), and the tool decides WHERE those residues come from. That is the
lever/symptom split in its cleanest form: releasing residues is what buys
closure, and WHICH residues you release is what costs interface, so a tool that
places the released set well is doing the allocation the LLM is bad at.

Every rule is a pure function of measured coordinates and the contact set. None
reads a closure result, so none is answer-informed except `hold_best_feasible`,
which is labelled as such because it consults the analytic feasibility bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import env as E


def _best_contact_window(t: E.LNRTarget, w: int) -> tuple[int, int]:
    """The length-w contiguous window containing the most interface contacts."""
    w = max(1, min(w, t.length))
    pc = t.per_res_contacts
    best_a, best_sum = 0, -1
    for a in range(0, t.length - w + 1):
        s = sum(pc[a:a + w])
        if s > best_sum:
            best_sum, best_a = s, a
    return best_a, best_a + w - 1


def hold_contact_core(t: E.LNRTarget, k: int) -> tuple[int, int]:
    """Release k residues; HOLD the contact-richest window of the rest.

    The naive instinct made contiguous: keep the residues that carry the
    interface fixed, release the rest for closure. Ignores whether the result
    closes -- which is exactly the failure this environment can expose, because
    the contact-richest residues are often the ones a closing backbone must
    move.
    """
    return _best_contact_window(t, t.length - k)


def release_termini(t: E.LNRTarget, k: int) -> tuple[int, int]:
    """Release k residues split across the two termini; hold the interior.

    Head-to-tail closure joins the termini, so freeing the ends is the
    geometrically natural way to spend a release budget. Ignores where the
    contacts are.
    """
    k = max(0, min(k, t.length - 1))
    n = k // 2
    c = k - n
    a = n
    b = t.length - 1 - c
    if b < a:
        b = a
    return a, b


def hold_retention_window(t: E.LNRTarget, k: int) -> tuple[int, int]:
    """Release k; hold the length-(L-k) window of MAX interface retention.

    Retention is not the contact count: it is the fraction of the FULL contact
    set that a given held window keeps, which weights by where the receptor
    partners sit. A pure coordinate quantity, no closure result consulted.
    """
    w = max(1, t.length - k)
    best, best_ret = (0, w - 1), -1.0
    for a in range(0, t.length - w + 1):
        r = E.retention_of(t, a, a + w - 1)
        if r > best_ret:
            best_ret, best = r, (a, a + w - 1)
    return best


def hold_best_feasible(t: E.LNRTarget, k: int) -> tuple[int, int]:
    """Release k; among length-(L-k) ANALYTICALLY FEASIBLE windows, max retention.

    Answer-informed: it reads the analytic closure bound, so it knows which
    windows can plausibly close before paying the torsion solver. Reported as
    such. It is the geometry-aware tool the free-hand LLM has to beat, the LNR
    counterpart of Stage 3's `rebuild_cheapest_2span`.
    """
    w = max(1, t.length - k)
    feas = []
    for a in range(0, t.length - w + 1):
        b = a + w - 1
        if E.analytic_feasible(t, a, b):
            feas.append((E.retention_of(t, a, b), a, b))
    if not feas:
        a, b = hold_retention_window(t, k)
        return a, b
    feas.sort(key=lambda x: -x[0])
    return feas[0][1], feas[0][2]


@dataclass(frozen=True)
class Tool:
    name: str
    doc: str
    place: Callable[[E.LNRTarget, int], tuple[int, int]]


TOOLS: tuple[Tool, ...] = (
    Tool("hold_contact_core",
         "Keep the contact-richest stretch of the peptide fixed and release "
         "the rest. Spends the release budget on whatever is left over after "
         "protecting the binding core. May not close if the binding core is "
         "exactly what the ring needs to move.",
         hold_contact_core),
    Tool("release_termini",
         "Free the two ends of the peptide and hold the interior. The natural "
         "way to close a head-to-tail ring, since the bond joins the termini. "
         "Ignores where the interface contacts sit.",
         release_termini),
    Tool("hold_retention_window",
         "Hold the window that keeps the largest FRACTION of the interface "
         "(not just the most contacts), releasing the rest. Weighs each "
         "contact by its receptor partner, but does not check closure.",
         hold_retention_window),
    Tool("hold_best_feasible",
         "Among windows of this size that can geometrically close, hold the "
         "one that keeps the most interface. Checks closure feasibility before "
         "choosing, so it spends the release budget where a ring actually fits.",
         hold_best_feasible),
)
BY_NAME = {t.name: t for t in TOOLS}
TOOL_NAMES = tuple(t.name for t in TOOLS)


def render_menu() -> str:
    return "\n\n".join(f"  {t.name}\n    {t.doc}" for t in TOOLS)


# ---------------------------------------------------------------------------
# The deterministic controls
# ---------------------------------------------------------------------------


def rule_min_release_feasible(t: E.LNRTarget, max_k: int | None = None) -> tuple[int, int]:
    """Smallest release budget at which a window analytically closes; among
    those, hold the max-retention feasible window. The smart deterministic
    policy -- spend the least closure budget, keep the most interface -- and
    the strong control the LLM arms have to beat."""
    L = t.length
    cap = max_k if max_k is not None else L - 1
    for k in range(0, cap + 1):
        w = L - k
        feas = [(E.retention_of(t, a, a + w - 1), a, a + w - 1)
                for a in range(0, L - w + 1)
                if E.analytic_feasible(t, a, a + w - 1)]
        if feas:
            feas.sort(key=lambda x: -x[0])
            return feas[0][1], feas[0][2]
    return 0, L - 1


def rule_hold_contact_core(t: E.LNRTarget, k: int = 2) -> tuple[int, int]:
    """Release a fixed small budget and hold the contact core. The naive
    'protect the binding residues' baseline -- the instinct the Stage 0/2/3
    story is about."""
    return hold_contact_core(t, k)


def rule_release_termini(t: E.LNRTarget, k: int = 2) -> tuple[int, int]:
    return release_termini(t, k)


def ranked_analytic_windows(t: E.LNRTarget) -> list[tuple[float, int, int]]:
    """Every analytically-feasible window as (retention, a, b), best retention
    first. The candidate list a budget-limited frontier search walks. Cached on
    the target object so repeated arms do not recompute it."""
    cache = getattr(t, "_ranked_cache", None)
    if cache is not None:
        return cache
    out = []
    for a in range(t.length):
        for b in range(a, t.length):
            if E.analytic_feasible(t, a, b):
                out.append((E.retention_of(t, a, b), a, b))
    out.sort(key=lambda x: (-x[0], (x[2] - x[1])))
    try:
        object.__setattr__(t, "_ranked_cache", out)
    except Exception:
        pass
    return out
