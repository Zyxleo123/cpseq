"""Run one LNR campaign per (target, arm): allocate a REAL evaluation budget.

The currency here is the real closure check. The analytic bound over-licenses
(it calls [3,9] on 1d4t feasible, where the torsion solver needs five residues
released), so a controller cannot read the answer off geometry -- it has to
SPEND closure evaluations, learn from (closed?, retention), and re-propose.
That is the agent setting with the modelling removed: each evaluation is an
expensive real experiment, and the controller's skill is choosing which window
to spend the next one on.

A campaign is T evaluations on one target. The controller proposes a held
window (directly, or by naming a tool and a release budget), the harness scores
it REALLY, and the controller sees the result before its next proposal. The
campaign's utility is the retention of the best window that actually CLOSED --
0 if none did. Arms are compared paired over targets at matched T, never
against a per-cell regret oracle; the exhaustive frontier is reported only as
context for how much retention was on the table.

Arms
----
  rule:contact_core   spend evaluation i holding the contact-richest window at
                      release = i. The 'protect the binding residues' instinct.
  rule:termini        release from the two ends, budget rising with i.
  rule:feasible       the analytic-feasible max-retention window at the
                      smallest release that is analytically feasible, then the
                      next, ... -- geometry-aware, the strong deterministic bar.
  router              deterministic selection over the tool menu.
  bo                  Bayesian optimisation (GP-UCB) over the held-window space,
                      same budget and feedback as the LLM. The published-method
                      baseline: is an LLM a better few-shot allocator than a
                      principled optimiser given identical information?
  llm-mask            the LLM authors [a,b] from the contact table, sees the
                      real result, re-authors. Stage 3's mask-authoring arm.
  llm-tool            the LLM names a tool and a release budget, sees the real
                      result, re-chooses. Stage 3's tool-selection arm.
  llm-info-mask       llm-mask, but also shown the analytic closure-feasibility
  llm-info-tool       frontier the geometry-aware rules (feasible /
                      placement_search / bo) read for free. The information-
                      matched arms: same budget and feedback as the plain LLM
                      arms, plus the one thing they were denied, so the pair
                      llm-info-* vs placement_search tests whether the LLM can
                      close the information-asymmetry gap to the strong bar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE.parent.parent / "pilot" / "stage0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lnr import env as E          # noqa: E402
from lnr import tools as T        # noqa: E402


# ---------------------------------------------------------------------------
# Rendering the real evidence a controller sees
# ---------------------------------------------------------------------------

def residue_table(t: E.LNRTarget) -> str:
    rows = ["  idx  aa  contacts",
            "  ---  --  --------"]
    for i in range(t.length):
        rows.append(f"  {i:3d}  {t.seq[i] if i < len(t.seq) else 'X'}   "
                    f"{t.per_res_contacts[i]:3d}")
    return "\n".join(rows)


def history_table(history: list[dict]) -> str:
    if not history:
        return "  (no windows tried yet)"
    rows = ["  held[a,b]   release  closes  retention",
            "  ----------  -------  ------  ---------"]
    for h in history:
        rows.append(f"  [{h['a']:2d},{h['b']:2d}]     {h['release']:3d}     "
                    f"{'yes' if h['closes'] else 'no ':>3s}     {h['retention']:.3f}")
    return "\n".join(rows)


def analytic_frontier_block(t: E.LNRTarget) -> str:
    """The analytic closure-feasibility frontier the INFO-matched arms are shown.

    For each release budget k (how many residues are freed), the highest-
    retention contiguous window the cheap analytic bound calls feasible. This is
    EXACTLY the geometric foreknowledge the geometry-aware rules read for free --
    `feasible`/`placement_search` walk this ladder (`hold_best_feasible` at an
    adaptive k) and `bo` uses the same analytic-feasible set as its candidate
    list and prior mean. The plain `llm-mask`/`llm-tool` arms are DENIED it; an
    `llm-info-*` arm is shown this block and nothing else different, so the pair
    isolates the information asymmetry -- can the LLM close the gap to the
    strong deterministic bar once it sees the same geometry?

    The bound is a true UPPER bound and it OVER-LICENSES (it calls windows
    feasible that the real torsion solve then rejects -- the `greedy_frontier`
    lesson), so the block says feasible != closes and points back at the real
    history. Nothing here is a closure result; it is a free coordinate screen.
    """
    L = t.length
    best: dict[int, tuple[float, int, int]] = {}
    for ret, a, b in T.ranked_analytic_windows(t):
        k = L - (b - a + 1)
        if k not in best or ret > best[k][0]:
            best[k] = (ret, a, b)
    rows = [f"  {'release':>7s}  {'best feasible held[a,b]':<23s}  retention",
            f"  {'-------':>7s}  {'-----------------------':<23s}  ---------"]
    for k in range(L):
        if k in best:
            ret, a, b = best[k]
            rows.append(f"  {k:7d}  {f'[{a},{b}]':<23s}  {ret:.3f}")
        else:
            rows.append(f"  {k:7d}  {'(no feasible window)':<23s}  --")
    table = "\n".join(rows)
    return (
        "Analytic closure-feasibility frontier (geometric foreknowledge, FREE -- "
        "costs no evaluation):\n"
        "A fast ideal-bond torsion test screens, for free, whether a held window "
        "COULD close before you spend an evaluation on it. It is an UPPER BOUND: "
        "a window it rejects truly cannot close, but a window it accepts only "
        "MIGHT -- the bound over-licenses, so an 'analytically feasible' window "
        "can still fail the real closure solve. For each release budget, the "
        "highest-retention window the bound accepts (and its retention):\n\n"
        f"{table}\n\n"
        "Use this to spend evaluations only where a ring can plausibly fit, but "
        "remember feasible is NOT closes -- the evaluated-windows table above is "
        "the ground truth."
    )


SYSTEM = (
    "You are the controller in a benchmark for cyclic-peptide design. A linear "
    "peptide is bound to a receptor in a known pose; you must make it a "
    "head-to-tail cyclic peptide. Closing the ring requires moving some "
    "residues off their bound positions, which costs interface contacts. You "
    "decide which residues to HOLD at their bound pose (a contiguous window "
    "[a,b], 0-indexed, inclusive); every residue outside the window is released "
    "so the backbone can close the ring. Holding more keeps more interface but "
    "may not close; holding less closes more easily but loses binding. Every "
    "quantity you are shown is measured from deposited coordinates. You never "
    "choose amino-acid identities; you choose only the window.\n\n"
    "Reason briefly, then end your reply with the required tag and nothing after it."
)


def mask_brief(t: E.LNRTarget, history: list[dict], T_left: int,
               info: bool = False) -> str:
    extra = ("\n" + analytic_frontier_block(t) + "\n") if info else ""
    return (
        f"Peptide length {t.length} (indices 0..{t.length-1}), "
        f"{t.n_contacts} interface contacts.\n\n"
        f"Per-residue interface contacts:\n{residue_table(t)}\n\n"
        f"Windows already evaluated this target:\n{history_table(history)}\n"
        f"{extra}\n"
        f"You have {T_left} closure evaluation(s) left. Choose the held window "
        f"to evaluate next -- the residues to KEEP at their bound pose. "
        f"Everything outside [a,b] is released to close the ring.\n\n"
        f"End with exactly:\n<hold>a,b</hold>"
    )


def tool_brief(t: E.LNRTarget, history: list[dict], T_left: int,
               info: bool = False) -> str:
    extra = ("\n" + analytic_frontier_block(t) + "\n") if info else ""
    return (
        f"Peptide length {t.length} (indices 0..{t.length-1}), "
        f"{t.n_contacts} interface contacts.\n\n"
        f"Per-residue interface contacts:\n{residue_table(t)}\n\n"
        f"Windows already evaluated this target:\n{history_table(history)}\n"
        f"{extra}\n"
        f"Each tool decides WHICH residues to hold; you choose the tool and how "
        f"many residues to RELEASE for closure (the release budget k, 0.."
        f"{t.length-1}). Holding is the complement of what you release.\n\n"
        f"Tools:\n\n{T.render_menu()}\n\n"
        f"You have {T_left} closure evaluation(s) left. "
        f"End with exactly:\n<tool>name k</tool>"
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

import re  # noqa: E402

_HOLD = re.compile(r"<hold>\s*(\d+)\s*,\s*(\d+)\s*</hold>", re.I)
_TOOL = re.compile(r"<tool>\s*([a-z_]+)\s*[, ]\s*(\d+)\s*</tool>", re.I)


def parse_hold(text: str, L: int):
    m = _HOLD.findall(text or "")
    if not m:
        return None, "no_tag"
    a, b = int(m[-1][0]), int(m[-1][1])
    if a > b:
        a, b = b, a
    a = max(0, min(a, L - 1))
    b = max(a, min(b, L - 1))
    return (a, b), "ok"


def parse_tool(text: str):
    m = _TOOL.findall(text or "")
    if not m:
        return None, "no_tag"
    name, k = m[-1][0].lower(), int(m[-1][1])
    if name not in T.BY_NAME:
        return None, "unknown_tool"
    return (name, k), "ok"


# ---------------------------------------------------------------------------
# One campaign
# ---------------------------------------------------------------------------

def _next_release(history: list[dict], L: int) -> int:
    """Binary-search the release budget for the closure threshold.

    Closure is MONOTONE in release: freeing more residues can only make the
    ring easier to close, never harder (the solver's own prune relies on this
    -- once a held window fails, every superset of it fails too). So the
    closing releases form an upper interval [k*, L-1] and the failing ones a
    lower interval [0, k*-1], and the smallest closing release k* is found by
    bisection in about log2(L) evaluations. k* is the budget that keeps the
    most interface while still closing, which is the whole objective, so a rule
    that finds it fast is the strong bar the LLM arms must match. The analytic
    bound cannot be trusted to name k* (it over-licenses), which is exactly why
    this searches with REAL closure results instead of reading geometry.
    """
    opened = [h["release"] for h in history if not h["closes"]]
    closed = [h["release"] for h in history if h["closes"]]
    lo = max(opened) if opened else -1        # largest failing release; k* > lo
    hi = min(closed) if closed else L - 1     # smallest closing release; k* <= hi
    if lo + 1 >= hi:
        return hi                              # converged on the minimum
    return (lo + 1 + hi) // 2


RULE_ARMS = ("contact_core", "termini", "feasible", "router",
             "greedy_frontier", "placement_search", "random", "bo")


# ---------------------------------------------------------------------------
# A published-method baseline: Bayesian optimisation (GP-UCB)
# ---------------------------------------------------------------------------
#
# The accepted bar for "is an LLM a good few-shot black-box optimiser" is
# Bayesian optimisation, and there is a literature benchmarking LLM-optimisers
# against exactly it (OPRO, Yang et al. 2024; LLAMBO). This arm is that bar,
# matched to the LLM arms on everything that matters: the SAME action space
# (contiguous held windows), the SAME evaluation budget T, and the SAME feedback
# -- only the (closed?, retention) rows of THIS campaign. It is NOT told that
# closure is monotone in release (the structural prior the hand-coded
# `placement_search` bakes in); it must learn the action->utility surface from
# the T solves alone. It differs from the LLM arms only in the controller: a
# Gaussian process instead of a language model. So `llm-* vs bo` isolates the
# one thing the project is about -- is a language model a better few-shot
# allocator than a principled optimiser given identical information and budget.
#
# Formulation. The expensive, budgeted oracle is the torsion closure solve;
# retention_of() and analytic_feasible() are cheap coordinate quantities (no
# solve). So -- exactly as feasible/placement_search/hold_best_feasible already
# do -- BO reads those for free: the candidate set is the analytic-feasible
# windows (the analytic bound is a true upper bound, so an analytic-infeasible
# window cannot close and is never worth a solve), and the GP's PRIOR MEAN at
# each window is its free retention. The T closure results are the data that
# correct the analytic bound's over-licensing: a window that was analytic
# -feasible but did NOT close is observed at utility 0, so its residual pulls the
# posterior down there and nearby. UCB over untried candidates picks the next
# solve. With no data the posterior mean equals the prior, so the first pick is
# the top-retention feasible window -- deterministic and reproducible, so the
# arm is perfectly paired across runs with no RNG.

_BO_LENGTHSCALE = 0.25      # RBF lengthscale in normalised (a,b) space
_BO_SIGNAL_SD = 0.30        # prior SD of utility about its (retention) mean
_BO_NOISE_SD = 0.03         # observation noise on a closure utility
_BO_BETA = 1.5              # UCB exploration weight


def _bo_window(t: E.LNRTarget, history: list[dict]):
    """One GP-UCB proposal over the held-window action space. numpy only."""
    import numpy as np

    L = t.length
    ranked = T.ranked_analytic_windows(t)          # (retention, a, b), cached
    if ranked:
        cand = [(a, b, ret) for ret, a, b in ranked]
    else:
        # No analytically-feasible window: fall back to the full space with a
        # flat prior, so BO degenerates to variance-driven space-filling.
        cand = [(a, b, 0.0) for a in range(L) for b in range(a, L)]

    tried = {(h["a"], h["b"]) for h in history}
    untried = [(a, b, m) for (a, b, m) in cand if (a, b) not in tried]
    if not untried:
        # Feasible set exhausted within the budget: re-propose the best closer
        # seen (a free repeat -- the score cache makes it cost nothing), else
        # the top candidate.
        closed = [h for h in history if h["closes"]]
        if closed:
            best = max(closed, key=lambda h: h["retention"])
            return best["a"], best["b"]
        return cand[0][0], cand[0][1]

    denom = max(1, L - 1)

    def feat(a, b):
        return (a / denom, b / denom)

    def kernel(p, q):
        dx, dy = p[0] - q[0], p[1] - q[1]
        return _BO_SIGNAL_SD ** 2 * np.exp(
            -(dx * dx + dy * dy) / (2.0 * _BO_LENGTHSCALE ** 2))

    # Observations: utility = retention if the window closed, else 0. retention
    # is stored for every evaluated window (it is the free coordinate quantity),
    # so the prior mean at an observed window is h["retention"]; the residual is
    # 0 for a true closer and -retention for an over-licensed non-closer.
    Xo = [feat(h["a"], h["b"]) for h in history]
    resid = np.array([(h["retention"] if h["closes"] else 0.0) - h["retention"]
                      for h in history], float)
    alpha = Kinv = None
    if Xo:
        n = len(Xo)
        K = np.array([[kernel(Xo[i], Xo[j]) for j in range(n)]
                      for i in range(n)], float) + (_BO_NOISE_SD ** 2) * np.eye(n)
        try:
            Kinv = np.linalg.inv(K)
            alpha = Kinv @ resid
        except np.linalg.LinAlgError:
            alpha = Kinv = None

    scored = []
    for (a, b, m) in untried:
        fx = feat(a, b)
        if alpha is not None:
            kstar = np.array([kernel(fx, xo) for xo in Xo], float)
            mean = m + float(kstar @ alpha)
            var = kernel(fx, fx) - float(kstar @ Kinv @ kstar)
        else:
            mean, var = m, kernel(fx, fx)
        acq = mean + _BO_BETA * (max(var, 0.0) ** 0.5)
        # Deterministic tie-break: acquisition, then prior retention, then
        # smaller release, then lower a -- so the arm is reproducible and paired.
        scored.append(((round(acq, 9), round(m, 9), -(L - (b - a + 1)), -a), a, b))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2]


def deterministic_window(arm: str, t: E.LNRTarget, attempt: int,
                         history: list[dict]):
    """The window a rule proposes on its `attempt`-th evaluation."""
    L = t.length
    if arm == "random":
        # The null baseline. A random contiguous window at each evaluation,
        # seeded by (target, attempt) so the floor is reproducible and paired
        # across arms. Without it "the LLM reaches 85% of the ceiling" has no
        # floor -- a legitimate comparison needs to know what chance alone
        # scores at the same evaluation budget.
        import hashlib
        h = hashlib.sha256(f"{t.target_id}|{attempt}".encode()).digest()
        u1 = int.from_bytes(h[:4], "big") / 2**32
        u2 = int.from_bytes(h[4:8], "big") / 2**32
        a = int(u1 * L)
        b = int(u2 * L)
        if a > b:
            a, b = b, a
        return a, min(b, L - 1)
    if arm == "bo":
        # The published-method baseline: GP-UCB over the held-window space,
        # same budget and feedback as the LLM. See _bo_window above.
        return _bo_window(t, history)
    if arm == "contact_core":
        # Naive escalating ladder: protect the contact core, release a little
        # more each time it fails to close. The 'binding residues' instinct,
        # given real feedback but no geometry.
        k = _next_release(history, L)
        return T.hold_contact_core(t, k)
    if arm == "termini":
        k = _next_release(history, L)
        return T.release_termini(t, k)
    if arm == "feasible":
        # Geometry-aware: at the adaptive release budget, hold the
        # analytic-feasible max-retention window. The strong deterministic bar.
        k = _next_release(history, L)
        return T.hold_best_feasible(t, k)
    if arm == "router":
        # Deterministic selection over the tool menu: feasible tool, adaptive k.
        k = _next_release(history, L)
        return T.hold_best_feasible(t, k)
    if arm == "greedy_frontier":
        # A plausible-but-weak rule, kept to make a methodological point: spend
        # each evaluation on the next-highest analytic-retention window. Because
        # the analytic bound over-licenses, the top-retention windows are
        # exactly the ones that do NOT close, so at a small budget this wastes
        # every evaluation above the closure threshold. Its failure is why the
        # monotone binary search below is the right strategy, not this.
        ranked = T.ranked_analytic_windows(t)
        tried = {(h["a"], h["b"]) for h in history}
        for _, a, b in ranked:
            if (a, b) not in tried:
                return a, b
        return (0, L - 1) if not ranked else (ranked[0][1], ranked[0][2])
    if arm == "placement_search":
        # The STRONG deterministic bar. Two phases, both off REAL feedback:
        #   phase 1  binary-search the release budget for the closure threshold
        #            k* (closure is monotone in release), holding the
        #            analytic-max-retention window at each probe.
        #   phase 2  once something has closed, spend the rest of the budget on
        #            the highest-retention analytic-feasible windows at release
        #            k*-1 and k* that have not been tried -- i.e. search WHERE
        #            to hold, not just HOW MUCH to release. This is the part the
        #            binary-search `feasible` rule skips and the part the LLM
        #            arm is really being tested on.
        closed = [h for h in history if h["closes"]]
        tried = {(h["a"], h["b"]) for h in history}
        if not closed:
            k = _next_release(history, L)
            return T.hold_best_feasible(t, k)
        k_star = min(h["release"] for h in closed)
        ranked = T.ranked_analytic_windows(t)
        for want_rel in (k_star - 1, k_star, k_star + 1):
            for _, a, b in ranked:
                if (L - (b - a + 1)) == want_rel and (a, b) not in tried:
                    return a, b
        for _, a, b in ranked:
            if (a, b) not in tried:
                return a, b
        return closed[0]["a"], closed[0]["b"]
    raise ValueError(arm)


def run_campaign(arm: str, t: E.LNRTarget, budget: int, llm=None,
                 max_retries: int = 1, n_restarts: int = 6,
                 max_seconds: float = 10.0, save_raw: bool = False) -> dict:
    history: list[dict] = []
    choices: list = []
    scored: dict = {}            # (a,b) -> WindowScore, so a repeat is free
    best = {"retention": 0.0, "a": -1, "b": -1, "closes": False}
    n_closed = 0
    declines = malformed = 0
    cost = 0.0
    calls = 0
    t_start = time.time()

    for attempt in range(budget):
        pick = None
        if arm in RULE_ARMS:
            pick = deterministic_window(arm, t, attempt, history)
            choices.append({"attempt": attempt, "window": list(pick)})
        else:
            is_tool = arm.endswith("tool")
            # `llm-info-mask` / `llm-info-tool`: same scaffold as the plain arm,
            # plus the analytic closure-feasibility frontier the geometry-aware
            # rules read for free. Everything else (budget, feedback, parsing)
            # is identical, so info-vs-plain isolates the information asymmetry.
            info = "info" in arm
            brief = (tool_brief if is_tool else mask_brief)(
                t, history, budget - attempt, info=info)
            msgs = [{"role": "user", "content": brief}]
            parsed = None
            for r in range(max_retries + 1):
                reply = llm.complete(SYSTEM, msgs)
                calls += 1
                cost += reply.cost_usd
                from lp2cp.enums import StepStatus
                if reply.status == StepStatus.DECLINED:
                    declines += 1
                    break
                if reply.status in (StepStatus.ERROR, StepStatus.BUDGET_CAP):
                    break
                if is_tool:
                    val, st = parse_tool(reply.text)
                else:
                    val, st = parse_hold(reply.text, t.length)
                if val is not None:
                    parsed = val
                    choices.append({"attempt": attempt, "raw": val,
                                    "text": reply.text[:500] if save_raw else ""})
                    break
                if r < max_retries:
                    msgs += [{"role": "assistant", "content": reply.text},
                             {"role": "user",
                              "content": f"Could not parse ({st}). End with "
                              + ("<tool>name k</tool>" if is_tool
                                 else "<hold>a,b</hold>") + " and nothing after."}]
            if parsed is None:
                malformed += 1
                continue
            if is_tool:
                name, k = parsed
                pick = T.BY_NAME[name].place(t, max(0, min(k, t.length - 1)))
            else:
                pick = parsed

        a, b = pick
        if (a, b) in scored:
            s = scored[(a, b)]
        else:
            s = E.score_window(t, a, b, n_restarts=n_restarts,
                               max_seconds=max_seconds)
            scored[(a, b)] = s
        history.append({"a": a, "b": b, "release": s.release_count,
                        "closes": s.closes, "retention": round(s.retention, 4),
                        "repeat": (a, b) in {(h["a"], h["b"]) for h in history}})
        if s.closes:
            n_closed += 1
            if s.retention > best["retention"]:
                best = {"retention": s.retention, "a": a, "b": b, "closes": True}

    return {
        "target_id": t.target_id, "arm": arm, "length": t.length,
        "n_contacts": t.n_contacts, "budget": budget,
        "best_retention": round(best["retention"], 4),
        "best_window": [best["a"], best["b"]] if best["closes"] else None,
        "any_closed": n_closed > 0, "n_closed": n_closed,
        "n_eval": len(history), "llm_calls": calls,
        "declines": declines, "malformed": malformed,
        "cost_usd": round(cost, 6), "seconds": round(time.time() - t_start, 1),
        "history": history, "choices": choices,
        "model_id": getattr(llm, "model_id", arm),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_llm(backend: str, model: str, effort: str, cap: float, max_tokens: int):
    if backend == "openai":
        from lp2cp.llm.openai_api import OpenAIClient
        return OpenAIClient(model=model, effort=effort, max_spend_usd=cap,
                            max_tokens=max_tokens).load()
    if backend == "api":
        from lp2cp.llm.anthropic_api import AnthropicClient
        return AnthropicClient(model=model, effort=effort, max_spend_usd=cap,
                               max_tokens=max_tokens).load()
    if backend == "scripted":
        from lp2cp.llm.anthropic_api import ScriptedClient
        return ScriptedClient(["<hold>2,8</hold>", "<tool>hold_best_feasible 5</tool>"],
                              model_id=model or "scripted")
    raise ValueError(backend)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--out", required=True, help="output jsonl (resumable)")
    ap.add_argument("--arms", default="contact_core,termini,feasible,router")
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--backend", default="", choices=["", "openai", "api", "scripted"])
    ap.add_argument("--model", default="")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-spend-usd", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--n-restarts", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    targets = E.load_manifest(args.manifest, args.pdb_dir)
    targets.sort(key=lambda t: t.target_id)
    if args.limit:
        targets = targets[: args.limit]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done.add((r["target_id"], r["arm"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"[lnr] {len(targets)} targets, arms={arms}, budget={args.budget}, "
          f"{len(done)} rows already done", flush=True)

    llm = None
    if args.backend and any(a.endswith(("mask", "tool")) for a in arms):
        llm = make_llm(args.backend, args.model, args.effort,
                       args.max_spend_usd, args.max_tokens)
        print(f"[lnr] llm backend={args.backend} model={args.model}", flush=True)

    with out.open("a") as fh:
        for t in targets:
            for arm in arms:
                if (t.target_id, arm) in done:
                    continue
                client = llm if arm.endswith(("mask", "tool")) else None
                if arm.endswith(("mask", "tool")) and client is None:
                    continue
                row = run_campaign(
                    arm, t, args.budget, llm=client,
                    max_retries=args.max_retries, n_restarts=args.n_restarts,
                    max_seconds=args.max_seconds, save_raw=args.save_raw)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"[lnr] {t.target_id:8s} {arm:14s} "
                      f"best_ret={row['best_retention']:.3f} "
                      f"closed={row['n_closed']}/{row['n_eval']} "
                      f"{row['seconds']:5.1f}s "
                      f"${row['cost_usd']:.4f}", flush=True)
    print("[lnr] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
