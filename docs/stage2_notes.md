# Stage 2 build notes — things measured while building, not results

*Written for: whoever picks this up next. These are the findings that came out
of making the agent run at all, as distinct from what the campaigns measured.
They are here because each one cost time to discover and would cost the same
time again.*

Date: 2026-11-10. Code: [`../agent/`](../agent/). Run artefacts:
`/zfsauton/scratch/yixiz/lp2cp_stage2/`.

---

## 1. Opus 5.5 declines this task category outright

**Measured.** `claude-opus-5-5` returned `stop_reason=refusal`, category
`bio`, on **68 of 68** LLM steps in the first full API run, and **8 of 8** in a
second run after the system prompt was rewritten to describe the task
accurately (offline methods benchmark · published PDB structures · the model
selects positions only and never emits a sequence · the objective is to
preserve an interaction the deposited structure already shows).

The reframing was a correction, not an evasion — the first prompt said "a
linear peptide binder is being converted into a cyclic analog that must keep
binding its target", which reads as de novo binder engineering and is not what
the system does. It did not change the outcome. **No further attempt was made
to get around it**, and none should be: two accurate framings producing the
same refusal is a consistent determination about the task category, and
engineering past it would be circumventing a safety system rather than fixing a
prompt.

How this relates to Stage 0: there, on the SKEMPI alanine slice, Opus and
Sonnet refused only 3–6 of 73 while **Haiku refused 67%**. So the refusal rate
is not a property of the model alone — it moves with the *domain*. Cyclic
peptide binder work draws a refusal from a model that was nearly compliant on
protein–protein alanine scanning.

**Consequence for the paper.** The plan's control loop has no provision for a
step that declines, and this makes that gap concrete rather than hypothetical:
a campaign is several LLM calls per iteration per parent, so a per-call refusal
rate near 1.0 does not degrade the agent, it removes it. The loop here handles
it — `StepStatus.DECLINED` with a declared per-step fallback, counted per arm —
but an arm whose model declines is not a measurement of that model's
allocation ability, and must not be reported as one.

**Consequence for arm design.** The model axis for this benchmark is: local 8B,
local 70B-4bit, Sonnet 5.5, Haiku 5.5. Opus is reported as *declined*, with the
count, not as a score.

## 2. `refusal` is two different things and must not be pooled

`stop_reason=refusal` carries a category, and the categories mean different
things:

| category | what it is | recorded as |
|---|---|---|
| `bio` | a genuine safety decline | `StepStatus.DECLINED` |
| `reasoning_extraction` | the thinking block could not be extracted | `StepStatus.ERROR` |
| `max_tokens` | output ran out mid-thinking | `StepStatus.ERROR` |

Sonnet 5.5 produced 6 `reasoning_extraction` refusals in 11 calls on the first
pass. Reporting those as "Sonnet refused 55% of the time" would have attributed
a transport problem to a model's safety behaviour — and the Stage 0 refusal
result is one of the things this project carries forward, so getting that
number right matters. `lp2cp/llm/anthropic_api.py` splits them.

## 3. What caused the `reasoning_extraction` failures

Two things, both ours:

1. **Two reasoning channels colliding.** The prompt asked for
   `<reasoning>...</reasoning>` tags while the model also emits native thinking
   blocks. Agent Rosetta's `{reasoning_formatting}` slot exists exactly to
   normalise reasoning and non-reasoning models to one *output* shape, so it is
   now set per backend: tagged for local instruct weights (which have no native
   channel and do need to be told to deliberate), native for API models. The
   forced-depth instruction itself is unchanged and still shared by every arm
   and every backend — that part must not vary, or a win is confounded with it.
2. **`max_tokens` too low.** 3000 truncated thinking at `effort=low`; raised to
   12000.

## 4. The one-call scaffold was silently a different action space

`--one-call-specify` was meant to be a cheaper scaffold (3 LLM calls per
iteration instead of 4) by skipping Agent Rosetta's operator-selection call.
As first written it passed a placeholder operator into the fill prompt, so the
model saw **only `REBUILD_SEGMENT`'s documentation** and a schema with
`REBUILD_SEGMENT` pre-filled. That is not a cheaper scaffold, it is a
five-times narrower action space, and comparing it against the two-call form
would have measured the narrowing. Caught before it reached a reported number;
`test_one_call_scaffold_offers_every_operator` pins it.

## 5. Benchmark calibration that had to be measured, not guessed

Three constants were set from the distribution over 150 CPSea parents rather
than chosen, and each one was wrong in a way that would have made the benchmark
degenerate:

- **Strain saturation.** The raw Ramachandran-deviation proxy is unbounded and
  ran 1.0–30.6 (median 7.2) against a 4.0 limit, so its *normalised* margin was
  a multiple of every other objective's and the defect tie-break never reached
  any other class: the parent set came out **87% `BACKBONE_STRAIN_AT_CLOSURE`**.
  Soft-saturating at 12.0 maps it to 1.0–8.6, median 4.5.
- **ddG weights.** At the first weights the median parent sat at −58 REU against
  a −30 ceiling, so ddG was unfailable and `INTERFACE_CONTACT_LOSS` could not
  occur. Halved, median now −29.6, straddling the ceiling.
- **Cyclisation BSA penalty.** At 0.30 no parent could ever fail
  `BURIAL_DEFICIT` — the defect the LP→CP conflict is most directly about.
  Raised to 0.75 against a measured closure/interface overlap of 0.10–0.46.

After all three: a balanced census of 8/7/7/7/3 across the five classes.

## 6. Parents must be NEAR MISSES or the benchmark measures nothing

With no cap on how many objectives a parent may fail, 19 of 32 selected parents
failed 3+ objectives and **26 of 32 had no intervention anywhere in a 141-cell
action space that produced a contract-holding repair**. Every controller then
sat within 4 utility points of doing nothing. Capping at ≤2 failing objectives
— which is what the plan's own iteration 0 means by "zero qualify; 14 near
misses" — fixed it. The resulting table has 162 `CONTRACT_HOLDING_REPAIR` cells
against 805 `DEFECT_RESOLVED_CONTRACT_BROKEN`, i.e. **the contract-breaking
shortcut is 5× more available than the honest repair**, which is the condition
that makes the 2→3 ablation measurable at all.

## 7. `k_spent0` was internally inconsistent and killed every coordinated move

Set to 2 to mimic the plan's worked example, which starts at 2 of 4 because its
parent came from a generative run that had already substituted twice. But here
the linear parent is *the deposited peptide's own sequence*, so the cyclic
parent is at edit distance 0 and nothing has been spent. Worse, with only 2
substitutions left, a 3-residue rebuild costs 2 and a rebuild-plus-substitution
costs 3 — so **every coordinated move in the grid was labelled INFEASIBLE**,
which is precisely the cell type Stage 0 said must be present (greedy optimal
only 45.8% of the time at k=2). Set to 0.

## 8. Cluster and environment facts

- **`/tmp` is node-local.** A job submitted from lov4 with a `--bench-dir` under
  this session's scratchpad gate-failed on lov5 because the path did not exist
  there. Benchmark artefacts live on `/zfsauton/scratch`.
- **a6000s are routinely GPU-free-but-RAM-full.** gpu29 and gpu30 both had idle
  cards with 2 GB and 4 GB of host RAM free. The 8B job's RAM request was cut
  from 64 G to 32 G (peak load is ~12 GB — four 4 GB shards streamed one at a
  time), which lands it on strictly more nodes.
- **API arms belong on the `cpu` partition.** They wait on the network, not on a
  card; a GPU request would hold a 48 GB a6000 idle for hours.
- **The API venv needs no scientific stack.** `numpy` is imported inside
  `diagnostics.measure`, which an API job never calls, so the dedicated venv
  with `anthropic` alone can run the campaign runner.

## 9. Spend

Stage 0 left ~$4.05 of $5. Stage 2 build + probes spent roughly **$0.95**, of
which **~$0.65 bought nothing** — the Opus refusal runs, which bill input
tokens for a declined call. That is itself worth knowing: a refusing model is
not free to discover.
