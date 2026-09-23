# Learn model development by following Marin's decisions

Research snapshot: **2026-09-22**. The
[Marin repository](https://github.com/marin-community/marin) moves quickly;
issue status and run claims below describe what was public at this cutoff, not a
permanent verdict. For LLM theory and interview derivations, use the
[case-based tutorial](./LLM_INTERVIEW_TUTORIAL_2026.md). For the team's
recurring habits, use
[Marin's development traits](./MARIN_MODEL_DEVELOPMENT_TRAITS.md).

This is a reading-and-analysis apprenticeship, not a recipe for reproducing a
multi-rack run. You need a browser, a notebook, and enough arithmetic to compare
budgets and measurements. Your output is a **casebook of decisions** that you
can defend in an interview.

## The method: reconstruct a decision, not a timeline

For each case, read in this order:

1. Start with a dated
   [weekly standup](https://github.com/marin-community/marin/issues/9324).
   Extract the decision or blocker, not every status line.
2. Open the linked experiment issue. Record its hypothesis, control, proposed
   intervention, promotion criterion, and what remained uncertain _before_ the
   result.
3. Read the linked PR and relevant config or report. Separate implemented
   behavior from proposals and issue comments.
4. Record the measured result with units, denominator, number of steps,
   hardware, and whether the comparison was matched.
5. Write the decision you would make at the information cutoff. Then read later
   comments and explain any change of mind.

Use these evidence labels throughout the casebook: **proposal**, **synthetic
diagnostic**, **controlled small-scale result**, **production observation**, and
**independent evaluation**. A merged PR proves code was integrated, not that a
model-quality claim is true. An open issue may contain strong measurements
without representing a team-wide decision.

### One-page decision memo

```text
Decision and cutoff date:
Model/data/code/run identity:
Question and pre-result prediction:
Control, treatment, matched variables, and confounders:
Primary metric and guardrails:
Observed result (numbers, units, sample size, links):
What the result does NOT establish:
My decision now; evidence that would reverse it:
What happened later:
Two-minute interview explanation:
```

Keep each memo to one page. A source link beside every number is more useful
than a long unsourced summary.

## Where to read code

Code answers “what would actually run?” Issues answer “why did they consider
it?” Read only the slice needed for the current decision.

| Question | Start here | What to trace |
|---|---|---|
| How is a run specified? | [Experiment definitions](https://github.com/marin-community/marin/tree/main/experiments), [pipeline library](https://github.com/marin-community/marin/tree/main/lib/marin) | Config identity, dependencies, data artifacts, checkpoint inputs, evaluation outputs |
| What computes the training step? | [Levanter](https://github.com/marin-community/marin/tree/main/lib/levanter) | Model/loss, optimizer, microbatching, parallel axes, checkpoint restore |
| What reaches the trainer? | [Zephyr](https://github.com/marin-community/marin/tree/main/lib/zephyr), [dataset experiments](https://github.com/marin-community/marin/tree/main/experiments/datasets) | Source filters, tokenizer, packing, shuffle, provenance |
| What makes it run at scale? | [Iris](https://github.com/marin-community/marin/tree/main/lib/iris) | Scheduling, task lifecycle, restart, telemetry |
| What is the current hero shape? | [535B MoE runbook](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/README.md), [hero issue #8435](https://github.com/marin-community/marin/issues/8435) | 384 routed experts/top-8, shared experts, sequence length, rack topology, precision, gates |

Do not treat `main` as a time machine. Pin a commit when reconstructing a
historical decision; the September snapshot inspected
[Marin commit `0541426`](https://github.com/marin-community/marin/commit/054142676a0f11a589557131da3495de76038a1e).

## Eight-week casebook

Plan on three to five hours a week: roughly one hour of first-principles work,
two hours following primary sources, and one hour writing and defending a memo.
The “interview test” is an oral prompt: answer without reciting issue
chronology.

### Week 1 — Model development is a portfolio of options

Read the
[8B retrospective](https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md),
[hero campaign #8435](https://github.com/marin-community/marin/issues/8435), and
[September 21 standup #9324](https://github.com/marin-community/marin/issues/9324).
Draw three nested loops: cheap experiment, production run, and program-level
learning. Mark which changes to a live run were controlled tests and which were
operational interventions. The 535B target is a campaign with a scale ladder and
possible context-extension phases, not proof that every planned phase has run.

**Deliverable:** a timeline with a separate column for decision, evidence
available then, and later outcome. **Interview test:** “When should you branch a
valuable checkpoint instead of starting over?”

### Week 2 — Scaling laws are forecasts with error bars

Read the
[hero iso-FLOP sweep #8003](https://github.com/marin-community/marin/issues/8003),
[fast-track experiment framework PR #9287](https://github.com/marin-community/marin/pull/9287),
and the
[feature-ablation proposal #9288](https://github.com/marin-community/marin/issues/9288).
Derive approximate dense training FLOPs `6ND`; explain why active parameters,
routing, attention length, and hardware utilization complicate the MoE version.
Compare **data-matched** and **compute-matched** experiments. PR #9287 provides
a 16k-BPE dense/MoE ladder at several widths; #9288 is a proposed screen, not a
result.

**Deliverable:** an experiment plan with two widths, equalized budget, promotion
threshold, and a rule for abandoning obsolete cells. **Interview test:** “A d512
win disappears at d1024. What do you infer?”

### Week 3 — Architecture, numerics, and runtime are one decision

Read the
[535B hero runbook](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/README.md),
[hero gradient-norm investigation #9148](https://github.com/marin-community/marin/issues/9148),
and
[coordinated-GC PR #9224](https://github.com/marin-community/marin/pull/9224).
In #9148, the `lm_head` dominated the squared gradient norm at ~100k steps, yet
fixed-state probes did not justify an immediate intervention while loss and
evaluations stayed stable. PR #9224 measured a 2.46% elapsed-time reduction over
300 diagnostic steps on 64 GPUs; the feature remained opt-in. These are
different kinds of evidence and different risk thresholds.

**Deliverable:** a fault tree separating changed training trajectory, numerical
implementation, measurement artifact, and harmless scale shift. **Interview
test:** “Would you change router arithmetic halfway through a costly run?” Use
the
[Sep 22 router analysis in #8435](https://github.com/marin-community/marin/issues/8435)
to defend the answer.

### Week 4 — Data is a versioned part of the model

Read the
[32B retrospective](https://github.com/marin-community/marin/blob/main/docs/reports/marin-32b-retro.md),
[hero mixture swap #9126](https://github.com/marin-community/marin/issues/9126),
and
[SFT overlap audit #9212](https://github.com/marin-community/marin/issues/9212).
The mixture study reports a d1536 main-phase BPB change from 0.92507 to 0.91844
and estimates a 1.20× **compute-equivalent** gain under equal throughput; that
is not a measured 1.20× wall-clock gain or proof of a completed production swap.
The SFT audit flagged 8.42M of 113.35M conversations as possible evaluation
overlap; a flag is not verified leakage, and counts are not token-weighted.

**Deliverable:** a data-card excerpt with source identity, filters, phase
weights, version, overlap protocol, and a counterexample to “lower loss means
better model.” **Interview test:** “How could a dedup filter remove useful
training data while a benchmark still appears to improve?”

### Week 5 — Long context is both a quality and systems claim

Read
[context-parallel training PR #9119](https://github.com/marin-community/marin/pull/9119),
[65k H100 diagnostics #9277](https://github.com/marin-community/marin/issues/9277),
and
[checkpoint handoff #8506](https://github.com/marin-community/marin/issues/8506).
PR #9119 reached 40 diagnostic updates at 262k context on 64 GB200 GPUs, with
~10.03% median MFU; it did not establish long-context task quality. #9277
reports 65k synthetic diagnostics with different batch sizes and parallel
layouts, so its MFU figures are not an isolated context-parallel speed
comparison. The hero handoff used a 200-step parent/child loss-overlap check and
checkpoint lineage; that establishes a narrower continuity claim than “the
entire future run is reproduced.”

**Deliverable:** a table with five distinct gates: fits in memory, finite steps,
throughput, loss continuity, and long-context capability. **Interview test:**
“Why is a 262k training step insufficient evidence that the model uses 262k
context?”

### Week 6 — Evaluation is an experimental protocol

Read
[Eval Policy v0.1 #9193](https://github.com/marin-community/marin/issues/9193),
[benchmark protocol PR #9145](https://github.com/marin-community/marin/pull/9145),
[standard-error fix PR #9196](https://github.com/marin-community/marin/pull/9196),
and
[Harbor output-budget fix PR #9327](https://github.com/marin-community/marin/pull/9327).
Specify the benchmark split, model checkpoint, prompt and reasoning mode,
generation budget, metric, seeds, attempted/full counts, and confidence interval
_before_ a comparison. Policy v0.1 remains a living issue, not an immutable
standard. AIME repeats over 30 items do not create 300 independent questions.

**Deliverable:** a one-page evaluation protocol and an example of how a serving
or harness change could masquerade as a model improvement. **Interview test:**
“A capped run scores higher. Can it replace the full result?”

### Week 7 — Post-training optimizes a narrower distribution

Read
[Snowball comparison #9225](https://github.com/marin-community/marin/issues/9225),
[async RL investigation #8936](https://github.com/marin-community/marin/issues/8936),
[successor program #8955](https://github.com/marin-community/marin/issues/8955),
and
[OPD/MOPD prototype #9250](https://github.com/marin-community/marin/issues/9250).
In #9225, SFT+RL scored 0.307 on the sampled SWE-bench Verified set versus 0.145
for its “base,” yet TB2 fell from 0.094 to 0.050. That “base” is already a
Stage-3 SFT checkpoint, not raw pretraining. Different trial counts and
concurrency prevent a clean universal ranking. In #8936, raw GSM8K reward hid
nontermination; completed-answer rate changed the interpretation. The OPD/MOPD
work is preliminary parity/prototype evidence, not a demonstrated general
replacement for RL.

**Deliverable:** a post-training scorecard with capability, completion rate,
cost, off-distribution regression, and evaluation comparability. **Interview
test:** “When is faster asynchronous RL worse despite higher training reward?”

### Week 8 — Hold a launch review

Use
[launch-readiness history #8233](https://github.com/marin-community/marin/issues/8233),
[current hero campaign #8435](https://github.com/marin-community/marin/issues/8435),
[run handoff #8506](https://github.com/marin-community/marin/issues/8506), and
the latest standup. Write a two-page recommendation to continue, branch, change,
or stop a hypothetical hero run. Include data identity, model/numerics, quality
gates, throughput, checkpoint recovery, evaluation protocol, owner, and rollback
trigger. Label every assertion by evidence level. This is a fictional review; do
not claim authority over Marin's actual run.

**Deliverable:** the review memo plus a five-minute oral defense. **Interview
test:** “What would make you delay a launch despite a promising validation
loss?”

## Keep the casebook fresh

Once a month, read the newest standup, then inspect only the two or three issues
that could change an existing memo. Record the new cutoff and source commit.
Preserve older conclusions as _then-known_ decisions; add a dated correction
instead of rewriting history. Prefer the latest issue comment, linked PR and run
report over an undated `main` snippet, and preserve disagreements between
sources.

The goal is to explain not just how a Transformer trains, but how a team spends
scarce compute to reduce uncertainty—and how it knows when a result is too weak
to promote.
