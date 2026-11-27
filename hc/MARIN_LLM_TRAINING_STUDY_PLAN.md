# Learn model development by following Marin's decisions

Research snapshot: **2026-11-27**. The
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
If your interview is before the end of 2026, use the tutorial's
[five-week route](./LLM_INTERVIEW_TUTORIAL_2026.md#five-week-route-for-a-december-2026-interview)
to prioritize these cases.

## The method: reconstruct a decision, not a timeline

For each case, read in this order:

1. Start with a dated
   [weekly standup](https://github.com/marin-community/marin/issues/11597).
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
historical decision; this snapshot inspected
[Marin commit `e19b299`](https://github.com/marin-community/marin/commit/e19b29901a8da95c5898d38973448ecf61509fa2).

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
and the active
[Marin Belay ladder #9451](https://github.com/marin-community/marin/issues/9451).
Derive approximate dense training FLOPs `6ND`; explain why active parameters,
routing, attention length, and hardware utilization complicate the MoE version.
Compare **data-matched**, **compute-matched**, and **fixed wall-clock**
experiments. PR #9287 provides the 16k-BPE dense/MoE ladder. Belay v1 selects
variants by Paloma macro loss after eight minutes of 8×H100 train-step time at
d512, then checks d768–d1280 and an extrapolated 1e24-FLOP point. Its current
candidate improves the measured small rungs, but d512 uses a different step
count and the 1e24 projection assumes a fixed loss asymptote. Hero-scale
training and inference throughput are outside this gate.

**Deliverable:** an experiment plan with two widths, equalized budget, promotion
threshold, and a rule for abandoning obsolete cells. Calculate how a fixed-time
win can arise from more tokens rather than a better per-token model. **Interview
test:** “A d512 win disappears at d1024. What do you infer?”

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
[4K/8K/16K hero diagnostics #9615](https://github.com/marin-community/marin/issues/9615),
and
[checkpoint handoff #8506](https://github.com/marin-community/marin/issues/8506).
PR #9119 reached 40 diagnostic updates at 262k context on 64 GB200 GPUs, with
~10.03% median MFU; it did not establish long-context task quality. #9615
restored the hero step-180000 checkpoint on one rack for three 100-step arms at
the same 4.19M tokens/update. Relative to 4K, 8K lost 1.6% tokens/s and 16K
lost 3.1%, while expert-drop fractions rose 4.5× and 16×. Sequence length and
QK scaling changed together; one-rack drops and 100-step stability do not
predict the 11-rack continuation. The [step-146139 cutover in #8506](https://github.com/marin-community/marin/issues/8506)
passed a 200-step overlap and improved observed 11-rack tokens/s from 2.83M to
3.14M, but later checkpoint stalls show why a good performance handoff still
needs storage and recovery gates.

The later [262K result #8435](https://github.com/marin-community/marin/issues/8435#issuecomment-6748198431)
and [recipe PR #11044](https://github.com/marin-community/marin/pull/11044)
complete the earlier decision chain. Log-scaled and global-1.84 recipes tied
on clean 262K macro loss (1.6434 vs 1.6433), while log scaling lowered
teacher-forced MRCR full-context NLL from 0.0252 to 0.0223. There was one
training seed per recipe; the NLL result is not a demonstrated gain in
generated retrieval accuracy. [Export schema 3 #11128](https://github.com/marin-community/marin/pull/11128)
and [Megatron support #11129](https://github.com/marin-community/marin/pull/11129)
show why training, export, SFT, and RL must implement the same attention scale.
For a systems confounder, [Snowball SFT's two-speed audit #11909](https://github.com/marin-community/marin/issues/11909)
found 44–55 versus 120–128 seconds/update correlated with cached versus in-job
compilation across audited 64-H100 runs. A related [autotune-dir pair #11793](https://github.com/marin-community/marin/issues/11793)
compiled 24 distinct executables and hung at update 1 with default per-process
dirs, versus one executable and three finite updates with a uniform populated
dir. There was one run per arm on different node sets, so the cause of the
original speed split and hang is not fully isolated.

**Deliverable:** a gate table for memory, finite steps, throughput, loss
continuity, expert drops, export/serving parity, and long-context capability.
**Interview test:** “Which November results go beyond the October 16K
diagnostics, and which questions still require generated-task evaluation?”

### Week 6 — Evaluation is an experimental protocol

Read
[October 16 Eval Policy #10197](https://github.com/marin-community/marin/issues/10197),
[verified-cohort gate PR #9461](https://github.com/marin-community/marin/pull/9461),
[GraphWalks failure analysis #9562](https://github.com/marin-community/marin/issues/9562),
and [base-model format study #10935](https://github.com/marin-community/marin/issues/10935).
Specify the benchmark split, model checkpoint, prompt and reasoning mode,
generation budget, metric, seeds, attempted/full counts, and confidence interval
_before_ a comparison. The October policy defines reportable benchmark settings
and distinguishes gradeable model failures from infrastructure absences. In
GraphWalks, doubling the output budget raised Grug's F1 from 0.073689 to
0.144746 on the **same 750 examples**; the original and rerun totals covered
920 and 750 examples, so
their overall scores cannot be compared directly. In #10935, adding Qwen3.5
base's assistant-turn stop token changed SWE-bench Verified from 0.05 to 0.55
without training, while verified-row coverage also changed from 20 to 40.
An evaluator score is a property of model, serving contract, harness, and
scored population together.

**Deliverable:** a one-page evaluation protocol and an example of how a serving
or harness change could masquerade as a model improvement. **Interview test:**
“The same weights score ten times higher after a stop-token change. What
exactly improved, and what must be held fixed to compare model quality?”

### Week 7 — Post-training optimizes a narrower distribution

Read
[Snowball comparison #9225](https://github.com/marin-community/marin/issues/9225),
[Mochi SFT weakness analysis #11723](https://github.com/marin-community/marin/issues/11723),
and [long-loop comparison #12201](https://github.com/marin-community/marin/issues/12201).
In #9225, SFT+RL scored 0.307 on the sampled SWE-bench Verified set versus 0.145
for its “base,” yet TB2 fell from 0.094 to 0.050. That “base” is already a
Stage-3 SFT checkpoint, not raw pretraining. Different trial counts and
concurrency prevent a clean universal ranking. #11723 found that Mochi's
thinking-off versus thinking-on probe changed deterministic-constraint
accuracy from 98/120 to 116/120. On missing-argument tool prompts, thinking
off over-called on 57/60; thinking on abstained on 59/59. A one-hour synthetic
multi-needle SFT continuation on Snowball did not improve full MRCR-65K
against its control, so that data stayed out of Mochi's mix. In #12201,
Mochi SFT→RL lost 6.82 pooled Micro210 reward points while Snowball
round 1 gained 5.03 under a different policy. Micro210 pools heterogeneous
partial-credit rewards, not pass@1. Their bases, recipes, judges,
serving stacks, and scored populations differ; this is not evidence that RL
causally hurts one base and helps the other. Mochi's checkpoint selector also
fell back to the final step because no holdout evaluation met its 180/200
coverage gate.

**Deliverable:** a post-training scorecard with capability, completion rate,
cost, off-distribution regression, and evaluation comparability. **Interview
test:** “Training reward rises, but no holdout evaluation meets the coverage
gate. Which checkpoint can you defend?”

### Week 8 — Hold a launch review

Use
[launch-readiness history #8233](https://github.com/marin-community/marin/issues/8233),
[completed hero pretraining record #8435](https://github.com/marin-community/marin/issues/8435),
[run handoff #8506](https://github.com/marin-community/marin/issues/8506), and
the [November 9 standup #11597](https://github.com/marin-community/marin/issues/11597).
Write a two-page retrospective on the pretraining decisions, then a hypothetical
promotion memo for the next SFT/RL stage. Include data identity, model/numerics,
quality gates, throughput, checkpoint recovery, evaluation protocol, owner,
and rollback trigger. Label every assertion by evidence level. This is a
fictional review; do not claim authority over Marin's actual run.

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
