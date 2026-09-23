<!-- markdownlint-disable MD013 -->

# How large language models are developed

## A case-based tutorial for 2026 interviews

Updated: **2026-09-22**
Marin evidence cutoff: **2026-09-22**. The companion reading guide also selects external frontier sources through **2026-09-22**; its selection is not a comprehensive survey of the month.

This tutorial teaches the reasoning used to develop a language model: how to
turn a capability goal into a training recipe, measure whether an idea works,
diagnose a failed run, and decide whether to scale it. It uses
[Marin](https://github.com/marin-community/marin) as a public case study because
its code, experiment issues, negative results, retrospectives, and launch gates
are visible.

The companion
[reading guide](https://github.com/voe09/codingmachine/blob/main/hc/spaces-ac-llm-reading-guide.md)
is an index of papers and implementations. Use it to choose sources. Use this
tutorial to learn how the pieces constrain one another.

You do not need to reproduce a large run. Small calculations, trace reading,
config comparison, and paper-to-issue reconstruction are enough to build the
judgment tested in research and systems interviews.

## What you should be able to do

After completing the tutorial, you should be able to:

1. derive the shapes, parameter count, training compute, memory, and KV-cache
   cost of a decoder Transformer;
2. explain the quality and systems tradeoffs among dense models, GQA, MLA,
   sparse MoE, long-context attention, and residual-routing variants;
3. design an iso-FLOP experiment and convert a loss improvement into an
   equivalent wall-clock gain;
4. triage loss spikes using evidence from data, optimizer state, attention
   statistics, precision, and distributed execution;
5. specify a reproducible data mixture, shuffle, deduplication, contamination,
   and checkpoint-lineage contract;
6. choose parallelism and serving strategies from the workload rather than
   model size alone;
7. derive SFT, preference, policy-gradient, RLVR, and distillation objectives
   and identify their failure modes;
8. design an evaluation that separates model quality from harness and serving
   failures; and
9. give an interview answer with a hypothesis, mechanism, measurement, decision
   rule, and caveat.

## The development loop

A model is the result of a coupled loop:

1. **Target:** define the users, capabilities, latency, context, safety, and
   cost envelope.
2. **Measurement:** choose validation losses, downstream tasks, serving
   metrics, and statistical decision rules before seeing results.
3. **Data:** define source identities, filters, deduplication, mixture weights,
   ordering, tokenizer, and contamination policy.
4. **Model:** choose objective, architecture, parameterization, optimizer, and
   precision.
5. **System:** make the recipe executable through kernels, parallelism,
   checkpointing, telemetry, and recovery.
6. **Experiment:** compare matched alternatives at more than one scale.
7. **Decision:** promote, revise, or reject using quality per wall-clock
   resource.
8. **Post-training and serving:** adapt behavior, evaluate complete
   trajectories, and feed failures back into earlier stages.

These stages constrain one another. GQA may lose some quality and still win
because it cuts KV cache. A better MoE router may lose because its all-to-all
traffic dominates. An apparent model regression may be a tool parser or timeout
change. A lower training loss may come from a different data phase.

---

## Part I — The accounting you must know

### 1. Objective, tokens, and metrics

For token sequence \(x_{1:T}\), an autoregressive model factorizes

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

The token-average negative log-likelihood is

$$
\mathcal L(\theta)
=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

If natural logarithms are used, token perplexity is
\(\mathrm{PPL}=\exp(\mathcal L)\). Perplexity is meaningful only with the same
tokenizer and token accounting. Bits per byte,

$$
\mathrm{BPB}
=-\frac{1}{N_{\text{bytes}}}\sum_t \log_2 p_\theta(x_t\mid x_{<t}),
$$

is more comparable across tokenizers, though normalization and corpus details
still matter.

#### Why tokenization is part of the model

A tokenizer determines sequence length, effective compute per character,
multilingual fragmentation, code and number boundaries, and the units that the
softmax predicts. A 0.1 lower token loss does not establish a better model if
the candidate tokenizer emits more tokens for the same text.

For any tokenizer comparison, report:

- bytes or characters per token by domain and language;
- vocabulary size and embedding/output-head cost;
- unknown or byte-fallback behavior;
- boundary behavior for code, whitespace, numbers, and Unicode;
- downstream quality at matched raw data and compute; and
- both inference tokens per task and tokens per second.

#### Causal masking and supervision masks

The causal attention mask prevents position \(t\) from attending to future
positions. A separate loss mask decides which tokens contribute to the
objective. During chat SFT, prompt tokens can remain visible as context while
only assistant tokens receive loss. Confusing these masks either leaks future
information or trains the model to imitate the user.

**Takeaway:** a loss number has no meaning without its tokenizer, data
distribution, masking policy, and normalization.

### 2. One decoder block from shapes to cost

Let the residual stream be \(X\in\mathbb R^{B\times T\times d}\). A pre-norm
decoder block can be written as

$$
H=X+\mathrm{Attention}(\mathrm{RMSNorm}(X)),
$$

$$
Y=H+\mathrm{MLP}(\mathrm{RMSNorm}(H)).
$$

With \(H_q\) query heads, \(H_{kv}\) key/value heads, and head dimension
\(d_h=d/H_q\):

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

After reshaping,

- \(Q\) has shape \(B\times H_q\times T\times d_h\);
- \(K,V\) have shape \(B\times H_{kv}\times T\times d_h\).

For each query head,

$$
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad O=AV.
$$

MHA has \(H_{kv}=H_q\), MQA has \(H_{kv}=1\), and GQA lies between them. GQA
shares keys and values among query-head groups. It reduces the cache and K/V
projection cost, but can remove representational capacity.

#### RoPE

RoPE rotates pairs of query and key coordinates by a position-dependent angle.
For one two-dimensional pair,

$$
R_m(\theta)=
\begin{bmatrix}
\cos m\theta&-\sin m\theta\\
\sin m\theta& \cos m\theta
\end{bmatrix}.
$$

Because \(R_m^\top R_n=R_{n-m}\), the query-key inner product contains relative
position. Long-context extensions change the mapping from position to rotation
frequency; they cannot by themselves prove that the model uses distant
evidence. Long-context evaluation must test retrieval, composition,
instruction-following, and degradation across positions.

#### SwiGLU

With hidden width \(m\),

$$
\mathrm{SwiGLU}(x)
=\left(\mathrm{SiLU}(xW_g)\odot xW_u\right)W_d.
$$

The three matrices contribute approximately \(3dm\) parameters per dense MLP.
RMSNorm contributes little to parameter count but affects numerical behavior.

#### Approximate parameters per dense block

Let \(r=H_{kv}/H_q\). Ignoring biases and norms:

$$
P_{\text{attention}}\approx 2(1+r)d^2,
$$

because Q and output projections each cost \(d^2\), while K and V each cost
\(rd^2\). Thus

$$
P_{\text{block}}\approx 2(1+r)d^2+3dm.
$$

For MHA, \(r=1\), giving \(4d^2\) attention parameters. For 4:1 GQA,
\(r=1/4\), giving \(2.5d^2\).

#### Worked calculation: a 1.2B-class model

Suppose:

- \(d=2048\);
- 24 layers;
- \(m=5632\);
- \(H_q=16\), \(H_{kv}=4\);
- vocabulary \(V=64{,}000\);
- tied input/output embeddings.

Per block:

$$
P_{\text{attention}}\approx2.5(2048)^2=10.49\text{M},
$$

$$
P_{\text{MLP}}\approx3(2048)(5632)=34.60\text{M}.
$$

Twenty-four blocks contain about
\(24(10.49+34.60)=1.082\)B parameters. The tied embedding contains
\(64{,}000(2048)=131.1\)M. Norms add little, so the total is approximately
1.21B.

This estimate is good enough to catch a wrong config before launch. Exact
accounting must follow the implementation: untied heads, biases, routed and
shared experts, and latent projections change the total.

### 3. Compute, memory, and KV cache

For a dense decoder, a common first-order training estimate is

$$
C_{\text{train}}\approx6ND,
$$

where \(N\) is the number of non-embedding parameters used per token and \(D\)
is the number of training tokens. The factor six approximates forward and
backward matrix multiplication. Attention's quadratic term, embeddings,
recomputation, sparsity, padding, and hardware inefficiency require a more
exact model for a launch.

For the 1.21B example and a \(3\times10^{20}\) FLOP budget:

$$
D\approx\frac{3\times10^{20}}{6(1.21\times10^9)}
\approx41.3\text{B tokens}.
$$

This calculation bounds one candidate allocation. Matched iso-FLOP runs are
needed to infer the compute-optimal model size.

#### Training-state memory

A conventional mixed-precision Adam configuration can require roughly:

- 2 bytes per parameter for bf16 weights;
- 2 bytes for bf16 gradients;
- 4 bytes for fp32 master weights;
- 4 bytes each for first and second moments.

That is about 16 bytes per parameter before activations, temporary buffers, and
fragmentation. A 7B model would require roughly 112 GB for unsharded model and
optimizer state. Exact bytes depend on optimizer and precision recipe. ZeRO/FSDP
shard state; tensor and pipeline parallelism partition compute; activation
checkpointing trades additional FLOPs for memory.

#### KV-cache formula

For batch \(B\), sequence length \(T\), layers \(L\), KV heads \(H_{kv}\), head
dimension \(d_h\), and \(s\) bytes per element:

$$
M_{\mathrm{KV}}
=B\,T\,L\,2H_{kv}d_hs.
$$

The factor two stores keys and values.

**Worked calculation.** With one 32,768-token request, 32 layers, 8 KV heads,
head dimension 128, and bf16:

$$
M_{\mathrm{KV}}
=1(32768)(32)(2)(8)(128)(2)
=4{,}294{,}967{,}296\text{ bytes}=4\text{ GiB}.
$$

MHA with 32 KV heads would use 16 GiB for the same request. This is why a model
team can retain GQA even when an MHA ablation has somewhat better loss.

#### Attention arithmetic

Projection and MLP work scale roughly as \(O(Td^2)\). Full attention score and
value aggregation scale as \(O(T^2d)\). FlashAttention computes exact softmax
attention in tiles, reducing HBM traffic and avoiding materializing the full
\(T\times T\) matrix. It changes the memory algorithm, not the attention
function.

**Competence check.** Why can a model with fewer FLOPs run slower?

**Answer.** FLOPs omit communication, memory traffic, kernel launch overhead,
padding, expert imbalance, and low accelerator occupancy. A sparse expert may
reduce arithmetic while adding all-to-all communication and small matrix
multiplications. Measure tokens/s, MFU, memory, and communication alongside
loss.

---

## Part II — Eight model-development cases

### Case 1: choose a scale and shape

#### The decision

You have a fixed training allocation. Choose model width, depth, active
parameters, total parameters, sequence length, batch, and token count. The
objective is the best target capability within wall-clock, memory, serving, and
reliability constraints.

#### Scaling laws are planning models

A loss law often has the form

$$
L(C)=L_\infty+AC^{-\alpha}.
$$

It summarizes a specified model family, optimizer, data distribution, and
training region. It does not establish that a new architecture, new mixture, or
larger extrapolation follows the same curve.

At a fixed compute budget, very small models are overtrained and very large
models are undertrained. Plotting final loss against size yields a U-shaped
iso-FLOP curve. Repeat at several budgets:

1. choose several plausible widths around the predicted optimum;
2. set tokens so each cell uses the same model FLOPs;
3. tune learning rate or use a scaling rule validated in that range;
4. keep tokenizer, mixture, sequence length, batch semantics, and evaluation
   fixed;
5. finish or censor cells according to a preregistered rule;
6. fit the optimum model size and token count as functions of compute;
7. confirm at a larger budget before committing a hero run.

Marin's [August hero iso-FLOP sweep](https://github.com/marin-community/marin/issues/8003)
used widths d512–d2048 and budgets \(10^{18}\)–\(3\times10^{20}\) FLOPs. Its
completed smaller budgets formed U-curves, while divergent cells and unfinished
high-budget runs kept the extrapolation provisional. That wording is good
scientific practice: a clean fit over finished cells does not erase missing or
failed cells.

#### Initialization and schedule transfer

Width, depth, batch, and token budget do not scale independently. Residual
scaling and initialization control how activation and gradient variance grow
with depth. MuP-style parameterizations try to make learning-rate and
initialization choices transfer across width, but transfer is an empirical
contract for a particular parameter grouping and optimizer.

A complete schedule specifies warmup in tokens, peak learning rate, decay
shape, minimum learning rate, weight decay, gradient accumulation, and cooldown
or continued-pretraining transitions. Comparing runs by step while changing
global batch compares different token counts. Comparing final loss after
different cooldown fractions also confounds the architecture with its schedule.

#### Dense versus sparse MoE

For \(E\) routed experts and top-\(k\) routing, distinguish:

- **total parameters:** storage and checkpoint burden;
- **active parameters:** parameters used for one token;
- **model FLOPs:** arithmetic from active paths;
- **capacity:** tokens an expert can accept;
- **communication:** token dispatch and combine traffic;
- **routing quality:** specialization, load distribution, and dropped tokens.

If \(n\) tokens enter an MoE layer, ideal average assignments per expert are
\(nk/E\). With capacity factor \(c\), a simple per-expert capacity is

$$
\mathrm{capacity}=\left\lceil c\frac{nk}{E}\right\rceil.
$$

A small \(c\) may drop or reroute tokens. A large \(c\) pads work and wastes
memory. Expert parallelism adds all-to-all communication, so active parameter
count alone cannot predict speed.

#### Worked decision

Candidate A is the 1.21B dense model above. Candidate B is an MoE with 5.0B
total parameters but 1.35B active per token. At the same \(3\times10^{20}\)
model-FLOP budget, the first-order token allocations are:

$$
D_A\approx41.3\text{B},\qquad
D_B\approx\frac{3\times10^{20}}{6(1.35\times10^9)}
\approx37.0\text{B}.
$$

Suppose a matched pilot finds:

| Candidate | Validation loss | Tokens/s | Peak memory | Target eval |
|---|---:|---:|---:|---:|
| Dense A | 2.91 | 430k | 71 GB | 42.0 |
| MoE B | 2.86 | 305k | 77 GB | 43.1 |

The MoE has better quality at the fixed FLOP budget but 29% lower throughput.
There is no justified launch decision yet. You need the baseline scaling curve
to convert the 0.05 loss gain into equivalent compute, plus serving measurements
and at least a second scale. The correct conclusion is “quality-positive,
wall-clock unresolved,” not “MoE wins.”

#### Answer pattern for scale selection

For “How would you choose the size of a model under a compute budget?”:

1. state the target capability and hard serving constraints;
2. use \(6ND\) only to bound a candidate region;
3. run matched iso-FLOP U-curves across several widths and budgets;
4. tune the hyperparameters that do not transfer reliably;
5. fit compute-optimal size and tokens with uncertainty;
6. correct model-FLOP gains by measured throughput;
7. validate the selected point at the next scale;
8. record failed and censored cells.

**Red flag:** citing Chinchilla as a universal parameter-to-token ratio without
checking data, architecture, or extrapolation.

**September case update.** The [535B-A23B hero campaign #8435](https://github.com/marin-community/marin/issues/8435) targets 18T tokens, but its scale ladder and context phases are conditional decisions, not a single fixed training recipe. The [fast-track framework PR #9287](https://github.com/marin-community/marin/pull/9287) adds matched dense/MoE experiments at d512–d1280 with a 16k-BPE tokenizer. A proposed [hold-one-feature-out grid #9288](https://github.com/marin-community/marin/issues/9288) is useful for designing a screen; do not cite it as a measured win. Interview question: what evidence at d512 and d1024 would justify spending the next order of magnitude of compute?

### Case 2: diagnose an unstable run

#### Start with the causal chain

A useful incident timeline separates:

1. **input:** batch identity, token statistics, masks, lengths;
2. **forward:** activation RMS, attention logits, router loads, loss;
3. **backward:** per-group gradient norms and nonfinite values;
4. **optimizer:** preconditioned update norms and learning rate;
5. **state transition:** parameter and optimizer-state checksums;
6. **distributed result:** collectives, stragglers, dropped experts;
7. **next step:** whether the anomaly persists or recovers.

A scalar global gradient norm is insufficient. Log per-layer or per-parameter
group distributions, update-to-weight ratios, attention-logit extrema, and the
batch identity.

#### AdamW precisely

Ignoring bias correction for readability:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\theta_{t+1}
=\theta_t-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-\eta_t\lambda\theta_t.
$$

The raw gradient norm and the preconditioned update norm can tell different
stories. A coordinate with a modest gradient and underestimated second moment
can receive a large adaptive update. Clipping only the raw gradient does not
guarantee a bounded parameter update.

#### Why QK normalization can help

Attention logits are

$$
z_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}.
$$

If query or key norms grow, logit magnitudes grow. Softmax becomes sharp,
gradients concentrate, and low-precision ranges are stressed. QK normalization
constrains vector magnitudes before the dot product, leaving direction to carry
the similarity signal. It creates headroom; it is not a proof that every
instability is caused by attention.

#### Fault tree

| Class | Evidence to inspect | Matched intervention |
|---|---|---|
| Data | batch IDs, source, length, token IDs, mask density, duplicates | replay the same batch; substitute a known-good batch |
| Optimizer | LR, moments, update/weight ratios, clipping | restore before anomaly; change one optimizer control |
| Architecture | Q/K norms, logits, residual RMS, router loads | add or remove the suspected normalization in a paired run |
| Precision | overflows, cast sites, accumulation dtype, reduction order | run a short higher-precision or deterministic comparison |
| Distributed system | rank divergence, collective errors, expert drops | compare per-rank state and single-host replay |
| Implementation | config diff, checkpoint conversion, masks, shapes | unit/reference test and minimal deterministic replay |

#### Marin 32B reconstruction

The [Marin 32B retrospective](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/docs/reports/marin-32b-retro.md) records a
valuable sequence:

1. training was usable for roughly 70k steps but had more spikes than smaller
   runs;
2. large update norms preceded gradient-norm and loss spikes;
3. tighter gradient clipping, update clipping, bad-step skipping, restarts, and
   optimizer changes did not provide durable stability;
4. the team warm-started a QK-normalized attention backbone from the 80k
   checkpoint;
5. loss paid a one-time penalty, recovered in about 10B tokens, and the spikes
   disappeared;
6. a later cooldown anomaly was traced separately to data ordering, not model
   instability.

The evidence supports “QK normalization fixed this run under this recipe.”
It does not support “large Llama-style models cannot train without QK
normalization.” The retrospective itself notes counterexamples and earlier
stable runs.

#### Solved incident

**Observation:** loss jumps at step 80,412. The preceding step has a normal
batch, normal raw gradient norm, a 12× update-norm outlier in Q/K projections,
and extreme attention logits in three late layers. Restarting from step 80,000
reproduces the event within 500 steps on different data. A higher-precision
replay changes the exact step but not the growing Q/K norms.

**Diagnosis:** data is weakened as the cause because the batch changes and a
restart reproduces the trend. A pure transient overflow is weakened because
higher precision delays but does not remove growth. The strongest current
hypothesis is attention-scale instability interacting with adaptive updates.

**Next experiment:** paired short runs from the same pre-instability checkpoint:
control versus QK normalization, identical data order, LR, optimizer state, and
precision. Compare loss, Q/K norms, logit percentiles, update norms, and time to
first spike. Do not simultaneously change optimizer and architecture.

**Promotion rule:** no spike over a horizon longer than the control's failure
time, bounded attention and update statistics, recovery of the one-time loss
penalty, and no material throughput regression.

#### Answer pattern for instability diagnosis

State the timeline and competing hypotheses before proposing a fix. Interviewers
are looking for causal isolation, checkpoint discipline, and the ability to
distinguish data phase shifts from numerical instability.

**September case update.** In [hero gradient-norm investigation #9148](https://github.com/marin-community/marin/issues/9148), the `lm_head` accounted for about 95% of squared total gradient norm around 100k steps. Fixed-checkpoint probes compared numerical paths and z-loss settings; they did not show a direct kernel bug that warranted interrupting a run with stable loss and evaluations. The later [router-precision check in #8435](https://github.com/marin-community/marin/issues/8435) changed top-8 order for 8.9853% of tokens, slowed steps by roughly 1.5–2%, and found no benefit over a 20-step continuation. That argues against a mid-run switch, not against testing higher precision in a fresh-run ablation. The interview skill is knowing when a surprising internal statistic is a monitoring signal rather than a mandate to intervene.

### Case 3: decide whether an architecture idea is real

#### Evaluate quality and runtime together

Suppose the baseline scaling law near a test point is

$$
L_b(C)=L_\infty+A_bC^{-\alpha}.
$$

At budget \(C_0\), recenter the coefficient through the observed baseline:

$$
A_b=(L_b(C_0)-L_\infty)C_0^\alpha.
$$

For a variant loss \(L_v\), the compute the baseline would need to match it is

$$
C_{\text{needed}}
=\left(\frac{A_b}{L_v-L_\infty}\right)^{1/\alpha}.
$$

The loss-only equivalent-compute gain is
\(C_{\text{needed}}/C_0\). Correct it by measured throughput:

$$
S_{\text{wall}}
=\frac{C_{\text{needed}}}{C_0}
\frac{\mathrm{TPS}_v}{\mathrm{TPS}_b}.
$$

#### Worked calculation from the Marin gate

The [Agent MoE playbook](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/experiments/grug/moe/agent.md) gives:

- \(C_0=3.82\times10^{17}\);
- \(L_\infty=1.6\);
- \(\alpha=0.0941\);
- baseline loss 3.5422 at 433,986 tokens/s;
- hypothetical variant loss 3.52 at 410,000 tokens/s.

The compute ratio can be simplified:

$$
\frac{C_{\text{needed}}}{C_0}
=\left(\frac{3.5422-1.6}{3.52-1.6}\right)^{1/0.0941}
\approx1.13.
$$

The throughput ratio is \(410000/433986\approx0.945\). Therefore:

$$
S_{\text{wall}}\approx1.13(0.945)\approx1.07.
$$

The candidate is about 7% faster to the matched loss despite being about 5.5%
slower per training step. This conclusion depends on the local scaling-law
shape and must be checked at another scale.

#### A negative result that teaches more

In Marin's
[Agent MoE digest](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/docs/reports/agent-moe-experiments.md), finer expert
granularity improved loss enough for a 1.04–1.07× model-FLOP gain. Throughput
fell by roughly 30%, producing only 0.72–0.76× wall-clock speed. The mechanism
was plausible and the loss improved, yet the model-development decision was to
reject it.

The same digest shows why one metric cannot define the recipe:

- MHA improved effective training speed but GQA remained attractive because
  its KV cache was four times smaller;
- full attention residuals improved loss but lost after a 9–14% throughput
  cost, while block residuals survived;
- some router changes won at one scale and lost at another;
- several “promising” results remained unpromoted because larger-scale anchors
  were missing.

#### A defensible gate

1. Freeze the baseline implementation and reference measurements.
2. Predeclare primary loss, throughput window, completion requirement, and
   failure handling.
3. Run two small compute-optimal scales.
4. Require effective wall-clock gain at both.
5. Run two larger scales only for survivors.
6. Fit a scaling law with an explicit asymptote policy.
7. Require favorable projections at two future budgets.
8. Check memory, inference, stability, and combination with the current recipe.

#### Solved exercise

**Candidate:** loss-only gain 1.18×, throughput ratio 0.82, peak memory +12%,
positive at d512 and negative at d768.

**Answer:** wall-clock gain at the positive cell is
\(1.18(0.82)=0.968\), already below one. The scale inversion is additional
negative evidence. Reject from the current recipe; retain the mechanism and
measurements as a negative result. Do not average the two scales into a win.

**September case update.** The [fast-track PR #9287](https://github.com/marin-community/marin/pull/9287) makes the control contract unusually explicit: data- versus compute-matched runs, width ladder, and tokenizer cache tags that prevent evaluation against the wrong vocabulary. The latter is not housekeeping; a cache mismatch can manufacture an architecture result. Treat the [feature-ablation proposal #9288](https://github.com/marin-community/marin/issues/9288) as an experiment design to critique, not a set of completed outcomes.

### Case 4: make data a reproducible experimental variable

#### Dataset identity

A dataset name is not an identity. A reproducible source needs:

- immutable input URIs or content hashes;
- code revision and complete preprocessing config;
- tokenizer revision;
- filter, classifier, and threshold versions;
- exact deduplication scope and index;
- included and excluded splits;
- output manifest and counts;
- mixture weights and sampling method;
- shuffle algorithm and seed;
- contamination audit version.

If a preprocessing fix is merged but old artifacts remain in a cache, the
training job may still consume contaminated data. Cache keys must include every
semantic input that changes output.

#### Mixtures change the gradient

For datasets \(D_i\) sampled with probabilities \(w_i\),

$$
\nabla\mathcal L_{\text{mix}}
=\sum_i w_i\,\mathbb E_{x\sim D_i}
[\nabla\ell(x;\theta)].
$$

Weights therefore define the optimization target. “Add more code” is not a
complete intervention: it reduces the relative mass of something else and may
alter length, packing, language, and duplication distributions.

At an 8M-token global batch with weights
\((0.50,0.20,0.15,0.10,0.05)\), the expected token allocations are
\((4.0,1.6,1.2,0.8,0.4)\)M. A direct sampler may hold those proportions stable,
but it does not guarantee good within-source ordering.

#### Deduplication and contamination

Deduplication can reduce memorization and wasted compute, but aggressive fuzzy
matching can delete legitimate templates, code idioms, or multilingual
variants. Report precision/recall audits by domain and evaluate retained-token
quality.

Contamination has at least four forms:

1. exact benchmark examples;
2. paraphrases, translations, or solutions;
3. benchmark-derived synthetic data;
4. task-format leakage that does not copy the answer but narrows the problem.

Keep a clean benchmark holdout before data collection when possible. Scan
before and after normalization, record matches rather than silently deleting
them, and distinguish pretraining from post-training exposure.

#### Packing, boundaries, and loss semantics

Packing concatenates documents to reduce padding. It can still be wrong in
three independent ways:

- attention crosses a document boundary when the recipe intended isolation;
- position IDs reset or continue differently between training and serving;
- end-of-document and loss masks teach unintended transitions.

Record useful-token fraction, padding, documents per sequence, boundary-token
frequency, and length distribution. For sequence parallel or FlashAttention
variants, confirm that the packed-sequence representation preserves the same
causal and document-boundary semantics as the reference implementation.

#### Shuffling is a statistical property

A permutation can be bijective and still produce harmful local correlation. A
linear modular traversal

$$
i_t=(a+ts)\bmod N,\qquad \gcd(s,N)=1
$$

visits every item, yet a structured source order and unfortunate stride can
create long phases. Test windowed source proportions, document adjacency,
length autocorrelation, duplicate distance, and loss by source over training
time.

The 32B Marin cooldown exposed both failure modes:

- a fixed preprocessing rule had not invalidated an already cached bundle,
  allowing GSM8K contamination;
- a cheap stateless linear permutation created correlated phases;
- evaluation losses remained comparatively flat during one training-loss
  shift, pointing toward changed input distribution rather than parameter
  instability;
- a Feistel-based shuffle and cleaner mixture removed the observed phase
  behavior.

#### Solved diagnosis

**Observation:** training loss drops abruptly, math benchmark rises, broad
validation is unchanged, gradient/update norms remain normal, and the change
aligns with an epoch boundary.

**Ranked hypotheses:**

1. data-order or mixture phase shift;
2. benchmark contamination or near-duplicate exposure;
3. intended curriculum boundary;
4. optimizer change;
5. spontaneous model improvement.

**Tests:** map step to sample IDs; plot source and length proportions around the
boundary; run contamination scans; replay the checkpoint on pre-boundary and
post-boundary batches; check config and optimizer hashes. The first action is
not an optimizer restart.

#### Data experiment contract

For a proposed mixture change, hold architecture and training budget fixed,
report per-domain held-out losses, downstream task groups, global validation,
tokens/s, and source-token counts. Require the hypothesized domain gain without
an unacceptable broad regression. If the candidate changes average token
length or packing efficiency, charge the wall-clock difference.

**September case update.** The [hero mixture swap #9126](https://github.com/marin-community/marin/issues/9126) completed a d1536 study: main-phase BPB changed from 0.92507 to 0.91844, with an estimated 1.20× **compute-equivalent** speedup if throughput is equal. Wikipedia was a small regression and code BPB regressed at some smaller widths. This does not measure a 1.20× wall-clock gain or prove that all proposed production phases were deployed. The [SFT audit #9212](https://github.com/marin-community/marin/issues/9212) scanned 113,347,086 normalized conversations and flagged 8,421,460 (7.43%) for possible eval overlap; after duplicate/overlap removal 104,878,042 remained. Flags are not verified leakage, counts are not token-weighted, and bounded dedup comparisons do not establish exhaustive recall. Good data work states the detection limits beside the impressive counts.

### Case 5: design the training system and recovery contract

#### Parallelism dimensions

- **Data parallelism:** replicas process different batches; gradients are
  reduced.
- **Fully sharded data parallelism:** parameters, gradients, and optimizer state
  are sharded and gathered around computation.
- **Tensor parallelism:** individual matrix operations are partitioned; it
  requires frequent collectives.
- **Pipeline parallelism:** layer groups occupy stages; bubbles and activation
  transfers matter.
- **Context/sequence parallelism:** sequence work is split for long contexts.
- **Expert parallelism:** experts are distributed; routing invokes all-to-all
  exchange.

The selection follows topology. Use fast links for communication-heavy tensor
or expert groups, and put lower-frequency data-parallel reductions across
slower links when possible.

#### Roofline reasoning

Arithmetic intensity is FLOPs per byte moved. A kernel is approximately:

- compute-bound when intensity is high enough to saturate matrix units;
- bandwidth-bound when HBM traffic dominates;
- latency/communication-bound when small kernels or collectives leave devices
  idle.

Model FLOP utilization is

$$
\mathrm{MFU}
=\frac{\text{achieved model FLOPs/s}}
{\text{theoretical accelerator FLOPs/s}}.
$$

MFU is useful only with a declared FLOP convention. Tokens/s and time-to-target
remain the operational outputs.

For a ring all-reduce over \(p\) ranks with message size \(M\), each rank moves
approximately \(2(p-1)M/p\) bytes, ignoring protocol overhead. Tensor
parallelism may pay such collectives inside every block; data parallelism
usually pays after backward; expert parallelism pays token-dependent all-to-all
traffic. Frequency is as important as byte volume.

#### Worked memory decision

A 7B model with a 16-byte Adam state estimate requires 112 GB before
activations. On eight 80-GB accelerators:

- pure data parallelism replicates the 112 GB and does not fit;
- full sharding divides the persistent state to about 14 GB per rank, before
  all-gather buffers and activations;
- long sequences may still OOM because activation memory scales with batch,
  length, layers, and hidden width;
- activation checkpointing reduces saved activations at the cost of forward
  recomputation.

The answer “use FSDP” remains incomplete. Specify wrapping granularity,
prefetch/all-gather behavior, checkpoint format, sequence length, microbatch,
gradient accumulation, and topology.

#### MoE system checks

For every routed layer, log:

- tokens assigned and accepted per expert;
- drop or reroute rate;
- capacity padding;
- dispatch/combine bytes and time;
- per-expert matrix shapes;
- straggler ranks;
- quality by token/source group when feasible.

A load-balancing loss can make counts uniform while routing semantically poor
tokens. A dropless router can preserve tokens while producing severe
stragglers. Balance, quality, and transport are separate measurements.

#### Checkpoint and resume correctness

A resumable training state includes model parameters, optimizer moments,
scheduler, step and token counters, data iterator and shuffle position, RNG
states, dynamic loss scale, and any adaptive router or curriculum state.

A recovery drill should:

1. stop at a controlled step;
2. resume on the intended topology;
3. compare the next batches and learning rates;
4. compare a short loss/update trajectory with uninterrupted training;
5. verify artifact names and retention;
6. time checkpoint save and restore;
7. exercise the policy for corrupted or partial checkpoints.

#### Hero-run thinking

Marin's [next hero run burndown](https://github.com/marin-community/marin/issues/8233)
shows that launch readiness is a proof across architecture, numerics, runtime,
data, evaluation, topology, and recovery. Every unresolved gate needs closure
or an explicit recorded decision. “The code runs on a small model” is not a
launch contract.

#### Solved system-design prompt

**Prompt:** throughput is 25% below projection for a top-4 MoE. Compute kernels
show acceptable utilization, but step-time variance is high.

**Answer:** split the step profile into router, dispatch, expert GEMM, combine,
and non-MoE blocks. Inspect per-expert load and rank stragglers, capacity
padding/drops, all-to-all bytes, topology placement, and matrix-size
distribution. Compare a dense or locally routed control. If the tail follows
the most-loaded expert rank, tune routing/capacity or expert placement. If all
ranks wait on the collective with balanced loads, investigate transport,
message sizes, and overlap. Do not optimize the expert GEMM based only on mean
utilization.

**September case update.** [Context-parallel PR #9119](https://github.com/marin-community/marin/pull/9119) ran 40 diagnostic updates at 262k context on 64 GB200 GPUs, reporting 10.03% median MFU and 1.42% token drops under that configuration. It establishes a finite training path, not long-context model quality. [H100 65k diagnostics #9277](https://github.com/marin-community/marin/issues/9277) report 15.523% MFU for PP24/EP8 at batch 384 versus 9.877% for PP24/EP4/CP2 at batch 96. Since batch and layout differ, the numbers are not an isolated CP speed comparison. [Hero handoff #8506](https://github.com/marin-community/marin/issues/8506) used a 200-step parent/child loss-overlap check (mean difference +0.000306) and explicit checkpoint lineage. Recovery, finite steps, throughput, and capability need separate gates.

### Case 6: separate model quality from serving and evaluation

#### Prefill and decode are different workloads

**Prefill** processes the prompt in parallel and is dominated by large
matrix operations and attention over the prompt. **Decode** adds one token per
sequence and repeatedly reads weights and KV cache; it is often bandwidth and
batching sensitive.

Report at least:

- time to first token;
- inter-token latency;
- end-to-end latency;
- prompt and output tokens/s;
- concurrency and request-length distribution;
- queue time, preemption, and KV utilization;
- accuracy under the exact serving configuration.

Continuous batching admits and retires requests between decode steps. Paged KV
allocation reduces fragmentation. Speculative decoding uses a draft model to
propose tokens and a target model to verify them; its gain depends on acceptance
length and verification cost, not draft speed alone.

#### From logits to tokens

For logits \(z_i\) and temperature \(\tau\),

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Greedy decoding chooses the maximum. Top-\(k\) keeps a fixed number of tokens;
top-\(p\) keeps the smallest set whose probability mass reaches \(p\).
Temperature and truncation interact, so they must be reported together with
seed and maximum output length. Beam search optimizes a sequence-score
approximation and is often useful for constrained or short tasks; for open-ended
chat it can reduce diversity and favor generic continuations. Repetition
penalties and grammar-constrained decoding change the effective distribution
and can break evaluation parity.

#### Quantization

Separate:

- weight-only versus weight-and-activation quantization;
- post-training quantization versus quantization-aware training;
- group size, scale granularity, outlier handling, and calibration data;
- prefill and decode speedups;
- quality by task and context length;
- memory saved after runtime buffers and KV cache.

An FP8 or FP4 label is not a recipe. Accumulation dtype, scaling, clipping,
which tensors remain higher precision, and kernel support determine the result.

#### Capacity calculation

Use the earlier 4-GiB KV example. If a deployment has 170 GiB available after
weights and non-KV runtime allocations, a naive upper bound is
\(\lfloor170/4\rfloor=42\) concurrent full-length sequences. Real capacity is
lower because allocator slack, temporary attention buffers, uneven sequence
lengths, and safety margin remain. At 8k tokens, the same cache shape is about
1 GiB per sequence, so length-aware admission control materially changes
throughput.

#### Evaluation is an estimator

For task distribution \(P(x)\), harness \(H\), model-serving configuration
\(S\), and scoring function \(r\), the measured quantity is

$$
\hat\mu=\frac1n\sum_{i=1}^{n}r(x_i,\tau_i;H,S).
$$

It estimates a specific system under a specific protocol. Record prompt
templates, few-shot examples, decoding, tool parser, context/output limits,
retries, timeouts, model revision, server flags, and environment version.

For binary pass rates, a rough standard error is

$$
\mathrm{SE}(\hat p)=\sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

For paired model comparison, retain per-example outcomes and bootstrap the
paired difference. For pass@\(k\), use the estimator matching the sampling
scheme; do not treat repeated attempts as independent users.

If \(n\) samples contain \(c\) correct solutions, the common unbiased estimator
for drawing \(k\) without replacement from those samples is

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

It is undefined for claims about a different sampler, prompt, or budget.
LLM-as-judge evaluation additionally needs position-order swaps, rubric
validation against humans or verifiers, judge-version pinning, and analysis of
self-preference, verbosity, and style bias.

#### Marin parity case

In [issue #7930](https://github.com/marin-community/marin/issues/7930), one
baseline family reproduced historical results while another failed three
parity checks despite more than 90% infrastructure-clean coverage. Trace and
serving evidence identified higher timeout rates, decode-bound requests,
different tool parsing, and a different output-token allowance.

The model-regression conclusion was therefore unsupported. The useful outcome
was a debugging protocol:

1. enforce an infrastructure-clean gate;
2. preserve run identity for selective retry;
3. compare effective harness configs, not intended configs;
4. condition scores on timeout and failure class;
5. inspect serving counters and traces;
6. retain trial artifacts before stopping a driver.

#### Solved parity calculation

Suppose historical reward is 0.70 on non-timeout trials and 0.05 on timeout
trials. Timeout rate rises from 30% to 55%.

Historical expected score:

$$
0.70(0.70)+0.05(0.30)=0.505.
$$

New expected score if conditional capability is unchanged:

$$
0.70(0.45)+0.05(0.55)=0.3425.
$$

A 0.1625 score drop can be explained entirely by the changed failure mix. This
does not prove the model is equal; it shows why the serving mismatch must be
fixed before attributing the aggregate drop to model weights.

#### Evaluation suite

A launch-quality suite contains:

- clean held-out language-model loss by domain;
- capability tasks with uncertainty and saturation checks;
- instruction, safety, and refusal behavior;
- long-context tasks across positions and lengths;
- agent trajectory and infrastructure metrics;
- efficiency under representative traffic;
- regression sets from known failures;
- contamination and benchmark-version records.

**September case update.** The living [Eval Policy v0.1 #9193](https://github.com/marin-community/marin/issues/9193) specifies benchmark versions, metrics, repeats, generation limits, and a protected out-of-distribution set. [Protocol PR #9145](https://github.com/marin-community/marin/pull/9145) records attempted/full counts and uncertainty; [standard-error PR #9196](https://github.com/marin-community/marin/pull/9196) corrected an AIME24 interval from a vacuous [0, 1] to approximately [0.129, 0.197] in a reported case. Repeated trials over 30 questions are not 300 independent questions. [Harbor PR #9327](https://github.com/marin-community/marin/pull/9327) forwards the requested generation-token limit to ordinary requests, illustrating how an output-budget bug can affect apparent score parity. Policy v0.1 is still evolving; quote the tested protocol, not merely its name.

### Case 7: choose a post-training signal

#### Supervised fine-tuning

For prompt \(x\), response \(y\), and assistant-token mask \(m_t\):

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

SFT teaches format, style, tool protocol, and demonstrated strategies. It is
limited by support: it can imitate only trajectories present in the data and
may overfit formatting artifacts or lose base capabilities.

Rejection sampling and best-of-\(n\) can improve SFT data by generating several
candidates and retaining verifier- or judge-approved responses. This shifts
quality toward the selector: false positives become training targets, and
diversity can collapse.

For parameter-efficient adaptation, LoRA represents an update to a frozen
matrix as

$$
W'=W+\frac{\alpha}{r}BA,
$$

with rank \(r\), \(A\in\mathbb R^{r\times d_{\mathrm{in}}}\), and
\(B\in\mathbb R^{d_{\mathrm{out}}\times r}\). It reduces trainable and optimizer
state, but base-model activations still dominate some workloads, target-module
choice matters, and low rank can limit large behavioral changes. Compare LoRA
and full fine-tuning at matched data and wall-clock budget, including serving
and adapter-management cost.

#### Preference modeling

A Bradley–Terry reward model can use

$$
\mathcal L_{\mathrm{RM}}
=-\log\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)).
$$

Preference data quality depends on annotator policy, pair difficulty, position
bias, length bias, prompt coverage, and whether preferences remain valid for
the current policy distribution.

For a KL-regularized objective,

$$
\max_\pi\;
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}),
$$

the optimal policy has

$$
\pi^*(y\mid x)\propto
\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta).
$$

DPO substitutes this relation into a pairwise likelihood. With
\[
\Delta_\theta=
\log\pi_\theta(y^+\mid x)-\log\pi_\theta(y^-\mid x)
\]
and the analogous \(\Delta_{\mathrm{ref}}\):

$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(\beta(\Delta_\theta-\Delta_{\mathrm{ref}})\right).
$$

DPO avoids an explicit reward-model-and-PPO loop, but it remains an offline
preference method. Distribution shift, pair construction, reference choice,
length bias, and excessive margins still matter.

#### Policy gradients, PPO, and GRPO

The policy-gradient identity is

$$
\nabla_\theta J
=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)A_t\right].
$$

PPO uses an importance ratio

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
$$

and clips large policy changes. Actor, critic, reward model, reference model,
rollout engine, and synchronization make the system expensive.

GRPO-style methods sample a response group for one prompt and normalize rewards
within the group:

$$
A_i=\frac{r_i-\bar r}{s_r+\epsilon}.
$$

This removes a learned critic in the simple form. It does not remove variance,
bad rewards, zero-variance groups, off-policy drift, length effects, or rollout
cost. DAPO-like recipes adjust clipping, sampling, token aggregation, and
overlong handling; interview answers should name the component rather than use
an algorithm label as explanation.

#### RL with verifiable rewards

RLVR uses a programmatic verifier for math, code, or constrained outputs.
Verifiability lowers reward ambiguity but does not guarantee a good objective.
Models can exploit parser gaps, tests, answer formats, timeouts, or partial
credit. Track:

- verifier validity and adversarial tests;
- reward distribution and zero-variance groups;
- train–evaluation overlap;
- response length and entropy;
- KL from reference;
- solve rate by difficulty;
- infrastructure failure classes;
- held-out verifiers.

#### Distillation and on-policy distillation

Offline distillation trains on a fixed teacher dataset. On-policy distillation
samples from the student and obtains token-, step-, or trajectory-level teacher
signals on states the student actually visits. This reduces a form of
distribution mismatch but makes teacher inference, synchronization, and
feedback design part of training.

OPD and SDPO remain active 2026 research directions; Marin's [September OPD/MOPD prototype #9250](https://github.com/marin-community/marin/issues/9250) supplies preliminary implementation evidence, not a settled recipe.
Treat reported gains as method evidence that still needs independent
replication. Distinguish:

- online versus cached/offline teacher feedback;
- same versus different teacher/student tokenizer;
- token versus reasoning-step supervision;
- forward versus reverse KL behavior;
- teacher consistency and latency;
- student-policy staleness;
- reasoning tasks versus long-horizon agent trajectories.

The companion reading guide links the current primary papers and repositories
for [SDPO](https://arxiv.org/abs/2601.20802),
[Lightning OPD](https://arxiv.org/abs/2604.13010), and later 2026 variants.

#### Marin SFT case

[Issue #8225](https://github.com/marin-community/marin/issues/8225) took a
cooled-down checkpoint through Chat, Thinking, and two task-specific third-stage
branches. The pipeline and durable artifacts succeeded. Math tasks improved
substantially, but among 51 sequential non-agentic score comparisons, 30
improved and 21 regressed. The right conclusion is capability redistribution
under a reproducible SFT pipeline, not monotonic improvement.

This reveals four development traits:

1. keep every intermediate checkpoint;
2. evaluate intermediate and final stages;
3. compare post-training cost with the alternative pretraining allocation;
4. treat agent infrastructure quality as a gate on model conclusions.

#### Solved method selection

**Goal:** improve a code model whose main failures are wrong tool protocol and
weak solutions on problems with executable tests.

**Plan:**

1. SFT on clean tool schemas and high-quality solution trajectories to teach the
   interface.
2. Evaluate protocol validity separately from code correctness.
3. Add RLVR on sandboxed tests after the model reliably emits executable tool
   calls.
4. Harden tests against reward hacking and keep private held-out tests.
5. Consider preference training for style or ambiguous quality dimensions not
   captured by tests.
6. Use online distillation only if a stronger teacher supplies useful feedback
   on student-visited failures and teacher cost is justified.

Starting with RL on an invalid tool protocol wastes rollout compute and confounds
interface errors with reasoning.

**September case update.** [Snowball comparison #9225](https://github.com/marin-community/marin/issues/9225) reports sampled SWE-bench Verified scores of 0.145 for its starting checkpoint and 0.307 for SFT+RL, while TB2 goes from 0.094 to 0.050. The starting checkpoint is already Stage-3 SFT; trial counts and concurrency differ, and the domains are not equivalent. This is evidence of capability redistribution, not “RL universally improves agents.” [Async RL #8936](https://github.com/marin-community/marin/issues/8936) found raw GSM8K reward could hide nonterminating answers (120/128 raw successes versus 64 completed in one arm); its [successor #8955](https://github.com/marin-community/marin/issues/8955) still studies safe staleness and quality gates. [OPD/MOPD prototype #9250](https://github.com/marin-community/marin/issues/9250) establishes preliminary parity tests, not a production replacement for RL.

### Case 8: develop retrieval, agents, and multimodal systems

#### Retrieval-augmented generation

A practical RAG path is:

query → query transformation → candidate retrieval → filtering/reranking →
context packing → generation → citation/verification.

Sparse retrieval captures exact terms. Dense retrieval captures learned
semantic similarity. Hybrid systems often need score calibration or rank
fusion. Measure components separately:

- retrieval recall@\(k\) against answer-bearing evidence;
- reranker precision or NDCG;
- answer accuracy conditional on retrieved evidence;
- citation entailment and coverage;
- abstention when evidence is absent;
- latency and token cost.

Chunking is a modeling choice. Small chunks raise retrieval precision but lose
context; large chunks improve continuity but dilute evidence and consume the
context window. Preserve titles, hierarchy, dates, permissions, and source
identity. Deduplicate overlapping chunks before packing.

#### Agents as partially observed control

At step \(t\), an agent observes context \(o_t\), chooses an action \(a_t\)
(text or tool call), receives tool output, updates working state, and continues.
The final reward depends on a trajectory \(\tau\), not one completion.

A reliable loop has explicit:

1. goal and constraints;
2. typed action schema;
3. observation normalization;
4. state and budget tracking;
5. error classification and retry policy;
6. verification before irreversible action;
7. stop conditions;
8. complete trajectory logging.

Evaluate success, tool-call validity, recovery, steps, tokens, latency, cost,
unsafe actions, environment failures, and judge/verifier agreement.

#### Memory

- **Working memory:** current prompt or summarized trajectory.
- **Episodic memory:** prior runs and outcomes.
- **Semantic memory:** retrieved facts and documentation.
- **Procedural memory:** skills, policies, and tool instructions.

Every memory write needs provenance, access control, retention, and a policy for
contradictions. More memory can amplify stale or malicious content.

#### Security boundary

Retrieved text and tool output are untrusted data. They must not silently
become higher-priority instructions. Use least-privilege tools, typed
parameters, domain and path allowlists, secret isolation, approval for
irreversible actions, output validation, and audit logs. Test indirect prompt
injection, data exfiltration, confused-deputy actions, malicious files, and
cross-user memory leakage.

#### Multimodal models

An understanding model typically maps modality encoders or tokenizers into a
shared sequence model, often through a projector or cross-attention interface.
Training may combine contrastive alignment, captioning, next-token prediction,
instruction tuning, and preference or RL objectives.

Important details include:

- image/audio/video tokenization and resolution;
- positional representation across space and time;
- modality balance and missing-modality behavior;
- projector bottlenecks;
- grounding and hallucination evaluation;
- input-specific safety;
- preprocessing parity between train and serve.

Autoregressive image token models and diffusion/flow models optimize different
generation processes. Diffusion language models are also an active alternative
to left-to-right generation, but their practical value depends on sampling
steps, likelihood/evaluation conventions, editing ability, and serving
throughput. Treat them as a frontier branch, not a replacement assumed in the
core curriculum.

#### Solved agent incident

**Observation:** a new agent model scores three points higher, but uses 40% more
tokens, times out twice as often, and ran with a different tool parser.

**Answer:** the comparison is invalid for a model-quality claim. Re-run under
the same parser, tool schema, context/output limits, retry policy, and budgets.
Report paired task outcomes, infrastructure-clean coverage, success conditional
on failure class, tokens and latency, and trajectory-level errors. If unequal
budgets are part of the product policy, report a quality–cost frontier instead
of one score.

---

## Part III — Frontier knowledge and evidence levels

### 9. Stability levels

Use three evidence levels in interviews.

#### Foundation

These are durable mechanisms you should derive and implement at small scale:

- autoregressive cross-entropy and masking;
- decoder attention, RoPE, RMSNorm, and gated MLPs;
- AdamW, gradient accumulation, clipping, and mixed precision;
- data sampling, packing, deduplication, and held-out evaluation;
- data/tensor/pipeline/expert parallelism;
- prefill, decode, KV cache, batching, and quantization basics;
- SFT, pairwise preference learning, policy gradients, and RAG.

#### Current engineering practice

These have broad practical evidence, though recipes differ:

- FlashAttention-style exact tiled attention;
- GQA and latent/cache compression methods;
- sparse MoE with explicit routing and communication metrics;
- scaling-law-guided sweeps rather than one guessed scale;
- QK normalization or related logit controls for additional stability;
- FP8 training/inference in selected tensors with higher-precision
  accumulation;
- RLVR for domains with robust programmatic verifiers;
- paged KV management and continuous batching;
- agent evaluation with infrastructure and trajectory gates.

#### Active frontier: selected external sources through 2026-09-22

| Direction | Mechanism to understand | Evidence question |
|---|---|---|
| Kimi Linear/KDA and gated Delta-style models | recurrent or linear attention state with gating | Does long-sequence quality survive at useful latency? |
| MLA and gated MLA | compress latent K/V representation | What cache reduction remains after kernels and quality constraints? |
| Attention Residuals and hyper-connections | learned aggregation across layer/residual states | Does loss gain repay extra memory and compute at scale? |
| Deep sparse MoE/LatentMoE variants | more total capacity with controlled active work | What are routing, drop, all-to-all, and serving costs? |
| Muon-family optimization and QK controls | structured matrix updates and logit stability | Does the optimizer transfer across scale, data, and parameter groups? |
| FP4-class training/inference | lower-precision compute and storage | Which tensors, scales, accumulators, and error controls preserve quality? |
| Disaggregated prefill/decode | place different workloads on different pools | Does transport and KV movement erase utilization gains? |
| OPD/SDPO and step-level feedback | teacher signal on student-visited states | Is the gain worth teacher cost and robust to staleness/tokenizer mismatch? |
| World-model and agentic training | model environment transitions or train on trajectories | Does simulated experience transfer to real tools and environments? |
| Diffusion language models | iterative denoising instead of strict left-to-right decoding | What is the quality–latency–editability frontier? |

Recent model reports such as
[DeepSeek-V4](https://arxiv.org/abs/2606.19348),
[Kimi K3](https://arxiv.org/abs/2607.24653), and
[Qwen-AgentWorld](https://arxiv.org/abs/2606.24597) are useful systems case
studies. Read them as bundles of choices. Separate the claimed mechanism,
ablation evidence, hardware/software stack, evaluation protocol, and facts that
have independent reproduction.

The September selections add two particularly useful interview contrasts.
[Qwen3.8-Next](https://arxiv.org/abs/2608.30320) jointly ablates hybrid
Gated DeltaNet/attention, sparse attention, Gated Residual, and optimizer
settings; its authors explicitly report that lower loss and downstream accuracy
can diverge. [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) reports a
causal encoder-decoder design and cross-layer/FP4 KV-cache compression for
input-heavy agent workloads. Treat its cost and capability numbers as
author-reported until independently measured. In post-training,
[Rethinking OPD II](https://arxiv.org/abs/2609.04172) reports that one query's
rollouts cover 71.5% of the states visited by its full-data experiment; the
interview question is whether state coverage or slow student alignment limits
progress, and whether that finding transfers beyond its tested settings.

### 10. How to read a paper, issue, or standup

Use this one-page extraction:

1. **Decision:** what choice was the team trying to make?
2. **Baseline:** exact code, config, data, budget, and metric.
3. **Hypothesis:** mechanism and predicted observable.
4. **Intervention:** smallest actual change.
5. **Controls:** what was held fixed; what was unintentionally changed?
6. **Execution:** completed, diverged, censored, or missing cells.
7. **Result:** raw loss, throughput, memory, downstream metrics, uncertainty.
8. **Gate:** predeclared or chosen after seeing results?
9. **Conclusion:** supported at tested scales.
10. **Next risk:** scale transfer, interaction, serving, data, or evaluation.

#### Worked extraction: August Marin standup

The [week of August 17 standup](https://github.com/marin-community/marin/issues/8394)
mentions SFT, agentic RL ablations, hero architecture, expert configurations,
ragged all-to-all performance, custom-kernel correctness, data deduplication,
and hero preparation.

Do not summarize this as “the team trained an MoE.” Reconstruct the coupled
decision:

- a candidate expert shape changes routing and all-to-all requirements;
- kernel correctness and MFU gate whether that shape is executable;
- data artifacts and deduplication must be ready before launch;
- small-scale exact-architecture and mixture ablations reduce extrapolation
  risk;
- the run contract records which unresolved system choices are acceptable;
- downstream SFT and agent evaluation reveal capability redistribution.

The model recipe is a chain of measured contracts across research and systems.

#### Worked extraction: September Marin standup

The [week of September 21 standup #9324](https://github.com/marin-community/marin/issues/9324) mentions 535B context extension, async RL weight sync, rapid dense/MoE screens, SFT filtering, and evaluator changes. Choose one claim and trace it to a narrower primary artifact. For example:

- “262k context works” means [PR #9119](https://github.com/marin-community/marin/pull/9119) ran 40 finite diagnostic updates at that length, not that the trained model solves 262k retrieval tasks.
- “Mixture gives 1.20× speedup” means [issue #9126](https://github.com/marin-community/marin/issues/9126) estimates a compute-equivalent advantage under equal throughput, not measured end-to-end wall-clock progress.
- “SFT overlap is 7.43%” means [issue #9212](https://github.com/marin-community/marin/issues/9212) flagged that share for possible overlap under a bounded comparison, not verified benchmark leakage.
- “The hero resumed correctly” means [issue #8506](https://github.com/marin-community/marin/issues/8506) passed a 200-step lineage/overlap check, not that all later training is numerically identical.

For each sentence, say what evidence would upgrade it to a stronger claim. That exercise is closer to a research interview than memorizing a model card.

---

## Part IV — A 16-week interview program

### 11. Weekly method

Budget 10–12 focused hours each week:

- **Mechanism, 3 hours:** derive the main equations without notes.
- **Primary evidence, 2 hours:** read one paper section and one Marin
  issue/report.
- **Code, 2 hours:** trace shapes and configs; no large run required.
- **Artifact, 3 hours:** produce a calculation, decision memo, incident
  timeline, or evaluation design.
- **Interview, 2 hours:** give one 20-minute answer, then handle changed
  constraints.

Every artifact must state assumptions, units, raw evidence, decision rule, and
one unresolved risk. Keep a dated correction log; silently replacing an old
belief destroys the evidence that you are learning.

### 12. The schedule

| Week | Focus | Required artifact |
|---|---|---|
| 1 | Cross-entropy, tokenization, masking | Compare two tokenizers by bytes/token and cost |
| 2 | Decoder block, RoPE, GQA, SwiGLU | Shape trace and parameter derivation |
| 3 | FLOPs, scaling, dense/MoE | Iso-FLOP design with candidate U-curves |
| 4 | AdamW, schedules, precision | Optimizer-state and update-norm worksheet |
| 5 | Stability | Marin 32B incident timeline and paired test |
| 6 | Data pipeline | Dataset identity and contamination contract |
| 7 | Mixtures and cooldown | Mixture-change decision memo |
| 8 | Parallelism and memory | Topology plan with memory/communication budget |
| 9 | Kernels and MoE systems | Step-time decomposition and bottleneck diagnosis |
| 10 | Inference | Prefill/decode/KV capacity model |
| 11 | Evaluation | Paired comparison with infrastructure-clean gate |
| 12 | SFT and preference learning | Derive DPO and list violated assumptions |
| 13 | PPO, GRPO, RLVR | Rollout/verifier failure analysis |
| 14 | OPD/SDPO | Teacher–student system and staleness memo |
| 15 | RAG and agents | Threat model plus trajectory evaluation |
| 16 | Synthesis | Full model-development review and two mock interviews |

#### Depth-track substitutions

Keep the shared spine through week 13. Then choose one:

- **Pretraining research:** more iso-FLOP fits, data ablations, optimizer and
  architecture interactions.
- **Training systems:** collective models, sharding, kernels, checkpoint
  consistency, and failure recovery.
- **Post-training:** preference data, RL estimators, verifiers, rollout
  orchestration, and distillation.
- **Inference:** quantization, scheduling, speculative decoding, distributed
  serving, and capacity engineering.
- **Applied/agents:** retrieval, tool protocols, environment design, security,
  and trajectory evaluation.

### 13. A capstone that requires no cluster

Choose one Marin experiment family and produce six linked artifacts:

1. **Experiment map:** parent question, branches, baselines, and chronology.
2. **Config audit:** architecture, optimizer, data, precision, and topology
   differences.
3. **Accounting sheet:** parameters, active parameters, FLOPs, memory, and
   throughput.
4. **Evidence table:** completed, failed, missing, and censored runs with raw
   metrics.
5. **Decision memo:** promote/reject/continue under an explicit gate.
6. **Interview defense:** 20-minute presentation plus written answers to three
   constraint changes.

Good families include the
[Agent MoE tracker](https://github.com/marin-community/marin/issues/4281),
[August iso-FLOP sweep](https://github.com/marin-community/marin/issues/8003),
[32B retrospective](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/docs/reports/marin-32b-retro.md), or
[Terminus parity experiment](https://github.com/marin-community/marin/issues/7930).

### 14. Interview answer rubric

A strong answer usually has six moves:

1. **Clarify the target:** workload, metric, budget, and hard constraints.
2. **Build a first-order model:** shapes, FLOPs, bytes, or estimator.
3. **Name the dominant tradeoff:** quality versus throughput, memory,
   communication, variance, or distribution shift.
4. **Propose a matched experiment:** controls and instrumentation.
5. **Set a decision rule:** threshold, uncertainty, and failure policy.
6. **Name the next risk:** scale transfer, interactions, serving parity, data,
   or security.

“I would try both” is incomplete until you say what is fixed, what is measured,
and what result changes the decision.

---

## Part V — Interview drills with answer keys

### 15. Foundations and architecture

#### 1. Why decoder-only?

**Answer contract:** causal next-token modeling maps directly to open-ended
generation, uses one stack and objective, and scales naturally on raw sequences.
Encoder–decoder models retain advantages when inputs and outputs have distinct
roles or bidirectional input encoding matters. The choice follows workload and
training data, not fashion.

#### 2. Why does RoPE struggle outside the training length?

**Answer contract:** relative phases at unseen distances and frequencies can
leave the trained distribution; attention and data may also lack long-range
behavior. Separate positional extrapolation, attention pattern, training
length, and the model's ability to use evidence. Evaluate across distance and
position, not with one needle score.

#### 3. GQA versus MHA versus MLA?

**Answer contract:** compare representational capacity, KV bytes, projection
cost, kernel support, quality, and conversion/training procedure. Use the KV
formula. MLA compresses keys/values into latent representations but introduces
projection and implementation choices. Give a serving workload before choosing.

#### 4. When does MoE help?

**Answer contract:** more total capacity at limited active compute can help if
routing learns specialization and expert computation amortizes dispatch. It
hurts under imbalance, small GEMMs, all-to-all bottlenecks, capacity drops,
memory pressure, or difficult serving. Report active and total parameters.

### 16. Optimization, data, and systems

#### 5. Loss spikes: what do you inspect first?

**Answer contract:** preserve checkpoint and batch identity; build a timeline
across forward, backward, optimizer, and ranks; compare gradient versus update
norms and attention/router statistics; generate data, precision, optimizer,
architecture, and distributed hypotheses; replay and change one factor.

#### 6. How do learning rate and global batch interact?

**Answer contract:** larger batches reduce stochastic noise and change steps per
token; LR transfer depends on optimizer, parameterization, warmup, sequence
length, and loss reduction. Linear or square-root scaling is a hypothesis.
Match tokens and tune around the prediction at more than one scale.

#### 7. How do you test a data-mixture change?

**Answer contract:** immutable datasets and tokenizer; fixed model, token
budget, and schedule; record actual source tokens; per-domain held-out loss and
downstream tasks; contamination check; throughput charge; multiple seeds or
paired checkpoints where noise justifies them; explicit broad-regression limit.

#### 8. Why can a valid shuffle be bad?

**Answer contract:** bijection guarantees coverage, not local independence.
Structured source order plus modular stride creates phases. Measure
autocorrelation and windowed source/length distribution; compare a stronger
stateless permutation such as a Feistel construction.

#### 9. How do you choose parallelism?

**Answer contract:** first calculate persistent and activation memory; then map
communication frequency and volume to topology. FSDP shards state, TP shards
operators, PP shards layers, CP shards sequence, EP shards experts. Include
microbatch, bubbles, collectives, checkpoint format, and failure domain.

### 17. Inference, evaluation, and post-training

#### 10. Why is decode often memory-bound?

**Answer contract:** one new token produces relatively small matrix work per
request while weights and growing KV state are read repeatedly. Batching raises
reuse but adds queueing. Discuss arithmetic intensity, KV bytes, continuous
batching, and latency SLOs.

#### 11. How do you prove an evaluation gain?

**Answer contract:** identical model/harness/serving configs except intended
change; paired examples; infrastructure-clean gate; uncertainty on paired
difference; failure slices; contamination audit; token/time budget; preserved
traces. A leaderboard delta alone is not proof.

#### 12. Derive DPO at a high level

**Answer contract:** begin with KL-regularized reward maximization, show the
optimal policy as reference times exponentiated reward, substitute log policy
ratios into a Bradley–Terry preference likelihood, and state assumptions:
offline pair distribution, reference dependence, preference consistency, and
temperature.

#### 13. PPO versus GRPO?

**Answer contract:** both are policy-gradient families using sampled
trajectories and controlled policy updates. PPO commonly learns a value
baseline; simple GRPO normalizes rewards among responses to the same prompt.
Compare estimator bias/variance, zero-variance groups, clipping, KL control,
token aggregation, rollout cost, and off-policy staleness.

#### 14. When is RLVR appropriate?

**Answer contract:** when reward can be checked reliably and cheaply enough:
tests, symbolic answers, or constraints. Audit verifier exploits, private
holdouts, ambiguous valid outputs, reward sparsity, timeouts, and distribution
coverage. SFT may be needed first for protocol competence.

#### 15. What is on-policy distillation buying?

**Answer contract:** teacher feedback covers states generated by the current
student, reducing fixed-dataset mismatch. Costs include teacher serving,
synchronization, policy staleness, tokenizer alignment, feedback granularity,
and possible teacher-error amplification. Compare with offline distillation
under a fixed total compute/cost budget.

#### 16. How do you evaluate an agent?

**Answer contract:** task success plus tool validity, recovery, trajectory
length, tokens, latency, cost, unsafe actions, and infrastructure failures.
Pin environment, tools, parser, context, retries, and model serving. Inspect
trajectories and use paired tasks; do not collapse harness failures into model
quality.

---

## Part VI — Source map and maintenance

### 18. Read these Marin artifacts in order

1. [Repository architecture](https://github.com/marin-community/marin#readme):
   identify Marin, Levanter, Iris, Zephyr, and the experiment layer.
2. [Agent MoE playbook](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/experiments/grug/moe/agent.md): learn the promotion
   gate and effective-speedup calculation.
3. [Agent MoE digest](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/docs/reports/agent-moe-experiments.md): compare positive,
   mixed, negative, unfinished, and scale-inverting results.
4. [August hero iso-FLOP issue](https://github.com/marin-community/marin/issues/8003):
   see compute-optimal planning with incomplete and divergent cells.
5. [32B retrospective](https://github.com/marin-community/marin/blob/e7c34f396f8f2780fc76bb60bcb7263900540534/docs/reports/marin-32b-retro.md): reconstruct stability,
   contamination, shuffle, and recovery decisions.
6. [Hero burndown](https://github.com/marin-community/marin/issues/8233): see
   launch readiness across research and systems.
7. [Three-stage SFT issue](https://github.com/marin-community/marin/issues/8225):
   see capability redistribution and artifact lineage.
8. [Terminus parity issue](https://github.com/marin-community/marin/issues/7930):
   separate model, harness, and serving causes.
9. [Standups](https://github.com/marin-community/marin/issues?q=repo%3Amarin-community%2Fmarin+standup):
   follow live constraints, then open the linked experiment issues for evidence.
10. [Current 535B hero campaign #8435](https://github.com/marin-community/marin/issues/8435), [handoff #8506](https://github.com/marin-community/marin/issues/8506), and [runbook](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/README.md): distinguish planned scale/context phases from measured continuation.
11. [Mixture #9126](https://github.com/marin-community/marin/issues/9126) and [SFT overlap #9212](https://github.com/marin-community/marin/issues/9212): practice compute-equivalent accounting and contamination caveats.
12. [Context-parallel PR #9119](https://github.com/marin-community/marin/pull/9119) and [H100 diagnostics #9277](https://github.com/marin-community/marin/issues/9277): distinguish finite-step throughput from long-context capability.
13. [Eval Policy #9193](https://github.com/marin-community/marin/issues/9193) and [Snowball comparison #9225](https://github.com/marin-community/marin/issues/9225): separate protocol design from capability-specific post-training outcomes.
14. [September 21 standup #9324](https://github.com/marin-community/marin/issues/9324): use it as a current index, then read linked primary artifacts.

### 19. Foundation sources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- [GQA](https://arxiv.org/abs/2305.13245)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [DoReMi](https://arxiv.org/abs/2305.10429)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [DPO](https://arxiv.org/abs/2305.18290)
- [PPO](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

The
[Scientific Spaces and SOTA reading guide](https://github.com/voe09/codingmachine/blob/main/hc/spaces-ac-llm-reading-guide.md)
adds the long-context, tokenizer, modern architecture, post-training, and
agentic branches in a recommended order.

### 20. Monthly freshness procedure

On the first study session of each month:

1. record the date and current revisions of the reading guide and Marin;
2. scan Marin's latest standup and open experiment trackers;
3. update only claims whose evidence changed;
4. move a frontier item toward current practice only after broader evidence;
5. keep negative and contradictory results;
6. attach primary sources and distinguish author-reported from independently
   reproduced results;
7. re-run the parameter, compute, memory, or estimator calculation affected by
   the change.

### Completion test

You are interview-ready when you can take an unfamiliar model report or
experiment issue and, within 45 minutes:

1. reconstruct the objective and config;
2. estimate parameters, FLOPs, memory, and KV cache;
3. identify the dominant quality–systems tradeoff;
4. find the missing control or ambiguous metric;
5. propose the next matched experiment;
6. state a decision rule and uncertainty;
7. explain what would fail at ten times the scale;
8. defend the answer when latency, budget, context, or data constraints change.

That is closer to model development than memorizing a list of architectures.
