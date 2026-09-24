# #levanter

Archive coverage: 2025-04-30–2026-09-09 (UTC); 825 messages, including 358 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1366985639743979541).

## Discussion and lessons

- Levanter discussion addressed JAX/TPU training usability, migration into the Marin monorepo, model training on multiple slices, and difficult numerical or dependency bugs. [Source](https://marin-discord.pages.dev/#1366985639743979541/1406768389342363668).
- A fused softmax loss became the default despite being slower than a reference in one measurement because it greatly reduced memory at large vocabulary sizes. Throughput and memory headroom traded off. [Discussion](https://marin-discord.pages.dev/#1366985639743979541/1467710224935817389).
- For a LoRA divergence on TPU, the debugging matrix separated dataset, gradient accumulation, LoRA versus full fine-tuning, fused cross-entropy, and topology. Cross-topology comparisons were used to narrow the cause rather than masking the divergence. [Discussion](https://marin-discord.pages.dev/#1366985639743979541/1494464722299519136).
