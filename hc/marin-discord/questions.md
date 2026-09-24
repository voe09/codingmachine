# #questions

Archive coverage: 2025-04-02–2026-09-22 (UTC); 292 messages, including 143 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1357080963472949428).

## Discussion and lessons

- The help channel mixed contributor project ideas with technical questions about pretraining metrics, architecture comparisons, evaluation flags, and JAX version compatibility. Answers were often specific to a branch or dependency version. [Source](https://marin-discord.pages.dev/#1357080963472949428/1407497543444594719).
- One JAX incompatibility discussion used an XLA cross-entropy path as a temporary diagnostic workaround; it should not be treated as a permanent supported configuration. [Source](https://marin-discord.pages.dev/#1476357075948011581/1477895277338820753).
- A discussion of pretraining observability proposed tracking parameter, activation, gradient, and attention-logit statistics, not just loss. These measurements help localize instability when the data mix, architecture, and optimizer change together. [Discussion](https://marin-discord.pages.dev/#1357080963472949428/1407497543444594719).
