# #sft

Archive coverage: 2025-05-16–2026-09-23 (UTC); 512 messages, including 228 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1355318637854199848).

## Discussion and lessons

- A Qwen2.5-7B-Instruct comparison showed noisy AIME trajectories, and an initial mixed dataset did not clearly beat OpenThoughts3 alone. Researchers checked the evaluation pipeline before drawing conclusions from end-of-run accuracy. [Discussion](https://marin-discord.pages.dev/#1355318637854199848/1440255448774348880).
- The initial long-context Marin 8B versus Qwen fine-tune scored 20.0% versus 53.3% on AIME24 and 13.3% versus 53.3% on AIME25. The hyperparameters had been tuned for Qwen, so this was a baseline comparison, not a controlled upper bound for Marin. [Discussion](https://marin-discord.pages.dev/#1355318637854199848/1458203611543900190).
- Reevaluating intermediate checkpoints with more trials showed Marin was more competitive earlier in training and then declined. The lesson is to inspect accuracy over training time, not only the final checkpoint; learning-rate schedule and overtraining remained hypotheses. [Discussion](https://marin-discord.pages.dev/#1355318637854199848/1465858152133169182).
- Chat-template application was a concrete prerequisite: missing or mismatched template tokens changed whether thinking delimiters appeared at all. Format errors can masquerade as failures of reasoning data. [Discussion](https://marin-discord.pages.dev/#1405655976832532560/1405993542245154938).
