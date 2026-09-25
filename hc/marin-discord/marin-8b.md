# #marin-8b

Archive coverage: 2025-04-24–2025-09-03 (UTC); 309 messages, including 59 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1355318614802042950).

## Training and evaluation trajectory

The first 8B model had seen about 5T tokens and was reported roughly 0.3 MMLU points below Llama 3.1 8B under Marin's evaluation, while its SFT variant lagged by about 1.5 on AlpacaEval under a matched recipe. A continuation on another roughly 5T tokens showed a 2.3-point MMLU gain before cooldown. By early May the group reported its cooled-down base slightly ahead of Llama 3.1 8B on MMLU. These are successive checkpoints and protocol-specific comparisons, not one contradictory claim about a fixed model. [Initial status](https://marin-discord.pages.dev/#1355318614802042950/1365063963427082322) · [Later base result](https://marin-discord.pages.dev/#1355318614802042950/1369391292500672583).

Post-training revealed a different bottleneck. Participants expected AlpacaEval to saturate without preference optimization; improved knowledge alone might not produce more preference-test wins on a small set. The speedrun projection target was later moved toward 1e22 FLOPs, nearer compute-optimal for 8B, instead of extrapolating directly to Marin's much more overtrained 6.5e23-FLOP 8B run. The lesson is to separate base-model knowledge, instruction behavior, and compute-optimal scaling when comparing recipes. [Preference-eval caveat](https://marin-discord.pages.dev/#1355318614802042950/1368019067310444598) · [Projection target](https://marin-discord.pages.dev/#1355318614802042950/1392292231746027654).
