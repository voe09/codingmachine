# #general

Archive coverage: 2025-04-01–2026-09-18 (UTC); 1,247 messages, including 421 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1354881461060243561).

## Discussion and lessons

- Marin's 2025 path ran from an 8B base model through a 32B base release, with open experiment records and later post-training still needed. The project-wide discussion treated a strong base checkpoint as a platform for data, evaluation, and RL work rather than a finished assistant. [Retrospective](https://marin-discord.pages.dev/#1354881461060243561/1457847114561229045); [32B release discussion](https://marin-discord.pages.dev/#1354881461060243561/1433121298049138782).
- The 2026 research agenda joined four dependencies: larger MoEs/hybrid architectures, data mixing and synthesis, RL environments, and reliable heterogeneous compute. Progress in one area alone would not produce a competitive open model. [Discussion](https://marin-discord.pages.dev/#1354881461060243561/1457847162783273202).
- A data-constrained pretraining paper shared in the channel argued that simply adding epochs or parameters can eventually overfit when the available web text is fixed. This helps explain the later emphasis on synthetic data, rewriting, and careful mixture schedules. [Discussion](https://marin-discord.pages.dev/#1354881461060243561/1418674334955733122).
- The finished 1e23-FLOP MoE had 130B total and 16B active parameters, trained on 1T tokens, and reached Paloma macro loss 2.234 against a preregistered 2.252 forecast from much smaller runs. The team reported roughly sixfold FLOP efficiency versus its dense Delphi formula; this is a within-project comparison. [Discussion](https://marin-discord.pages.dev/#1354881461060243561/1506713859824816269).
