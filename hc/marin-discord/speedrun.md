# #speedrun

Archive coverage: 2025-04-28–2026-05-15 (UTC); 656 messages, including 54 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1364826908994568282).

## Discussion and lessons

- Speedrun served as a lower-cost gate for training ideas and a contributor entry point. Discussion ranged from memory-saving activations and optimizer changes to leaderboard participation and how small a model can still provide useful evidence. [Source](https://marin-discord.pages.dev/#1364826908994568282/1466159874546077787).
- Participants cautioned that even FLOP counting and scaling-law fitting need a reproducible method before small-run wins are promoted to larger experiments. [Source](https://marin-discord.pages.dev/#1364826908994568282/1480569693805744219).
- Early GPU trials quantified entry cost: 50M and 75M parameter models ran 1,000 steps in about 900 and 1,330 seconds respectively on their stated A100 setups. Memory limits constrained batch size before theoretical FLOP budgets did. [Discussion](https://marin-discord.pages.dev/#1364826908994568282/1372119403734237254).
- The leaderboard's hardware-FLOP measure included optimizer work and actual train-step time; it was not identical to a theoretical model-FLOP count. Comparing methods at several scales helps avoid promoting a one-off small-run win. [Discussion](https://marin-discord.pages.dev/#1364826908994568282/1414997323620159628).
