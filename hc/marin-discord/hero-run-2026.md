# #hero-run-2026

Archive coverage: 2026-08-21–2026-09-18 (UTC); 161 messages, including 98 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1540486417078296576).

## Discussion and lessons

- This channel tracked an ongoing production-scale run, with checkpointed changes to attention-gate/router weight decay, expert parallelism, PJRT/NCCL wheels, and the data mix. Throughput claims were tied to specific redeployments rather than a stable final recipe. [Source](https://marin-discord.pages.dev/#1540486417078296576/1550174441433206785).
- When moving from a 24-layer scaling ladder to a 48-layer hero model, early-layer outputs were much smaller than late-layer outputs despite similar gradients. Attention-gate/router decay was introduced mid-run as a monitored intervention, not a proven explanation of the discrepancy. [Discussion](https://marin-discord.pages.dev/#1544415299191709716/1544731188898439359).
- The ragged expert-parallel redeployment targeted about 10% higher throughput and a reduction in dropped token assignments from roughly 3% to 0.01%. These were expectations at deployment time; later quality and hang monitoring remained necessary. [Discussion](https://marin-discord.pages.dev/#1540486417078296576/1544864246897315921).
