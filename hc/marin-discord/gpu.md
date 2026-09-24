# #gpu

Archive coverage: 2025-07-30–2026-09-18 (UTC); 119 messages, including 64 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1399998407657001062).

## Discussion and lessons

- Muon orthogonalization wants complete matrices local to each device, making an expert-stacked sharding layout attractive. On one two-node run it consumed roughly 500 ms of a 1,500 ms step; the team temporarily dropped it when a 10% step-count gain did not repay that cost. [Discussion](https://marin-discord.pages.dev/#1399998407657001062/1518855948100177991).
- A later PGLE flag helped a 360B-A23B setup reach roughly 25 MFU even with Muon and several architectural features. Hardware results changed with kernels and layout, so an earlier wall-clock verdict did not settle the optimizer question forever. [Discussion](https://marin-discord.pages.dev/#1399998407657001062/1530411117279711383).
