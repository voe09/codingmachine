# #hero-run-2026

Archive coverage: 2026-08-21–2026-09-18 (UTC); 161 messages, including 98 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1540486417078296576).

## Discussions and learnings

- This channel tracked an ongoing production-scale run, with checkpointed changes to attention-gate/router weight decay, expert parallelism, PJRT/NCCL wheels, and the data mix. Throughput claims were tied to specific redeployments rather than a stable final recipe. [Source](https://marin-discord.pages.dev/#1540486417078296576/1550174441433206785).
- A ragged expert-parallel backend was expected to reduce token-assignment drops from about 3% to 0.01%; subsequent messages monitored quality and hangs, so this should be read as run history, not a closed benchmark. [Source](https://marin-discord.pages.dev/#1540486417078296576/1544864246897315921).

## Month-by-month source trail

These are selected entry points into the archived conversation, not a list of every message. The discussion summary above distinguishes reported results from proposals and unresolved issues.

- 2026-08: 24 messages; [2026-08-24](https://marin-discord.pages.dev/#1540486417078296576/1541483657041158164), [2026-08-23](https://marin-discord.pages.dev/#1540486417078296576/1541185526269354214).
- 2026-09: 137 messages; [2026-09-01](https://marin-discord.pages.dev/#1540486417078296576/1544415299191709716), [2026-09-18](https://marin-discord.pages.dev/#1540486417078296576/1550304318769340488).
