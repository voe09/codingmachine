# #infra

Archive coverage: 2025-04-30–2026-08-25 (UTC); 7,369 messages, including 2,445 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1364827114670657616).

## Discussions and learnings

- Infrastructure discussion spanned Ray and Iris scheduling, TPU preemption, cluster/controller incidents, storage costs, observability, and migration of jobs from Ray commands to Iris. The high message volume includes many transient operations reports. [Source](https://marin-discord.pages.dev/#1364827114670657616/1476412061339619469).
- Recurring lessons were to make crashes visible without relying only on W&B, make recovery and retention deliberate, and verify evaluation code after integration changes; an Evalchemy update reported roughly 10x speedup plus bug fixes. [Source](https://marin-discord.pages.dev/#1364827114670657616/1471664293479972886).
- A Ray token-authentication migration required a planned cluster reboot and explicit checkpoint/stop/resubmit instructions for running jobs. [Source](https://marin-discord.pages.dev/#1364827114670657616/1456402718481711337).
- Checkpoint storage was estimated at roughly $60k per month. Retention proposals prompted concern that one-month deletion could remove baselines needed for three- to six-month research projects; warning and recovery policy were still under discussion. [Cost estimate](https://marin-discord.pages.dev/#1364827114670657616/1490767485547319388); [research concern](https://marin-discord.pages.dev/#1364827114670657616/1490771304188481657).

## Month-by-month source trail

These are selected entry points into the archived conversation, not a list of every message. The discussion summary above distinguishes reported results from proposals and unresolved issues.

- 2025-04: 56 messages; [2025-04-30](https://marin-discord.pages.dev/#1364827114670657616/1367262649653661757), [2025-04-30](https://marin-discord.pages.dev/#1364827114670657616/1367200372283801711).
- 2025-05: 463 messages; [2025-05-14](https://marin-discord.pages.dev/#1364827114670657616/1372317550075187312), [2025-05-10](https://marin-discord.pages.dev/#1364827114670657616/1370904887558275173).
- 2025-06: 279 messages; [2025-06-09](https://marin-discord.pages.dev/#1364827114670657616/1381674313345929386), [2025-06-27](https://marin-discord.pages.dev/#1364827114670657616/1388226591179669685).
- 2025-07: 517 messages; [2025-07-20](https://marin-discord.pages.dev/#1364827114670657616/1396310453981810759), [2025-07-03](https://marin-discord.pages.dev/#1364827114670657616/1390249092298313779).
- 2025-08: 416 messages; [2025-08-25](https://marin-discord.pages.dev/#1364827114670657616/1409642552960880780), [2025-08-07](https://marin-discord.pages.dev/#1364827114670657616/1403127593732931706).
- 2025-09: 1,114 messages; [2025-09-12](https://marin-discord.pages.dev/#1364827114670657616/1415856696152035492), [2025-09-26](https://marin-discord.pages.dev/#1364827114670657616/1421180132831264908).
- 2025-10: 698 messages; [2025-10-07](https://marin-discord.pages.dev/#1364827114670657616/1424936448221778054), [2025-10-01](https://marin-discord.pages.dev/#1422607680253464616/1423022712850350231).
- 2025-11: 357 messages; [2025-11-03](https://marin-discord.pages.dev/#1364827114670657616/1434754936687825007), [2025-11-05](https://marin-discord.pages.dev/#1364827114670657616/1435754744596529163).
- 2025-12: 303 messages; [2025-12-23](https://marin-discord.pages.dev/#1364827114670657616/1452938027356917760), [2025-12-13](https://marin-discord.pages.dev/#1364827114670657616/1449224391106756672).
- 2026-01: 607 messages; [2026-01-01](https://marin-discord.pages.dev/#1364827114670657616/1456402718481711337), [2026-01-24](https://marin-discord.pages.dev/#1464369718457798819/1464445011503611957).
- 2026-02: 478 messages; [2026-02-13](https://marin-discord.pages.dev/#1364827114670657616/1471664293479972886), [2026-02-26](https://marin-discord.pages.dev/#1364827114670657616/1476412061339619469).
- 2026-03: 339 messages; [2026-03-19](https://marin-discord.pages.dev/#1364827114670657616/1484211957421703310), [2026-03-20](https://marin-discord.pages.dev/#1364827114670657616/1484638373128699986).
- 2026-04: 787 messages; [2026-04-14](https://marin-discord.pages.dev/#1364827114670657616/1493692493374554293), [2026-04-07](https://marin-discord.pages.dev/#1364827114670657616/1491210597172252762).
- 2026-05: 466 messages; [2026-05-21](https://marin-discord.pages.dev/#1364827114670657616/1507158614127808693), [2026-05-28](https://marin-discord.pages.dev/#1509597879214543031/1509599190005715026).
- 2026-06: 303 messages; [2026-06-22](https://marin-discord.pages.dev/#1364827114670657616/1518613966849445948), [2026-06-03](https://marin-discord.pages.dev/#1364827114670657616/1511845749430816879).
- 2026-07: 183 messages; [2026-07-06](https://marin-discord.pages.dev/#1364827114670657616/1523698536921825310), [2026-07-10](https://marin-discord.pages.dev/#1364827114670657616/1525035072414023750).
- 2026-08: 3 messages; [2026-08-25](https://marin-discord.pages.dev/#1364827114670657616/1541617452386095185), [2026-08-23](https://marin-discord.pages.dev/#1364827114670657616/1540875073102290974).
