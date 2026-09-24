# #moe

Archive coverage: 2025-04-29–2026-09-19 (UTC); 1,734 messages, including 559 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1365044508546568372).

## Discussions and learnings

- MoE work established iterative iso-FLOP baselines, routing and initialization ablations, load balancing, learning-rate schedule comparisons, and large-run scaling. The stated process was to promote a change only after it beats the current baseline under comparable conditions. [Source](https://marin-discord.pages.dev/#1365044508546568372/1481488141012697190).
- At 3e18 FLOPs, a linear-decay schedule lowered reported BPB from 1.047 to 1.016. QB load balancing then allowed expert padding to fall from 0.25x to zero with a reported 10–15% speedup and negligible BPB difference. These are configuration-specific results. [Schedule](https://marin-discord.pages.dev/#1365044508546568372/1485898126454820907); [load balancing](https://marin-discord.pages.dev/#1365044508546568372/1487197835580412055).
- MuonH showed roughly 20–30% step-count improvement versus AdamH in a reported sweep, but hardware efficiency and scale remained separate questions. [Source](https://marin-discord.pages.dev/#1365044508546568372/1502824423936491550).
- Router vectors in one large run became strongly aligned, motivating investigation of whether this helped or hurt specialization. A later d768 ladder test showed router-metric excursions with the responsible change not yet isolated. [Router analysis](https://marin-discord.pages.dev/#1365044508546568372/1489131651920564244); [open investigation](https://marin-discord.pages.dev/#1365044508546568372/1550547629958504620).

## Month-by-month source trail

These are selected entry points into the archived conversation, not a list of every message. The discussion summary above distinguishes reported results from proposals and unresolved issues.

- 2025-04: 19 messages; [2025-04-30](https://marin-discord.pages.dev/#1365044508546568372/1367289246196568215), [2025-04-29](https://marin-discord.pages.dev/#1365044508546568372/1366843605913829376).
- 2025-05: 56 messages; [2025-05-08](https://marin-discord.pages.dev/#1365044508546568372/1369826734647803944), [2025-05-09](https://marin-discord.pages.dev/#1365044508546568372/1370246089533423647).
- 2025-06: 7 messages; [2025-06-30](https://marin-discord.pages.dev/#1365044508546568372/1389087430812110951), [2025-06-04](https://marin-discord.pages.dev/#1365044508546568372/1379877103503147052).
- 2025-07: 21 messages; [2025-07-26](https://marin-discord.pages.dev/#1365044508546568372/1398504982000701480), [2025-07-25](https://marin-discord.pages.dev/#1365044508546568372/1398344985782517962).
- 2025-08: 27 messages; [2025-08-27](https://marin-discord.pages.dev/#1365044508546568372/1410155862533603339), [2025-08-28](https://marin-discord.pages.dev/#1365044508546568372/1410515320250306601).
- 2025-09: 8 messages; [2025-09-11](https://marin-discord.pages.dev/#1365044508546568372/1415499293191831713), [2025-09-24](https://marin-discord.pages.dev/#1365044508546568372/1420508884597997589).
- 2025-10: 86 messages; [2025-10-29](https://marin-discord.pages.dev/#1365044508546568372/1432964417078693928), [2025-10-12](https://marin-discord.pages.dev/#1365044508546568372/1426774363365179492).
- 2025-11: 50 messages; [2025-11-30](https://marin-discord.pages.dev/#1365044508546568372/1444807598871150754), [2025-11-11](https://marin-discord.pages.dev/#1365044508546568372/1437887227098169396).
- 2025-12: 8 messages; [2025-12-15](https://marin-discord.pages.dev/#1365044508546568372/1450227018938449930), [2025-12-19](https://marin-discord.pages.dev/#1365044508546568372/1451450439706083358).
- 2026-01: 238 messages; [2026-01-31](https://marin-discord.pages.dev/#1365044508546568372/1467100228162162801), [2026-01-27](https://marin-discord.pages.dev/#1365044508546568372/1465504028917108822).
- 2026-02: 185 messages; [2026-02-09](https://marin-discord.pages.dev/#1365044508546568372/1470278088041562346), [2026-02-27](https://marin-discord.pages.dev/#1365044508546568372/1477023596760141929).
- 2026-03: 258 messages; [2026-03-19](https://marin-discord.pages.dev/#1365044508546568372/1484269845758349412), [2026-03-02](https://marin-discord.pages.dev/#1365044508546568372/1477949253299994706).
- 2026-04: 255 messages; [2026-04-02](https://marin-discord.pages.dev/#1365044508546568372/1489356958967398480), [2026-04-02](https://marin-discord.pages.dev/#1365044508546568372/1489131651920564244).
- 2026-05: 243 messages; [2026-05-10](https://marin-discord.pages.dev/#1365044508546568372/1502824423936491550), [2026-05-09](https://marin-discord.pages.dev/#1365044508546568372/1502491222105915472).
- 2026-06: 152 messages; [2026-06-02](https://marin-discord.pages.dev/#1365044508546568372/1511227646715887626), [2026-06-29](https://marin-discord.pages.dev/#1365044508546568372/1521008406184460288).
- 2026-07: 39 messages; [2026-07-01](https://marin-discord.pages.dev/#1365044508546568372/1521726976120066048), [2026-07-01](https://marin-discord.pages.dev/#1365044508546568372/1521988804641030164).
- 2026-08: 66 messages; [2026-08-22](https://marin-discord.pages.dev/#1540202379301879838/1540568271361679420), [2026-08-13](https://marin-discord.pages.dev/#1365044508546568372/1537559415052439692).
- 2026-09: 16 messages; [2026-09-18](https://marin-discord.pages.dev/#1365044508546568372/1550547629958504620), [2026-09-18](https://marin-discord.pages.dev/#1550547629958504620/1550559747042840706).
