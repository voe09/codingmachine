# #reinforcement-learning

Archive coverage: 2025-05-22–2026-09-23 (UTC); 1,852 messages, including 1,010 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1374989195109466122).

## Discussions and learnings

- RL discussion moved from algorithm trade-offs and vLLM rollout integration to stable Iris/Fray training and post-training Snowball models. A key systems result was that replacing a failing Ray execution path with on-demand workers allowed a 500-step stability test. [Source](https://marin-discord.pages.dev/#1374989195109466122/1486914443060449320).
- Reported Snowball RL tests improved math benchmarks, and a later stack comparison raised held-out r2egym pass@1 from 50.1% to 59.8%; transfer to other agent benchmarks was not uniform. [Source](https://marin-discord.pages.dev/#1374989195109466122/1549558259713703986).
- Before Iris/Fray stabilization, vLLM was integrated into the RL rollout pipeline with region-specific images so inference jobs did not reinstall it at startup. [Source](https://marin-discord.pages.dev/#1374989195109466122/1430965818241454254).
- A Snowball RL report measured AIME24 at 17.67%→27%, MATH-500 at 64%→78%, and OlympiadBench at 12.67%→20% for one experiment. The report scoped these to its checkpoints and setup; they are not general RL gains. [Source](https://marin-discord.pages.dev/#1374989195109466122/1542681739497832499).

## Month-by-month source trail

These are selected entry points into the archived conversation, not a list of every message. The discussion summary above distinguishes reported results from proposals and unresolved issues.

- 2025-05: 14 messages; [2025-05-29](https://marin-discord.pages.dev/#1374989195109466122/1377493531018395770), [2025-05-29](https://marin-discord.pages.dev/#1374989195109466122/1377685855514394714).
- 2025-06: 58 messages; [2025-06-20](https://marin-discord.pages.dev/#1374989195109466122/1385442628694183996), [2025-06-29](https://marin-discord.pages.dev/#1374989195109466122/1388765139213484033).
- 2025-07: 87 messages; [2025-07-03](https://marin-discord.pages.dev/#1390036988975120535/1390202913888145499), [2025-07-01](https://marin-discord.pages.dev/#1374989195109466122/1389427590376325131).
- 2025-08: 7 messages; [2025-08-14](https://marin-discord.pages.dev/#1405688723827331264/1405689463878975549), [2025-08-14](https://marin-discord.pages.dev/#1405688723827331264/1405693369006424167).
- 2025-09: 97 messages; [2025-09-30](https://marin-discord.pages.dev/#1374989195109466122/1422534500931469353), [2025-09-30](https://marin-discord.pages.dev/#1374989195109466122/1422709309023850612).
- 2025-10: 437 messages; [2025-10-23](https://marin-discord.pages.dev/#1374989195109466122/1430965818241454254), [2025-10-14](https://marin-discord.pages.dev/#1374989195109466122/1427792970547204126).
- 2025-11: 169 messages; [2025-11-16](https://marin-discord.pages.dev/#1374989195109466122/1439700923654864998), [2025-11-10](https://marin-discord.pages.dev/#1374989195109466122/1437505134841036831).
- 2025-12: 39 messages; [2025-12-16](https://marin-discord.pages.dev/#1374989195109466122/1450368585796030544), [2025-12-03](https://marin-discord.pages.dev/#1374989195109466122/1445586750264311929).
- 2026-01: 131 messages; [2026-01-25](https://marin-discord.pages.dev/#1374989195109466122/1464917052182630596), [2026-01-24](https://marin-discord.pages.dev/#1374989195109466122/1464539202359136492).
- 2026-02: 9 messages; [2026-02-24](https://marin-discord.pages.dev/#1457901307103936670/1475942171159957597), [2026-02-09](https://marin-discord.pages.dev/#1374989195109466122/1470299955145998531).
- 2026-03: 31 messages; [2026-03-27](https://marin-discord.pages.dev/#1374989195109466122/1486914443060449320), [2026-03-31](https://marin-discord.pages.dev/#1374989195109466122/1488591394116468746).
- 2026-04: 8 messages; [2026-04-13](https://marin-discord.pages.dev/#1374989195109466122/1493297286338318406), [2026-04-29](https://marin-discord.pages.dev/#1374989195109466122/1498842492110373067).
- 2026-05: 4 messages; [2026-05-05](https://marin-discord.pages.dev/#1374989195109466122/1501359531660152944), [2026-05-06](https://marin-discord.pages.dev/#1374989195109466122/1501439083585601551).
- 2026-06: 222 messages; [2026-06-22](https://marin-discord.pages.dev/#1518700249902743773/1518725530751602758), [2026-06-09](https://marin-discord.pages.dev/#1513172211991838901/1513838512670249102).
- 2026-07: 116 messages; [2026-07-22](https://marin-discord.pages.dev/#1374989195109466122/1529589529852248275), [2026-07-28](https://marin-discord.pages.dev/#1374989195109466122/1531644649230438591).
- 2026-08: 116 messages; [2026-08-27](https://marin-discord.pages.dev/#1374989195109466122/1542681739497832499), [2026-08-31](https://marin-discord.pages.dev/#1374989195109466122/1544006720496599131).
- 2026-09: 307 messages; [2026-09-17](https://marin-discord.pages.dev/#1374989195109466122/1550202720433086504), [2026-09-09](https://marin-discord.pages.dev/#1374989195109466122/1547120190758584320).
