# #data-mixing

Archive coverage: 2026-01-20–2026-09-15 (UTC); 706 messages, including 259 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1462895580064911522).

## Discussion and lessons

- Mixing work aimed to optimize proportions across multiple training phases while accounting for epoching and quality upsampling. Researchers compared proportional, UniMax, and Olmix-like mixtures and debated phase interactions and diminishing returns. [Source](https://marin-discord.pages.dev/#1462895580064911522/1463172424257245349).
- A later operational requirement was publishing enough dataset metadata for others to reconstruct a mix from licensed original providers without rehosting the documents. [Source](https://marin-discord.pages.dev/#1462895580064911522/1542677539799961770).
- For mixture sweeps, multiple-choice evaluation by letter log probability was noisier than a verbalized-choice variant; the changed metric reduced reported variance. Evaluation noise can swamp small mixture effects. [Source](https://marin-discord.pages.dev/#1462895580064911522/1486640925328408669).
- The technical objection to a purely greedy online mixer was temporal: high-quality data may be most valuable late in training, so a policy that spends it early can miss phase effects and diminishing returns. [Discussion](https://marin-discord.pages.dev/#1462895580064911522/1476399039917457659).
- In a StarCoder comparison, a two-phase learned schedule reported loss 0.910 versus 1.044 for proportional mixing. In the three-phase test, the learned schedule reached 0.870 versus 0.903 for phase-local UniMax and 1.013 for proportional mixing. The comparison is specific to that data domain and evaluation. [Discussion](https://marin-discord.pages.dev/#1462895580064911522/1480799818430545982).
- The group opposed selecting only data that correlates with downstream benchmarks: such correlation is evidence to include a domain, but exclusive selection risks narrowing general capability and overfitting the evaluation set. [Discussion](https://marin-discord.pages.dev/#1462895580064911522/1498811239927906405).
- Small-model mixture studies picked tasks with enough signal and avoided floor-prone frontier benchmarks. A low score on an impossible task cannot distinguish two candidate mixes. [Discussion](https://marin-discord.pages.dev/#1462895580064911522/1499201070063419432).
