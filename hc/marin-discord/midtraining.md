# #midtraining

Archive coverage: 2026-03-17–2026-09-11 (UTC); 328 messages, including 183 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1483266366772351067).

## Discussion and lessons

- The channel noted that stale or unexplained evaluations make midtraining conclusions hard to compare; deduplication and up-to-date experiment records were part of the lesson. [Source](https://marin-discord.pages.dev/#1483266366772351067/1521407119423836262).
- Nemotron CC-math continued-pretraining sweeps varied learning rate, warmup, data ratio, and token budget on a Delphi model ladder. One run accidentally used a 3.4% rather than 10% warmup, showing why experiment metadata matters to comparisons. [Discussion](https://marin-discord.pages.dev/#1483266366772351067/1507658133844263036).
- A 10-gram contamination check found GSM8K overlap of 2.43% test and 1.95% train, but roughly 28% of MATH-500 examples overlapped with documents containing both question and solution. Math gains on that benchmark need this caveat. [Discussion](https://marin-discord.pages.dev/#1483266366772351067/1523154815599902791).
- The Snowball 67B-A2B cooldown became an end-to-end post-training base: SFT, chat templates, serving, and evaluations ran, including a 700,000-example SFT in under an hour. This is an operational milestone rather than a downstream-quality verdict. [Discussion](https://marin-discord.pages.dev/#1483266366772351067/1527726088799649953).
