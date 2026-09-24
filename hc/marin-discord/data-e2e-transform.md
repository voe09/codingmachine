# #data-e2e-transform

Archive coverage: 2026-01-19–2026-07-18 (UTC); 77 messages, including 2 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1462896888000024711).

## Discussion and lessons

- The project used a rephraser to turn domain text into training data. Preliminary midtraining tests favored a Nemotron cooldown over mixing in DCLM or rephraser data, while math-forum question/reasoning/answer extraction improved GSM8K and MATH in a separate SFT test. [Source](https://marin-discord.pages.dev/#1462896888000024711/1478952804830679142).
- The rephraser did not beat a Nemotron cooldown in preliminary continued-pretraining comparisons, while extracting question/reasoning/answer tuples from a math forum improved GSM8K and MATH in a separate SFT test. The same transform can have different value by domain and training phase. [Discussion](https://marin-discord.pages.dev/#1462896888000024711/1481525022534139996).
