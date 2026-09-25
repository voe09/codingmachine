# #train-test-overlap

Archive coverage: 2025-05-07–2025-08-03 (UTC); 122 messages, including 28 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1369724540590227596).

## Detecting contamination without overinterpreting a match

The initial audit reported that all GSM8K questions were found in a Dolmino source. That is an alarming *overlap* result, but the question alone does not reveal whether the matching document included the answer, whether that document entered a particular model's training stream, or how much an evaluation score was affected. The channel linked per-example n-gram match records back to input paths so researchers could inspect the actual matched contexts, rather than treating a binary hit as the final verdict. [GSM8K finding](https://marin-discord.pages.dev/#1369724540590227596/1373561759704678420) · [Per-example trace](https://marin-discord.pages.dev/#1369724540590227596/1373564519825150022).

The pipeline also had an operational dimension: decontamination over large shards could read from GCS directly or through `gcsfuse`, with cost depending on region and data movement. A later estimate put `gcsfuse` at about four times the local-storage cost but only around five cents for one DCLM global shard under the stated conditions. Those figures are specific to the 2025 deployment and should not be reused as current cloud pricing. [Cost analysis](https://marin-discord.pages.dev/#1369724540590227596/1401671179395268658).
