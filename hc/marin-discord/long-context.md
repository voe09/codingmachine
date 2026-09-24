# #long-context

Archive coverage: 2025-06-08–2026-09-22 (UTC); 293 messages, including 72 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1380235124011958313).

## Discussion and lessons

- The immediate long-context target was agentic work over repositories, not generic long-document performance. The plan was to benchmark the existing 8B instruct model, then test SWE-Smith fine-tuning before trying more exotic positional changes. [Source](https://marin-discord.pages.dev/#1380235124011958313/1394527166674243606).
- An early SWE-Smith fine-tune produced 14 resolved issues out of 500 on SWE-Bench Verified, establishing a nonzero agentic baseline. This was not evidence that context extension alone caused the improvement. [Discussion](https://marin-discord.pages.dev/#1380235124011958313/1396039727081324644).
- The repository-concatenation hypothesis was that code understanding requires multiple files together because individual files often fit within short contexts. Random within-repo concatenation was the baseline; topology-aware ordering was a next experiment. [Discussion](https://marin-discord.pages.dev/#1397996868503605308/1398070439162413098).
