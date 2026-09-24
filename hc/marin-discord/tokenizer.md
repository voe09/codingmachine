# #tokenizer

Archive coverage: 2026-05-04–2026-09-18 (UTC); 127 messages, including 21 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1500987824206254120).

## Discussion and lessons

- Tokenizer experiments compared vocabulary sizes, numeric-token handling, compression, compatibility with other models, and evaluation diagnostics. A prior numeric-token gain required rerunning because the wrong model size had been supplied. [Source](https://marin-discord.pages.dev/#1500987824206254120/1525252850408620142).
- A purported numeric-token improvement was invalidated when the wrong model size was used. Repeating the run with model configuration checked was necessary before claiming a tokenizer gain. [Discussion](https://marin-discord.pages.dev/#1500987824206254120/1525252850408620142).
- The BPE vocabulary sweep on the hero mixture showed limited compression gains beyond 16k tokens. That result should be judged against downstream capability and serving cost, rather than declaring 16k universally optimal. [Discussion](https://marin-discord.pages.dev/#1500987824206254120/1550036882480566272).
