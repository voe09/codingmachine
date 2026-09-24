# #automate-research

Archive coverage: 2026-01-19–2026-09-01 (UTC); 236 messages, including 46 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1462884917292699669).

## Discussion and lessons

- The proposed research workflow uses natural-language GitHub issues and coding agents to implement ideas, run experiments, and report evidence. A recurring goal was making framework-to-Grug ports and routine dataset onboarding reproducible rather than ad hoc. [Source](https://marin-discord.pages.dev/#1462884917292699669/1462893070809960459).
- The discussion also exposed a verification lesson: an agent's apparent RunAI loading slowdown included XLA compilation; checking the underlying logs reversed that conclusion. [Source](https://marin-discord.pages.dev/#1462884917292699669/1486094524139835432).
- A proposed first agent task was porting PyTorch or NanoGPT ideas into Grug/JAX with a reproducible test, rather than letting an agent merely describe a paper. One apparent model-loading regression later vanished when logs showed that XLA compilation had been included in the timing. [Discussion](https://marin-discord.pages.dev/#1462884917292699669/1464705505690783754).
