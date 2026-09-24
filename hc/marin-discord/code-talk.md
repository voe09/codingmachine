# #code-talk

Archive coverage: 2025-04-29–2026-08-24 (UTC); 1,474 messages, including 400 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1366632114316906506).

## Discussion and lessons

- The Levanter-to-Marin monorepo migration established a uv workspace, making experiments, training code, and library packaging part of one build. Subsequent discussion asked which experiment defaults were reusable library APIs versus project-specific code. [Discussion](https://marin-discord.pages.dev/#1366632114316906506/1435085750231240714).
- A vLLM checkpoint-identity bug interpreted Llama 3 as Mistral and silently masked attention to tokens earlier than a 4,096-token sliding window. This was a correctness defect, not merely a serving optimization issue. [Discussion](https://marin-discord.pages.dev/#1366632114316906506/1489495767130443857).
