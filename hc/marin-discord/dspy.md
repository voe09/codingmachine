# #dspy

Archive coverage: 2025-11-07–2026-07-06 (UTC); 116 messages, including 21 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1436412621040648222).

## Discussion and lessons

- The channel scoped evaluation and training for DSPy-style programmatic language-model use, including LangProBe and output-format robustness. It distinguished simple pretraining/generation tasks from interactive post-training environments. [Source](https://marin-discord.pages.dev/#1436412621040648222/1439286738714628156).
- A central open question was whether fine-tuning can make task performance invariant to required formats such as TOON, rather than merely teaching one format. [Source](https://marin-discord.pages.dev/#1436412621040648222/1463298066839900304).
- Format is not neutral: the channel questioned whether TOON or CSV separates related fields too far for robust use compared with JSON. The proposed research test was whether SFT could preserve task accuracy across output formats, not just teach one serialization. [Discussion](https://marin-discord.pages.dev/#1436412621040648222/1437867376405250161).

One small-scale follow-up used DSPy's GEPA prompt optimization as a bridge into RL. A standard prompt gave the largest Delphi model too little initial reward for RL to learn; after GEPA found a better prompt, the same style of RL run showed a learning signal. This was an exploratory observation, not a controlled proof that GEPA is generally required. It illustrates that prompt/program optimization can change the initial reward distribution enough to make a later training experiment feasible. [GEPA-to-RL observation](https://marin-discord.pages.dev/#1436412621040648222/1520172887133257779).
