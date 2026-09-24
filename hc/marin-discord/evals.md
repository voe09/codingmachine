# #evals

Archive coverage: 2025-04-30–2026-09-18 (UTC); 1,319 messages, including 410 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1356487738840318002).

## Discussion and lessons

- Evaluation discussion pushed beyond familiar multiple-choice benchmarks toward fresh, diverse perplexity tests that better capture broad modeling capability and resist overfitting. The proposed 'uncheatable' measure showed encouraging correlations, with caveats about the fitted model set. [Source](https://marin-discord.pages.dev/#1356487738840318002/1418685458988269569).
- The group worked toward a common stack: lm-evaluation-harness for pretraining/log-probability and generation tasks, with environment-based evaluation for post-training. Protocol versioning and pinned versus current-stack configurations remained important for comparability. [Source](https://marin-discord.pages.dev/#1356487738840318002/1461766966040596644).
- The motivating problem was that multiple-choice accuracy entangles raw modeling ability with domain exposure and answer formatting. The proposed fresh-text perplexity suite used recent arXiv, code, and news to reduce benchmark gaming; early correlation plots were encouraging but depended on the model set used for fitting. [Discussion](https://marin-discord.pages.dev/#1356487738840318002/1417394447129382943).
- AIME initially looked poor in one harness because chat templates were applied incorrectly and exact-match scoring did not extract mathematical answers. An evaluation adapter can change the measured outcome without any model-weight change. [Discussion](https://marin-discord.pages.dev/#1356487738840318002/1427789130460561528).
- Benchmark definitions need versioning: a corrected benchmark should coexist with the original erroneous version so past results remain interpretable, with corrections checked against independent evidence. [Discussion](https://marin-discord.pages.dev/#1528226796729008288/1529535129498685471).
- The common-stack proposal separated log-probability/generation evaluation from interactive environment evaluation. Comparisons across models also need matched training stage: an unmidthtrained base is not a fair baseline for a model that has received additional domain training. [Discussion](https://marin-discord.pages.dev/#1356487738840318002/1507184887084482671).
