# #scaling-laws

Archive coverage: 2025-05-19–2026-08-30 (UTC); 947 messages, including 357 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1356490712199462912).

## Discussion and lessons

- Scaling-law work compared Chinchilla fitting approaches, numerical stability, critical batch size, and early-loss forecasts. Reparameterization and grid search were emphasized to avoid unstable multi-parameter fits and local minima. [Source](https://marin-discord.pages.dev/#1475498065333850224/1479137523165954078).
- The team distinguished compute-optimal model/token allocation from the narrower task of predicting the loss of an already chosen run. A Chinchilla fitting method can be biased for allocation yet forecast Delphi loss well under a better-controlled IsoFLOP setup. [Discussion](https://marin-discord.pages.dev/#1356490712199462912/1488870819349270660).
- A reported reanalysis estimated material compute waste from a biased two-stage parabola fit at Llama-3-like scale, but the dollar figure came from that analysis and is not a direct Marin expenditure. [Discussion](https://marin-discord.pages.dev/#1356490712199462912/1483148794232176851).
- Scaling behavior was not perfectly invariant: smaller WSD runs showed re-warmup spikes that seemed to fade at larger sizes, and AdamH showed some hyperparameter transfer even without the expected parameterization. Both observations argue for checking transfer empirically. [Discussion](https://marin-discord.pages.dev/#1356490712199462912/1375167314483347646).
