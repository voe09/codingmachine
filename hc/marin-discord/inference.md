# #inference

Archive coverage: 2025-06-20–2026-08-29 (UTC); 798 messages, including 290 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1385733711013871729).

## Discussion and lessons

- Trainer-versus-server differences were traced partly to ragged paged attention and a layer-fold implementation, motivating parity checks before attributing quality changes to weights. [Source](https://marin-discord.pages.dev/#1385733711013871729/1437252801750106185).
- The 2.6x vLLM throughput comparison held workload settings fixed; absolute numbers were 5,427 versus 1,000 tokens/s. A separate multi-host TPU test reached about 1,536 generation tokens/s on v5p-16, but neither result is a universal serving-speed ratio. [Discussion](https://marin-discord.pages.dev/#1385733711013871729/1427163698522488832).
- Serving quality required prompt and stop-condition parity. Fixing padding in stop-sequence handling, chat-template application, and few-shot formatting helped reproduce Llama 3.1 8B Instruct GSM8K at 85.6 versus a reported 84.5. [Discussion](https://marin-discord.pages.dev/#1385733711013871729/1420612223822200934).
- Head-dimension constraints were model-specific: an early serving path handled the 8B model's 128-dimensional heads but not the 1B model's 64-dimensional heads. The team treated this as an engine limitation, not a model-quality defect. [Discussion](https://marin-discord.pages.dev/#1385733711013871729/1420198905940934686).
