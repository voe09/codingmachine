# #architecture

Archive coverage: 2026-07-17–2026-09-23 (UTC); 150 messages, including 79 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1527756652890161292).

## Discussion and lessons

- Architecture ideas were treated as measured ablations against a shared baseline, with attention to whether a change helps at MoE scale rather than only in NanoGPT. The channel discussed attention residual/identity HC, latent or shared expert normalization, and looped transformers. [Source](https://marin-discord.pages.dev/#1527756652890161292/1529186868166524969).
- Several proposals targeted inference cost as well as training loss: YOCO-style shared KV cache could speed prefill, but MLA-related alternatives may raise KV-cache requirements. These remained design discussions, not established Marin recipe changes. [Source](https://marin-discord.pages.dev/#1527756652890161292/1537565848108015687).
- Identity-HC attention residuals showed a reported 10% step-wise gain in an early test, but a related singular-value constraint that helped NanoGPT had not improved MoE tests. Small-model architectural wins needed MoE-scale validation. [Discussion](https://marin-discord.pages.dev/#1527756652890161292/1528576455469174784).
- One MLA normalization idea could avoid attention-logit growth but was estimated to increase KV-cache size by about a third in the discussed configuration. Architecture comparisons therefore include serving memory, not just training loss. [Discussion](https://marin-discord.pages.dev/#1527756652890161292/1551997294805520455).
