# #architecture

Archive coverage: 2026-07-17–2026-09-23 (UTC); 150 messages, including 79 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1527756652890161292).

## Discussions and learnings

- Architecture ideas were treated as measured ablations against a shared baseline, with attention to whether a change helps at MoE scale rather than only in NanoGPT. The channel discussed attention residual/identity HC, latent or shared expert normalization, and looped transformers. [Source](https://marin-discord.pages.dev/#1527756652890161292/1529186868166524969).
- Several proposals targeted inference cost as well as training loss: YOCO-style shared KV cache could speed prefill, but MLA-related alternatives may raise KV-cache requirements. These remained design discussions, not established Marin recipe changes. [Source](https://marin-discord.pages.dev/#1527756652890161292/1537565848108015687).

## Month-by-month source trail

These are selected entry points into the archived conversation, not a list of every message. The discussion summary above distinguishes reported results from proposals and unresolved issues.

- 2026-07: 59 messages; [2026-07-22](https://marin-discord.pages.dev/#1529186064537882635/1529619976850374830), [2026-07-21](https://marin-discord.pages.dev/#1527756652890161292/1529186868166524969).
- 2026-08: 61 messages; [2026-08-13](https://marin-discord.pages.dev/#1527756652890161292/1537565848108015687), [2026-08-11](https://marin-discord.pages.dev/#1527756652890161292/1536880244156272801).
- 2026-09: 30 messages; [2026-09-01](https://marin-discord.pages.dev/#1527756652890161292/1544406869693571173), [2026-09-22](https://marin-discord.pages.dev/#1527756652890161292/1551997294805520455).
