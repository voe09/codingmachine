# #deduplication

Archive coverage: 2025-11-21–2026-08-31 (UTC); 270 messages, including 179 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1441211384279994529).

## Discussion and lessons

- The group studied large-scale document deduplication, tooling, and where it belongs in the data pipeline. A central caveat was that deduplicating a mixture can downweight high-quality documents that occur in several curated datasets. [Source](https://marin-discord.pages.dev/#1441211384279994529/1443154063263928351).
- The pipeline-scale run counted about 15.1 billion input documents. At that size, representation, shard throughput, and cross-region data movement become material parts of a deduplication experiment. [Discussion](https://marin-discord.pages.dev/#1441211384279994529/1508887247796178985).
