# #marin-32b

Archive coverage: 2025-04-30–2026-06-13 (UTC); 972 messages, including 103 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1367246647062564995).

## Discussion and lessons

- The model was released and used as a base for later post-training; reports of positive model quality should be read with the channel's specific eval protocols and later comparisons. [Source](https://marin-discord.pages.dev/#1367246647062564995/1431414820800040971).
- The stability investigation proposed replaying batches before update-norm blowups and comparing optimizer state, then cooling down before and after spikes to test downstream impact. Later QK-normalization warm-start work was reported stable, but not all early hypotheses were established causes. [Discussion](https://marin-discord.pages.dev/#1367246647062564995/1379675233610502217).
- Release comparisons required protocol context: one later table ranked Marin 32B variants differently and showed their mean reciprocal ranks below Qwen2.5-32B Base in that particular comparison. A single favorable benchmark was not a comprehensive quality judgment. [Discussion](https://marin-discord.pages.dev/#1367246647062564995/1432164181699727380).
- Data license constraints affected the training recipe: Nemotron CC-v2 was flagged as potentially incompatible with Marin's intended openness, so the team could not treat technical quality as the only selection criterion. [Discussion](https://marin-discord.pages.dev/#1367246647062564995/1413766160150954075).
