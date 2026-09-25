# #data-e2e-transform

Archive coverage: 2026-01-19–2026-07-18 (UTC); 77 messages, including 2 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1462896888000024711).

## Rephrasing data for different training stages

The end-to-end transform project tested whether rephrasing source documents could make their information more useful to a language model. In preliminary continued-pretraining comparisons, a Nemotron-only cooldown beat mixtures that added either DCLM or the rephraser output. At roughly comparable output-token counts, the rephraser looked better than DCLM in an aggregate measure, yet DCLM won 7 of 11 individual tasks, and reported answer-probability differences were around 0.001. The author explicitly questioned whether this was strong enough evidence for a claim. [Preliminary comparison](https://marin-discord.pages.dev/#1462896888000024711/1478952804830679142).

A different transform extracted question/reasoning/answer tuples from math-forum material for SFT and reportedly improved GSM8K and MATH. This is why transform value cannot be reduced to a single “rewriting helps” verdict: domain, target format, and stage of training matter. The channel later considered locality-sensitive hashing for documents with more than 300 paragraphs, an implementation detail for scaling transforms over very long inputs. [Math SFT result](https://marin-discord.pages.dev/#1462896888000024711/1481525022534139996) · [Long-document processing](https://marin-discord.pages.dev/#1462896888000024711/1528146605637632071).
