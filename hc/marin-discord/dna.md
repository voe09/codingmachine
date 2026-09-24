# #dna

Archive coverage: 2025-09-19–2026-09-11 (UTC); 373 messages, including 225 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1418673157585502370).

## Discussion and lessons

- MarinDNA discussion adapted language-model scaling, tokenization, and objectives to genomic sequence. Participants stressed that lower sequence loss need not improve zero-shot biological tasks, and compared causal versus masked modeling, repeat downweighting, and sliding-window attention. [Source](https://marin-discord.pages.dev/#1418673157585502370/1432595415076831232).
- A concrete sweep compared masked and causal objectives, model size, repeat downweighting, and sliding-window attention on animal promoter sequences. It did not assume that a text-LM default or lower held-out loss would necessarily improve biological tasks. [Discussion](https://marin-discord.pages.dev/#1418673157585502370/1448363912092192939).
- DNA sequences can contain tens or hundreds of repeated bases in a row, depending on species and preprocessing. Tokenization and attention stability therefore need tests tailored to sequence data rather than ordinary prose. [Discussion](https://marin-discord.pages.dev/#1442534344622215178/1443099373432209408).
