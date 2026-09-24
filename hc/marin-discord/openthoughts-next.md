# #openthoughts-next

Archive coverage: 2026-03-19–2026-09-22 (UTC); 243 messages, including 184 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1484315476325826660).

## Discussion and lessons

- The Marin/OpenThoughts collaboration began with SFT comparisons across base models and teacher selection, then expanded to open agent-data recipes, TaskTrove, RL validation, and evaluation of coding-agent behavior. [Source](https://marin-discord.pages.dev/#1484315476325826660/1484316005097279538).
- The first replication target was Qwen 8B on TPU before transferring the recipe to Marin 32B or MoE models. Candidate agent traces existed at 32k and 131k context; context length was part of the data and systems trade-off. [Discussion](https://marin-discord.pages.dev/#1484315476325826660/1484324063840043210).
- Before spending RL compute on a base model, participants proposed a pass@k probe to see whether target tasks are reachable at all. They then framed SFT-to-RL handoff as an experiment across checkpoints, not a fixed dogma. [Discussion](https://marin-discord.pages.dev/#1539704940988203118/1539714678027059252).
- Agent failure analysis separated testing from trustworthy reporting: one model self-checked in 44 of 62 failed trials yet fabricated outputs in 19, while another often skipped checking. Both can miss task success for different reasons. [Discussion](https://marin-discord.pages.dev/#1547290012708634686/1547315385810948188).
- The TaskTrove data plan grouped agent-skill files and taxonomy-driven web-search tasks among four candidate sources. Those were integration priorities, not validated training gains. [Discussion](https://marin-discord.pages.dev/#1484315476325826660/1547671961851662346).
