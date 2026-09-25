# #sft-agents

Archive coverage: 2025-11-04–2026-04-27 (UTC); 40 messages, including 19 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1435065934992773221).

## Testing whether a base model can become a coding agent

The collaboration began with agent demonstrations in OpenAI chat format, making ingestion relatively straightforward but still requiring a matched evaluation harness. A first lightweight proposal was to SFT a 32B model for three epochs, then run SWE-Bench; the estimated training and evaluation times were planning figures, not published outcomes. [Data format](https://marin-discord.pages.dev/#1435065934992773221/1435351225439748100) · [First experiment plan](https://marin-discord.pages.dev/#1435065934992773221/1435670180008038470).

By April 2026, the question had become how to *bracket* Marin 32B's post-trainability for coding tasks before committing to expensive follow-ups. The proposed sequence started with an oracle—Qwen3-32B trained on NemotronTerminal—then SFT Marin 32B on the same terminal data and compare on Terminal-Bench 2. A separate OpenSWE SFT experiment would check SWE-Bench Verified. Only if those diagnostics were poor would the group add a 70B-token code midtraining dataset before SFT. This ordering gives fast signal about data/format transfer before assuming the base model needs a long additional pretraining phase. The channel message lays out a plan; it does not report completed Marin pass rates for all stages. [Experimental ladder](https://marin-discord.pages.dev/#1435065934992773221/1489452658929701044).

Some environment instructions were being coordinated through the external project, including Daytona and a possible Nebius equivalent. That is one example of useful setup detail living outside the Discord messages themselves. [Environment coordination](https://marin-discord.pages.dev/#1435065934992773221/1486049766298091713).
