# #data-rewriting

Archive coverage: 2025-09-05–2026-09-22 (UTC); 67 messages, including 15 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1413654082115342386).

## Open data and the format confounder

Rewriting was proposed to expand the useful training supply from openly licensed and public-domain seed data, which is much smaller than the total text available on the web. The group estimated the Common Pile at roughly 500B unique tokens in this discussion and wanted to test whether transformations could increase useful variety without losing provenance or license clarity. [Open-data motivation](https://marin-discord.pages.dev/#1413654082115342386/1415066746037211237).

An early rephrasing experiment exposed a measurement trap: changing text into multiple-choice-question format improved MMLU scores and made MMLU accuracy predictable at smaller model sizes. That does not by itself prove a semantic improvement in the underlying knowledge. A fairer comparison would train alternative content mixtures and then adapt each model to the same evaluation format before comparing. Content quality and familiarity with the answer format must be separated. [Rephrasing result and proposed control](https://marin-discord.pages.dev/#1413654082115342386/1416097374442356817).

Later posts asked for a canonical dataset→teacher→property rewrite workflow and shared a first GLM-generated task/RL-environment example for feedback before scale-up. The archive does not establish that this later task-generation process produced validated training gains. [Workflow question](https://marin-discord.pages.dev/#1413654082115342386/1461175734826434641) · [Task example](https://marin-discord.pages.dev/#1413654082115342386/1551989461401735188).
