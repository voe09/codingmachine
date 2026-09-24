# #marinfold

Archive coverage: 2026-06-15–2026-09-21 (UTC); 450 messages, including 214 thread replies. [Open channel in Discord](https://discord.com/channels/1354881461060243556/1516150256499163166).

## Discussion and lessons

- MarinFold explored protein structure from sequence through contact prediction and later structure-oriented training. Early contacts-v1 modeling failed, while a subsequent sweep produced meaningful contact precision; evaluation by sequence-identity bucket helped distinguish generalization from nearest-neighbor memorization. [Source](https://marin-discord.pages.dev/#1516150256499163166/1520077768807678135).
- The later ProteinMPNN-expanded synthetic set delivered a reported best model on de novo and legacy evaluations; smaller promising tricks, such as token smearing or soft targets, did not automatically translate into full-run gains. [Source](https://marin-discord.pages.dev/#1516150256499163166/1549131668957298799).
- The first contacts-v1 model performed poorly; a hyperparameter sweep, especially more passes over data, produced a model above 0.4 contact R-precision on 554 proteins. The team reported a change in behavior after roughly 23B training tokens. [Discussion](https://marin-discord.pages.dev/#1516150256499163166/1521231463318818956).
- A nearest-neighbor baseline beat MarinFold for proteins close to its training set but fell to zero on distant ones where MarinFold still made some correct predictions. Stratifying by sequence identity exposed generalization that an aggregate score would hide. [Discussion](https://marin-discord.pages.dev/#1516150256499163166/1520077768807678135).
- For unordered contact sets, one proposed loss avoids penalizing arbitrary serialization order and orientation. The implementation ran end to end, but the cited update did not yet establish that it improved structure prediction. [Discussion](https://marin-discord.pages.dev/#1516150256499163166/1531345956321230975).
