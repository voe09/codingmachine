# #idea-board

Forum coverage: nine posts visible in the Discord app on 2026-09-23, posted between August 2025 and February 2026. The public archive used for the text-channel files did not include this forum. [Open the forum](https://discord.com/channels/1354881461060243556/1404917882093043832).

## Discussions and learnings

- The forum was set up as a source of concrete, newcomer-friendly project ideas. Most posts framed a first implementation step and an experiment to evaluate it. They are proposals, not evidence that the idea succeeded.
- The [UniMax post](https://discord.com/channels/1354881461060243556/1404955814896341132) proposed computing corpus sampling weights from token counts, training budget, and an epoch cap. A contributor opened [PR #1514](https://github.com/marin-community/marin/pull/1514); replies identified an additional integration helper and a controlled comparison with a regular Dolma mix as follow-up work. The final forum reply requested merge follow-up, so the thread alone does not establish that integration or evaluation happened.
- The [UL2R post](https://discord.com/channels/1354881461060243556/1404953327510487190) proposed denoising and infilling as data-efficiency experiments. Replies recommended starting with infilling, reserving task/mask tokens, and comparing mixtures of UL2R and causal-LM objectives on the scaling ladder. Random-access dataset semantics and attention/loss masks were implementation risks.
- The [style/register/source-token post](https://discord.com/channels/1354881461060243556/1404953723188547634) proposed prepending a source or style prompt during pretraining and masking that prompt from loss. Replies compared caching prompted documents with just-in-time insertion; the latter offers flexibility but complicates random access and offsets.
- Other posts proposed launching experiment jobs from reviewed PRs, a dynamic artifact-like executor, per-layer learning rates, and averaging-based training without a cooldown. Later posts invited collaborators on looped language models and training-data attribution for social reasoning. The visible forum cards did not report validated results for those ideas.
