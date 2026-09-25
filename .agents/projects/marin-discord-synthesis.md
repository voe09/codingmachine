# Marin Discord synthesis

## Objective

Replace selective channel notes in `hc/marin-discord/` with learning-focused summaries grounded in a chronological review of every exported message body and thread reply. Ignore greetings, routine coordination, and bot digests; retain distinct technical questions, methods, results, failures, corrections, and open decisions. Cite the messages that support each substantive claim.

## Source and current state

The local public-archive export from September 22, 2026 contains 36,257 messages. It excludes private, deleted, or unindexed messages. Some image-only posts have expired attachments. The README identifies the seven channels whose exported message bodies have been reviewed end to end: `data-rewriting`, `mtp`, `train-test-overlap`, `tokenizer`, `scaling-suite`, `sft-agents`, and `optimizers` (877 archived messages in those channels). All other channel files remain selective, even where later gaps have been filled.

The live archive has changed since the export. Finish the frozen snapshot first, then refresh and review messages added after it. Do not change an archive-size header or claim full current-server coverage based on the older export.

## Review method

1. For each remaining text channel, read its exported messages in chronological date slices, including thread replies. Check empty-body posts for attachments and record inaccessible image-only evidence as a limitation.
2. Group each conversation by research question. Separate reported measurements, proposed experiments, hypotheses, and later corrections. Follow adjacent replies before interpreting a headline number.
3. Rewrite the channel Markdown around the distinct technical learnings. Link to original message IDs. Treat external papers, issues, plots, and reports as pointers unless their contents have been separately read.
4. Run the local citation/relative-link validator and `git diff --check`, inspect the diff, publish a coherent batch, then add the channel to the README's fully reviewed list. Never use the channel's total message count as a proxy for review coverage.

## Next batches

Prioritize channels with large research discussions: `data-curation`, `data-mixing`, `evals`, `scaling-laws`, `inference`, `moe`, `reinforcement-learning`, `marin-32b`, `speedrun`, `levanter`, and `infra`. `infra` and `reinforcement-learning` have already received selective expansions; they still need chronological review. Work through the smaller remaining technical channels afterward, using the README as the channel inventory. The forum pages require a separate UI audit because the export lacks complete forum topology.
