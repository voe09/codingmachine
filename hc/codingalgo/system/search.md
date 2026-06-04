This is a comprehensive, rigorous breakdown of potential system design and ML design problems for modern search, retrieval, and agent platforms.

A useful theme across every problem below is evidence-aware reasoning. Many production systems optimize primarily for relevance or engagement; here, the emphasis is factuality, source quality, and reasoning support. That changes the architecture in meaningful ways.

Part 1: Backend & Infrastructure Engineering (Systems Focus)
Focus: Scale, Latency, Consistency, Rust/C++, Distributed Systems

1. The "Live" Web Index (Ingestion & Freshness)
The Problem: Design a crawling and ingestion system that indexes the web, specifically optimizing for breaking news and fast-moving social or publisher feeds so they are available to the LLM within seconds.

Hard Constraints:

Throughput: 100k+ new documents/sec (posts, news, financial reports).

Latency: Time-to-index < 5 seconds.

Deduplication: A breaking story generates 10,000 duplicate articles/posts instantly. How do you deduplicate in stream without blocking ingestion?

Key Concepts:

Log-Structured Merge-trees (LSM) for write-heavy loads.

Stream processing (Kafka/Spark Structured Streaming).

Bloom filters / MinHash / SimHash for ultra-fast near-duplicate detection.

Design Twist: How do you prioritize authoritative publishers or first-party sources in the ingestion queue during a high-load event?

2. Trillion-Scale Hybrid Search (Vector + Keyword)
The Problem: Pure vector search (semantic) is bad at exact matches (names, dates, error codes). Pure keyword search misses context. Design a unified search engine that queries both an Inverted Index (BM25) and a Vector Index (HNSW) and merges results.

Hard Constraints:

Scale: 1 Trillion Documents.

Storage: Multi-Petabyte scale (cannot fit in RAM).

Latency: p95 < 200ms.

Key Concepts:

Sharding Strategy: Document-based vs. Term-based sharding.

Index compression: Product Quantization (PQ), Scalar Quantization (SQ8).

Posting List intersection: How to efficiently intersect a list of 10M documents containing a named entity with a vector search for a related concept.

Design Twist: Agents generate complex boolean filters ("Find PDF, published > 2024, author=X"). How does your engine handle heavy post-filtering without timing out?

3. The "Infinite" Context Window Memory System
The Problem: An assistant needs to remember previous conversations or massive uploaded codebases. Design a hierarchical memory system that acts as an external hard drive for the LLM.

Hard Constraints:

Retrieval: Must retrieve specific code blocks or chat turns with 100% precision (no fuzzy match errors for variable names).

Lifecycle: Data is mutable (user edits a file). How do you update the index cheapy?

Key Concepts:

Content-Addressable Storage (CAS).

Hierarchical Navigable Small World (HNSW) graphs with deletion support (hard).

Design Twist: How do you prevent poisoning? If a user uploads a malicious document that contradicts trusted information, how do you isolate that memory so it does not bleed into the shared retrieval layer?

4. Distributed Embedding Generation Service
The Problem: You have 1 PB of raw HTML/Images. You need to convert them to embeddings using a heavy model (e.g., SigLIP or a 7B LLM).

Hard Constraints:

GPU Utilization: Keep GPUs at 100% utilization despite variable input sizes (text vs 4k video).

Fault Tolerance: Nodes die constantly.

Key Concepts:

Batching strategies (dynamic batching vs. padding).

Model parallelism vs. Data parallelism.

Design Twist: Multimodal Alignment. How do you ensure the video embedding service and text embedding service are synchronized so that "A dog running" (text) and a video of a dog running map to the same vector space?

Part 2: Machine Learning Engineering (Modeling & Algo Focus)
Focus: Ranking, Embeddings, Precision, Factuality, Agentic Flows

5. The Factuality-Aware Reward Model (Ranking)
The Problem: Most rankers predict click probability or short-term satisfaction. Design a ranker that predicts factual accuracy.

Hard Constraints:

No Labels: You don't have human labels for every fact.

Adversarial Web: The web is full of SEO spam that looks authoritative.

Key Concepts:

Learning to Rank (LTR): Pointwise vs. Pairwise vs. Listwise loss functions.

Distillation: Using a larger teacher model to label data to train a smaller model for the real-time ranker.

Design Twist: Consensus Ranking. If 5 sources say X and 1 source says Y, usually X is true. But if the 1 source is the primary source (for example, an official announcement), Y may be the better answer. How do you encode source authority into the model?

6. Retrieval for "Reasoning" Agents (Chain-of-Thought Retrieval)
The Problem: An agent solving a math or physics problem needs to search iteratively. Design a retrieval loop.

Hard Constraints:

Query Formulation: The agent's internal thought is "I need to find the mass of the sun." The search query should be "Mass of Sun kg NASA". How do you train the query re-writer?

Stopping Condition: When does the agent know it has enough info?

Key Concepts:

Dense Retrieval vs. Sparse Retrieval: When to use which?

Relevance Feedback: Using the first search result to generate the second search query.

Design Twist: Negative Constraints. The agent needs to verify a fact is not true. How do you design a retrieval system that proves a negative?

7. Multimodal Granularity (The "Needle in a Haystack" Video Search)
The Problem: A user asks: "Find the moment in a launch video where stage separation happens."

Hard Constraints:

Granularity: You can't just return the whole video. You need the specific timestamp (seconds 140-145).

Cost: You can't run a heavy vision encoder on every frame of every YouTube video.

Key Concepts:

Hierarchical Indexing: Indexing whole videos -> scenes -> shots -> frames.

Late Interaction Models (ColBERT): Preserving token-level interactions for higher precision.

Design Twist: Aligning audio transcripts with visual frames to maximize factual consistency. For example, the video shows a failure, but the narrator says "success." How does the model resolve the conflict?

8. Optimizing Retrieval-Augmented Generation (RAG) Latency
The Problem: The "Time to First Token" (TTFT) for the user includes: Query Rewrite -> Search -> Rerank -> LLM Generation. This is too slow.

Hard Constraints:

Reduce end-to-end latency by 50% without losing accuracy.

Key Concepts:

Speculative Retrieval: Start searching while the user is still typing?

Async Loading: Stream the LLM answer while the search results are still being verified in the background?

Design Twist: Confidence Scoring. If the retrieval model is 99% confident, skip the expensive re-ranker. How do you calibrate that confidence score well enough to trust it?

How to "Practice Carefully" (Action Plan)
Pick One from Each List: Start with #2 (Hybrid Search) and #5 (Factuality-Aware Reward Model). These are strong practice candidates.

Whiteboard the "Happy Path": Draw the flow. User Query -> API -> Pre-processor -> Index -> Ranker -> Response.

Break it (The "Senior" Step):

"What if the index server crashes?" -> Replica strategy.

"What if a high-profile post drives 1M queries/sec for the same key?" -> Caching/Hot-key handling.

"What if the accepted answer changes as new information arrives?" -> Cache invalidation strategy.

Math Check:

Calculate storage: 1T docs * 1KB vector = 1PB. Can you put this in RAM? No. You need DiskANN or SSD-based storage.

Calculate QPS: If 10M users, 10% active, 1 query/min = ~16k QPS. Can one machine handle this? No. You need load balancers and sharding.
