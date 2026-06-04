Great — I’ll design this as a high-quality system design interview problem, structured from two perspectives:

Backend Engineer

Machine Learning Research Scientist

The system:

A search system that returns product images and descriptions based on user-posted media (image or video).

🧩 Problem Title

Design a Multimodal Product Visual Search System

📌 Product Context

Users upload:

📷 An image (e.g., outfit, furniture, gadget)

🎥 A short video (e.g., someone walking with a handbag)

The system returns:

Similar product images

Product descriptions

Purchase links

Optional: ranked by relevance and personalization

Think of:

Google Lens

Pinterest Lens

Amazon visual search

TikTok Shop

🎯 Core Requirements
Functional

Accept image or video upload

Return top-K product matches

Sub-second latency (P95 < 500ms)

Scale to 100M+ products

Handle 1M+ daily queries

Non-Functional

High availability (99.99%)

Low latency

Fresh index (new products searchable within minutes)

Cost-efficient

Safe (no inappropriate content)

🏗️ Part 1: Backend Engineer Perspective

You are evaluated on:

Distributed systems

Indexing & search architecture

Scalability

Tradeoffs

API & infra design

1️⃣ High-Level Architecture
Client
   ↓
API Gateway
   ↓
Media Preprocessing Service
   ↓
Embedding Service
   ↓
Vector Search Service
   ↓
Ranking Service
   ↓
Product Metadata Service
   ↓
Response

2️⃣ Major Components
1. Media Ingestion Service

Handles image/video upload

Validates format

Extracts frames (for video)

Stores raw media in object storage (e.g., S3 equivalent)

2. Embedding Service

Converts image/video into vector embedding

GPU-backed inference cluster

Caches frequent queries

Outputs:

query_embedding ∈ R^d

3. Vector Search System

Core of retrieval.

Options:

FAISS-based ANN cluster

Managed vector DB (e.g., Pinecone, Milvus)

Custom HNSW implementation

Stores:

product_id → embedding


Scales to:

100M products

512-dim embeddings

Latency target:

< 50ms ANN lookup

4. Ranking Layer

Two-stage ranking:

Stage 1: ANN Retrieval

Top 1000 candidates

Stage 2: Learned Ranking

Cross-modal reranker

Incorporates:

Text similarity

Price

CTR

Personalization

Popularity

5. Metadata Service

Stores:

Product images

Description

Price

Inventory

Brand

Backed by:

Distributed KV store

Search index (Elastic-like)

6. Video Handling Strategy

Option A:

Extract keyframes

Average embeddings

Option B:

Temporal transformer encoder

Tradeoff:

Accuracy vs latency

3️⃣ Scaling Considerations
Data Volume

Assume:

100M products

512-d float32

Memory:

100M × 512 × 4 bytes ≈ 200GB


With HNSW index overhead → ~300–400GB

Solution:

Shard across nodes

Replicate for availability

4️⃣ Freshness Strategy

When new product added:

Compute embedding async

Insert into vector index

Update metadata DB

Warm caches

Use:

Kafka-like event pipeline

5️⃣ Caching Strategy

Embedding cache (popular queries)

ANN result cache

Final ranking cache

6️⃣ Failure Handling

Fallback to:

Text search

Category search

Graceful degradation if embedding service down

7️⃣ Tradeoffs (Backend Focus)
Decision	Tradeoff
Precompute vs real-time embedding	Freshness vs cost
HNSW vs IVF	Latency vs memory
Multi-modal unified embedding vs dual index	Simplicity vs flexibility
GPU inference vs CPU inference	Cost vs latency
🧠 Part 2: ML Research Scientist Perspective

You are evaluated on:

Model architecture

Training data

Multimodal representation learning

Evaluation metrics

Bias and failure modes

1️⃣ Core ML Problem

Learn a shared embedding space:

f(image/video) ≈ g(product_image + description)


So relevant products are close in embedding space.

2️⃣ Model Architecture Options
Option A: CLIP-style Model

Inspired by:

OpenAI CLIP

Train contrastively:

(image, product_text)


Loss:

InfoNCE

Advantages:

Strong cross-modal alignment

Scalable

Option B: Video-Text Contrastive Model

Use:

Frame encoder (ViT)

Temporal pooling

Text transformer

Option C: Two-Tower Model

Tower 1:

User media encoder

Tower 2:

Product encoder

Trained with:

Click logs

Purchase logs

3️⃣ Training Data

Sources:

Product catalog

Click-through logs

Add-to-cart

Purchase

Human relevance labels

Hard negative mining:

Visually similar but wrong brand

Same category but different product

4️⃣ Loss Functions

Contrastive loss

Triplet loss

Pairwise ranking loss

Listwise ranking loss (LambdaRank)

5️⃣ Evaluation Metrics

Offline:

Recall@K

NDCG@K

MRR

Online:

CTR uplift

Conversion rate

Revenue per query

6️⃣ Video-Specific Challenges

Motion blur

Multiple objects

Temporal ambiguity

Partial visibility

Strategies:

Object detection pre-filter

Saliency-guided frame selection

Multi-instance learning

7️⃣ Cold Start Problem

New product:

Few clicks

Few interactions

Solutions:

Text-image pretraining

Zero-shot transfer

Synthetic augmentation

8️⃣ Bias & Safety

Brand bias

Skin tone bias

Gender bias in fashion

Counterfeit detection

NSFW filtering

Add:

Moderation model before pipeline

📊 System Design Interview Extensions

Good follow-ups interviewer may ask:

Backend

How do you support 1B products?

How do you reduce memory footprint?

How to support hybrid (text + image) queries?

How do you handle region-specific catalogs?

ML

How to improve fine-grained similarity?

How to handle occlusion?

How to support compositional queries (“red Nike shoes”)?

How to distill large models into mobile models?

⭐ What Makes This a Strong Interview Problem

It tests:

Backend

Distributed systems

Search infrastructure

Scaling ANN

Latency tradeoffs

ML

Representation learning

Contrastive training

Retrieval systems

Multimodal modeling

If you'd like, I can next:

Turn this into a mock Meta-level interview question

Or provide a full 60-minute interview solution walkthrough

Or create a whiteboard-ready system diagram

Or convert this into a realistic FAANG-style interview script

Which direction would you like?