# 科学空间 + SOTA LLM 阅读路线

更新日期：**2026-05-28**

这份文档的目标，不再是“先看 Su，再看站外论文”，而是把它们**揉成一条可执行的阅读路线**。  
核心思路是：

1. 先用经典论文建立骨架。
2. 再用苏剑林的博客把关键机制真正想明白。
3. 再补 2025-2026 的最新技术报告与论文，知道现在 frontier 到了哪里。
4. 最后对照官方代码仓库和 benchmark，避免停留在“只会读 paper”。

如果只用一句话概括：  
**Su 最强的是把机制讲透；SOTA 论文最强的是告诉你工业界最近到底在做什么。两者必须交叉读。**

## 怎么用这份路线

我统一用下面四种标签：

- `骨架`：建立全局框架，先读。
- `深挖`：苏剑林最有价值的“把问题讲透”的文章。
- `前沿`：2025-2026 还很新的方向或技术报告。
- `代码`：最值得直接读实现的官方 repo / benchmark / trainer。

阅读时最好按每个主题里给出的**顺序**走，不要只挑最新的读。  
你这次提的要求其实很关键：**“最新”不等于“最好入口”**。真正高效的做法是“骨架 -> 深挖 -> 前沿 -> 代码”。

## 一张总图

如果你想获得“广而深”的理解，我建议按下面 8 个分支读：

1. Transformer 主干、Decoder-only、位置编码与 RoPE
2. 长上下文与长度外推
3. Attention 效率、MLA、线性注意力、Residual 新结构
4. MoE、训练稳定性、优化器、缩放规律
5. Tokenizer、随机分词、token-free 路线
6. 预训练数据、数据配比、continued pretraining / mid-training
7. Post-training：SFT、DPO、online preference、reasoning RL、OPD、SDPO
8. Agentic training、coding agents、agent benchmark、agent training framework

下面每一节都把 **Su + 论文 + 代码** 混在同一条顺序里。

## 先看这张对照表

如果你不想一上来就从头顺读，可以先按“我眼下最想搞懂什么”来选主线：

- **主干为什么是 Decoder-only + RoPE？**  
  `AIAYN -> Su 8130 / 8231 / 8265 -> RoFormer -> Su 9529 / 10347 / 9675`
- **长上下文到底怎么扩，RoPE 为什么老是绕不过去？**  
  `Su T7 / T8 / T9 / T12 / T15 / T16 -> YaRN -> LongRoPE -> yarn 代码`
- **KV cache、attention 效率、MLA、线性注意力最近怎么演化？**  
  `GQA -> Su 10091 -> DeepSeek-V2 -> Su T20 / T21 -> Kimi Linear -> Attention Residuals -> 对应 repo`
- **MoE 和大规模训练 recipe 为什么现在又变成重点？**  
  `Switch / GLaM / Mixtral -> Su MoE 环游记 -> Scaling Laws / Chinchilla -> Muon / QK-Clip / K2`
- **Tokenizer 还是不是核心问题？**  
  `BPE / SentencePiece -> Su BytePiece / 随机分词 -> ByT5 / MEGABYTE / BLT`
- **pretraining / mid-training 不是只靠堆数据，那怎么读？**  
  `The Pile / Dedup / DoReMi -> OLMo / DataComp-LM -> Don’t Stop Pretraining -> DeepSeekMath / Qwen3`
- **post-training 现在最前沿的主线是什么？**  
  `InstructGPT -> DPO / ORPO / SimPO -> Online Preference -> DeepSeek-R1 -> SDPO -> OPD / Lightning OPD`
- **agentic training 到底是在训什么？**  
  `Kimi K2 / Qwen3 / Qwen3-Coder-Next -> Agent Lightning -> MAD-OPD / OPAD -> tau2-bench / SWE-bench`

---

## 1. Transformer 主干、Decoder-only、位置编码与 RoPE

这一支解决的问题是：**今天的 LLM 主干为什么几乎都长成 Decoder-only + RoPE 这一套？**

### 推荐顺序

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) `骨架`  
   Transformer 的起点。
2. [让研究人员绞尽脑汁的Transformer位置编码](https://spaces.ac.cn/archives/8130) `深挖`  
   先把位置编码全景图立起来。
3. [Transformer升级之路：1、Sinusoidal位置编码追根溯源](https://spaces.ac.cn/archives/8231) `深挖`  
   适合建立 Su 后续文章的数学风格。
4. [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265) `深挖`
5. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) `骨架`  
   这里建议和上面第 4 篇对读。
6. [为什么现在的LLM都是Decoder-only的架构？](https://spaces.ac.cn/archives/9529) `深挖`
7. [《为什么现在的LLM都是Decoder-only的架构？》FAQ](https://spaces.ac.cn/archives/9547) `深挖`
8. [Decoder-only的LLM为什么需要位置编码？](https://spaces.ac.cn/archives/10347) `深挖`
9. [Transformer升级之路：10、RoPE是一种β进制编码](https://spaces.ac.cn/archives/9675) `深挖`  
   这是整条 RoPE 主线最关键的一篇。
10. [Transformer升级之路：12、无限外推的ReRoPE？](https://spaces.ac.cn/archives/9708) `深挖`
11. [Transformer升级之路：16、“复盘”长度外推技术](https://spaces.ac.cn/archives/9948) `深挖`

### 这一节读完，你应该能回答

1. 为什么 RoPE 会比“纯 absolute PE”更适合 LLM？
2. 为什么主流大模型还是以 Decoder-only 为主？
3. 为什么 RoPE 几乎天然会把你带进“长上下文”问题？

---

## 2. 长上下文与长度外推

这一支解决的问题是：**上下文窗口为什么难扩？能扩到多长？哪些方法真的有工程价值？**

### 推荐顺序

1. [Transformer升级之路：7、长度外推性与局部注意力](https://spaces.ac.cn/archives/9431) `深挖`
2. [Transformer升级之路：8、长度外推性与位置鲁棒性](https://spaces.ac.cn/archives/9444) `深挖`
3. [Transformer升级之路：9、一种全局长度外推的新思路](https://spaces.ac.cn/archives/9603) `深挖`
4. [Transformer升级之路：10、RoPE是一种β进制编码](https://spaces.ac.cn/archives/9675) `深挖`
5. [Transformer升级之路：12、无限外推的ReRoPE？](https://spaces.ac.cn/archives/9708) `深挖`
6. [Transformer升级之路：14、当HWFA遇见ReRoPE](https://spaces.ac.cn/archives/9731) `深挖`
7. [Transformer升级之路：15、Key归一化助力长度外推](https://spaces.ac.cn/archives/9859) `深挖`
8. [Transformer升级之路：16、“复盘”长度外推技术](https://spaces.ac.cn/archives/9948) `深挖`
9. [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) `骨架`
10. [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307) `骨架`
11. [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) `前沿`

### 代码 / 工具

- [jquesnelle/yarn](https://github.com/jquesnelle/yarn) `代码`

### 这一节的阅读心法

这条线不要一开始就只盯着“谁扩到了 1M / 2M token”。  
真正该先想明白的是：

1. 是**位置编码失效**，还是**attention 结构本身不鲁棒**？
2. 是**训练长度不够**，还是**推理时插值方法有问题**？
3. 是**paper 里的 window**，还是**模型真的会用 long context**？

Su 在这里的价值，比很多综述都高，因为他会强迫你问“为什么这个方法会有效”。

---

## 3. Attention 效率、MLA、线性注意力、Residual 新结构

这一支解决的问题是：**算力和 KV cache 不够时，主干结构怎么改？**

### 推荐顺序

1. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) `骨架`
2. [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091) `深挖`
3. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) `骨架`  
   MLA 在这里正式进入主流叙事。
4. [Transformer升级之路：20、MLA好在哪里?（上）](https://spaces.ac.cn/archives/10907) `深挖`
5. [Transformer升级之路：21、MLA好在哪里?（下）](https://spaces.ac.cn/archives/11111) `深挖`
6. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) `前沿`  
   这是 MLA + DeepSeekMoE 真正走到前线的报告。
7. [Transformer升级之路：3、从Performer到线性Attention](https://spaces.ac.cn/archives/8338) `深挖`
8. [线性Transformer应该不是你要等的那个模型](https://spaces.ac.cn/archives/8610) `深挖`
9. [线性注意力简史：从模仿、创新到反哺](https://spaces.ac.cn/archives/11033) `深挖`
10. [为什么DeltaNet要加L2 Normalize？](https://spaces.ac.cn/archives/11486) `深挖`  
    这是 Su 在 2025 年末补的一个很好的“线性注意力细节课”。
11. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) `前沿`  
    虽然是 2025-10-30 提交，但它代表的是 2025 下半年之后最值得关注的高效注意力路线。
12. [Attention Residuals](https://arxiv.org/abs/2603.15031) `前沿`  
    2026-03-16 提交。这个工作不改 QKV 本身，而是改“层间信息如何聚合”。
13. [Attention Residuals 回忆录](https://spaces.ac.cn/archives/11664) `深挖`  
    2026-03-19。非常适合和上面的 tech report 对读。
14. [低精度Attention可能存在有偏的舍入误差](https://spaces.ac.cn/archives/11371) `深挖`  
    2025-10-27。适合在你开始关心 BF16 / FP8 / low precision inference 时读。

### 代码 / kernel / 实现

- [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) `代码`
- [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) `代码`
- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) `代码`
- [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) `代码`

### 这一节的阅读心法

这里不要把“MLA、linear attention、residual routing”看成互斥路线。  
它们分别在回答不同问题：

- GQA / MLA：主要解决 **KV cache 与推理效率**
- DeltaNet / Kimi Linear：主要解决 **长序列效率与表达能力**
- Attention Residuals：主要解决 **层间信息混合方式**

---

## 4. MoE、训练稳定性、优化器、缩放规律

这一支解决的问题是：**大模型怎么在更大规模上稳定训练、便宜训练、稀疏训练？**

### 推荐顺序

1. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) `骨架`
2. [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) `骨架`
3. [Mixtral of Experts](https://arxiv.org/abs/2401.04088) `骨架`
4. [MoE环游记：1、从几何意义出发](https://spaces.ac.cn/archives/10699) `深挖`
5. [MoE环游记：2、不患寡而患不均](https://spaces.ac.cn/archives/10735) `深挖`
6. [MoE环游记：3、换个思路来分配](https://spaces.ac.cn/archives/10757) `深挖`
7. [MoE环游记：4、难处应当多投入](https://spaces.ac.cn/archives/10815) `深挖`
8. [MoE环游记：5、均匀分布的反思](https://spaces.ac.cn/archives/10945) `深挖`
9. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) `骨架`
10. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) `骨架`
11. [DeepSeek LLM: Scaling Open-Source Language Models with Longtermism](https://arxiv.org/abs/2401.02954) `骨架`
12. [Muon续集：为什么我们选择尝试Muon？](https://spaces.ac.cn/archives/10739) `深挖`
13. [QK-Clip：让Muon在Scaleup之路上更进一步](https://spaces.ac.cn/archives/11126) `深挖`
14. [Muon优化器指南：快速上手与关键细节](https://spaces.ac.cn/archives/11416) `深挖`
15. [滑动平均视角下的权重衰减和学习率](https://spaces.ac.cn/archives/11459) `深挖`
16. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) `前沿`  
    它的重要性不只在 agentic，它也展示了 MuonClip / QK-Clip 在大规模训练里的现实价值。
17. [基于流式幂迭代的Muon实现：2. 加速](https://spaces.ac.cn/archives/11673) `深挖`
18. [基于流式幂迭代的Muon实现：3. 雕琢](https://spaces.ac.cn/archives/11697) `深挖`

### 代码 / 系统

- [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) `代码`
- [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) `代码`
- [MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) `代码`

### 这一节的阅读心法

这一节最好分两层理解：

1. **稀疏路由层**：MoE 为什么能更大但不必更贵？
2. **训练系统层**：Muon、QK-Clip、WD/LR schedule 为什么能让大规模训练更稳？

Su 在第 2 层非常有价值，尤其是 Muon 和训练细节这一串 2025-2026 新文章。

---

## 5. Tokenizer、随机分词、token-free 路线

这一支解决的问题是：**tokenizer 到底是工具，还是模型能力的一部分？**

### 推荐顺序

1. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) `骨架`
2. [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) `骨架`
3. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/) `骨架`
4. [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267) `骨架`
5. [BytePiece：更纯粹、更高压缩率的Tokenizer](https://spaces.ac.cn/archives/9752) `深挖`
6. [大词表语言模型在续写任务上的一个问题及对策](https://spaces.ac.cn/archives/9762) `深挖`
7. [随机分词浅探：从Viterbi Decoding到Viterbi Sampling](https://spaces.ac.cn/archives/9768) `深挖`
8. [随机分词再探：从Viterbi Sampling到完美采样算法](https://spaces.ac.cn/archives/9811) `深挖`
9. [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) `骨架`
10. [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) `骨架`
11. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) `前沿`

### 代码 / tokenizer / token-free 实现

- [google/sentencepiece](https://github.com/google/sentencepiece) `代码`
- [facebookresearch/blt](https://github.com/facebookresearch/blt) `代码`

### 这一节的阅读心法

这里最容易犯的错，是把 tokenizer 当成“只影响输入长度”的小问题。  
实际上这条线至少牵涉四件事：

1. 训练成本
2. 生成行为
3. 鲁棒性
4. 多语言与长尾泛化

Su 在“随机分词 / 大词表副作用 / BytePiece”上很有个人视角；  
站外论文则告诉你为什么 byte-level / patch-level 路线在 2025 后重新变得重要。

---

## 6. 预训练数据、数据配比、continued pretraining / mid-training

这一支解决的问题是：**模型不是只靠结构长出来的，数据和中继训练到底怎么做？**

### 推荐顺序

1. [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) `骨架`
2. [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) `骨架`
3. [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) `骨架`
4. [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) `骨架`
5. [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) `前沿`
6. [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) `骨架`
7. [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) `骨架`
8. [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045) `骨架`
9. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) `前沿`
10. [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464) `前沿`
11. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) `前沿`  
    对“thinking / non-thinking 一体化”以及蒸馏式构建小模型很关键。
12. [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905) `前沿`  
    适合看“数据质量 + 合成数据 + curriculum”。

### 代码 / 数据 / pipeline

- [allenai/OLMo](https://github.com/allenai/OLMo) `代码`
- [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) `代码`

### 这一节的阅读心法

Su 的博客在这一支**不是没有价值**，但它不是他最系统的主线。  
如果你主要想建立对 modern pretraining recipe 的理解，这一节应该更偏向读 paper / tech report。

---

## 7. Post-training：SFT、DPO、online preference、reasoning RL、OPD、SDPO

这是这次重写里最重要的一节。  
你提到的 **post-training、OPD、新 area**，主要都在这里。

### 7.1 先建立经典骨架

1. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) `骨架`  
   InstructGPT，所有后续 post-training 的出发点。
2. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) `骨架`
3. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) `骨架`
4. [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) `骨架`
5. [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) `骨架`
6. [Human Alignment of Large Language Models through Online Preference Optimisation](https://arxiv.org/abs/2403.08635) `骨架`
7. [OPTune: Efficient Online Preference Tuning](https://arxiv.org/abs/2406.07657) `前沿`
8. [Online Preference Alignment for Language Models via Count-based Exploration](https://arxiv.org/abs/2501.12735) `前沿`

### 7.2 再进入 reasoning post-training

9. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) `前沿`  
   2025-01-22。reasoning post-training 的里程碑。
10. [DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1) `代码`
11. [Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802) `前沿`  
    2026-01-28。重点不是 preference，而是把 rich feedback 变成 dense signal。
12. [lasgroup/SDPO](https://github.com/lasgroup/SDPO) `代码`

### 7.3 再进入 OPD 主线

13. [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626) `前沿`  
    2026-04-01。适合先把 OPD 的设计空间、失败模式、与 KL-RL 的关系扫一遍。
14. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) `骨架`  
    OPD 的起点。
15. [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125) `前沿`  
    2026-02-12。G-OPD / ExOPD，把 OPD 放回 KL 约束 RL 框架里理解。
16. [RUCBM/G-OPD](https://github.com/RUCBM/G-OPD) `代码`
17. [Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation](https://arxiv.org/abs/2604.13010) `前沿`  
    2026-04-14。offline OPD + teacher consistency，是现在很值得重点读的一个点。
18. [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/abs/2603.25562) `前沿`
19. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://github.com/thunlp/OPD) `代码`
20. [hhh675597/revisiting_opd](https://github.com/hhh675597/revisiting_opd) `代码`
21. [On-Policy Distillation (OPD) — verl 文档](https://verl.readthedocs.io/en/latest/algo/opd.html) `代码`
22. [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) `代码`
23. [huggingface/trl Distillation Trainer](https://github.com/huggingface/trl/blob/main/docs/source/distillation_trainer.md) `代码`
24. [MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate](https://arxiv.org/abs/2605.01347) `前沿`  
    2026-05-02。把 OPD 正式拉进了 **agentic tasks**。

### 7.4 Su 在这一支怎么读？

Su 在 post-training 这一支**没有像 RoPE 那样的完整长系列**，所以不要硬找一一对应。  
更合理的搭配是：

1. 用上面的 paper 把 post-training 大框架搭起来。
2. 再用 Su 的 PEFT / 训练细节文章补“局部机理”：
   - [梯度视角下的LoRA：简介、分析、猜测及推广](https://spaces.ac.cn/archives/9590)
   - [配置不同的学习率，LoRA还能再涨一点？](https://spaces.ac.cn/archives/10001)
   - [对齐全量微调！这是我看过最精彩的LoRA改进（一）](https://spaces.ac.cn/archives/10226)
   - [对齐全量微调！这是我看过最精彩的LoRA改进（二）](https://spaces.ac.cn/archives/10266)

### 这一节真正该想明白的事

1. DPO / ORPO / SimPO 解决的是哪一层问题？
2. online preference optimization 比 offline 方法多解决了什么？
3. RLVR / reasoning RL 和 preference optimization 的关系是什么？
4. OPD / SDPO 为什么会在 2026 变得这么重要？

如果这四个问题想明白了，你对 modern post-training 的理解会比只读 benchmark 表格强很多。

---

## 8. Agentic training、coding agents、agent benchmark、agent training framework

这一支解决的问题是：**模型不只是会答题，而是会在环境里行动、调工具、读仓库、修 bug。**

### 推荐顺序

1. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) `前沿`  
   2025-07-28。它是 open agentic LLM 里非常重要的一份报告。
2. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) `前沿`
3. [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729) `前沿`  
   2026-02-28。它明确把“coding agents + executable environments + RL/mid-training”放到中心位置。
4. [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) `前沿`  
   2026-02-02。把多模态 agentic intelligence 推得更完整。
5. [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680) `前沿`  
   偏 framework / system，但很适合从“训练 agent”角度读。
6. [MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate](https://arxiv.org/abs/2605.01347) `前沿`  
   这篇和上面的 Agent Lightning 一起看，很容易看出 2026 的趋势：  
   **agent training 正在和 OPD / self-distillation / multi-agent orchestration 合流。**

### 代码 / agent 工具 / benchmark

- [MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) `代码`
- [QwenLM/Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) `代码`
- [QwenLM/qwen-code](https://github.com/QwenLM/qwen-code) `代码`
- [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) `代码`
- [tau2-bench 官方仓库](https://github.com/sierra-research/tau2-bench) `代码`
- [SWE-bench 网站与评测入口](https://swebench.lol/) `代码`
- [SWE-bench 网站仓库](https://github.com/SWE-bench/swe-bench.github.io) `代码`

### 这一节的阅读心法

agentic training 不是“在 prompt 里加一句你是 agent”这么简单。  
真正关键的是三层：

1. **模型能力层**：code / tool use / long context / reasoning
2. **环境反馈层**：执行结果、测试结果、runtime error、judge feedback
3. **训练算法层**：RL、OPD、SDPO、multi-agent debate、trajectory decomposition

如果你把这一节和第 7 节一起读，就会很自然地理解为什么 2026 的 agent 训练论文越来越像“post-training 论文”，而不是“prompt engineering 论文”。

---

## 我会怎么安排实际阅读顺序

如果是我自己从零开始补这条线，我会按下面顺序读：

1. **主干与位置编码**  
   `AIAYN -> Su 位置编码总览 -> T1/T2 -> RoFormer -> T10`
2. **长上下文**  
   `T7 -> T9 -> T12 -> T16 -> YaRN -> LongRoPE`
3. **效率与结构改造**  
   `GQA -> MHA/MQA/GQA/MLA -> DeepSeek-V2 -> T20/T21 -> 线性注意力简史 -> Kimi Linear -> Attention Residuals`
4. **MoE + pretraining + 优化器**  
   `Switch -> GLaM -> Mixtral -> MoE环游记 -> Scaling Laws -> Chinchilla -> DeepSeek LLM -> Muon续集 -> QK-Clip -> Muon指南`
5. **Tokenizer**  
   `BPE -> SentencePiece -> Subword Reg -> BytePiece -> 随机分词 -> BLT`
6. **数据与 mid-training**  
   `The Pile -> Dedup -> DoReMi -> OLMo -> DataComp-LM -> Don’t Stop Pretraining -> Self-Instruct -> DeepSeekMath -> Magpie`
7. **post-training 核心**  
   `InstructGPT -> Constitutional AI -> DPO -> ORPO -> SimPO -> Online Preference Optimisation`
8. **reasoning / RLVR / OPD / SDPO**  
   `DeepSeek-R1 -> SDPO -> OPD origin -> G-OPD -> Lightning OPD -> thunlp/OPD -> MAD-OPD`
9. **agentic training**  
   `Kimi K2 -> Qwen3 -> Qwen3-Coder-Next -> Kimi K2.5 -> Agent Lightning -> tau2-bench / SWE-bench / qwen-code`

---

## 如果你时间很少，只读这些

如果你只能抽时间读一个“最小但不失真”的版本，我建议：

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)
3. [Transformer升级之路：10、RoPE是一种β进制编码](https://spaces.ac.cn/archives/9675)
4. [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091)
5. [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
6. [Transformer升级之路：20、MLA好在哪里?（上）](https://spaces.ac.cn/archives/10907)
7. [MoE环游记：1、从几何意义出发](https://spaces.ac.cn/archives/10699)
8. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
9. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
10. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
11. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
12. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
13. [Reinforcement Learning via Self-Distillation (SDPO)](https://arxiv.org/abs/2601.20802)
14. [Lightning OPD](https://arxiv.org/abs/2604.13010)
15. [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)

---

## 这次重写补了什么

相对上一版，这一版主要修了四个问题：

1. **补上了 2025-2026 的新内容**  
   特别是 Kimi K2.5、Qwen3-Coder-Next、Attention Residuals、SDPO、Lightning OPD、MAD-OPD、G-OPD。
2. **补上了 Su 的最新博客**  
   特别是 Attention Residuals、DeltaNet L2 Normalize、Muon 指南、QK-Clip、流式幂迭代 Muon 等。
3. **不再把 Su 和 SOTA 分开**  
   现在每条主题线都是交叉排序。
4. **把 post-training / OPD / agentic training 拉成主线**  
   而不是只停留在架构、RoPE、MoE。

## 最后一句

如果你真正想把这批材料读透，最有价值的姿势不是“刷 paper 标题”，而是：

**每个主题都同时看一篇骨架 paper、一篇 Su 深挖、一篇 2025-2026 更新、一个官方 repo。**
