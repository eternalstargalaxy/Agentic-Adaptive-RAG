# Agentic Adaptive RAG 框架说明、医学域化设计与简历点映射

## 1. 项目定位

这个项目当前的目标，是在原始 `Agentic Adaptive RAG` 的基础上，逐步演进成一个更贴近你简历描述的 `医学域自适应 RAG 系统`。它不是单纯的“先检索、再生成”的固定流水线，而是一个由 `LangGraph` 驱动的、可根据问题复杂度动态改变策略的闭环系统。

项目最核心的智能化设计点是：

- `基于 LLM 的 Query Router 动态路由机制`

系统会根据查询复杂度做三级分流：

1. `No Retrieval`
   - 对低风险、低复杂度、可由模型参数知识直接回答的问题，跳过检索直接回答。
   - 目标是降低时延，提高响应速度。

2. `Single-Step`
   - 对简单事实型问题，走一次 `BGE-M3 + BM25 + RRF` 的单轮混合检索。
   - 目标是用最小检索成本获得高精度证据。

3. `Multi-Hop`
   - 对复杂医学问题，例如多药联用、综合诊疗方案、需要多源交叉验证的问题，进入 LangGraph 的多节点闭环推理。
   - 目标是用更强的检索、重试、联网补证和答案校验来保证可靠性。

这个三级路由机制已经重新纳入主工作流，不只是概念保留，而是实际的代码路径。

## 2. 当前整体架构

当前项目的主流程如下：

1. `rewrite_query`
2. `LLM Query Router`
3. 三路分流
   - `No Retrieval -> generate -> evaluate`
   - `Single-Step -> retrieve -> grade_documents -> generate -> evaluate`
   - `Multi-Hop -> retrieve -> grade_documents -> retrieve/websearch/generate -> evaluate -> retry`

其中：

- `No Retrieval` 保留了直答能力
- `Single-Step` 保留了单步精准检索
- `Multi-Hop` 保留了复杂问题闭环推理

因此，这个项目当前不是一个普通的检索增强生成 demo，而是一个带有 `动态路由策略层` 的 agentic RAG。

## 3. 已实现的关键能力

### 3.1 LLM Query Router 三级动态分流

当前状态：

- 已实现

模块位置：

- 路由策略定义：`graph/consts.py`
- 路由判断：`graph/chains/router.py`
- 路由结果写入 state：`graph/nodes/rewrite_query.py`
- 图结构分流：`graph/graph.py`
- single-step 与 multi-hop 分化：`graph/nodes/grade_documents.py`
- no-retrieval 失败后升级为 single-step：`graph/nodes/evaluate_generation.py`
- single-step 失败后升级为 multi-hop：`graph/nodes/evaluate_generation.py`

作用说明：

- 这是系统智能化的核心。
- 它让系统能根据问题复杂度，选择最合适的成本与可靠性平衡点。
- 同时保留了强叙事性：
  - 简单问题快答
  - 简单事实单步检索
  - 复杂医学问题进入多节点闭环

### 3.2 HyDE 辅助检索

当前状态：

- 已实现

模块位置：

- HyDE 链：`graph/chains/hyde.py`
- 检索节点接入：`graph/nodes/retrieve.py`

作用说明：

- 对短 query、模糊 query、医学术语不完整 query，生成假设性文档来扩大召回空间。
- 这对症状描述类和缩写类 query 特别重要。

### 3.3 单轮混合检索：BM25 + BGE-M3 + RRF

当前状态：

- 已实现

模块位置：

- 本地向量库与检索基座：`ingestion.py`
- 稠密 embedding：`graph/embeddings/bge_m3.py`
- embedding 接入：`model.py`

作用说明：

- `BM25` 负责关键词、医学术语、缩写匹配。
- `BGE-M3` 负责语义召回。
- `RRF` 融合两路召回结果，降低单一检索路径的偏置。

### 3.4 ChromaDB 本地向量库

当前状态：

- 已实现

模块位置：

- `ingestion.py`

作用说明：

- 提供本地持久化向量存储。
- 支持不同 corpus profile 的独立 collection。
- 与当前混合检索和后续 query tower LoRA 推理保持兼容。

### 3.5 多轮信息缺口分析与自适应重试

当前状态：

- 已实现

模块位置：

- 信息缺口分析：`graph/chains/gap_analyzer.py`
- 文档打分与下一步决策：`graph/nodes/grade_documents.py`
- 生成后评估：`graph/evaluation.py`
- 生成后升级与重试：`graph/nodes/evaluate_generation.py`

作用说明：

- 系统不会因为第一次检索不足就立即失败。
- 对于 multi-hop 问题，会先分析是否应该：
  - 再做一轮本地检索
  - 直接联网搜索
  - 或直接生成

### 3.6 Tavily API 网络搜索与医学白名单

当前状态：

- 已实现

模块位置：

- 搜索 provider：`graph/search/providers.py`
- 工具注册：`graph/mcp/registry.py`
- 搜索节点：`graph/nodes/web_search.py`

当前白名单包括：

- `medlineplus.gov`
- `nih.gov`
- `ncbi.nlm.nih.gov`
- `pubmed.ncbi.nlm.nih.gov`
- `cdc.gov`
- `who.int`
- `mayoclinic.org`

作用说明：

- 所有网络搜索统一以 `Tavily API` 为主入口。
- 医学场景下优先限制在高可信医学来源。
- 这能显著降低噪声来源和错误证据。

### 3.7 PubMed / 医学网页 / 通用网页三类搜索能力

当前状态：

- 已实现

模块位置：

- `graph/search/providers.py`
- `graph/mcp/registry.py`

当前工具包括：

- `search_web_general`
- `search_medical_web`
- `search_pubmed`

作用说明：

- 对外部知识源做结构化区分。
- 为后续 source policy、citation 机制和高风险问题保守回答打基础。

### 3.8 MCP Client 抽象层

当前状态：

- 已实现第一阶段 client 抽象

模块位置：

- `graph/mcp/client.py`
- `graph/mcp/registry.py`

作用说明：

- 当前还不是完整的分布式 MCP server 方案，但已经把工具调用抽象成统一边界。
- 后续如果切换到真正的 MCP transport，不需要重写 graph 主逻辑。

### 3.9 RAGAS Faithfulness 优先评估

当前状态：

- 已实现

模块位置：

- `graph/evaluation.py`

作用说明：

- 优先使用 `RAGAS faithfulness` 评估 grounding。
- 当本地环境不具备 ragas 运行条件时，才回退到 LLM grader。

### 3.10 BEIR nfcorpus 检索评测骨架

当前状态：

- 已补齐脚本骨架

模块位置：

- `scripts/run_beir_nfcorpus_eval.py`
- `graph/retrieval_metrics.py`

作用说明：

- 为医学域检索建立标准 benchmark 入口。
- 后续可以真实统计 `Recall@5`、`NDCG@10`、`MRR@10`。

### 3.11 Synthetic Test Set 生成评测骨架

当前状态：

- 已补齐模板与脚手架

模块位置：

- synthetic 数据模板：`data/eval/synthetic_generation_eval_template.jsonl`
- 评测脚本：`scripts/evaluate_synthetic_generation.py`
- 指标函数：`graph/generation_metrics.py`

作用说明：

- 用于评估生成阶段的质量。
- 为 `ROUGE-L` 和 `BERTScore` 提供可重复的评测载体。

### 3.12 BGE-M3 Query Tower LoRA 训练方案与入口

当前状态：

- 已落地训练方案、训练脚本和数据格式
- 尚未在当前环境完成真实训练

模块位置：

- 训练脚本：`scripts/train_query_tower_lora.py`
- 训练数据模板：`data/train/query_tower_lora_template.jsonl`
- 配置示例：`configs/query_tower_lora.example.json`
- LoRA 推理接入：`graph/embeddings/bge_m3.py`
- 环境变量入口：`model.py`

作用说明：

- 第二阶段重点增强 query 侧对医学语料空间的适配能力。
- 保持 doc tower 稳定，先做 query tower 的低成本可解释增强。

当前训练目标已经明确为 `InfoNCE 对比学习`，不是普通分类微调。

训练输入：

- `query`
- `correct document`
- `confusable negative document`

训练输出：

- `query embedding`
- `positive document embedding`
- `negative document embedding`

训练目标：

- 让模型在医学易混淆知识之间拉开距离。
- 让 `query` 更接近正确文档，远离易混淆错误文档。

## 4. 简历点逐条对照

下面按你的简历描述逐条核对当前项目状态。

### 4.1 基于 LLM 的 Query Router 动态路由机制

当前状态：

- 已实现并保留

对应模块：

- `graph/chains/router.py`
- `graph/nodes/rewrite_query.py`
- `graph/graph.py`
- `graph/nodes/evaluate_generation.py`

说明：

- 当前路由明确支持：
  - `No Retrieval`
  - `Single-Step`
  - `Multi-Hop`

### 4.2 针对短/模糊 Query 使用 HyDE

当前状态：

- 已实现

对应模块：

- `graph/chains/hyde.py`
- `graph/nodes/retrieve.py`

### 4.3 No Retrieval：跳过检索直接回答

当前状态：

- 已实现

对应模块：

- 路由定义：`graph/chains/router.py`
- 主图分流：`graph/graph.py`
- direct route generate：`graph/nodes/generate.py`
- no-retrieval 失败后升级：`graph/nodes/evaluate_generation.py`

说明：

- 如果 direct answer 质量不够，会自动升级到 `Single-Step`。

### 4.4 Single-Step：微调后的 BGE-M3 双塔模型做单步混合检索

当前状态：

- 已实现基础结构
- 已接入 BGE-M3 双塔 embedding
- 已落地 query tower LoRA 训练入口
- 真实微调结果仍需在本地或服务器训练后验证

对应模块：

- embedding：`graph/embeddings/bge_m3.py`
- ChromaDB + hybrid retrieval：`ingestion.py`
- LoRA 训练：`scripts/train_query_tower_lora.py`

说明：

- 当前 single-step 路线不会进入 multi-hop gap analysis，而是走单轮检索后直接生成。
- 这里的“微调后”当前已经对应到 `BGE-M3 query tower LoRA` 的训练脚手架。
- 训练方式明确为 `InfoNCE` 三元组对比学习。

### 4.5 Multi-Hop：复杂医学问题触发 LangGraph 多 Agent 推理闭环

当前状态：

- 已实现

对应模块：

- 图结构：`graph/graph.py`
- gap analysis：`graph/chains/gap_analyzer.py`
- 文档打分：`graph/nodes/grade_documents.py`
- 网络补证：`graph/nodes/web_search.py`
- 生成评估：`graph/evaluation.py`

说明：

- 当前的 multi-hop 不是多进程 agent，而是多节点职责分离的 LangGraph 闭环。
- 从工程角度看，这已经满足“多 agent 风格推理闭环”的结构要求。

### 4.6 网络搜索采用 Tavily API

当前状态：

- 已实现

对应模块：

- `graph/search/providers.py`

### 4.7 本地向量库采用 ChromaDB

当前状态：

- 已实现

对应模块：

- `ingestion.py`

### 4.8 RAGAS Faithfulness

当前状态：

- 已实现

对应模块：

- `graph/evaluation.py`

### 4.9 BEIR nfcorpus

当前状态：

- 已补齐脚手架
- 尚未实跑 benchmark 结果

对应模块：

- `scripts/run_beir_nfcorpus_eval.py`

### 4.10 Synthetic Test Set + ROUGE-L + BERTScore

当前状态：

- 已补齐模板和脚手架

对应模块：

- `data/eval/synthetic_generation_eval_template.jsonl`
- `scripts/evaluate_synthetic_generation.py`
- `graph/generation_metrics.py`

### 4.11 BGE-M3 query tower LoRA 通过对比学习 InfoNCE 微调实现

当前状态：

- 已补齐为显式 InfoNCE 三元组训练实现

对应模块：

- `scripts/train_query_tower_lora.py`
- `data/train/query_tower_lora_template.jsonl`
- `configs/query_tower_lora.example.json`

说明：

- 输入：`(query, 正确文档, 易混淆错误文档)`
- 输出：`3 个向量`
- 目标：让模型分得清医学易混淆知识

## 5. 为什么当前阶段仍然先做 LoRA，而不是直接上更复杂的 Adaptive 方案

当前更推荐的顺序是：

1. 先用当前脚手架拿到训练前 baseline
2. 再对 `BGE-M3 query tower` 做 `LoRA`
3. 再比较：
   - `Recall@5`
   - `NDCG@10`
   - `MRR@10`
   - 各类 query 切片表现
4. 如果 LoRA 提升有限，再考虑更复杂的 adaptive 方案

原因：

1. 你现在更需要“能解释提升来自哪里”，而不是“先把训练做复杂”。
2. LoRA 更适合作为第一轮低风险、可归因的实验。

## 6. 当前仍未完全实现或仍需强化的点

下面这些点还没有完全到你简历中“最终结果”的强度：

1. `BGE-M3 query tower` 还没有完成真实医学数据上的 LoRA 训练
2. `BEIR nfcorpus` 还没有产出真实 benchmark 数值
3. `Synthetic generation eval` 还没有产出真实 `ROUGE-L / BERTScore` 数值
4. 还没有单独的 `医学缩写扩展模块`
5. 还没有 `cross-encoder reranker`
6. 还没有完整的 `citation 展示机制`

也就是说：

- 结构已经到位
- 训练入口已经到位
- benchmark 脚手架已经到位
- 但真实实验结果还需要后续跑通

## 7. 当前验证状态

本轮已经完成的验证：

1. `python -m compileall .`
2. 纯逻辑 smoke test：
   - corpus profile
   - retrieval metrics
   - 路由相关状态分流逻辑

本轮还没有完成的验证：

1. BGE-M3 模型下载与真实建库
2. Tavily API 联网实跑
3. BEIR nfcorpus 联网 benchmark
4. Query Tower LoRA 实际训练

## 8. 建议的下一步

建议下一步继续按下面顺序推进：

1. 准备真实医学训练数据
2. 实跑 `scripts/train_query_tower_lora.py`
3. 实跑 `scripts/run_beir_nfcorpus_eval.py`
4. 用训练前/训练后结果对比：
   - `Single-Step` 路线提升了哪些 query
   - `No Retrieval` 升级到 `Single-Step` 的比例
   - `Single-Step` 升级到 `Multi-Hop` 的比例
5. 如果 single-step 提升稳定，再补：
   - 缩写扩展
   - reranker
   - citation
