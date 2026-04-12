# Agentic Adaptive RAG 框架说明、医学域化设计与简历点对照

## 1. 项目定位

这个项目当前的目标，不再是一个通用 demo，而是逐步演进为一个更贴近你简历描述的 `医学域自适应 RAG 系统`。它的核心思路是：

1. 用 `LangGraph` 组织多节点闭环，而不是单次检索生成。
2. 用 `混合检索 + 多层评估 + 自适应重试` 来提升答案准确性和相关性。
3. 在医学场景下，优先保证：
   - 检索来源可信
   - query 对医学术语空间的对齐能力
   - 生成答案的可验证性

因此，本项目当前分成两批推进：

1. 第一批：医学域底座搭建
   - 医学语料接入
   - `ChromaDB` 本地向量库
   - `BM25 + BGE-M3 + RRF`
   - Tavily 网络搜索和医学白名单
   - MCP client 抽象层
   - retrieval / generation eval 脚手架

2. 第二批：表示学习增强
   - `BGE-M3 query tower LoRA`
   - 训练数据格式
   - baseline 对比与提升归因

## 2. 当前整体架构

当前工作流主线如下：

1. `rewrite_query`
2. `route`
3. `retrieve`
4. `grade_documents`
5. `retrieve / websearch / generate`
6. `evaluate_generation`
7. `rewrite / regenerate / websearch / end`

其中，第一轮检索已经对齐到：

- `BM25 稀疏检索`
- `BGE-M3 稠密检索`
- `RRF 融合`
- `HyDE 查询扩展`

多轮补检和网络搜索也已经具备：

- LLM 信息缺口分析
- 子查询补全
- Tavily 搜索回退
- 医学站点白名单控制

## 3. 简历描述逐点对照与模块映射

这一节按你的简历描述逐条检查，说明是否实现、对应模块在哪里、这一步的作用是什么。

### 3.1 通过混合检索、多层评估和自适应重试机制提高 RAG 系统的准确性和相关性

实现说明：

- 已实现。
- 当前系统已经是一个闭环式的 adaptive RAG，而不是单步流水线。

模块位置：

- 工作流编排：`graph/graph.py`
- 共享状态：`graph/state.py`
- 生成评估：`graph/evaluation.py`
- 生成重试：`graph/nodes/evaluate_generation.py`

对应作用：

- 让系统在答案不可靠时继续补证据或重写问题。
- 将“相关性”和“真实性”纳入控制流，而不是只看一次生成结果。

### 3.2 首轮采用 RRF 融合 BM25 稀疏检索与 BGE-M3 稠密向量检索

实现说明：

- 已实现。
- 本地向量库使用 `ChromaDB`。
- 稀疏检索使用 `BM25Retriever`。
- 稠密检索使用 `BGE-M3`。
- 两路召回通过 `RRF` 融合。

模块位置：

- 稠密模型：`model.py`
- BGE-M3 双塔 embedding 封装：`graph/embeddings/bge_m3.py`
- ChromaDB 建库与混合检索：`ingestion.py`

对应作用：

- `BM25` 擅长关键词和医学术语匹配。
- `BGE-M3` 更擅长语义对齐和口语 query。
- `RRF` 融合可以降低单一路径召回不稳定的问题。

### 3.3 针对短 / 模糊 Query 采用 HyDE 方法辅助检索

实现说明：

- 已实现。

模块位置：

- HyDE 生成链：`graph/chains/hyde.py`
- 检索节点中接入：`graph/nodes/retrieve.py`

对应作用：

- 当 query 太短、缺少关键词、医学实体不完整时，用假设性回答扩展检索空间。
- 这对症状描述类 query 尤其有帮助。

### 3.4 次轮由 LLM 自动识别信息缺口并生成子查询定向补全

实现说明：

- 已实现。

模块位置：

- 信息缺口分析：`graph/chains/gap_analyzer.py`
- 文档分级和二轮检索决策：`graph/nodes/grade_documents.py`

对应作用：

- 不是检索失败后直接联网，而是先判断是否还能通过第二轮本地检索补齐证据。
- 这可以减少无效联网，提高闭环的可解释性。

### 3.5 多轮未达标时触发网络搜索模块保证答案准确性

实现说明：

- 已实现。
- 网络搜索统一通过 `Tavily API` 接入。

模块位置：

- 网络搜索节点：`graph/nodes/web_search.py`
- MCP 工具注册：`graph/mcp/registry.py`
- Tavily provider 与医学白名单：`graph/search/providers.py`

对应作用：

- 当本地知识不足时，用外部信息补充。
- 保证网络搜索不是完全开放域，而是有医学域名白名单约束。

### 3.6 Multi-Agent：基于 LangGraph 实现问题改写、网络搜索、结果生成、幻觉检查和结果相关性检验模块

实现说明：

- 已实现。
- 当前虽然没有使用多个独立进程 agent，但在 `LangGraph` 中已经实现了多节点、分职责的 agentic workflow。

模块位置：

- 问题改写：`graph/nodes/rewrite_query.py`
- 网络搜索：`graph/nodes/web_search.py`
- 结果生成：`graph/nodes/generate.py`
- 幻觉评估：`graph/evaluation.py`
- 结果相关性检验：`graph/chains/answer_grader.py`
- 总图结构：`graph/graph.py`

对应作用：

- 让每个模块的职责单独可控、可替换、可评估。
- 这比把所有判断都塞进一个 prompt 更适合工程化扩展。

### 3.7 未通过则触发改写重试闭环

实现说明：

- 已实现。

模块位置：

- 闭环入口和重试路由：`graph/graph.py`
- 生成后决策：`graph/nodes/evaluate_generation.py`

对应作用：

- 当答案虽然有内容，但没真正回答问题时，系统不是原地重复生成，而是先重写问题后重试。

### 3.8 引入 RAGAS faithfulness 指标替代 LLM 自评，量化幻觉率

实现说明：

- 已补齐。
- 当前评估逻辑已经优先使用 `RAGAS faithfulness`。
- 仅在本地环境没有 `ragas` 或运行失败时，才回退到 LLM grader。

模块位置：

- 主评估逻辑：`graph/evaluation.py`

对应作用：

- 降低完全依赖 LLM 自评带来的不稳定性。
- 让幻觉率评估更接近标准化指标。

### 3.9 BEIR nfcorpus：基于复杂医学类数据库做验证

实现说明：

- 已补充脚本骨架。
- 当前还没有在本地环境完成在线实跑，但已具备评估入口。

模块位置：

- BEIR nfcorpus 评测脚本：`scripts/run_beir_nfcorpus_eval.py`
- 检索指标函数：`graph/retrieval_metrics.py`

对应作用：

- 为医学域检索阶段提供标准 benchmark。
- 后续可以真实统计 `Recall@5`、`NDCG@10`、`MRR@10`。

### 3.10 独立构建 Synthetic Test Set 用于评估答案生成质量

实现说明：

- 已补充模板和评测脚本骨架。

模块位置：

- Synthetic 数据模板：`data/eval/synthetic_generation_eval_template.jsonl`
- 生成评测脚本：`scripts/evaluate_synthetic_generation.py`
- 生成指标函数：`graph/generation_metrics.py`

对应作用：

- 将生成阶段评估从“主观感觉”转成可批量统计的指标。
- 为后续 ROUGE-L / BERTScore 对比提供载体。

### 3.11 生成阶段 ROUGE-L / BERTScore

实现说明：

- 已补充指标计算骨架。

模块位置：

- 指标函数：`graph/generation_metrics.py`
- 评估脚本：`scripts/evaluate_synthetic_generation.py`

对应作用：

- 让生成质量可以量化比较。
- 为后续和不同 prompt / retriever / tower 版本做对照提供基础。

### 3.12 Query Tower 医学微调

实现说明：

- 已补充第二阶段训练方案、训练脚本和数据格式。
- 目前属于“已落地方案与训练入口，待你准备医学训练数据后启动训练”。

模块位置：

- Query tower LoRA 训练脚本：`scripts/train_query_tower_lora.py`
- 训练数据模板：`data/train/query_tower_lora_template.jsonl`
- LoRA 推理接入口：`graph/embeddings/bge_m3.py`
- 运行时 adapter 入口：`model.py` 中 `QUERY_TOWER_ADAPTER_PATH`
- 训练配置示例：`configs/query_tower_lora.example.json`

对应作用：

- 用最小改动先提升 query 侧对医学语料空间的适配能力。
- 不必一开始就重建整个 doc tower 和向量库。

### 3.13 本地向量库用 ChromaDB

实现说明：

- 已实现。

模块位置：

- `ingestion.py`

对应作用：

- 提供本地持久化向量存储。
- 支持后续不同语料 profile 的独立 collection 和持久化目录。

### 3.14 网络搜索接口用 Tavily API，并加入医学域名白名单

实现说明：

- 已实现。
- 当前通用搜索和医学域搜索都走 `Tavily API`。
- 医学搜索加入白名单。

模块位置：

- Tavily 搜索 provider：`graph/search/providers.py`
- 白名单：
  - `MEDICAL_DOMAIN_WHITELIST`
  - `PUBMED_DOMAIN_WHITELIST`

当前白名单包括：

- `medlineplus.gov`
- `nih.gov`
- `ncbi.nlm.nih.gov`
- `pubmed.ncbi.nlm.nih.gov`
- `cdc.gov`
- `who.int`
- `mayoclinic.org`

对应作用：

- 降低医学场景下开放网页噪声。
- 提升检索来源可信度。

## 4. 第一阶段医学域化改造已经完成的内容

### 4.1 医学语料 Profile

已完成：

- `medical_demo` profile
- 医学语料默认激活
- 不同 profile 独立 Chroma collection

模块位置：

- `graph/corpus_profiles.py`
- `ingestion.py`

### 4.2 MCP Client 抽象

已完成：

- 轻量 `InProcessMCPClient`
- 统一 tool register / call

模块位置：

- `graph/mcp/client.py`
- `graph/mcp/registry.py`

### 4.3 PubMed / 医学网页 / 通用网页搜索

已完成：

- `search_pubmed`
- `search_medical_web`
- `search_web_general`

模块位置：

- `graph/search/providers.py`
- `graph/mcp/registry.py`
- `graph/nodes/web_search.py`

### 4.4 Retrieval Eval 骨架

已完成：

- `Recall@k`
- `NDCG@k`
- `MRR@k`
- JSONL 模板
- CLI 脚本

模块位置：

- `graph/retrieval_metrics.py`
- `scripts/retrieval_eval.py`

## 5. 第二阶段：BGE-M3 Query Tower LoRA 训练方案

### 5.1 为什么是 Query Tower LoRA

当前医学检索最可能出现的问题不是“文档库完全没有相关内容”，而是：

1. query 太口语化
2. 医学缩写与正文术语不对齐
3. 检查项、药名、疾病分型表达不一致

这些问题首先体现在 `query embedding` 质量上，因此第二阶段更适合先做：

- 保持 doc tower 稳定
- 只训练 query tower
- 用 `LoRA` 先做低成本可解释增强

### 5.2 训练数据格式

当前训练数据模板路径：

- `data/train/query_tower_lora_template.jsonl`

字段设计：

- `query`
- `positive_passage`
- `hard_negative_passages`
- `query_type`
- `medical_entities`
- `source`
- `split`

这样设计的原因：

1. `positive_passage` 直接提供正样本证据。
2. `hard_negative_passages` 用于构造对比学习难负样本。
3. `query_type` 方便后续做训练后分桶分析。
4. `medical_entities` 方便分析哪些实体类型最受益。

### 5.3 训练脚本

当前训练入口：

- `scripts/train_query_tower_lora.py`

当前实现方式：

1. doc tower 冻结
2. query tower 加 LoRA
3. query 与候选段落做对比打分
4. 正样本放在候选第一位
5. 用交叉熵训练 query tower 选择正样本

### 5.4 推理接入方式

当前推理接入口：

- `graph/embeddings/bge_m3.py`
- `model.py`

方式：

1. 文档 embedding 使用基础 BGE-M3 编码器
2. query embedding 可加载 `QUERY_TOWER_ADAPTER_PATH`
3. 通过同一 embedding 接口接入 ChromaDB 查询

这意味着后续训练完成后，不需要改 graph 主流程，只要：

1. 保存 adapter
2. 设置 `QUERY_TOWER_ADAPTER_PATH`

就能直接对比训练前后的检索结果。

## 6. 为什么当前阶段不直接上 Adaptive LoRA

当前更推荐先 `LoRA`，再考虑更复杂的 adaptive 方案，原因是：

1. 现在最重要的是建立训练前 baseline
2. 现在更缺的是评测闭环，而不是参数量
3. LoRA 更适合作为第一轮可解释对比实验

只有在下面条件同时满足后，再考虑 adaptive 更合理：

1. baseline 已稳定
2. LoRA 提升有限
3. 已确认瓶颈来自表示学习，而不是 reranker 或 source policy

## 7. 仍需继续精进的点

后续建议继续推进：

1. `reranker`
   - 当前还没有 cross-encoder reranker
   - 加入后往往能比纯 embedding 微调更稳定

2. `医学缩写扩展`
   - 当前 rewrite 已保留医学实体，但还没有专门的缩写展开模块

3. `citation 机制`
   - 当前答案生成和 grounding 已有，但证据展示粒度还可以继续细化

4. `高风险问题保守回答`
   - 医学治疗和诊断问题还可以增加更严格的安全策略

## 8. 当前验证状态

本轮已完成的离线验证：

1. `python -m compileall .`
2. 纯逻辑 smoke test：
   - corpus profile
   - retrieval metrics

本轮尚未完成的在线验证：

1. BGE-M3 模型下载与建库实跑
2. Tavily API 联网实跑
3. BEIR nfcorpus benchmark 联网执行
4. query tower LoRA 实际训练

建议下一步验证顺序：

1. 安装 `requirements.txt`
2. 配置 `.env`
3. 执行 `python ingestion.py`
4. 执行 `python main.py`
5. 执行 `python scripts/retrieval_eval.py --dataset data/eval/medical_retrieval_eval_template.jsonl`
6. 执行 `python scripts/run_beir_nfcorpus_eval.py`
7. 准备训练集后执行 `python scripts/train_query_tower_lora.py`
