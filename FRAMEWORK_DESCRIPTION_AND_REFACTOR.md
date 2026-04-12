# Agentic Adaptive RAG 框架说明与医学域化改造记录

## 1. 项目当前特点与基础定位

这个仓库的核心价值，不在于它已经是一个完整的医学 RAG 成品，而在于它已经具备了一个非常适合继续扩展的 `LangGraph 自适应闭环骨架`。当前项目最有价值的几个特点是：

1. 不是“检索一次、生成一次”的静态 RAG，而是有状态的闭环工作流。
2. 已经具备 `query rewrite -> retrieve -> grade -> generate -> evaluate -> retry` 的基本控制流。
3. 检索和生成之间不是硬连接，而是通过多个 grader 和条件跳转来决定下一步动作。
4. 当前代码已经具备继续扩展成“多轮检索 + 多工具协同 + 评测驱动优化”的工程基础。

这意味着它非常适合从一个通用 demo 版本，继续演进到你简历里那类“面向医学复杂查询的自适应 RAG 系统”。

换句话说，这个项目现在的价值是：

- `图结构已经对`
- `闭环已经有`
- `后续增强点很明确`

所以这轮改造的目标，不是直接把它变成最终版，而是先把它改造成一个更合理的“医学域化第一阶段底座”。

## 2. 为什么先做医学域化，而不是直接训练 Query Tower

你前面提到希望在 embed 过程里加入双塔结构，并针对 query 侧的 `BGE-M3 tower` 做医学数据集微调。这个方向是合理的，但不适合作为第一步直接落地，原因有三点：

1. 当前仓库还缺少稳定的医学语料基础。
   如果 corpus 还不是医学语料，或者医学语料组织方式还不稳定，先训练 query tower，最后很难判断收益到底来自模型，还是来自语料变化。

2. 当前仓库还缺少统一的 retrieval benchmark。
   如果没有训练前 baseline，例如 `Recall@5`、`NDCG@10`、`MRR@10`，后面即使训出一个 query tower，也很难清楚说明“为什么要训”“训前差在哪里”“提升来自哪里”。

3. 当前外部知识接入方式还没有分层。
   如果后续要接 PubMed、权威医学网页、甚至院内知识库，最好先把工具层抽象干净，再考虑更强的表示学习。

所以这一轮更合理的顺序是：

1. 先完成医学语料接入
2. 先补 retrieval eval 骨架
3. 先把外部搜索能力抽象成 MCP client 边界
4. 先建立“训练前 baseline”
5. 第二批再做 `BGE-M3 query tower LoRA`

这个顺序的好处是，后面你就能比较完整地回答：

- 为什么需要训练 query tower
- 不训练时检索错误主要发生在哪里
- 训练后到底提升了哪些 query 类型
- 是缩写问题、术语问题、症状问法问题，还是多跳问题得到改善

## 3. 医学域化方案设计

### 3.1 Corpus 设计

这一阶段的 corpus 目标不是一步到位接入超大规模医学库，而是先把项目改造成“能够稳定承载医学语料”的结构。

本轮采用两层设计：

1. `medical_demo` 语料配置
   当前先接入高可信公开医学页面，优先选择 `MedlinePlus / NIH / CDC / WHO` 一类来源，作为演示版医学 corpus。

2. 后续可扩展语料层
   后续第二阶段可以继续接入：
   - PubMed 摘要库
   - BEIR `nfcorpus`
   - 自建医学 QA 检索集
   - 指南类 PDF / HTML 知识库

这样做的原因是：

1. 先用高可信公开医学页面完成结构迁移，风险低、可运行性高。
2. 后续接入 PubMed 和 benchmark 时，不需要推翻现有 ingestion 和 retriever 结构。

对应作用：

- 让项目从“AI 博客 demo 语料”切换为“医学域可扩展语料”
- 为后续 query tower 微调准备更合理的数据环境
- 为后续医学 benchmark 和 source policy 打基础

### 3.2 Benchmark 设计

医学域化后，项目不能只看“能不能答出来”，必须明确区分 `检索质量` 和 `生成质量`。

因此建议 benchmark 分两层：

1. 检索层 benchmark
   - `Recall@5`
   - `NDCG@10`
   - `MRR@10`

2. 生成层 benchmark
   - `faithfulness`
   - `answer relevancy`
   - `citation precision`

同时再做几类专项切片：

1. 医学缩写
   例如：`COPD`、`HFpEF`、`RA`
2. 症状口语化问法
   例如：“胸闷喘不过气”“经常口渴尿多”
3. 检查项与实验室指标
   例如：`HbA1c`、`CBC`
4. 疾病分型或治疗路径
   例如：慢阻肺分级、心衰类型

这样设计的原因是：

1. 医学 query 的失败模式高度集中，不同错误类型的优化方向不同。
2. 只有把检索和生成拆开评，后面训练 query tower 时才知道收益到底落在哪一层。

对应作用：

- 让后续训练和改造可以用数据说话
- 避免“整体看着变好，但不知道为什么”
- 为论文式/简历式表述提供可复用指标

### 3.3 Query Tower 训练策略

这部分属于第二批改造，但这次已经先把设计方案明确下来。

#### 为什么是 Query Tower，而不是一开始同时训练双塔

因为你的目标更像是“让用户 query 更好对齐医学表达空间”，而不是一开始就重做整个向量库。医学场景里，最常见的问题往往是：

1. 用户使用口语表达，文档使用标准医学术语
2. 用户使用缩写，语料使用全称
3. 用户 query 很短、模糊、关键词不足

这些问题首先影响的是 `query side representation`，所以第一步更适合先做 query tower 适配。

#### 为什么第二阶段建议先用 LoRA

当前更建议：

- 第一版先用 `LoRA`
- 暂时不直接上更复杂的 adaptive 方案

原因：

1. `LoRA` 更稳，实验可复现性更好。
2. 工程接入成本更低，更适合当前项目阶段。
3. 更容易和 baseline 对比，清楚说明“训练带来了什么变化”。
4. 当前项目的主要问题还不在参数规模不足，而在语料、工具和评测闭环还没补齐。

#### 什么情况下再考虑 adaptive

当下面条件同时出现时，再考虑 `AdaLoRA` 或其它 adaptive 方案会更合理：

1. 已经有稳定 baseline
2. LoRA 提升有限
3. 已经确认收益瓶颈来自表示学习，而不是 reranker、source policy 或 query rewrite

#### 第二批训练建议

1. 先冻结 doc tower
2. 只对 query tower 做 `LoRA`
3. 训练数据优先使用：
   - 正样本：医学 query 与 gold evidence
   - 负样本：BM25 hard negatives + dense hard negatives
4. 训练后重新评估：
   - `Recall@5`
   - `NDCG@10`
   - query 类型切片表现

对应作用：

- 先用最小训练改动换取最可解释的提升
- 避免过早把系统复杂度堆高
- 为后续是否继续 joint tuning 提供依据

### 3.4 MCP 接入边界设计

MCP 这部分本轮不追求“一步到位的完整分布式协议系统”，而是先做清楚边界。

本轮的设计原则是：

1. 先做 `MCP client 抽象层`
2. 先把外部工具调用统一走 client
3. 暂时不把整个项目暴露成 MCP server

本轮纳入 MCP 边界的工具包括：

1. `search_web_general`
   通用网页搜索
2. `search_medical_web`
   限定在高可信医学站点的网页搜索
3. `search_pubmed`
   PubMed 文献搜索

这一阶段刻意不做的部分：

1. 不把本地 hybrid retriever 暴露成 MCP server
2. 不把 generation/evaluation 节点对外 server 化
3. 不引入更复杂的远程 transport 和多进程调度

这样做的原因是：

1. 当前最重要的是先把工具层从 workflow 中解耦。
2. 只要 client 边界先建立起来，后面切换成真正的 MCP transport 就不会动到 graph 的上层逻辑。
3. 对简历叙事也更清楚：先做“可扩展工具调用抽象”，再做“协议级平台化”。

对应作用：

- 降低 workflow 与外部工具的耦合
- 让 PubMed / 医学网页 / 通用 web 搜索走统一调用接口
- 为后续真实 MCP server 接入预留结构位置

## 4. 第一批改造实施记录

这一轮已经按上面的设计，完成了第一批工程改造。

### 4.1 医学语料接入

新增或修改：

- `graph/corpus_profiles.py`
- `ingestion.py`

本轮做了什么：

1. 引入 `corpus profile` 机制
2. 增加 `medical_demo` 语料配置
3. 将持久化向量库改成按 profile 组织
4. 默认激活医学 profile，而不是继续只使用 AI 博客语料

为什么这样改：

1. 让项目从结构上具备“多领域/多语料配置”的能力
2. 避免后续换医学语料时推翻整个 ingestion 逻辑

对应作用：

- 让当前项目正式进入“医学域版本”
- 为后续 benchmark 和 query tower 训练提供语料基础

### 4.2 MCP Client 抽象层

新增：

- `graph/mcp/client.py`
- `graph/mcp/registry.py`

本轮做了什么：

1. 建立了一个轻量的 `InProcessMCPClient`
2. 用统一的 tool register / call 接口管理搜索工具
3. 让 graph 上层只关心“调用工具”，不关心具体 provider 是谁

为什么这样改：

1. 当前最需要的是工具调用边界，而不是复杂的协议实现细节
2. 先把依赖倒置做对，后面接真正的 MCP server 才会顺

对应作用：

- workflow 不再直接绑定单一搜索实现
- 搜索工具可以继续扩展，而不需要频繁修改 graph 节点

### 4.3 PubMed 与医学搜索接入

新增或修改：

- `graph/search/providers.py`
- `graph/nodes/web_search.py`

本轮做了什么：

1. 增加 `PubMedSearchProvider`
2. 增加 `MedicalWebSearchProvider`
3. 将原来写死的 Tavily web search，改成通过 MCP client 调度：
   - PubMed
   - 医学权威网页搜索
   - 通用网页搜索

为什么这样改：

1. 医学场景不能继续只依赖开放域 web search
2. 需要先让高可信来源进入工具层

对应作用：

- 提高医学场景下外部知识补充的可信度
- 为未来加入 source policy 和 citation 机制打基础

### 4.4 Retrieval Eval 脚本骨架

新增：

- `graph/retrieval_metrics.py`
- `scripts/retrieval_eval.py`
- `data/eval/medical_retrieval_eval_template.jsonl`

本轮做了什么：

1. 增加检索评价指标函数：
   - `Recall@k`
   - `NDCG@k`
   - `MRR@k`
2. 增加 evaluation 脚本骨架
3. 增加医学检索模板数据格式

为什么这样改：

1. 不先建立检索基线，后面训练 query tower 的收益就说不清楚
2. 评测脚本越早落地，后续每一轮改造越容易归因

对应作用：

- 为“训练前 baseline”提供第一版载体
- 让第二批 LoRA 训练有明确对照组

### 4.5 路由与改写提示词同步医学化

修改：

- `graph/chains/router.py`
- `graph/chains/query_rewriter.py`
- `graph/nodes/rewrite_query.py`
- `graph/state.py`

本轮做了什么：

1. router 现在会基于当前 corpus profile 理解本地知识库范围
2. query rewrite prompt 会显式保留医学实体、缩写、检查项、药名
3. state 中增加了：
   - `corpus_profile`
   - `preferred_search_tools`

为什么这样改：

1. 如果语料已经医学化，而 router 和 rewrite 还是通用 AI 语境，流程会失真

对应作用：

- 让 query rewrite、route、web search 三者围绕同一领域配置工作

## 5. 第二批改造计划：Query Tower LoRA

等第一批改造跑通并拿到 baseline 后，第二批建议按下面顺序做：

1. 准备医学 retrieval benchmark
2. 统计 baseline：
   - `Recall@5`
   - `NDCG@10`
   - `MRR@10`
3. 对 `BGE-M3 query tower` 做 `LoRA`
4. 训练数据使用：
   - 正样本：query 与 gold evidence
   - 负样本：BM25 hard negatives + dense hard negatives
5. 再做训练后对比

这样设计的价值是：

1. 能明确回答“为什么训练”
2. 能明确回答“训练前到底差在哪里”
3. 能明确回答“训练后提升来自哪类 query”

这也是为什么当前阶段不直接上 adaptive 方案，而是建议 `LoRA first`。

## 6. 目前仍需继续精进的点

除了这轮已经做的内容，我认为后续还值得优先推进的点包括：

1. `reranker`
   很多时候，`hybrid retrieval + reranker` 的收益比直接训练 embedding 更稳定。

2. `source policy`
   医学场景下应该明确区分：
   - PubMed
   - 指南/权威站点
   - 普通网页

3. `citation 机制`
   医学 RAG 不能只给答案，最好能给出证据来源和证据粒度。

4. `医学缩写扩展`
   rewrite 阶段可以进一步加入缩写展开和标准术语映射。

5. `高风险问题保守回答`
   对诊断、治疗、用药问题，应该增加更保守的回答策略。

## 7. 当前验证状态

本轮已经完成的是：

1. 医学域化方案设计
2. 医学语料 profile 接入
3. MCP client 抽象层
4. PubMed / 医学搜索 provider
5. retrieval eval 脚本骨架

当前仍受本地环境限制的部分：

1. 本地尚未完整安装运行依赖
2. 网络环境无法稳定连接 GitHub
3. PubMed / Tavily 搜索属于联网能力，当前未做在线实跑验证

建议后续验证顺序：

1. 安装依赖
2. 配置 `.env`
3. 执行 `python ingestion.py`
4. 执行 `python main.py`
5. 执行 `python scripts/retrieval_eval.py --dataset data/eval/medical_retrieval_eval_template.jsonl`
6. 在 baseline 稳定后，再开始第二批 query tower LoRA
