# 医学域自适应 RAG 面试深挖 Q&A

## 1. 使用说明

这份文档不是给你“照着念”的，而是给你在面试时快速组织回答结构用的。

建议每个问题都按下面这个顺序回答：

1. 先给一句话结论
2. 再讲设计动机
3. 再讲工程实现
4. 最后指出代码位置

如果面试官继续追问，你就顺着“为什么这么设计”“失败会怎样”“和别的方案比有什么取舍”往下讲。

---

## 2. 项目总览类问题

### Q1. 你这个项目一句话怎么介绍？

推荐短答：

这是一个基于 LangGraph 的医学域自适应 RAG 系统，核心不是“先检索再生成”，而是通过三级动态路由、混合检索、风险守卫、多轮补证和 Query Tower LoRA 训练闭环，把医学问答做成一个更可靠的 agentic workflow。

展开回答：

- 它保留了 `No Retrieval / Single-Step / Multi-Hop` 三种路径。
- 简单低风险问题可以快速直答。
- 一般医学事实问题走单轮混合检索。
- 复杂或高风险问题进入多跳补证和重试闭环。
- 在工程上，我还补了医疗安全边界、版本化语料治理、MCP 接入边界，以及训练前后 baseline 对比链路。

代码位置：

- 图主流程：`graph/graph.py` 中的 `workflow`、`app`
- 路由与改写入口：`graph/nodes/rewrite_query.py` 中的 `rewrite_query`
- 项目文档总说明：`FRAMEWORK_DESCRIPTION_AND_REFACTOR.md`

### Q2. 这个项目和普通 RAG 最大的区别是什么？

推荐短答：

普通 RAG 是固定“检索再生成”，我的系统是“先判断问题复杂度和风险，再决定是否检索、检索几轮、要不要联网、失败后怎么升级”。

展开回答：

- 系统不是所有问题都走同一条路径。
- 它有显式的控制流，不是简单链式调用。
- 它会记录路由历史、重试次数、检索轮次、风险等级和评估结果。
- 所以它本质上是一个“策略层 + 证据层 + 控制层 + 评测层”组成的闭环系统。

代码位置：

- 图条件分流：`graph/graph.py` 中的 `route_after_rewrite`、`route_after_document_grading`、`route_after_evaluation`
- 状态定义：`graph/state.py` 中的 `GraphState`

### Q3. 为什么你会选 LangGraph，而不是普通 LangChain chain？

推荐短答：

因为这个项目不是单步链式处理，而是明确需要状态、条件分支、失败升级和多轮重试，LangGraph 更适合表达这种控制流。

展开回答：

- 我需要显式的节点和边，而不是隐式串联。
- 我需要把 `rewrite -> route -> retrieve -> grade -> generate -> evaluate` 变成可控的状态机。
- 我还需要在失败时从 `No Retrieval` 升级到 `Single-Step`，从 `Single-Step` 升级到 `Multi-Hop`，这在图结构里更自然。

代码位置：

- 图构建：`graph/graph.py`
- 共享状态：`graph/state.py`

---

## 3. 路由与闭环类问题

### Q4. 为什么要做三级动态路由，而不是统一都走检索？

推荐短答：

因为医学问答的复杂度差异非常大，统一都走检索会浪费时延和成本，但统一直答又不安全，所以我做了三级分流。

展开回答：

- `No Retrieval` 用来处理低风险定义型问题。
- `Single-Step` 用来解决大部分简单事实型问题。
- `Multi-Hop` 用来兜底复杂、高风险、多实体、多证据问题。
- 这个设计的目标是做“成本和可靠性的平衡”，而不是无脑堆最重路径。

代码位置：

- 路由提示词：`graph/chains/router.py` 中的 `RouteQuery`、`question_router`
- 图分流：`graph/graph.py`
- 路由写入状态：`graph/nodes/rewrite_query.py`

### Q5. No Retrieval 在医学场景不是很危险吗？为什么还保留？

推荐短答：

我保留它是为了系统效率，但它只允许低风险问题使用，而且外面套了一层风险守卫，不是让模型随便直答。

展开回答：

- 定义类、术语解释类问题可以直答，比如 “HbA1c 是什么”。
- 但只要涉及诊断、治疗、药物联用、剂量、禁忌、特殊人群、急症信号，就不允许停留在 `No Retrieval`。
- 所以 `No Retrieval` 不是裸奔的，它是“受风险守卫约束的低成本路径”。

代码位置：

- 风险守卫：`graph/risk_guardrails.py` 中的 `assess_medical_risk`
- 路由修正：`graph/nodes/rewrite_query.py`
- 失败升级：`graph/nodes/evaluate_generation.py`

### Q6. 既然已经有 Query Router，为什么还要额外做风险路由层？

推荐短答：

因为 Query Router 是复杂度判断，不等于安全判断；医学场景里有些问题语言形式很简单，但业务风险很高。

展开回答：

- 比如“华法林和阿司匹林能不能联用”，形式上像个简单问句。
- 但它属于高风险药物联用问题，不应该让模型按低成本直答处理。
- 所以我把“复杂度路由”和“安全约束”拆开，风险守卫只做一件事：高风险问题禁止 `No Retrieval`。

代码位置：

- 风险模式库与判断逻辑：`graph/risk_guardrails.py`
- 风险结果写入状态：`graph/nodes/rewrite_query.py`

### Q7. 你的风险守卫具体识别哪些问题？

推荐短答：

它识别药物联用、剂量、禁忌、诊断和治疗决策、特殊人群、急症症状、高风险药物、多实体复杂语境这些信号。

展开回答：

- 这不是靠一个单一关键词，而是一个信号库。
- 除了明确模式匹配，还会统计疾病实体和药物实体数量。
- 如果一个问题中同时出现多个疾病或多个药物，也会提高风险等级。

代码位置：

- 风险信号库：`graph/risk_guardrails.py` 中的 `HIGH_RISK_SIGNAL_LIBRARY`
- 医疗实体计数：`graph/risk_guardrails.py` 中的 `MEDICAL_ENTITY_PATTERNS`、`MEDICATION_ENTITY_PATTERNS`
- 总入口：`graph/risk_guardrails.py` 中的 `assess_medical_risk`

### Q8. 系统失败后是怎么升级路线的？

推荐短答：

我把升级写成了显式控制逻辑，不是靠模型自由发挥。

展开回答：

- 如果 `No Retrieval` 没答到点，就强制升级成 `Single-Step`。
- 如果 `Single-Step` 不够 grounded 或没回答完整，就升级成 `Multi-Hop`。
- 如果 `Multi-Hop` 还是证据不足，就继续重试或联网搜索。

代码位置：

- 升级主逻辑：`graph/nodes/evaluate_generation.py`
- 图跳转：`graph/graph.py` 中的 `route_after_evaluation`

### Q9. 你为什么要在 state 里保留 route_history？

推荐短答：

因为我要统计真实执行路径，而不是只看初始路由标签。

展开回答：

- 面试里你可以强调，这样做是为了支持“升级率评测”。
- 如果只看 router 初判，就不知道系统后面有没有被迫升级。
- 保留 `route_history` 之后，我能统计 `No Retrieval -> Single-Step` 和 `Single-Step -> Multi-Hop` 的真实比例。

代码位置：

- 状态字段：`graph/state.py`
- 写入逻辑：`graph/nodes/rewrite_query.py`
- 统计脚本：`scripts/compute_route_upgrade_stats.py`

---

## 4. 检索与证据类问题

### Q10. 为什么不用纯向量检索，而是 BM25 + BGE-M3 + RRF？

推荐短答：

因为医学场景既有严格术语和缩写匹配问题，也有语义改写问题，稀疏和稠密召回各有优势，RRF 用来做稳健融合。

展开回答：

- `BM25` 擅长术语、缩写、药名、检查项的关键词匹配。
- `BGE-M3` 擅长语义表达和问法变化。
- 单独依赖哪一边都容易偏。
- `RRF` 的价值是降低单路召回偏置，而不是手工调一个复杂融合权重。

代码位置：

- 混合检索实现：`ingestion.py` 中的 `HybridRetriever`
- RRF 融合：`ingestion.py` 中的 `_reciprocal_rank_fusion`
- 调用入口：`graph/nodes/retrieve.py`

### Q11. 为什么选 ChromaDB？

推荐短答：

因为这个项目当前目标是快速搭建可本地持久化、可版本化、可对接实验脚本的向量检索基座，ChromaDB 足够轻且工程接入简单。

展开回答：

- 它方便本地持久化和按 profile/version 管理 collection。
- 我当前更关注系统闭环和实验对比，不是极致检索基础设施。
- 后续如果做更大规模语料，可以替换后端，但接口层已经分出来了。

代码位置：

- collection 与 persist dir：`ingestion.py` 中的 `_collection_name`、`_persist_directory`
- 打开向量库：`ingestion.py` 中的 `get_vectorstore`、`open_vectorstore`

### Q12. 你为什么还要再加一层 reranker？

推荐短答：

因为召回解决的是“找得到”，重排解决的是“排得准”，真正影响生成质量的是前几条证据的排序质量。

展开回答：

- 混合检索召回完后，候选里通常会混进语义近但不够准的文档。
- 如果不重排，后面的 LLM 很容易拿错证据。
- 我当前用 BGE-M3 做 bi-encoder 风格重排，先用低成本方式把前排质量提上来。

代码位置：

- 重排器：`graph/rerankers/bge_m3.py` 中的 `BGEM3Reranker`
- 接入点：`graph/nodes/retrieve.py`
- 分数使用逻辑：`graph/nodes/grade_documents.py`

### Q13. 为什么当前不用 cross-encoder reranker？

推荐短答：

因为当前阶段更追求低成本、可落地、可批量实验的精排增强，bi-encoder 重排的工程性价比更高。

展开回答：

- cross-encoder 通常更准，但更重、吞吐更低。
- 当前这个项目已经有很多路径要跑，先把“可持续评测”的方案落下来更重要。
- 我把它留作下一步高 ROI 增强项，而不是第一阶段就上。

代码位置：

- 当前重排实现：`graph/rerankers/bge_m3.py`
- 后续优化方向说明：`FRAMEWORK_DESCRIPTION_AND_REFACTOR.md`

### Q14. 为什么 Single-Step 不是“LLM 全量筛文档”？

推荐短答：

因为那样既贵又慢，而且会让检索系统的收益被 LLM 吞掉，所以我改成“检索 + 重排 + 仅在不确定时少量 LLM 判别”。

展开回答：

- 如果召回完后把所有文档都交给 LLM 判别，本质上检索系统价值会被弱化。
- 我现在先看重排分数是否足够自信。
- 只有分数不够稳的时候，才对少量候选调用 LLM 过滤。
- 这样能同时兼顾效率、可解释性和训练收益观察。

代码位置：

- 自信阈值逻辑：`graph/nodes/grade_documents.py` 中的 `confident_rerank`
- LLM 过滤：`graph/nodes/grade_documents.py` 中的 `_llm_filter_documents`
- 路由分支：`graph/nodes/grade_documents.py`

### Q15. HyDE 在这里具体解决什么问题？

推荐短答：

HyDE 主要解决短查询、模糊查询、症状式查询和缩写式查询的召回不足问题。

展开回答：

- 如果用户问题里没有直接出现标准疾病名，向量召回和关键词召回都可能不稳。
- HyDE 会先生成一个“假设性证据段”，把可能相关的术语和上下文扩进去，再送去检索。
- 但它不是默认总开，因为显式疾病问题没必要多绕一圈。

代码位置：

- HyDE 提示词：`graph/chains/hyde.py`
- 是否启用：`graph/chains/query_rewriter.py` 中的 `use_hyde`
- 检索接入：`graph/nodes/retrieve.py`

### Q16. 你怎么避免多轮检索后文档重复？

推荐短答：

我在检索节点里做了按 `source + content 前缀` 的去重合并，避免多轮检索把同类文档越堆越多。

展开回答：

- 这个去重不是数据库级去重，而是 workflow 级去重。
- 它保证多轮补证时文档集合是累积但不过度重复的。
- 否则后续重排和生成的上下文会被冗余内容污染。

代码位置：

- 去重逻辑：`graph/nodes/retrieve.py` 中的 `_merge_documents`
- 检索级去重：`ingestion.py` 中的 `_dedupe_documents`

---

## 5. 提示词与生成类问题

### Q17. 你认为系统里最关键的 prompt 是哪些？

推荐短答：

最关键的是路由、查询改写、生成、缺口分析和三类评估 prompt，因为它们分别控制“走哪条路、拿什么证据、怎么回答、缺什么、答得行不行”。

展开回答：

- Router prompt 决定成本层级。
- Rewrite prompt 决定检索入口质量。
- Generation prompt 决定回答风格、安全性和证据整合方式。
- Gap analyzer 决定继续本地检索还是联网。
- Retrieval / Answer / Hallucination grader 决定证据过滤和失败升级。

代码位置：

- 路由：`graph/chains/router.py`
- 改写：`graph/chains/query_rewriter.py`
- 生成：`graph/chains/generation.py`
- 缺口分析：`graph/chains/gap_analyzer.py`
- 评估：`graph/chains/retrieval_grader.py`、`graph/chains/answer_grader.py`、`graph/chains/hallucination_grader.py`
- 公共安全策略：`graph/prompt_defaults.py`

### Q18. 生成 prompt 里你最看重什么？

推荐短答：

我最看重三点：同语言回答、明确不确定性、禁止编造高风险医学信息。

展开回答：

- 它要求优先使用上下文。
- 它明确禁止编造剂量、禁忌、相互作用、引用。
- 它根据 route strategy 调整回答风格。
- 它还规定了输出结构，尽量让回答更短、更稳、更像医学信息说明而不是随意聊天。

代码位置：

- 生成 prompt：`graph/chains/generation.py`
- 统一安全约束：`graph/prompt_defaults.py`

### Q19. 你怎么判断生成结果是不是“答到了点”？

推荐短答：

我把“有没有事实依据”和“有没有回答问题”拆成两个评估器，分别看 groundedness 和 answer coverage。

展开回答：

- `answer_grader` 只判断答案是否真正回应了问题。
- 它不负责判断证据真伪。
- 这样可以避免把“说了很多但没回答”和“回答了但没证据”混成一件事。

代码位置：

- 答题完整性评估：`graph/chains/answer_grader.py`
- 总评估入口：`graph/evaluation.py` 中的 `evaluate_generation`

### Q20. 你怎么判断有没有幻觉？

推荐短答：

优先用 RAGAS faithfulness，如果本地环境跑不了，再退回到 LLM groundedness grader。

展开回答：

- 我不想完全依赖 LLM 自评，所以优先用结构化指标。
- 但考虑到环境依赖，做了一个 fallback。
- 同时为了让 LLM fallback 更稳，我把证据串成带标题和来源的文本块再送进去判别。

代码位置：

- RAGAS fallback：`graph/evaluation.py` 中的 `_safe_ragas_faithfulness`
- LLM groundedness 判别：`graph/chains/hallucination_grader.py`
- 文档串接：`graph/evaluation.py` 中的 `_stringify_documents`

### Q21. 为什么在 no_retrieval 路线里 grounded 直接设成 true？

推荐短答：

因为这条路线没有检索证据可供 groundedness 判别，所以我把它当成“参数知识回答”处理，重点只检查是否回答到了问题，并由风险守卫保证它只能用于低风险场景。

展开回答：

- 这不是说它一定真实，而是“没有外部证据可比对”。
- 真正的安全边界不在这里，而是在前面的风险守卫。
- 如果 no-retrieval 答得不好，系统会立刻升级。

代码位置：

- no-retrieval 评估逻辑：`graph/evaluation.py` 中的 `route_strategy == "no_retrieval"` 分支
- 升级逻辑：`graph/nodes/evaluate_generation.py`

---

## 6. 训练与检索优化类问题

### Q22. 为什么你只训练 query tower，不训练 document tower？

推荐短答：

因为当前最主要的问题不是文档没有入库，而是查询表达不稳定，先训 query tower 更便于控制变量和解释收益。

展开回答：

- 文档向量已经建入 ChromaDB。
- 如果动 document tower，就要重建向量库，实验变量会变多。
- 只动 query side 更适合做 before/after baseline 对照。

代码位置：

- 查询塔加载：`graph/embeddings/bge_m3.py`
- 模型装配：`model.py`
- 训练器：`scripts/train_query_tower_lora.py`

### Q23. 为什么是 LoRA，不是全量微调？

推荐短答：

因为我要的是低成本、可插拔、便于对照实验的 query-side 适配，而不是重训练整个编码器。

展开回答：

- LoRA 更适合第一轮实验。
- 它方便快速迭代和 ablation。
- 也能和当前“文档塔固定”的工程假设保持一致。

代码位置：

- LoRA 配置：`scripts/train_query_tower_lora.py` 中的 `LoraConfig`
- 查询塔适配器路径：`model.py`、`graph/embeddings/bge_m3.py`

### Q24. 你的训练目标为什么选 InfoNCE？

推荐短答：

因为我要做的是“把 query 拉近正确文档、远离易混文档”，本质是对比学习问题，不是分类问题。

展开回答：

- 输入是 `(query, positive, confusable negative)`。
- 输出是三组向量。
- 损失函数直接约束正负相对距离。
- 这和医学域里的“相似病、相似药、相似指标”区分任务很匹配。

代码位置：

- 训练目标说明：`scripts/train_query_tower_lora.py` 中的 `QueryTowerLoRATrainer`
- InfoNCE 计算：`scripts/train_query_tower_lora.py` 中的 `_compute_info_nce_loss`

### Q25. 你的 hard negative 是怎么来的？为什么重要？

推荐短答：

我的 hard negative 不是随机采样，而是优先从当前检索器最容易混淆的候选里挖出来的，所以它更贴近真实错误模式。

展开回答：

- 随机负例太容易，训练意义有限。
- 医学检索真正难的是“像但不对”的知识点。
- 比如心衰和肺炎都可能出现气短，但证据重点不同。
- 所以我用当前检索器先跑，再从易混候选里选负例。

代码位置：

- 正例解析：`scripts/build_query_tower_training_data.py` 中的 `_resolve_positive_document`
- 负例挖掘：`scripts/build_query_tower_training_data.py` 中的 `_mine_negative_document`
- 三元组构建：`scripts/build_query_tower_training_data.py` 中的 `build_triplets`

### Q26. 你怎么回答“为什么要训、训前 baseline 是多少、提升来自哪里”？

推荐短答：

我把这个问题拆成了训练前检索指标、训练后检索指标、以及路由升级率变化三层证据，不只看一个总分。

展开回答：

- 为什么要训：因为医学 query 表达不稳定，首轮检索排序不够稳。
- 训前 baseline：用 `retrieval_eval.py` 或 `compare_query_tower_baseline.py` 在相同语料和相同召回设置下评估。
- 提升来自哪里：不仅看 `Recall@5 / NDCG@10 / MRR@10`，还看 `recovered_from_miss`、`improved_top1`，以及路由升级率下降没有。

代码位置：

- 检索基线：`scripts/retrieval_eval.py`
- 训练前后对比：`scripts/compare_query_tower_baseline.py`
- 升级率：`scripts/compute_route_upgrade_stats.py`、`scripts/compare_route_upgrade_baseline.py`
- 总报告：`scripts/compare_training_before_after.py`

### Q27. 你怎么统计路由升级率？

推荐短答：

我是统计真实 graph 执行后的 `route_history`，不是静态看 router 初始预测。

展开回答：

- 这样统计更真实。
- 可以看到“系统原本打算怎么走”和“最后被迫走成什么样”。
- 这对衡量 single-step 改善是否真的减少 multi-hop 依赖很关键。

代码位置：

- 路由历史：`graph/state.py`
- 写入：`graph/nodes/rewrite_query.py`
- 统计：`scripts/compute_route_upgrade_stats.py`

### Q28. 你怎么做训练前后统一总报告？

推荐短答：

我写了一个统一脚本，把检索指标和路由升级率一起汇总到同一个 JSON 里。

展开回答：

- 单看检索指标不够，因为系统是带路由的。
- 单看路由升级率也不够，因为你不知道是不是单步检索本身变强了。
- 所以要把 single-step 检索效果和 route upgrade 一起看。

代码位置：

- 总报告脚本：`scripts/compare_training_before_after.py`

---

## 7. 数据治理与评测类问题

### Q29. 为什么要做 versioned corpus，而不是直接放几份文档进去？

推荐短答：

因为我要把“服务语料”“评测语料”“不同版本语料”分开治理，不然你后面很难解释实验结果是哪里来的。

展开回答：

- profile 负责领域级配置。
- corpus_version 负责版本管理。
- manifest 负责资产组织。
- asset usage 把 serve 和 eval 分开。

代码位置：

- profile：`graph/corpus_profiles.py` 中的 `CorpusProfile`
- manifest 解析：`graph/corpus_pipeline.py`

### Q30. 为什么要把 serving corpus 和 eval corpus 分开？

推荐短答：

因为线上回答所依赖的知识源，和离线 benchmark 注册表，不应该混在同一个语料池里。

展开回答：

- 否则你很难知道系统是不是把评测集本身“当知识库用了”。
- 我现在把 `guideline_pages` 和 `pubmed_abstracts` 用于服务。
- 把 `nfcorpus_eval_registry` 单独放在 eval 用途里。

代码位置：

- manifest：`data/corpus/medical_demo/v1/manifest.json`
- 资产加载：`graph/corpus_pipeline.py` 中的 `load_corpus_documents`
- 当前 profile：`graph/corpus_profiles.py`

### Q31. 你的评测体系为什么不是只看一个最终准确率？

推荐短答：

因为这个系统是分层的，如果只报一个总分，你根本无法解释提升来自路由、检索还是生成。

展开回答：

- 检索层看 Recall/NDCG/MRR。
- 路由层看升级率和路径分布。
- 生成层看 ROUGE-L/BERTScore。
- 真实性层看 RAGAS faithfulness 或 fallback grader。

代码位置：

- 检索评测：`scripts/retrieval_eval.py`
- 路由评测：`scripts/compute_route_upgrade_stats.py`
- 生成评测：`scripts/evaluate_synthetic_generation.py`
- 真实性评估：`graph/evaluation.py`

### Q32. 你现在真实跑出了哪些数值，哪些还没跑？

推荐短答：

框架和评测入口都已经落地了，但真实 benchmark 数值目前还需要依赖模型权重、API 和环境进一步实跑。

展开回答：

- 已落地的是工程闭环和评测脚本。
- 还没完全固化的是真实的 `BEIR nfcorpus`、`Synthetic generation`、`Query Tower LoRA` 的最终结果。
- 面试时你要诚实说“方法和闭环已经做完，结果还在补实跑”。

代码位置：

- 评测脚本入口：`scripts/run_beir_nfcorpus_eval.py`、`scripts/evaluate_synthetic_generation.py`
- 文档说明：`FRAMEWORK_DESCRIPTION_AND_REFACTOR.md`

---

## 8. MCP 与工程边界类问题

### Q33. 你为什么要把 MCP 加进这个项目？

推荐短答：

MCP 在这个项目里不是推理核心，而是统一工具接入边界，让搜索、知识服务和外部能力调用更标准化。

展开回答：

- 我不想让所有工具都直接写死在 graph 节点里。
- MCP 让本地工具和远程工具可以用统一方式注册和调用。
- 这对后面扩展 citation、文献服务、规则服务很有帮助。

代码位置：

- 默认注册：`graph/mcp/registry.py`
- 传输层：`graph/mcp/transports.py`

### Q34. 你现在的 MCP 到什么程度了？

推荐短答：

已经从“只有抽象接口”升级到“抽象层 + 真实 stdio transport + 本地示例 server”的程度。

展开回答：

- 默认模式下可以用 in-process 工具。
- 如果配置环境变量，可以接外部 stdio MCP server。
- 我还补了一个本地 `local_medical_mcp_server.py`，让项目在不依赖外部 API 时也能演示 MCP 能力。

代码位置：

- in-process 注册：`graph/mcp/registry.py` 中的 `get_default_mcp_client`
- stdio transport：`graph/mcp/transports.py`
- 本地示例 server：`scripts/local_medical_mcp_server.py`
- env 示例：`configs/mcp_stdio.example.env`

### Q35. 本地 MCP 示例 server 具体提供了什么？

推荐短答：

它提供了语料描述、语料资产枚举、问题风险评估和本地语料搜索四类工具，主要是为了让 MCP 在本地也能真实跑起来。

展开回答：

- `describe_active_corpus`
- `list_corpus_assets`
- `assess_question_risk`
- `search_local_corpus`

这很好用在面试时证明：“我不是只会写接口名，我真的把工具边界跑通了。”

代码位置：

- `scripts/local_medical_mcp_server.py`

---

## 9. 诚实边界与取舍类问题

### Q36. 你觉得当前项目还有哪些明显没做完？

推荐短答：

目前框架已经比较完整，但 citation 展示、医学缩写扩展、cross-encoder reranker 和真实 benchmark 结果仍然是下一步重点。

展开回答：

- 现在的生成上下文已经有来源元数据，但还没有真正输出 citation。
- HyDE 有了，但没有单独的医学缩写扩展模块。
- reranker 目前是 BGE-M3 双编码器重排，不是更强的 cross-encoder。
- benchmark 入口都在，但最终数值还没全部封版。

代码位置：

- 生成上下文：`graph/nodes/generate.py`
- 当前重排：`graph/rerankers/bge_m3.py`
- 后续规划说明：`FRAMEWORK_DESCRIPTION_AND_REFACTOR.md`

### Q37. 如果你继续迭代，你优先做什么？

推荐短答：

我会优先做 citation 输出和医学缩写扩展，其次再考虑 cross-encoder reranker 和更强的 MCP 治理。

展开回答：

- citation 会直接提升可解释性和展示效果。
- 医学缩写扩展对检索提升的 ROI 很高。
- cross-encoder 适合在 single-step 已经稳定后继续抬精度。
- MCP 治理则是让系统更工程化。

代码位置：

- 当前生成链：`graph/chains/generation.py`、`graph/nodes/generate.py`
- 当前检索改写链：`graph/chains/query_rewriter.py`、`graph/chains/hyde.py`

### Q38. 你最大的工程取舍是什么？

推荐短答：

最大的取舍是我没有一开始就把所有东西做重，而是优先保证“闭环能跑、路径可解释、收益能量化”。

展开回答：

- 所以我先做 query tower LoRA，不先训 doc tower。
- 我先做 bi-encoder rerank，不先上 cross-encoder。
- 我先做 versioned corpus，不先上大规模数据平台。
- 我先做 stdio MCP，而不是一上来做复杂分布式工具总线。

代码位置：

- 查询塔实现：`scripts/train_query_tower_lora.py`、`graph/embeddings/bge_m3.py`
- 重排：`graph/rerankers/bge_m3.py`
- 语料治理：`graph/corpus_profiles.py`、`graph/corpus_pipeline.py`
- MCP：`graph/mcp/transports.py`

---

## 10. 面试官高压追问时的答题策略

### Q39. 如果面试官说“你这些都是概念，代码里到底哪里体现了？”

推荐应对：

你不要再泛泛解释，直接按模块回指：

1. 路由和图结构：`graph/graph.py`、`graph/nodes/rewrite_query.py`
2. 风险守卫：`graph/risk_guardrails.py`
3. 检索和重排：`ingestion.py`、`graph/nodes/retrieve.py`、`graph/rerankers/bge_m3.py`
4. 提示词：`graph/chains/*.py`
5. 训练：`scripts/train_query_tower_lora.py`
6. 评测：`scripts/retrieval_eval.py`、`scripts/compare_query_tower_baseline.py`、`scripts/compute_route_upgrade_stats.py`
7. MCP：`graph/mcp/*.py`、`scripts/local_medical_mcp_server.py`

### Q40. 如果面试官问“你这里最能体现你个人思考的地方是什么？”

推荐短答：

最能体现我个人思考的不是单个模块，而是三件事：把 Query Router 和风险守卫拆开、把 Single-Step 改成“检索 + 重排层 + 少量 LLM”、以及把训练收益闭环成检索指标加升级率。

展开回答：

- 这是系统级思考，不是单点优化。
- 它体现的是：我不是只会加模块，而是会考虑安全边界、成本结构和评测闭环。

对应代码位置：

- 风险守卫：`graph/risk_guardrails.py`
- 单步检索重构：`graph/nodes/grade_documents.py`
- 训练收益闭环：`scripts/compare_query_tower_baseline.py`、`scripts/compare_route_upgrade_baseline.py`、`scripts/compare_training_before_after.py`

---

## 11. 代码索引速查表

如果面试时你来不及展开，可以直接报下面这些文件：

### 11.1 架构主线

- `graph/graph.py`
- `graph/state.py`
- `graph/nodes/rewrite_query.py`
- `graph/nodes/evaluate_generation.py`

### 11.2 路由与安全

- `graph/chains/router.py`
- `graph/risk_guardrails.py`

### 11.3 检索与重排

- `ingestion.py`
- `graph/nodes/retrieve.py`
- `graph/nodes/grade_documents.py`
- `graph/rerankers/bge_m3.py`
- `graph/chains/hyde.py`

### 11.4 提示词与生成

- `graph/chains/query_rewriter.py`
- `graph/chains/generation.py`
- `graph/chains/gap_analyzer.py`
- `graph/chains/retrieval_grader.py`
- `graph/chains/answer_grader.py`
- `graph/chains/hallucination_grader.py`
- `graph/prompt_defaults.py`

### 11.5 训练与评测

- `scripts/train_query_tower_lora.py`
- `scripts/build_query_tower_training_data.py`
- `scripts/retrieval_eval.py`
- `scripts/compare_query_tower_baseline.py`
- `scripts/compute_route_upgrade_stats.py`
- `scripts/compare_training_before_after.py`

### 11.6 语料治理与 MCP

- `graph/corpus_profiles.py`
- `graph/corpus_pipeline.py`
- `graph/mcp/registry.py`
- `graph/mcp/transports.py`
- `scripts/local_medical_mcp_server.py`

---

## 12. 最后建议

真正面试时，你不要试图一次把所有内容讲全，而是按下面这个节奏来：

1. 先讲“系统为什么不是普通 RAG”
2. 再讲“三级路由 + 风险守卫”
3. 再讲“单步检索为什么要混合检索 + 重排层”
4. 再讲“Query Tower LoRA 为什么只训 query side”
5. 最后讲“我怎么证明收益，而不是只说我加了模块”

如果你能把这五步讲顺，绝大多数技术面试官都会觉得你是真的做过，而不是只会堆概念。
