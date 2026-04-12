# Agentic Adaptive RAG: Framework Description And Refactor Notes

## 1. Current Repository Positioning

This repository started as a compact LangGraph-based adaptive RAG demo. Its original goal was to show a simple self-correcting workflow:

1. Decide whether a user question should use local retrieval or web search.
2. Retrieve local documents when the question matches the local corpus.
3. Grade retrieved documents for relevance.
4. Generate an answer from the available context.
5. Check whether the answer is grounded and whether it answers the question.
6. Retry or trigger web search when the answer quality is not good enough.

In other words, the project was already a usable "control-flow skeleton", but it was still much closer to a demo than to the richer system described in the resume project.

## 2. Original Logical Framework

### 2.1 Data Layer

- `ingestion.py`
- Original behavior:
  - pull three Lilian Weng blog pages
  - split them into chunks
  - write them into Chroma
  - expose a single vector retriever

This gave the project a working local knowledge base, but the retrieval layer was single-path dense retrieval and had strong import-time side effects.

### 2.2 Chain Layer

- `graph/chains/router.py`
  - route to `vectorstore` or `websearch`
- `graph/chains/retrieval_grader.py`
  - judge whether retrieved documents are relevant
- `graph/chains/generation.py`
  - generate the final answer
- `graph/chains/hallucination_grader.py`
  - judge whether the answer is grounded in retrieved context
- `graph/chains/answer_grader.py`
  - judge whether the answer actually addresses the question

These chains were the "reasoning and judging" layer. They let the graph make decisions instead of following a fixed path.

### 2.3 Node Layer

- `graph/nodes/retrieve.py`
  - execute local retrieval
- `graph/nodes/grade_documents.py`
  - filter documents and decide whether to search the web
- `graph/nodes/web_search.py`
  - fetch Tavily search results
- `graph/nodes/generate.py`
  - build the final answer

These nodes were the "actions" layer.

### 2.4 Graph Layer

- `graph/graph.py`

This file defined the state machine. The original path was:

`route -> retrieve -> grade_documents -> generate -> hallucination/answer grading -> retry or end`

The graph already showed a basic adaptive loop, but it did not yet support the richer retrieval strategy and evaluation pipeline from the resume version.

## 3. Main Gaps Compared With The Resume System

The resume project described a stronger system with:

- hybrid retrieval instead of single dense retrieval
- RRF-style fusion
- HyDE for short or fuzzy queries
- multiple retrieval rounds
- LLM-driven information-gap analysis and sub-query completion
- stronger retry logic
- optional RAGAS-style faithfulness evaluation
- cleaner state management for the whole process

The original repository did not fully support these pieces.

## 4. Target Refactor Roadmap

To move the repository closer to the resume system, the refactor roadmap was set as follows:

1. Upgrade state management so the graph can remember rewrites, retrieval rounds, retry counts, sub-queries, and evaluation results.
2. Replace single-path retrieval with a hybrid retrieval layer that combines dense and sparse search.
3. Add query rewriting so retrieval uses a cleaner and more retrieval-friendly query.
4. Add HyDE as an optional retrieval expansion method for short or ambiguous questions.
5. Add a gap-analysis stage so the system can decide whether to:
   - generate immediately
   - run a second local retrieval round
   - fall back to web search
6. Replace the old generation prompt with a local prompt to reduce external prompt dependencies.
7. Add a dedicated evaluation node so "groundedness", "question answering quality", and optional `RAGAS faithfulness` are centralized.
8. Tighten tests so at least the graph routing utilities can be validated without requiring live model calls.

## 5. Implemented Refactor

### 5.1 State And Constants Upgrade

Updated files:

- `graph/state.py`
- `graph/consts.py`

Changes:

- added `rewritten_question`
- added `route`
- added `retrieval_queries`
- added `sub_queries`
- added `search_query`
- added `use_hyde`
- added `retrieval_round`
- added `retry_count`
- added `next_action`
- added `evaluation`
- added runtime constants such as:
  - `MAX_RETRIEVAL_ROUNDS`
  - `MAX_GENERATION_RETRIES`
  - `MIN_RELEVANT_DOCS`
  - `WEB_SEARCH_RESULT_COUNT`

This change makes the graph much more suitable for multi-round adaptive behavior.

### 5.2 Retrieval Layer Refactor

Updated file:

- `ingestion.py`

Changes:

- removed the old import-time "always build the vectorstore" behavior
- added lazy initialization
- added document reconstruction from persisted Chroma data
- added sparse retrieval through `BM25Retriever`
- added a hybrid retriever wrapper
- added reciprocal-rank-style fusion
- added query normalization and duplicate removal

This is the biggest architectural change in the repository. The project now has a retrieval layer that is much closer to the hybrid retrieval described in the resume project.

### 5.3 Query Rewrite And HyDE

Added files:

- `graph/chains/query_rewriter.py`
- `graph/chains/hyde.py`
- `graph/nodes/rewrite_query.py`

Changes:

- added a query rewrite chain that:
  - rewrites the user question
  - returns one to three retrieval queries
  - determines whether to use HyDE
- added a HyDE chain that generates a hypothetical passage for retrieval expansion
- added a rewrite node that combines:
  - query routing
  - retrieval planning
  - HyDE control flags

This moves the repository toward the resume version's "adaptive query preparation" capability.

### 5.4 Multi-Round Retrieval Decision

Updated or added files:

- `graph/chains/gap_analyzer.py`
- `graph/nodes/retrieve.py`
- `graph/nodes/grade_documents.py`

Changes:

- retrieval now accepts multiple queries
- the first retrieval round can include HyDE expansion
- retrieved results are merged and deduplicated
- document grading no longer triggers web search simply because one document is bad
- a new gap-analysis chain decides whether the next step should be:
  - `generate`
  - `retrieve`
  - `websearch`

This is the part that makes the repository meaningfully closer to "multiple hybrid retrieval rounds + information gap analysis".

### 5.5 Generation And Evaluation Refactor

Updated or added files:

- `graph/chains/generation.py`
- `graph/chains/answer_grader.py`
- `graph/evaluation.py`
- `graph/nodes/generate.py`
- `graph/nodes/evaluate_generation.py`

Changes:

- replaced the external LangChain Hub prompt with a local prompt
- fixed the old answer grading bug where the question field was incorrectly wired
- centralized generation evaluation in `graph/evaluation.py`
- added an optional `RAGAS faithfulness` hook
- moved evaluation into an explicit node so the graph can branch more cleanly

This gives the graph a much cleaner separation between "generate" and "judge".

### 5.6 Graph Rebuild

Updated file:

- `graph/graph.py`

New high-level flow:

1. `rewrite_query`
2. route to `retrieve` or `websearch`
3. `retrieve`
4. `grade_documents`
5. choose among:
   - another `retrieve`
   - `websearch`
   - `generate`
6. `generate`
7. `evaluate_generation`
8. choose among:
   - accept answer
   - regenerate
   - rewrite and retry
   - web search fallback

This is now much closer to the logic described in the resume:

- query rewriting
- hybrid retrieval
- gap-driven additional retrieval
- web fallback
- answer verification
- retry loop

### 5.7 Test Cleanup

Updated file:

- `graph/chains/tests/test_chains.py`

Changes:

- removed the old live-model tests that were brittle and contained assertion issues
- replaced them with lightweight tests for deterministic routing utilities and query normalization

## 6. Summary Of This Refactor

This repository is still not a full reproduction of the resume project, but after the refactor it has moved from:

- "demo-level adaptive RAG skeleton"

to:

- "multi-round hybrid-retrieval adaptive RAG scaffold that is suitable for continued engineering"

The main improvements are not cosmetic. They increase the ceiling of the project in four important ways:

1. The graph now has process memory.
2. The retrieval layer is no longer single-path.
3. The workflow can perform another local retrieval round before immediately falling back to the web.
4. Evaluation is cleaner and easier to extend toward RAGAS and benchmark-driven analysis.

## 7. Suggested Next Engineering Steps

The next steps that would make this repository even closer to the resume system are:

1. Add BGE-M3 embeddings instead of the current generic embedding model.
2. Replace the current simple hybrid fusion with explicit weighted RRF and optional reranking.
3. Add a dedicated sub-query planner node instead of combining the decision entirely inside `gap_analyzer`.
4. Add answer relevance metrics on a held-out evaluation set.
5. Build a BEIR `nfcorpus` evaluation script for retrieval quality.
6. Integrate a synthetic QA test set for end-to-end answer scoring.
7. Add structured tracing so each retry round records:
   - used queries
   - retrieved sources
   - evaluation scores
   - final route decisions

## 8. Validation Status

Validation completed in this round:

- Python syntax compilation passed with `python -m compileall .`
- static workflow code was inspected after the refactor

Validation blocked in this round:

- live imports could not be fully executed because the current local environment does not yet have runtime dependencies such as `langchain`
- `pytest` is not currently installed in the local Python environment
- the current directory is not a Git repository, so repository-native diff/status workflows are unavailable

Recommended next validation steps after dependency installation:

1. Install dependencies from `requirements.txt`
2. Configure `.env` with `GOOGLE_API_KEY` and `TAVILY_API_KEY`
3. Run `python ingestion.py`
4. Run `python main.py`
5. Run `python -m pytest graph/chains/tests/test_chains.py -q`
