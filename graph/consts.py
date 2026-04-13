REWRITE_QUERY = "rewrite_query"
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
EVALUATE_GENERATION = "evaluate_generation"
WEBSEARCH = "websearch"

VECTORSTORE = "vectorstore"
LLM = "llm"
WEB_SEARCH_TOOL = "search_web_general"
MEDICAL_WEB_SEARCH_TOOL = "search_medical_web"
PUBMED_SEARCH_TOOL = "search_pubmed"
NO_RETRIEVAL = "no_retrieval"
SINGLE_STEP = "single_step"
MULTI_HOP = "multi_hop"

MAX_RETRIEVAL_ROUNDS = 2
MAX_GENERATION_RETRIES = 2
MIN_RELEVANT_DOCS = 2
SINGLE_STEP_MIN_RELEVANT_DOCS = 1
WEB_SEARCH_RESULT_COUNT = 3
RERANK_TOP_K = 8
SINGLE_STEP_RERANK_TOP_K = 4
SINGLE_STEP_LLM_GRADING_MAX_DOCS = 2
MULTI_HOP_LLM_GRADING_MAX_DOCS = 6
SINGLE_STEP_CONFIDENT_RERANK_SCORE = 0.45
SINGLE_STEP_CONFIDENT_MARGIN = 0.05

"""
Centralized workflow names and runtime limits for the adaptive RAG graph.
"""
