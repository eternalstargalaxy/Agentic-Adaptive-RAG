from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    """
    Shared state carried across the LangGraph workflow.
    """

    question: str
    rewritten_question: str
    route: str
    route_strategy: str
    forced_route_strategy: str
    route_rationale: str
    route_history: List[str]
    risk_level: str
    risk_rationale: str
    risk_signals: List[str]
    corpus_profile: str
    corpus_version: str
    corpus_manifest_path: str
    retrieval_queries: List[str]
    sub_queries: List[str]
    search_query: str
    preferred_search_tools: List[str]
    use_hyde: bool
    generation: str
    documents: List[Any]
    web_search: bool
    retrieval_round: int
    retry_count: int
    next_action: str
    rerank_applied: bool
    screening_mode: str
    evaluation: Dict[str, Any]


"""
The refactored state tracks both user-facing outputs and process metadata,
including rewrite results, multi-round retrieval queries, retry counters, and
evaluation details used for control flow.
"""
