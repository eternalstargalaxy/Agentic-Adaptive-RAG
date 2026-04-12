from typing import Any, Dict, List, TypedDict


class GraphState(TypedDict, total=False):
    """
    Shared state carried across the LangGraph workflow.
    """

    question: str
    rewritten_question: str
    route: str
    retrieval_queries: List[str]
    sub_queries: List[str]
    search_query: str
    use_hyde: bool
    generation: str
    documents: List[Any]
    web_search: bool
    retrieval_round: int
    retry_count: int
    next_action: str
    evaluation: Dict[str, Any]


"""
The refactored state tracks both user-facing outputs and process metadata,
including rewrite results, multi-round retrieval queries, retry counters, and
evaluation details used for control flow.
"""
