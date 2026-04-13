from typing import Any, Dict

from graph.chains.gap_analyzer import gap_analyzer
from graph.chains.retrieval_grader import retrieval_grader
from graph.consts import (
    GENERATE,
    MAX_RETRIEVAL_ROUNDS,
    MIN_RELEVANT_DOCS,
    MULTI_HOP,
    MULTI_HOP_LLM_GRADING_MAX_DOCS,
    RETRIEVE,
    SINGLE_STEP,
    SINGLE_STEP_CONFIDENT_MARGIN,
    SINGLE_STEP_CONFIDENT_RERANK_SCORE,
    SINGLE_STEP_LLM_GRADING_MAX_DOCS,
    SINGLE_STEP_MIN_RELEVANT_DOCS,
    SINGLE_STEP_RERANK_TOP_K,
    WEBSEARCH,
)
from graph.state import GraphState


def _rerank_score(document: Any) -> float:
    metadata = getattr(document, "metadata", {}) or {}
    return float(metadata.get("rerank_score", 0.0))


def _llm_filter_documents(question: str, documents: list, max_docs: int) -> list:
    filtered_docs = []
    for document in documents[:max_docs]:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        if bool(score.binary_score):
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return filtered_docs


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Filter retrieved documents and decide whether to generate, retrieve again,
    or fallback to web search.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])
    route_strategy = state.get("route_strategy", MULTI_HOP)

    screening_mode = "llm_full_scan"
    if route_strategy == SINGLE_STEP:
        candidate_documents = documents[:SINGLE_STEP_RERANK_TOP_K]
        top_score = _rerank_score(candidate_documents[0]) if candidate_documents else 0.0
        second_score = (
            _rerank_score(candidate_documents[1])
            if len(candidate_documents) > 1
            else 0.0
        )
        confident_rerank = (
            top_score >= SINGLE_STEP_CONFIDENT_RERANK_SCORE
            or (top_score - second_score) >= SINGLE_STEP_CONFIDENT_MARGIN
        )

        if confident_rerank and candidate_documents:
            filtered_docs = candidate_documents[: max(SINGLE_STEP_MIN_RELEVANT_DOCS, 1)]
            screening_mode = "rerank_only"
        else:
            filtered_docs = _llm_filter_documents(
                question,
                candidate_documents,
                max_docs=SINGLE_STEP_LLM_GRADING_MAX_DOCS,
            )
            screening_mode = "rerank_then_llm"
            if not filtered_docs and candidate_documents and top_score >= 0.30:
                filtered_docs = [candidate_documents[0]]
                screening_mode = "rerank_fallback"
    else:
        filtered_docs = _llm_filter_documents(
            question,
            documents,
            max_docs=MULTI_HOP_LLM_GRADING_MAX_DOCS,
        )
        screening_mode = "rerank_then_llm"

    relevant_doc_count = len(filtered_docs)
    if route_strategy == SINGLE_STEP:
        next_action = GENERATE if relevant_doc_count >= SINGLE_STEP_MIN_RELEVANT_DOCS else WEBSEARCH
    else:
        next_action = GENERATE if relevant_doc_count >= MIN_RELEVANT_DOCS else WEBSEARCH
    sub_queries = []
    search_query = state.get("rewritten_question", question)

    if filtered_docs:
        evidence_summary = "\n\n".join(
            document.page_content[:500] for document in filtered_docs[:4]
        )
    else:
        evidence_summary = "No relevant local evidence found."

    if route_strategy == SINGLE_STEP:
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": next_action == WEBSEARCH,
            "next_action": next_action,
            "sub_queries": [],
            "search_query": search_query,
            "screening_mode": screening_mode,
        }

    if relevant_doc_count < MIN_RELEVANT_DOCS:
        analysis = gap_analyzer.invoke(
            {
                "question": question,
                "route": state.get("route", "vectorstore"),
                "retrieval_round": state.get("retrieval_round", 0),
                "relevant_doc_count": relevant_doc_count,
                "documents": evidence_summary,
            }
        )
        sub_queries = analysis.sub_queries
        search_query = analysis.search_query or search_query
        if (
            analysis.next_action == "retrieve"
            and state.get("retrieval_round", 0) < MAX_RETRIEVAL_ROUNDS
            and sub_queries
        ):
            next_action = RETRIEVE
        elif analysis.next_action == "generate" and filtered_docs:
            next_action = GENERATE
        else:
            next_action = WEBSEARCH

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": next_action == WEBSEARCH,
        "next_action": next_action,
        "sub_queries": sub_queries,
        "search_query": search_query,
        "screening_mode": screening_mode,
    }
