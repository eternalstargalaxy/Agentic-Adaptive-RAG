from typing import Any, Dict

from graph.chains.gap_analyzer import gap_analyzer
from graph.chains.retrieval_grader import retrieval_grader
from graph.consts import (
    GENERATE,
    MAX_RETRIEVAL_ROUNDS,
    MIN_RELEVANT_DOCS,
    MULTI_HOP,
    RETRIEVE,
    SINGLE_STEP,
    SINGLE_STEP_MIN_RELEVANT_DOCS,
    WEBSEARCH,
)
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Filter retrieved documents and decide whether to generate, retrieve again,
    or fallback to web search.
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state.get("documents", [])
    route_strategy = state.get("route_strategy", MULTI_HOP)

    filtered_docs = []
    for document in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        if score.binary_score.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(document)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

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
    }
