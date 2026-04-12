from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import (
    EVALUATE_GENERATION,
    GENERATE,
    GRADE_DOCUMENTS,
    RETRIEVE,
    REWRITE_QUERY,
    VECTORSTORE,
    WEBSEARCH,
)
from graph.nodes.evaluate_generation import evaluate_generation_node
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.rewrite_query import rewrite_query
from graph.nodes.web_search import web_search
from graph.state import GraphState

load_dotenv()


def route_after_rewrite(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    if state["route"] == VECTORSTORE:
        print("---ROUTE QUESTION TO HYBRID RAG---")
        return RETRIEVE
    print("---ROUTE QUESTION TO WEB SEARCH---")
    return WEBSEARCH


def route_after_document_grading(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    next_action = state.get("next_action", GENERATE)

    if next_action == RETRIEVE:
        print("---DECISION: RUN ANOTHER LOCAL RETRIEVAL ROUND---")
        return RETRIEVE
    if next_action == WEBSEARCH:
        print("---DECISION: FALL BACK TO WEB SEARCH---")
        return WEBSEARCH

    print("---DECISION: GENERATE---")
    return GENERATE


def route_after_evaluation(state: GraphState) -> str:
    evaluation = state.get("evaluation", {})
    next_action = state.get("next_action", "end")

    if next_action == GENERATE:
        print("---DECISION: REGENERATE WITH CURRENT CONTEXT---")
        return GENERATE
    if next_action == WEBSEARCH:
        print("---DECISION: SEEK MORE EVIDENCE FROM WEB SEARCH---")
        return WEBSEARCH
    if next_action == REWRITE_QUERY:
        print("---DECISION: REWRITE QUESTION AND RETRY---")
        return REWRITE_QUERY

    print(
        f"---DECISION: ANSWER ACCEPTED "
        f"(grounded={evaluation.get('grounded')}, "
        f"addresses_question={evaluation.get('addresses_question')})---"
    )
    return END


workflow = StateGraph(GraphState)

workflow.add_node(REWRITE_QUERY, rewrite_query)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)
workflow.add_node(EVALUATE_GENERATION, evaluate_generation_node)

workflow.set_entry_point(REWRITE_QUERY)

workflow.add_conditional_edges(
    REWRITE_QUERY,
    route_after_rewrite,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    route_after_document_grading,
    {
        RETRIEVE: RETRIEVE,
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, EVALUATE_GENERATION)
workflow.add_conditional_edges(
    EVALUATE_GENERATION,
    route_after_evaluation,
    {
        GENERATE: GENERATE,
        WEBSEARCH: WEBSEARCH,
        REWRITE_QUERY: REWRITE_QUERY,
        END: END,
    },
)

app = workflow.compile()
