from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.consts import WEB_SEARCH_RESULT_COUNT
from graph.state import GraphState
from ingestion import normalize_queries

load_dotenv()

web_search_tool = TavilySearch(max_results=WEB_SEARCH_RESULT_COUNT)


def _append_documents(existing: List[Document], additions: List[Document]) -> List[Document]:
    merged = []
    seen = set()
    for document in existing + additions:
        content = (document.page_content or "").strip()
        source = document.metadata.get("source", "")
        key = f"{source}::{content[:400]}"
        if key in seen or not content:
            continue
        seen.add(key)
        merged.append(document)
    return merged


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    search_queries = normalize_queries(
        state.get("sub_queries")
        or state.get("search_query")
        or state.get("rewritten_question")
        or question
    )
    documents = state.get("documents", [])
    web_documents = []

    for query in search_queries[:WEB_SEARCH_RESULT_COUNT]:
        tavily_results = web_search_tool.invoke({"query": query})["results"]
        joined_result = "\n".join(result["content"] for result in tavily_results)
        web_documents.append(
            Document(
                page_content=joined_result,
                metadata={"source": "tavily", "query": query},
            )
        )

    merged_documents = _append_documents(documents, web_documents)

    return {
        "documents": merged_documents,
        "question": question,
        "search_query": search_queries[0] if search_queries else question,
        "sub_queries": [],
    }
