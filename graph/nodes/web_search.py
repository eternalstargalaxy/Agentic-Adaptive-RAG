from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.schema import Document

from graph.consts import WEB_SEARCH_RESULT_COUNT
from graph.corpus_profiles import get_active_corpus_profile
from graph.mcp.registry import get_default_mcp_client
from graph.state import GraphState
from ingestion import normalize_queries

load_dotenv()


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
    profile = get_active_corpus_profile()
    mcp_client = get_default_mcp_client()

    search_queries = normalize_queries(
        state.get("sub_queries")
        or state.get("search_query")
        or state.get("rewritten_question")
        or question
    )
    preferred_tools = state.get("preferred_search_tools") or list(
        profile.preferred_search_tools
    )

    documents = state.get("documents", [])
    web_documents: List[Document] = []
    used_tools: List[str] = []

    for tool_name in preferred_tools:
        for query in search_queries[:WEB_SEARCH_RESULT_COUNT]:
            try:
                result = mcp_client.call_tool(
                    tool_name,
                    {"query": query, "max_results": WEB_SEARCH_RESULT_COUNT},
                )
            except Exception:
                continue

            hits = result.structured_content.get("results", [])
            if hits:
                used_tools.append(tool_name)

            for hit in hits:
                web_documents.append(
                    Document(
                        page_content=hit.get("content", ""),
                        metadata={
                            "source": hit.get("source", tool_name),
                            "title": hit.get("title", ""),
                            "url": hit.get("url", ""),
                            "query": query,
                            "tool_name": tool_name,
                            "corpus_profile": profile.name,
                        },
                    )
                )

            if len(web_documents) >= WEB_SEARCH_RESULT_COUNT:
                break

        if len(web_documents) >= WEB_SEARCH_RESULT_COUNT:
            break

    merged_documents = _append_documents(documents, web_documents)

    return {
        "documents": merged_documents,
        "question": question,
        "search_query": search_queries[0] if search_queries else question,
        "sub_queries": [],
        "preferred_search_tools": used_tools or preferred_tools,
    }
