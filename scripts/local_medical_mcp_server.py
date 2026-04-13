from __future__ import annotations

from typing import Any, Dict, List

from graph.consts import NO_RETRIEVAL
from graph.corpus_pipeline import describe_corpus_manifest, iter_corpus_assets
from graph.corpus_profiles import CORPUS_PROFILES, CorpusProfile, get_active_corpus_profile
from graph.risk_guardrails import assess_medical_risk
from ingestion import get_hybrid_retriever

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError(
        "The optional 'mcp' package is required to run the local stdio MCP server."
    ) from exc


mcp = FastMCP("local_medical_demo_server")


def _resolve_profile(profile_name: str | None = None) -> CorpusProfile:
    if profile_name:
        return CORPUS_PROFILES.get(profile_name, get_active_corpus_profile())
    return get_active_corpus_profile()


def _document_payload(document: Any, rank: int) -> Dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    return {
        "rank": rank,
        "title": metadata.get("title", "") or "unknown",
        "source": metadata.get("source", "") or metadata.get("url", "") or "unknown",
        "url": metadata.get("url", ""),
        "content": getattr(document, "page_content", ""),
        "metadata": metadata,
    }


@mcp.tool()
def describe_active_corpus(profile_name: str = "") -> Dict[str, Any]:
    """
    Return the active or named corpus manifest summary.
    """

    profile = _resolve_profile(profile_name or None)
    return describe_corpus_manifest(profile)


@mcp.tool()
def list_corpus_assets(profile_name: str = "", usage: str = "") -> Dict[str, Any]:
    """
    List corpus assets registered in the active or named profile.
    """

    profile = _resolve_profile(profile_name or None)
    assets = list(iter_corpus_assets(profile, usage=usage or None))
    return {
        "profile_name": profile.name,
        "corpus_version": profile.corpus_version,
        "usage_filter": usage or None,
        "assets": [
            {
                "name": asset.name,
                "kind": asset.kind,
                "usage": asset.usage,
                "description": asset.description,
                "path": asset.path,
                "urls": list(asset.urls),
            }
            for asset in assets
        ],
    }


@mcp.tool()
def assess_question_risk(question: str, proposed_route: str = NO_RETRIEVAL) -> Dict[str, Any]:
    """
    Run the built-in medical risk guardrail against a question.
    """

    assessment = assess_medical_risk(
        question=question,
        rewritten_question=question,
        route_strategy=proposed_route,
    )
    return {
        "question": question,
        "proposed_route": proposed_route,
        "risk_level": assessment.risk_level,
        "matched_signals": assessment.matched_signals,
        "rationale": assessment.rationale,
        "adjusted_route_strategy": assessment.adjusted_route_strategy,
        "prohibit_no_retrieval": assessment.prohibit_no_retrieval,
    }


@mcp.tool()
def search_local_corpus(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the local ChromaDB + BM25 hybrid corpus without external APIs.
    """

    try:
        retriever = get_hybrid_retriever()
        documents = retriever.invoke(query)[:max(1, max_results)]
    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "results": [],
            "content": "",
        }

    results: List[Dict[str, Any]] = [
        _document_payload(document, rank=index)
        for index, document in enumerate(documents, start=1)
    ]
    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        "content": "\n\n".join(
            f"[{row['rank']}] {row['title']} | {row['source']}\n{row['content']}"
            for row in results
        ),
    }


if __name__ == "__main__":
    mcp.run()
