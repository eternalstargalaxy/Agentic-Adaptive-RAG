from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from graph.consts import MEDICAL_WEB_SEARCH_TOOL, PUBMED_SEARCH_TOOL, WEB_SEARCH_TOOL
from graph.mcp.client import InProcessMCPClient
from graph.search.providers import (
    MedicalWebSearchProvider,
    PubMedSearchProvider,
    SearchHit,
    TavilySearchProvider,
)


def _hits_to_payload(tool_name: str, hits: List[SearchHit]) -> Dict[str, object]:
    return {
        "content": "\n\n".join(hit.content for hit in hits),
        "results": [hit.to_dict() for hit in hits],
        "tool_name": tool_name,
    }


@lru_cache(maxsize=1)
def get_default_mcp_client() -> InProcessMCPClient:
    client = InProcessMCPClient()
    tavily_provider = TavilySearchProvider()
    medical_web_provider = MedicalWebSearchProvider(tavily_provider=tavily_provider)
    pubmed_provider = PubMedSearchProvider()

    client.register_tool(
        name=WEB_SEARCH_TOOL,
        description="通用网页搜索工具，适合最新信息和开放域资料补充。",
        handler=lambda query, max_results=3: _hits_to_payload(
            WEB_SEARCH_TOOL,
            tavily_provider.search(query=query, max_results=max_results),
        ),
    )
    client.register_tool(
        name=MEDICAL_WEB_SEARCH_TOOL,
        description="受限于权威医学网站的网页搜索工具，用于医学场景下的高可信资料补充。",
        handler=lambda query, max_results=3: _hits_to_payload(
            MEDICAL_WEB_SEARCH_TOOL,
            medical_web_provider.search(query=query, max_results=max_results),
        ),
    )
    client.register_tool(
        name=PUBMED_SEARCH_TOOL,
        description="PubMed 检索工具，用于检索医学文献标题和摘要。",
        handler=lambda query, max_results=3: _hits_to_payload(
            PUBMED_SEARCH_TOOL,
            pubmed_provider.search(query=query, max_results=max_results),
        ),
    )
    return client
