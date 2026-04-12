from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

MEDICAL_DOMAIN_WHITELIST = (
    "medlineplus.gov",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
)
PUBMED_DOMAIN_WHITELIST = (
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
)


@dataclass
class SearchHit:
    title: str
    content: str
    source: str
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TavilySearchProvider:
    def search(self, query: str, max_results: int = 3) -> List[SearchHit]:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        results = tool.invoke({"query": query}).get("results", [])
        hits = []
        for result in results:
            hits.append(
                SearchHit(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    source="tavily",
                    url=result.get("url", ""),
                    metadata={"score": result.get("score")},
                )
            )
        return hits


class MedicalWebSearchProvider:
    """
    Restrict general web retrieval to high-trust medical websites.
    """

    def __init__(self, tavily_provider: TavilySearchProvider | None = None) -> None:
        self.tavily_provider = tavily_provider or TavilySearchProvider()

    def search(self, query: str, max_results: int = 3) -> List[SearchHit]:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        results = tool.invoke(
            {
                "query": query,
                "include_domains": list(MEDICAL_DOMAIN_WHITELIST),
            }
        ).get("results", [])
        hits = []
        for result in results:
            hits.append(
                SearchHit(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    source="medical_web",
                    url=result.get("url", ""),
                    metadata={"score": result.get("score")},
                )
            )
        if hits:
            return hits

        medical_query = " ".join(f"site:{domain}" for domain in MEDICAL_DOMAIN_WHITELIST)
        hits = self.tavily_provider.search(
            f"{medical_query} {query}",
            max_results=max_results,
        )
        for hit in hits:
            hit.source = "medical_web"
        return hits


class PubMedSearchProvider:
    def search(self, query: str, max_results: int = 3) -> List[SearchHit]:
        from langchain_tavily import TavilySearch

        tool = TavilySearch(max_results=max_results)
        results = tool.invoke(
            {
                "query": query,
                "include_domains": list(PUBMED_DOMAIN_WHITELIST),
            }
        ).get("results", [])
        hits = []
        for result in results:
            hits.append(
                SearchHit(
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    source="pubmed",
                    url=result.get("url", ""),
                    metadata={"score": result.get("score")},
                )
            )
        return hits
