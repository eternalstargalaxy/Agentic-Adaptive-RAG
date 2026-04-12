from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen

from langchain_tavily import TavilySearch


PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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
        medical_query = (
            f"(site:nih.gov OR site:medlineplus.gov OR site:cdc.gov OR "
            f"site:who.int) {query}"
        )
        hits = self.tavily_provider.search(medical_query, max_results=max_results)
        for hit in hits:
            hit.source = "medical_web"
        return hits


class PubMedSearchProvider:
    def __init__(self, email: str | None = None) -> None:
        self.email = email or "adaptive-rag@example.com"

    def _fetch_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urlencode(params)
        with urlopen(f"{PUBMED_BASE_URL}/{endpoint}?{query}") as response:
            return json.loads(response.read().decode("utf-8"))

    def _fetch_text(self, endpoint: str, params: Dict[str, Any]) -> str:
        query = urlencode(params)
        with urlopen(f"{PUBMED_BASE_URL}/{endpoint}?{query}") as response:
            return response.read().decode("utf-8")

    def _fetch_abstract(self, pmid: str) -> str:
        return self._fetch_text(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "id": pmid,
                "rettype": "abstract",
                "retmode": "text",
            },
        ).strip()

    def search(self, query: str, max_results: int = 3) -> List[SearchHit]:
        search_payload = self._fetch_json(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
                "tool": "adaptive-rag",
                "email": self.email,
            },
        )
        pmids = search_payload.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summary_payload = self._fetch_json(
            "esummary.fcgi",
            {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
                "tool": "adaptive-rag",
                "email": self.email,
            },
        )
        summaries = summary_payload.get("result", {})

        hits = []
        for pmid in pmids:
            summary = summaries.get(pmid, {})
            title = summary.get("title", f"PubMed article {pmid}")
            abstract = self._fetch_abstract(pmid)
            hits.append(
                SearchHit(
                    title=title,
                    content=abstract or title,
                    source="pubmed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    metadata={
                        "pmid": pmid,
                        "pubdate": summary.get("pubdate", ""),
                        "authors": summary.get("authors", []),
                    },
                )
            )
        return hits
