from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import BM25Retriever

from graph.corpus_profiles import CorpusProfile, get_active_corpus_profile
from model import embed_model

load_dotenv()

DEFAULT_TOP_K = 6


def _dedupe_documents(documents: Iterable[Document]) -> List[Document]:
    unique_documents = []
    seen = set()
    for document in documents:
        content = (document.page_content or "").strip()
        source = document.metadata.get("source", "")
        key = f"{source}::{content[:400]}"
        if key in seen or not content:
            continue
        seen.add(key)
        unique_documents.append(document)
    return unique_documents


def _reciprocal_rank_fusion(result_sets: Sequence[Sequence[Document]], k: int = 60) -> List[Document]:
    scores = {}
    document_lookup = {}

    for result_set in result_sets:
        for rank, document in enumerate(result_set, start=1):
            content = (document.page_content or "").strip()
            source = document.metadata.get("source", "")
            key = f"{source}::{content[:400]}"
            document_lookup[key] = document
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered_keys = sorted(scores, key=scores.get, reverse=True)
    return [document_lookup[key] for key in ordered_keys]


def _collection_name(profile: CorpusProfile) -> str:
    return f"rag-chroma-{profile.name}"


def _persist_directory(profile: CorpusProfile) -> Path:
    return Path("./.chroma") / profile.name


def get_collection_name(profile: CorpusProfile | None = None) -> str:
    return _collection_name(profile or get_active_corpus_profile())


def get_persist_directory(profile: CorpusProfile | None = None) -> Path:
    return _persist_directory(profile or get_active_corpus_profile())


def _load_seed_documents(urls: Sequence[str]) -> List[Document]:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )
    return text_splitter.split_documents(docs_list)


def _load_documents_from_vectorstore(vectorstore: Chroma) -> List[Document]:
    raw = vectorstore.get()
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    return [
        Document(page_content=page_content, metadata=metadata or {})
        for page_content, metadata in zip(documents, metadatas)
    ]


def load_documents_from_vectorstore(vectorstore: Chroma) -> List[Document]:
    return _load_documents_from_vectorstore(vectorstore)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    profile = get_active_corpus_profile()
    persist_directory = get_persist_directory(profile)
    collection_name = get_collection_name(profile)

    if persist_directory.exists():
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embed_model,
            persist_directory=str(persist_directory),
        )
        existing = vectorstore.get()
        if existing.get("ids"):
            return vectorstore

    seed_documents = _load_seed_documents(profile.urls)
    return Chroma.from_documents(
        documents=seed_documents,
        collection_name=collection_name,
        embedding=embed_model,
        persist_directory=str(persist_directory),
    )


@lru_cache(maxsize=1)
def get_seed_documents() -> List[Document]:
    vectorstore = get_vectorstore()
    documents = _load_documents_from_vectorstore(vectorstore)
    if documents:
        return documents
    return _load_seed_documents(get_active_corpus_profile().urls)


@dataclass
class HybridRetriever:
    vectorstore: Chroma | None = None
    documents: List[Document] | None = None
    dense_k: int = DEFAULT_TOP_K
    sparse_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        self.vectorstore = self.vectorstore or get_vectorstore()
        self.documents = self.documents or get_seed_documents()
        self.sparse_retriever = BM25Retriever.from_documents(self.documents)
        self.sparse_retriever.k = self.sparse_k

    def invoke(self, queries: str | Sequence[str]) -> List[Document]:
        normalized_queries = normalize_queries(queries)
        result_sets: List[List[Document]] = []

        for query in normalized_queries:
            dense_docs = self.vectorstore.similarity_search(query, k=self.dense_k)
            sparse_docs = self.sparse_retriever.invoke(query)
            result_sets.append(dense_docs)
            result_sets.append(sparse_docs)

        fused_documents = _reciprocal_rank_fusion(result_sets)
        return _dedupe_documents(fused_documents)[: self.dense_k + self.sparse_k]


def normalize_queries(queries: str | Sequence[str] | None) -> List[str]:
    if queries is None:
        return []
    if isinstance(queries, str):
        queries = [queries]

    cleaned = []
    for query in queries:
        query = (query or "").strip()
        if query and query not in cleaned:
            cleaned.append(query)
    return cleaned


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()


def open_vectorstore(
    embedding_function=None,
    profile: CorpusProfile | None = None,
) -> Chroma:
    profile = profile or get_active_corpus_profile()
    return Chroma(
        collection_name=get_collection_name(profile),
        embedding_function=embedding_function or embed_model,
        persist_directory=str(get_persist_directory(profile)),
    )


def get_local_documents(
    embedding_function=None,
    profile: CorpusProfile | None = None,
) -> List[Document]:
    vectorstore = open_vectorstore(
        embedding_function=embedding_function,
        profile=profile,
    )
    documents = load_documents_from_vectorstore(vectorstore)
    if documents:
        return documents

    raise RuntimeError(
        "No local corpus documents were found in the Chroma collection. "
        "Run ingestion.py first to build the local medical corpus index."
    )


def ingest_documents(urls: Sequence[str] | None = None) -> Chroma:
    profile = get_active_corpus_profile()
    urls = urls or profile.urls
    documents = _load_seed_documents(urls)
    return Chroma.from_documents(
        documents=documents,
        collection_name=_collection_name(profile),
        embedding=embed_model,
        persist_directory=str(_persist_directory(profile)),
    )


class RetrieverProxy:
    def invoke(self, queries: str | Sequence[str]) -> List[Document]:
        return get_hybrid_retriever().invoke(queries)


retriever = RetrieverProxy()


if __name__ == "__main__":
    ingest_documents()
