from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from graph.corpus_profiles import CorpusProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CorpusAsset:
    name: str
    kind: str
    usage: str
    description: str
    path: str = ""
    urls: Sequence[str] = ()


@dataclass(frozen=True)
class CorpusManifest:
    profile_name: str
    version: str
    description: str
    assets: Sequence[CorpusAsset]
    manifest_path: Path


def _text_splitter():
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=50,
    )


def _manifest_base_path(manifest_path: Path) -> Path:
    return manifest_path.parent


def _resolve_manifest_path(profile: CorpusProfile) -> Path | None:
    if not profile.manifest_path:
        return None
    path = Path(profile.manifest_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_corpus_manifest(profile: CorpusProfile) -> CorpusManifest | None:
    manifest_path = _resolve_manifest_path(profile)
    if manifest_path is None or not manifest_path.exists():
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = [
        CorpusAsset(
            name=asset["name"],
            kind=asset["kind"],
            usage=asset["usage"],
            description=asset.get("description", ""),
            path=asset.get("path", ""),
            urls=tuple(asset.get("urls", [])),
        )
        for asset in payload.get("assets", [])
    ]
    return CorpusManifest(
        profile_name=payload.get("profile_name", profile.name),
        version=payload.get("version", profile.corpus_version),
        description=payload.get("description", profile.description),
        assets=assets,
        manifest_path=manifest_path,
    )


def describe_corpus_manifest(profile: CorpusProfile) -> Dict[str, object]:
    manifest = load_corpus_manifest(profile)
    if manifest is None:
        return {
            "profile_name": profile.name,
            "version": profile.corpus_version,
            "manifest_path": None,
            "assets": [],
        }

    return {
        "profile_name": manifest.profile_name,
        "version": manifest.version,
        "manifest_path": str(manifest.manifest_path),
        "assets": [
            {
                "name": asset.name,
                "kind": asset.kind,
                "usage": asset.usage,
                "description": asset.description,
                "path": asset.path,
                "urls": list(asset.urls),
            }
            for asset in manifest.assets
        ],
    }


def _load_jsonl_documents(asset: CorpusAsset, manifest_path: Path, profile: CorpusProfile) -> List:
    from langchain.schema import Document

    asset_path = _manifest_base_path(manifest_path) / asset.path
    splitter = _text_splitter()
    documents: List = []
    with asset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            base_document = Document(
                page_content=row.get("content", ""),
                metadata={
                    "source": row.get("source", row.get("url", asset.name)),
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "doc_id": row.get("doc_id", ""),
                    "asset_name": asset.name,
                    "asset_usage": asset.usage,
                    "corpus_profile": profile.name,
                    "corpus_version": profile.corpus_version,
                },
            )
            documents.extend(splitter.split_documents([base_document]))
    return documents


def _load_web_documents(asset: CorpusAsset, profile: CorpusProfile) -> List:
    from langchain_community.document_loaders import WebBaseLoader

    docs = [WebBaseLoader(url).load() for url in asset.urls]
    docs_list = [item for sublist in docs for item in sublist]
    splitter = _text_splitter()
    documents = splitter.split_documents(docs_list)
    for document in documents:
        document.metadata = {
            **(document.metadata or {}),
            "asset_name": asset.name,
            "asset_usage": asset.usage,
            "corpus_profile": profile.name,
            "corpus_version": profile.corpus_version,
        }
    return documents


def load_corpus_documents(profile: CorpusProfile, usage: str = "serve") -> List:
    manifest = load_corpus_manifest(profile)
    if manifest is None:
        return []

    documents: List = []
    for asset in manifest.assets:
        if asset.usage != usage:
            continue
        if asset.kind == "jsonl_documents":
            documents.extend(_load_jsonl_documents(asset, manifest.manifest_path, profile))
        elif asset.kind == "web_urls":
            documents.extend(_load_web_documents(asset, profile))
    return documents


def iter_corpus_assets(profile: CorpusProfile, usage: str | None = None) -> Iterable[CorpusAsset]:
    manifest = load_corpus_manifest(profile)
    if manifest is None:
        return []
    if usage is None:
        return list(manifest.assets)
    return [asset for asset in manifest.assets if asset.usage == usage]
