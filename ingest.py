"""Ingestion pipeline: load → chunk → embed → store."""
import os
import argparse
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import config

_BATCH_SIZE = 2000  # safe limit for SQLite variable count


def load_documents(source_dir: str) -> list:
    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
        "**/*.md": TextLoader,
    }
    docs = []
    for glob_pattern, loader_cls in loaders.items():
        loader = DirectoryLoader(source_dir, glob=glob_pattern, loader_cls=loader_cls, silent_errors=True)
        docs.extend(loader.load())
    return docs


def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def get_vectorstore(
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def embed_and_store(
    chunks: list,
    persist_dir: str,
    collection_name: str = config.CHROMA_COLLECTION,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Chroma:
    """Embed chunks in batches and store in ChromaDB. Calls progress_callback(done, total)."""
    vs = get_vectorstore(persist_dir, collection_name)
    total = len(chunks)
    embed_batch = 100  # chunks per Ollama embedding call batch
    for i in range(0, total, embed_batch):
        batch = chunks[i: i + embed_batch]
        vs.add_documents(batch)
        if progress_callback:
            progress_callback(min(i + embed_batch, total), total)
    return vs


def ingest(
    source_dir: str = config.DOCS_DIR,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    print(f"Loading documents from {source_dir} ...")
    docs = load_documents(source_dir)
    if not docs:
        raise ValueError(f"No documents found in {source_dir}")
    print(f"  Loaded {len(docs)} document(s).")

    print(f"Chunking (size={chunk_size}, overlap={chunk_overlap}) ...")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"  Created {len(chunks)} chunk(s).")

    print("Embedding and storing in Chroma ...")
    embed_and_store(chunks, persist_dir, collection_name, progress_callback=progress_callback)
    print(f"  Done. Vectorstore persisted to {persist_dir}")
    return len(chunks)


def _iter_all_in_batches(collection, include: list[str]) -> list[dict]:
    """Fetch all records from a ChromaDB collection in safe-sized batches."""
    offset = 0
    all_ids, all_docs, all_metas = [], [], []
    while True:
        batch = collection.get(limit=_BATCH_SIZE, offset=offset, include=include)
        if not batch["ids"]:
            break
        all_ids.extend(batch["ids"])
        if "documents" in include:
            all_docs.extend(batch.get("documents") or [])
        if "metadatas" in include:
            all_metas.extend(batch.get("metadatas") or [])
        offset += _BATCH_SIZE
    return {"ids": all_ids, "documents": all_docs, "metadatas": all_metas}


def get_all_chunks(
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
) -> dict:
    """Return all chunks with ids, documents, and metadatas from ChromaDB."""
    vs = get_vectorstore(persist_dir, collection_name)
    return _iter_all_in_batches(vs._collection, include=["documents", "metadatas"])


def list_sources(
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
) -> list[str]:
    """Return sorted list of unique source filenames stored in ChromaDB."""
    vs = get_vectorstore(persist_dir, collection_name)
    data = _iter_all_in_batches(vs._collection, include=["metadatas"])
    sources = {
        Path(m["source"]).name
        for m in data["metadatas"]
        if m and "source" in m
    }
    return sorted(sources)


def delete_by_source(
    source_name: str,
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
) -> int:
    """Delete all chunks whose source filename matches source_name. Returns count deleted."""
    vs = get_vectorstore(persist_dir, collection_name)
    data = _iter_all_in_batches(vs._collection, include=["metadatas"])
    ids_to_delete = [
        doc_id
        for doc_id, m in zip(data["ids"], data["metadatas"])
        if m and Path(m.get("source", "")).name == source_name
    ]
    for i in range(0, len(ids_to_delete), _BATCH_SIZE):
        vs._collection.delete(ids=ids_to_delete[i: i + _BATCH_SIZE])
    return len(ids_to_delete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--source", default=config.DOCS_DIR)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--persist-dir", default=config.CHROMA_DIR)
    args = parser.parse_args()

    ingest(args.source, args.chunk_size, args.chunk_overlap, args.persist_dir)
