"""Parse a Langflow JSON export and run Classic RAG locally with LangChain."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

import config


@dataclass
class FlowSettings:
    llm_model: str = config.LLM_MODEL
    llm_base_url: str = config.OLLAMA_BASE_URL
    embed_model: str = config.EMBED_MODEL
    embed_base_url: str = config.OLLAMA_BASE_URL
    collection_name: str = config.CHROMA_COLLECTION
    persist_dir: str = config.CHROMA_DIR
    top_k: int = config.TOP_K
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP
    prompt_template: str = (
        "{context}\n\n---\n\n"
        "Given the context above, answer the question as best as possible.\n\n"
        "Question: {question}\n\nAnswer: "
    )
    source_file: str = ""
    warnings: list[str] = field(default_factory=list)


def _first_model_name(model_value) -> str | None:
    """Extract model name from Langflow's model field (list of dicts or plain string)."""
    if isinstance(model_value, list) and model_value:
        return model_value[0].get("name")
    if isinstance(model_value, str) and model_value:
        return model_value
    return None


def parse_langflow_json(path: str) -> FlowSettings:
    """Read a Langflow flow JSON and extract RAG pipeline settings."""
    with open(path) as f:
        flow = json.load(f)

    settings = FlowSettings(source_file=Path(path).name)
    nodes = flow.get("data", {}).get("nodes", [])

    for node in nodes:
        node_type = node["data"]["type"]
        tmpl = node["data"]["node"]["template"]

        if node_type == "LanguageModelComponent":
            model_name = _first_model_name(tmpl.get("model", {}).get("value"))
            if model_name:
                # Strip ":latest" suffix for Ollama compatibility
                settings.llm_model = model_name.replace(":latest", "")
            base_url = tmpl.get("ollama_base_url", {}).get("value", "")
            if base_url and not base_url.startswith("OLLAMA"):
                settings.llm_base_url = base_url
            else:
                settings.warnings.append(
                    "Language Model base URL was a placeholder — using default http://localhost:11434"
                )

        elif node_type == "EmbeddingModel":
            model_name = _first_model_name(tmpl.get("model", {}).get("value"))
            if model_name:
                settings.embed_model = model_name.replace(":latest", "")
            base_url = tmpl.get("ollama_base_url", {}).get("value", "")
            if base_url:
                settings.embed_base_url = base_url

        elif node_type == "Chroma":
            settings.collection_name = tmpl.get("collection_name", {}).get("value", settings.collection_name)
            settings.persist_dir = tmpl.get("persist_directory", {}).get("value", settings.persist_dir)
            top_k = tmpl.get("number_of_results", {}).get("value", 0)
            if top_k:
                settings.top_k = int(top_k)

        elif node_type == "Prompt":
            template = tmpl.get("template", {}).get("value", "")
            if template:
                settings.prompt_template = template

        elif node_type == "SplitText":
            chunk_size = tmpl.get("chunk_size", {}).get("value", 0)
            chunk_overlap = tmpl.get("chunk_overlap", {}).get("value", 0)
            if chunk_size:
                settings.chunk_size = int(chunk_size)
            if chunk_overlap:
                settings.chunk_overlap = int(chunk_overlap)

    return settings


def ask_from_flow(question: str, settings: FlowSettings) -> dict:
    """Run a Classic RAG query using settings extracted from a Langflow JSON."""
    embeddings = OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
    )
    vectorstore = Chroma(
        persist_directory=settings.persist_dir,
        embedding_function=embeddings,
        collection_name=settings.collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate.from_template(settings.prompt_template)
    llm = OllamaLLM(model=settings.llm_model, base_url=settings.llm_base_url)
    answer = (prompt | llm).invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [
            {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
            for doc in docs
        ],
        "model_used": settings.llm_model,
        "prompt_template": settings.prompt_template,
    }
