"""Query pipeline: embed → retrieve top-k → prompt Ollama → return answer."""
from __future__ import annotations

import re
import time as _time

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import config
from langflow_client import AgenticRAGTrace, LoopIteration, RetrievedChunk

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_WITH_CITATIONS = """You are a helpful assistant. Answer the question using ONLY the provided context.
Cite each source inline as [1], [2], etc. matching the context numbers.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore(
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


def ask(
    question: str,
    top_k: int = config.TOP_K,
    persist_dir: str = config.CHROMA_DIR,
    collection_name: str = config.CHROMA_COLLECTION,
) -> dict:
    vectorstore = load_vectorstore(persist_dir, collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)

    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in docs
        ],
    }


def ask_agentic(
    question: str,
    top_k: int = config.TOP_K,
    persist_dir: str = config.CHROMA_DIR,
) -> AgenticRAGTrace:
    """Agentic RAG with query-reformulation loop and LLM reranking fallback.

    Loop up to MAX_AGENTIC_LOOPS times; on each pass retrieve with similarity
    scores and check against SIMILARITY_THRESHOLD (L2 distance, lower = better).
    If the threshold is never met, gather all candidate chunks and ask the LLM
    to rerank them before generating the final answer with inline citations.
    """
    t0 = _time.time()
    trace = AgenticRAGTrace(query=question)
    vectorstore = load_vectorstore(persist_dir)
    llm = OllamaLLM(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)

    current_query = question
    winning_chunks: list[RetrievedChunk] = []

    for loop_num in range(1, config.MAX_AGENTIC_LOOPS + 1):
        results = vectorstore.similarity_search_with_score(current_query, k=top_k)
        chunks = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=float(score),
            )
            for doc, score in results
        ]
        best_score = min((c.score for c in chunks), default=float("inf"))
        passed = best_score < config.SIMILARITY_THRESHOLD

        trace.iterations.append(
            LoopIteration(
                loop_num=loop_num,
                query_used=current_query,
                chunks=chunks,
                best_score=best_score,
                passed=passed,
            )
        )

        if passed:
            winning_chunks = chunks
            break

        if loop_num < config.MAX_AGENTIC_LOOPS:
            current_query = _rephrase_query(llm, question, loop_num)

    if not winning_chunks:
        # Collect unique chunks across all iterations then rerank
        trace.reranked = True
        seen: set[str] = set()
        candidates: list[RetrievedChunk] = []
        for it in trace.iterations:
            for c in it.chunks:
                if c.content not in seen:
                    seen.add(c.content)
                    candidates.append(c)
        winning_chunks = _rerank_chunks(llm, question, candidates)[:top_k]

    trace.citations = winning_chunks

    # Build numbered context for citation-aware generation
    context = "\n\n---\n\n".join(
        f"[{i}] {c.content}" for i, c in enumerate(winning_chunks, 1)
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_CITATIONS)
    trace.answer = (prompt | llm).invoke({"context": context, "question": question})
    trace.elapsed_ms = (_time.time() - t0) * 1000
    return trace


# ── Helpers ───────────────────────────────────────────────────────────────────

_REPHRASE_PROMPTS = [
    "Rephrase this question more specifically to improve document search results. Output ONLY the rephrased question.\nOriginal: {q}\nRephrased:",
    "Rewrite this question using different keywords to surface relevant documents. Output ONLY the rewritten question.\nOriginal: {q}\nRewritten:",
]


def _rephrase_query(llm: OllamaLLM, question: str, attempt: int) -> str:
    template = _REPHRASE_PROMPTS[min(attempt - 1, len(_REPHRASE_PROMPTS) - 1)]
    try:
        return llm.invoke(template.format(q=question)).strip()
    except Exception:
        return question


def _rerank_chunks(llm: OllamaLLM, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Ask the LLM to rank chunks by relevance in a single call."""
    if not chunks:
        return chunks
    n = len(chunks)
    passages = "\n".join(f"[{i + 1}] {c.content[:250]}" for i, c in enumerate(chunks))
    prompt = (
        f"Given the question, rank these {n} passages by relevance (most relevant first).\n"
        f"Output ONLY a comma-separated list of numbers, e.g.: 2,1,3\n\n"
        f"Question: {question}\n\nPassages:\n{passages}\n\nRanking:"
    )
    try:
        response = llm.invoke(prompt).strip()
        indices = [int(x.strip()) - 1 for x in re.split(r"[,\s]+", response) if x.strip().isdigit()]
        valid = [i for i in indices if 0 <= i < n]
        missing = [i for i in range(n) if i not in valid]
        return [chunks[i] for i in valid + missing]
    except Exception:
        return chunks


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = ask(question)
    print("\nAnswer:", result["answer"])
    print("\nSources used:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']}: {src['content'][:120]}...")
