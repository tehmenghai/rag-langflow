"""Query pipeline: embed → retrieve top-k → prompt Ollama → return answer."""
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import config

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

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


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = ask(question)
    print("\nAnswer:", result["answer"])
    print("\nSources used:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']}: {src['content'][:120]}...")
