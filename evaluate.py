"""RAGAS evaluation module — run a golden set against Classic and Agentic RAG pipelines."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

# ── Compatibility shim: ragas 0.2.x imports langchain_community.chat_models.vertexai
# which was removed in langchain-community 0.3+. Stub it out before importing ragas.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_core.language_models.chat_models import BaseChatModel

    _stub = ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAI(BaseChatModel):
        def _generate(self, *args, **kwargs):
            raise NotImplementedError("Stub — install langchain-google-vertexai for VertexAI support.")

        @property
        def _llm_type(self):
            return "vertexai"

    _stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from langchain_ollama import OllamaEmbeddings, OllamaLLM

import config

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ALL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def load_golden_set(path: str | Path = GOLDEN_SET_PATH) -> list[dict]:
    """Load the golden set JSON. Returns list of {question, ground_truth, source_hint}."""
    with open(path) as f:
        return json.load(f)


def _make_judge():
    """Build RAGAS-compatible LLM and embedding wrappers using local Ollama."""
    llm = OllamaLLM(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    emb = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def run_ragas_eval(
    golden_set: list[dict],
    pipeline_fn: Callable[[str], dict],
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """
    Run RAGAS evaluation against pipeline_fn.

    pipeline_fn(question: str) → dict with keys:
        - "answer": str
        - "sources": list of {"content": str, "source": str}

    Returns:
        {
          "scores": {"faithfulness": float, "answer_relevancy": float, ...},
          "per_question": [{"question", "answer", "scores": {...}}],
          "elapsed_ms": float,
          "error": str | None,
        }
    """
    t0 = time.time()
    samples = []
    per_question = []
    total = len(golden_set)

    for i, item in enumerate(golden_set):
        question = item["question"]
        ground_truth = item["ground_truth"]
        try:
            result = pipeline_fn(question)
            answer = result.get("answer", "")
            contexts = [s["content"] for s in result.get("sources", [])]
        except Exception as exc:
            answer = f"[pipeline error: {exc}]"
            contexts = []

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        per_question.append({"question": question, "answer": answer, "scores": {}})

        if progress_callback:
            progress_callback(i + 1, total)

    dataset = EvaluationDataset(samples=samples)
    judge_llm, judge_emb = _make_judge()

    try:
        result = evaluate(
            dataset,
            metrics=ALL_METRICS,
            llm=judge_llm,
            embeddings=judge_emb,
            show_progress=False,
            raise_exceptions=False,
        )
        df = result.to_pandas()

        aggregate: dict[str, float] = {}
        for col in METRIC_NAMES:
            if col in df.columns:
                aggregate[col] = round(float(df[col].mean()), 4)
            else:
                aggregate[col] = float("nan")

        for i, row in df.iterrows():
            for col in METRIC_NAMES:
                per_question[i]["scores"][col] = round(float(row.get(col, float("nan"))), 4)

        weakest = min(
            (k for k in aggregate if aggregate[k] == aggregate[k]),  # skip NaN
            key=lambda k: aggregate[k],
            default=None,
        )

        return {
            "scores": aggregate,
            "weakest": weakest,
            "per_question": per_question,
            "elapsed_ms": (time.time() - t0) * 1000,
            "error": None,
        }

    except Exception as exc:
        return {
            "scores": {k: float("nan") for k in METRIC_NAMES},
            "weakest": None,
            "per_question": per_question,
            "elapsed_ms": (time.time() - t0) * 1000,
            "error": str(exc),
        }


if __name__ == "__main__":
    import query

    golden = load_golden_set()
    print(f"Loaded {len(golden)} golden questions.\n")

    def classic_fn(q: str) -> dict:
        return query.ask(q, top_k=config.TOP_K, persist_dir=config.CHROMA_DIR,
                         collection_name=config.CHROMA_COLLECTION)

    print("Running evaluation against Classic RAG …")
    res = run_ragas_eval(golden, classic_fn, progress_callback=lambda d, t: print(f"  {d}/{t}"))

    if res["error"]:
        print(f"Error: {res['error']}")
    else:
        print("\n=== Scorecard ===")
        for metric, score in res["scores"].items():
            tag = " <-- weakest" if metric == res["weakest"] else ""
            print(f"  {metric:25s} {score:.4f}{tag}")
        print(f"\nElapsed: {res['elapsed_ms']:.0f} ms")
        print("\n=== Per-question breakdown ===")
        for i, pq in enumerate(res["per_question"], 1):
            scores_str = "  ".join(f"{k[:2].upper()}:{v:.2f}" for k, v in pq["scores"].items())
            print(f"  Q{i:02d} [{scores_str}] {pq['question'][:60]}")
