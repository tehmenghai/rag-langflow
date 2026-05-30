"""Langflow REST API client for Classic RAG and Agentic RAG flows."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from langchain_core.documents import Document

import config


# ── Custom exceptions ─────────────────────────────────────────────────────────

class LangflowError(Exception):
    """Base Langflow error."""


class LangflowConnectionError(LangflowError):
    """Langflow server is unreachable."""


class LangflowFlowError(LangflowError):
    """Flow execution returned a non-2xx response."""


# ── Typed result dataclasses ──────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: Optional[float] = None


@dataclass
class ClassicRAGTrace:
    query: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    model_used: str = ""
    prompt_template: str = ""
    answer: str = ""
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


@dataclass
class AgentStep:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class LoopIteration:
    loop_num: int
    query_used: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = float("inf")
    passed: bool = False


@dataclass
class AgenticRAGTrace:
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: list[LoopIteration] = field(default_factory=list)
    reranked: bool = False
    answer: str = ""
    citations: list[RetrievedChunk] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


# ── Client ────────────────────────────────────────────────────────────────────

class LangflowClient:
    def __init__(self, base_url: str = config.LANGFLOW_BASE_URL, timeout: int = config.LANGFLOW_REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ── Connection ────────────────────────────────────────────────────────────

    def test_connection(self) -> tuple[bool, str]:
        """Returns (success, message). Never raises."""
        try:
            resp = httpx.get(f"{self.base_url}/api/v1/flows/", timeout=5)
            resp.raise_for_status()
            flows = resp.json()
            count = len(flows) if isinstance(flows, list) else 0
            return True, f"Connected — {count} flow(s) found"
        except httpx.ConnectError:
            return False, f"Cannot reach Langflow at {self.base_url}. Is Langflow Desktop running?"
        except httpx.TimeoutException:
            return False, f"Connection timed out reaching {self.base_url}."
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def get_flows(self) -> list[dict]:
        """Return list of {id, name, description} dicts."""
        try:
            resp = self._client.get(f"{self.base_url}/api/v1/flows/")
            resp.raise_for_status()
            raw = resp.json()
            flows = raw if isinstance(raw, list) else raw.get("flows", [])
            return [
                {
                    "id": f.get("id", ""),
                    "name": f.get("name", "Unnamed"),
                    "description": f.get("description", ""),
                }
                for f in flows
            ]
        except httpx.ConnectError as e:
            raise LangflowConnectionError(str(e)) from e
        except Exception as e:
            raise LangflowError(str(e)) from e

    # ── Core HTTP ─────────────────────────────────────────────────────────────

    def _run_flow(self, flow_id: str, query: str, output_type: str = "all") -> dict:
        """POST to /api/v1/run/{flow_id} and return raw JSON response."""
        if not flow_id:
            raise LangflowFlowError("No flow ID provided.")
        url = f"{self.base_url}/api/v1/run/{flow_id}"
        payload = {
            "input_value": query,
            "input_type": "chat",
            "output_type": output_type,
        }
        try:
            resp = self._client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.ConnectError as e:
            raise LangflowConnectionError(str(e)) from e
        except httpx.HTTPStatusError as e:
            raise LangflowFlowError(f"HTTP {e.response.status_code}: {e.response.text[:200]}") from e
        except Exception as e:
            raise LangflowError(str(e)) from e

    # ── Public run methods ────────────────────────────────────────────────────

    def run_classic_rag(self, flow_id: str, query: str) -> ClassicRAGTrace:
        """Run a Classic RAG flow and return a structured trace."""
        try:
            raw = self._run_flow(flow_id, query, output_type="all")
            return self._parse_classic_trace(query, raw)
        except Exception as e:
            return ClassicRAGTrace(query=query, error=str(e))

    def run_agentic_rag(self, flow_id: str, query: str) -> AgenticRAGTrace:
        """Run an Agentic RAG flow and return a structured trace."""
        try:
            raw = self._run_flow(flow_id, query, output_type="all")
            return self._parse_agentic_trace(query, raw)
        except Exception as e:
            return AgenticRAGTrace(query=query, error=str(e))

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_classic_trace(self, query: str, raw: dict, debug: bool = False) -> ClassicRAGTrace:
        trace = ClassicRAGTrace(query=query)
        try:
            outputs = raw.get("outputs", [{}])[0].get("outputs", [])
        except (IndexError, KeyError, TypeError):
            trace.error = "Unexpected response structure from Langflow."
            return trace

        for comp in outputs:
            name = comp.get("component_display_name", "").lower()
            results = comp.get("results", {})

            # Retriever / vector store component — parse JSON into LangChain Documents
            if any(k in name for k in ("chroma", "retriev", "vector", "search")):
                docs = results.get("documents", results.get("data", {}).get("documents", []))
                if isinstance(docs, list):
                    for doc in docs:
                        if isinstance(doc, dict):
                            content = doc.get("page_content", doc.get("text", ""))
                            source = (doc.get("metadata") or {}).get("source", "unknown")
                            score = doc.get("score")
                        else:
                            content = str(doc)
                            source = "unknown"
                            score = None
                        # Use LangChain Document to normalise the Langflow JSON payload
                        lc_doc = Document(
                            page_content=content,
                            metadata={"source": source, "score": score},
                        )
                        trace.retrieved_chunks.append(
                            RetrievedChunk(
                                content=lc_doc.page_content,
                                source=lc_doc.metadata["source"],
                                score=lc_doc.metadata.get("score"),
                            )
                        )

            # LLM / model component → answer
            elif any(k in name for k in ("chat output", "chatoutput", "llm", "model", "text output")) and "input" not in name:
                msg = results.get("message", {})
                if isinstance(msg, dict):
                    trace.answer = msg.get("text", msg.get("data", {}).get("text", ""))
                elif isinstance(msg, str):
                    trace.answer = msg
                if not trace.answer:
                    trace.answer = results.get("text", "")

            # Prompt component
            elif "prompt" in name:
                tmpl = results.get("template", results.get("text", ""))
                if tmpl:
                    trace.prompt_template = tmpl

            # Model name hint from any component
            if not trace.model_used:
                model_name = comp.get("params", {}).get("model_name", "") or comp.get("params", {}).get("model", "")
                if model_name:
                    trace.model_used = model_name

        if debug:
            trace._raw = raw  # type: ignore[attr-defined]

        return trace

    def _parse_agentic_trace(self, query: str, raw: dict) -> AgenticRAGTrace:
        trace = AgenticRAGTrace(query=query)
        try:
            outputs = raw.get("outputs", [{}])[0].get("outputs", [])
        except (IndexError, KeyError, TypeError):
            trace.error = "Unexpected response structure from Langflow."
            return trace

        for comp in outputs:
            name = comp.get("component_display_name", "").lower()
            results = comp.get("results", {})

            # Final answer from chat output
            if any(k in name for k in ("chat output", "chatoutput", "text output")) and "input" not in name:
                msg = results.get("message", {})
                if isinstance(msg, dict):
                    trace.answer = msg.get("text", msg.get("data", {}).get("text", ""))
                elif isinstance(msg, str):
                    trace.answer = msg

            # Agent component — parse ReAct reasoning steps
            if "agent" in name:
                messages = results.get("messages", [])
                full_text = ""
                for m in messages:
                    content = m.get("content", m.get("text", "")) if isinstance(m, dict) else str(m)
                    full_text += content + "\n"
                if full_text.strip():
                    trace.steps.extend(_parse_react_steps(full_text))

        # If no steps parsed but we have an answer, create a single summary step
        if not trace.steps and trace.answer:
            trace.steps.append(AgentStep(thought="Agent processed the query.", observation=trace.answer[:300]))

        return trace


# ── ReAct step parser ─────────────────────────────────────────────────────────

_REACT_PATTERN = re.compile(
    r"(?:Thought:\s*(.*?))?(?:Action:\s*(.*?))?(?:Action Input:\s*(.*?))?(?:Observation:\s*(.*?))?(?=Thought:|Action:|$)",
    re.DOTALL,
)


def _parse_react_steps(text: str) -> list[AgentStep]:
    """Best-effort parse of ReAct-formatted agent log text into AgentStep objects."""
    steps: list[AgentStep] = []

    # Split on known ReAct keywords
    segments = re.split(r"(?=Thought:|Action:|Observation:)", text.strip())

    current: dict[str, str] = {}
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("Thought:"):
            if current:
                steps.append(_dict_to_step(current))
                current = {}
            current["thought"] = segment[len("Thought:"):].strip()
        elif segment.startswith("Action Input:"):
            current["action_input"] = segment[len("Action Input:"):].strip()
        elif segment.startswith("Action:"):
            current["action"] = segment[len("Action:"):].strip()
        elif segment.startswith("Observation:"):
            current["observation"] = segment[len("Observation:"):].strip()

    if current:
        steps.append(_dict_to_step(current))

    # Fallback: no structured format found — return as one step
    if not steps:
        steps.append(AgentStep(thought=text.strip()[:600]))

    return steps


def _dict_to_step(d: dict) -> AgentStep:
    return AgentStep(
        thought=d.get("thought", ""),
        action=d.get("action", ""),
        action_input=d.get("action_input", ""),
        observation=d.get("observation", ""),
    )
