from typing import TypedDict, Dict, Any, List, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ── 1. Input ──────────────────────────────────────────────────────────────
    raw_input: Dict[str, Any]
    borrower_name: str            # Optional display name for personalized narrative

    # ── 2. ML Prediction ─────────────────────────────────────────────────────
    risk_score: float
    risk_class: str               # Low / Medium / High
    feature_contributions: List[Dict[str, Any]]  # ranked list from ml/predict.py

    # ── 3. Risk Interpretation ────────────────────────────────────────────────
    risk_drivers_explanation: str

    # ── 4. RAG ───────────────────────────────────────────────────────────────
    rag_query: str
    retrieved_documents: str
    rag_sources: List[str]

    # ── 5. LLM Reasoning ─────────────────────────────────────────────────────
    llm_reasoning: str
    llm_decision: str

    # ── 6. Domain Guardrail ───────────────────────────────────────────────────
    domain_blocked: bool
    domain_block_reason: str

    # ── 7. Decision Guardrail (Hybrid) ────────────────────────────────────────
    final_decision: str
    is_overridden: bool

    # ── 8. Formatted Output ───────────────────────────────────────────────────
    structured_output: Dict[str, Any]

    # ── 9. Chat History ───────────────────────────────────────────────────────
    messages: Annotated[Sequence[BaseMessage], add_messages]
