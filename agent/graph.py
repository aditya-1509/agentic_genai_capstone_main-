from __future__ import annotations
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from ml.predict import predict_risk
from rag.retriever import get_relevant_regulations

load_dotenv()
_api_key = os.environ.get("GROQ_API_KEY", "")

# Fast model for domain classification
llm_fast = ChatGroq(model_name="llama-3.1-8b-instant", api_key=_api_key, temperature=0)

# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIAL PIPELINE (Non-conversational)
# ─────────────────────────────────────────────────────────────────────────────
# This replaces the old app_graph. It runs the ML prediction and hybrid guardrail,
# and returns the structured output for the top Dashboard widgets.

def run_ml_pipeline(raw_input: Dict[str, Any], borrower_name: str) -> Dict[str, Any]:
    # 1. Structural Check
    required = {"person_income", "loan_amnt", "loan_int_rate", "loan_grade"}
    missing  = required - set(raw_input.keys())
    if missing:
        return {
            "domain_blocked": True,
            "domain_block_reason": f"Missing required fields: {', '.join(missing)}"
        }

    # 2. Semantic Domain Check (Fail-Open)
    if _api_key:
        check_prompt = (
            f"Is this a valid credit risk lending context?\n"
            f"Income: {raw_input.get('person_income')}, Loan: {raw_input.get('loan_amnt')}, "
            f"Grade: {raw_input.get('loan_grade')}, Purpose: {raw_input.get('loan_intent')}\n"
            f"Respond with ONE word only: VALID or INVALID"
        )
        try:
            resp = llm_fast.invoke([
                SystemMessage(content="You are a strict domain classifier. Reply ONLY with VALID or INVALID."),
                HumanMessage(content=check_prompt),
            ]).content.strip().upper()
            label = resp.split()[0] if resp else "VALID"
            if "INVALID" in label and "VALID" not in label.replace("INVALID", ""):
                 return {"domain_blocked": True, "domain_block_reason": "LLM classified this input as off-domain."}
        except Exception:
            pass

    # 3. ML Prediction
    ml_result = predict_risk(raw_input)
    score = ml_result["risk_score"]
    
    # 4. Hybrid Decision Guardrail
    if score > 0.70:
        final_decision = "Reject"
    elif score >= 0.40:
        final_decision = "Review"
    else:
        final_decision = "Approve"

    # 5. Formatter
    name = borrower_name.strip() or "Borrower"
    inc = raw_input.get("person_income", 0)
    
    output = {
        "profile_summary": (
            f"{name} | Age {raw_input.get('person_age')} | "
            f"{raw_input.get('person_emp_length')} yrs employment | "
            f"Income ${inc:,.0f} | {raw_input.get('person_home_ownership')} | "
            f"Grade {raw_input.get('loan_grade')} loan"
        ),
        "risk_score": f"{score:.3f}",
        "risk_class": ml_result["risk_class"],
        "decision": final_decision,
        "feature_contributions": ml_result["feature_contributions"],
        "confidence": "High — hybrid rule-based guardrail applied",
    }
    
    return {"domain_blocked": False, "structured_output": output}

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERSATIONAL RAG AGENT
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_regulations_tool(query: str) -> str:
    """Retrieve RBI credit risk regulations and policies from the knowledge base. 
    Use this tool whenever asked about regulations, rules, compliance, or guidelines."""
    docs, _ = get_relevant_regulations(query)
    return docs

@tool
def search_web_tool(query: str) -> str:
    """Search the live internet for general credit risk information, news, or rules.
    CRITICAL INSTRUCTION: Use this tool ONLY as a fallback if `search_regulations_tool` returns 'No regulatory guidelines found', or if the user explicitly asks for live web information."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        runner = DuckDuckGoSearchRun()
        return runner.run(query)
    except ImportError:
        return "Web search is unavailable (duckduckgo-search package not installed)."
    except Exception as e:
        return f"Web search failed: {e}"

SYSTEM_PROMPT = """You are an Intelligent Lending Decision Support Agent.
You are engaged in a conversation with an underwriter to help them understand a borrower's risk profile.
You have access to `search_regulations_tool` (primary RBI knowledge) and `search_web_tool` (fallback internet search).

STRICT RULES:
- Operate EXCLUSIVELY in the domain of credit risk, lending decisions, compliance, and financial underwriting.
- If asked about topics outside of this, gracefully decline to answer.
- Base your advice on the user's ML predictions provided in the chat context.
- ALWAYS try `search_regulations_tool` first for rule queries. Only use `search_web_tool` if the regulations tool fails or has no info.
- Do NOT make definitive approve/reject actions (only advise)."""

llm_smart = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=_api_key, temperature=0.2)
chat_agent = create_react_agent(llm_smart, tools=[search_regulations_tool, search_web_tool], prompt=SYSTEM_PROMPT)
