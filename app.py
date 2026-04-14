import os
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT — supports both local .env and Streamlit Cloud secrets
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass  # Will surface an error later from LLM call

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Credit Risk Scoring System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-BUILD RAG INDEX — runs once on first deploy if chroma_db is absent
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists("rag/chroma_db") or not os.listdir("rag/chroma_db"):
    with st.spinner("Building regulatory knowledge base from PDF documents (first run only)..."):
        from rag.build_index import build_index
        build_index()
    st.success("Knowledge base ready.")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS — after RAG check to avoid slow startup on first deploy
# ─────────────────────────────────────────────────────────────────────────────
from ui.sidebar import render_sidebar
from ui.dashboard import render_dashboard
from agent.graph import run_ml_pipeline, chat_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

/* Header */
.header-bar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 22px 40px;
    border-radius: 0 0 16px 16px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
}
.header-title  { color: #fff; font-size: 21px; font-weight: 700; letter-spacing: 0.5px; }
.header-sub    { color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px; }
.header-badge  { background: rgba(255,255,255,0.12); color: #fff; padding: 6px 16px;
                  border-radius: 20px; font-size: 12px; font-weight: 500; }

/* Cards */
.card { background: #fff; border-radius: 12px; padding: 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 16px;
        border: 1px solid #E9ECEF; }
.card-header { font-size: 12px; font-weight: 600; color: #000000;
               text-transform: uppercase; letter-spacing: 1px;
               margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }

/* Labels */
.section-label { font-size: 11px; font-weight: 600; color: #000000;
                 text-transform: uppercase; letter-spacing: 1.2px; margin: 18px 0 8px 0; }
.field-label   { font-size: 13px; font-weight: 600; color: #000000;
                 margin: 12px 0 2px 0; letter-spacing: 0.2px; }

/* Decision banners */
.decision-approve    { background: linear-gradient(135deg,#d4edda,#c3e6cb); border-left: 5px solid #28a745;
                        padding: 20px 24px; border-radius: 8px; margin: 16px 0; }
.decision-approve .label  { font-size: 12px; font-weight: 600; color: #155724;
                              text-transform: uppercase; letter-spacing: 1px; }
.decision-approve .value  { font-size: 28px; font-weight: 800; color: #155724; margin-top: 4px; }

.decision-conditional { background: linear-gradient(135deg,#fff3cd,#ffeeba); border-left: 5px solid #ffc107;
                         padding: 20px 24px; border-radius: 8px; margin: 16px 0; }
.decision-conditional .label { font-size: 12px; font-weight: 600; color: #856404;
                                 text-transform: uppercase; letter-spacing: 1px; }
.decision-conditional .value { font-size: 28px; font-weight: 800; color: #856404; margin-top: 4px; }

.decision-decline   { background: linear-gradient(135deg,#f8d7da,#f5c6cb); border-left: 5px solid #dc3545;
                       padding: 20px 24px; border-radius: 8px; margin: 16px 0; }
.decision-decline .label { font-size: 12px; font-weight: 600; color: #721c24;
                             text-transform: uppercase; letter-spacing: 1px; }
.decision-decline .value { font-size: 28px; font-weight: 800; color: #721c24; margin-top: 4px; }

/* Content boxes */
.reasoning-box  { background: #F8F9FA; border-radius: 8px; padding: 16px 20px;
                   font-size: 14px; color: #000000; line-height: 1.75; }
.reg-item       { background: #F8F9FA; border-left: 3px solid #0f3460; padding: 10px 16px;
                   border-radius: 0 6px 6px 0; margin-bottom: 8px; font-size: 13px; color: #000000; }
.disclaimer-box { background: #F8F9FA; border: 1px solid #000000; border-radius: 8px;
                   padding: 16px 20px; font-size: 12px; color: #000000;
                   line-height: 1.6; margin-top: 18px; }

/* Force Chat Input to be static/inline instead of sticking to the bottom viewport */
div[data-testid="stChatInput"] {
    position: relative !important;
    bottom: auto !important;
    padding-bottom: 16px !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <div>
        <div class="header-title">Next-Gen Credit Intelligence: Agentic AI for Risk-Aware Lending</div>
    </div>
    <div class="header-badge">Agentic Workflow Active</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_gap, col_right = st.columns([4, 0.3, 7])

with col_left:
    form_data = render_sidebar()

    if form_data:
        borrower_name = form_data.pop("borrower_name", "")
        input_data    = form_data

        if st.session_state.get("input_data") != input_data:
            # Re-run ML Pipeline when form input changes
            with st.spinner("Agent running: ML prediction & Guardrails..."):
                try:
                    result = run_ml_pipeline(input_data, borrower_name)
                    st.session_state["ml_output"] = result
                    st.session_state["input_data"] = input_data
                    
                    # Initialize conversational agent memory (empty by default to remove clutter)
                    st.session_state["messages"] = []
                except Exception as e:
                    st.error(f"ML execution error: {e}")

with col_right:
    # If no data exists yet, show empty dashboard prompt
    if not st.session_state.get("ml_output"):
        render_dashboard(None, None)
    else:
        # ── SEPARATE EXPERIENCES INTO TABS TO REDUCE COGNITIVE LOAD ──────────
        tab1, tab2 = st.tabs(["ML Risk Analysis Dashboard", "Agentic RAG Copilot"])
        
        with tab1:
            render_dashboard(
                st.session_state.get("ml_output"),
                st.session_state.get("input_data"),
            )

        with tab2:
            if "messages" in st.session_state and st.session_state.get("ml_output", {}).get("domain_blocked") != True:
                st.markdown(
                    '#### AGENTIC RAG COPILOT \n'
                    '<div style="font-size:14px; color:#000000; margin-bottom:24px;">'
                    'Chat directly with the agent about this borrower to uncover hidden risks, or ask questions about regulatory guidelines (RAG). <i>Example: "Does this income meet the RBI minimum threshold for this loan grade?"</i></div>',
                    unsafe_allow_html=True
                )
                
                # Chat Input inline at top
                user_query = st.chat_input("Ask about regulations or borrower risk...")
                if user_query:
                    st.session_state["messages"].append(HumanMessage(content=user_query))
                    
                    with st.spinner("Agent reasoning & checking documents..."):
                        try:
                            context_msg = SystemMessage(content=f"Current Output Data: {st.session_state['ml_output']}")
                            payload = {"messages": [context_msg] + st.session_state["messages"]}
                            
                            agent_response = chat_agent.invoke(payload)
                            final_msg = agent_response["messages"][-1]
                            
                            st.session_state["messages"].append(final_msg)
                        except Exception as e:
                            st.error(f"Agent error: {e}")
                    st.rerun()  # Forces layout reset
                
                # Output existing chat messages (Newest Pair Top, Oldest Pair Bottom)
                messages_list = st.session_state["messages"]
                # Group into chunks of 2 (User, AI)
                message_pairs = [messages_list[i:i+2] for i in range(0, len(messages_list), 2)]
                
                for pair in reversed(message_pairs):
                    for msg in pair:
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        with st.chat_message(role):
                            st.write(msg.content)
