import streamlit as st
from ml.predict import predict_risk


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _decision_css(decision: str) -> str:
    if decision == "Approve":
        return "decision-approve"
    elif decision == "Review":
        return "decision-conditional"
    return "decision-decline"


def _risk_bar(score_val: float):
    needle = score_val * 100
    st.markdown(
        f"""<div style="position:relative; margin:10px 0 4px 0;">
            <div style="display:flex; height:30px; border-radius:6px; overflow:hidden;">
                <div style="flex:40; background:#28a745; display:flex; align-items:center;
                            justify-content:center; color:white; font-size:11px; font-weight:600;">LOW</div>
                <div style="flex:30; background:#ffc107; display:flex; align-items:center;
                            justify-content:center; color:#333; font-size:11px; font-weight:600;">MEDIUM</div>
                <div style="flex:30; background:#dc3545; display:flex; align-items:center;
                            justify-content:center; color:white; font-size:11px; font-weight:600;">HIGH</div>
            </div>
            <div style="position:absolute; top:-4px; left:{needle:.1f}%;
                        width:3px; height:38px; background:#1a1a2e; border-radius:2px; z-index:10;">
                <div style="position:absolute; top:-5px; left:-4px; width:11px; height:11px;
                            background:#1a1a2e; border-radius:50%;"></div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#000000; margin-top:2px;">
            <span>0%</span><span>40%</span><span>70%</span><span>100%</span>
        </div>""",
        unsafe_allow_html=True,
    )


def _feature_bars(contributions: list):
    if not contributions:
        return
    st.markdown(
        '<div class="card"><div class="card-header">'
        'RISK DRIVER ANALYSIS (Model Feature Importance)</div>',
        unsafe_allow_html=True,
    )
    for c in contributions[:6]:
        pct       = int(c["risk_level"] * 100)
        imp_pct   = c["importance"] * 100
        bar_color = "#dc3545" if c["direction"] == "risk" else "#28a745"
        label_color = "#721c24" if c["direction"] == "risk" else "#155724"
        st.markdown(
            f"""<div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; font-size:12px;
                            font-weight:600; color:#000000; margin-bottom:4px;">
                    <span>{c["feature"]} <span style="color:#000000; font-weight:400;">
                        ({c["value"]})</span></span>
                    <span style="color:{label_color};">
                        Model Weight: {imp_pct:.1f}%</span>
                </div>
                <div style="background:#E9ECEF; border-radius:4px; height:22px; overflow:hidden;">
                    <div style="width:{pct}%; background:{bar_color}; height:100%;
                                display:flex; align-items:center; justify-content:flex-end;
                                padding-right:8px; color:white; font-size:11px; font-weight:600;
                                border-radius:4px; min-width:30px;">
                        {pct}%
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _what_if_section(original_input: dict, original_score: float, original_decision: str):
    st.markdown(
        '<div class="card"><div class="card-header">'
        'WHAT-IF SCENARIO ANALYSIS</div>'
        '<div style="font-size:13px; color:#000000; margin-bottom:16px;">'
        'Adjust parameters below to instantly recalculate risk using the ML model only (no LLM call).'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="field-label">Income ($)</div>', unsafe_allow_html=True)
        wi_income = st.slider(
            "WI Income", 10000, 500000,
            int(original_input.get("person_income", 55000)),
            step=1000, key="wi_income", label_visibility="collapsed",
        )
    with c2:
        st.markdown('<div class="field-label">Loan Amount ($)</div>', unsafe_allow_html=True)
        wi_loan = st.slider(
            "WI Loan", 1000, 50000,
            int(original_input.get("loan_amnt", 10000)),
            step=500, key="wi_loan", label_visibility="collapsed",
        )
    with c3:
        st.markdown('<div class="field-label">Loan Grade</div>', unsafe_allow_html=True)
        grade_options = ["A", "B", "C", "D", "E", "F", "G"]
        orig_grade    = original_input.get("loan_grade", "C")
        wi_grade = st.selectbox(
            "WI Grade", grade_options,
            index=grade_options.index(orig_grade) if orig_grade in grade_options else 2,
            key="wi_grade", label_visibility="collapsed",
        )

    wi_input = original_input.copy()
    wi_input["person_income"] = wi_income
    wi_input["loan_amnt"]     = wi_loan
    wi_input["loan_grade"]    = wi_grade

    wi_result   = predict_risk(wi_input)
    wi_score    = wi_result["risk_score"]
    wi_decision = wi_result["risk_class"]
    delta       = (wi_score - original_score) * 100
    delta_color = "#dc3545" if delta > 0 else "#28a745"
    delta_sign  = "+" if delta > 0 else ""

    # Hybrid decision for what-if
    if wi_score > 0.70:
        wi_verdict = "Reject"
        verdict_color = "#dc3545"
    elif wi_score >= 0.40:
        wi_verdict = "Review"
        verdict_color = "#ffc107"
    else:
        wi_verdict = "Approve"
        verdict_color = "#28a745"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Original Score", f"{original_score * 100:.1f}%")
    with m2:
        st.metric("Adjusted Score", f"{wi_score * 100:.1f}%", delta=f"{delta_sign}{delta:.1f}%")
    with m3:
        st.markdown(
            f'<div style="text-align:center; padding-top:16px;">'
            f'<span style="background:{verdict_color}; color:white; padding:6px 14px; '
            f'border-radius:16px; font-size:13px; font-weight:700;">{wi_verdict}</span></div>',
            unsafe_allow_html=True,
        )
    with m4:
        change = "improves" if delta < -0.5 else ("worsens" if delta > 0.5 else "unchanged")
        st.markdown(
            f'<div style="font-size:12px; color:#000000; padding-top:18px;">'
            f'Profile risk <b style="color:{delta_color};">{change}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def _download_report(out: dict, original_input: dict) -> str:
    score_pct = float(out.get("risk_score", 0)) * 100
    sources   = "\n  - ".join(out.get("sources", []))
    contribs  = out.get("feature_contributions", [])
    drivers_text = "\n  ".join(
        [f"{c['feature']} ({c['value']}): {c['direction'].upper()} — "
         f"model weight {c['importance']*100:.1f}%, risk level {int(c['risk_level']*100)}%"
         for c in contribs[:5]]
    ) or "None"

    return f"""CREDIT RISK ASSESSMENT REPORT
{'=' * 60}

BORROWER SNAPSHOT
{'-' * 60}
  Profile:          {out.get('profile_summary', 'N/A')}
  Age:              {original_input.get('person_age')}
  Annual Income:    ${original_input.get('person_income', 0):,.2f}
  Employment:       {original_input.get('person_emp_length')} years
  Home Ownership:   {original_input.get('person_home_ownership')}
  Loan Amount:      ${original_input.get('loan_amnt', 0):,.2f}
  Interest Rate:    {original_input.get('loan_int_rate')}%
  Loan Purpose:     {original_input.get('loan_intent')}
  Loan Grade:       {original_input.get('loan_grade')}
  Loan-to-Income:   {original_input.get('loan_amnt', 0) / max(original_input.get('person_income', 1), 1) * 100:.1f}%

RISK ANALYSIS
{'-' * 60}
  Default Probability:  {score_pct:.1f}%
  Risk Class:           {out.get('risk_class', 'N/A')}
  Confidence:           {out.get('confidence', 'N/A')}

TOP RISK DRIVERS (Model Feature Importance)
{'-' * 60}
  {drivers_text}

RECOMMENDED ACTION
{'-' * 60}
  {out.get('decision', 'N/A')}
  (Threshold: <40% Approve | 40-70% Review | >70% Reject)

AGENT REASONING NARRATIVE
{'-' * 60}
  {out.get('reason', 'N/A')}

REGULATORY SOURCES CONSIDERED (RAG)
{'-' * 60}
  - {sources if sources else 'None'}

DISCLAIMER
{'-' * 60}
  {out.get('disclaimer', 'AI-generated recommendation.')}
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(agent_output: dict, original_input: dict):
    if not agent_output:
        st.markdown(
            """<div class="card" style="min-height:480px; display:flex; align-items:center;
                        justify-content:center; flex-direction:column;">
                <div style="font-size:48px; color:#000000; margin-bottom:16px;">&#9671;</div>
                <div style="font-size:16px; color:#000000; font-weight:500;">
                    Enter borrower details and click Analyze</div>
                <div style="font-size:13px; color:#000000; margin-top:6px;">
                    The agentic decision summary will appear here</div>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    out = agent_output.get("structured_output", {})

    # ── Domain Guardrail Blocked State ────────────────────────────────────────
    if agent_output.get("domain_blocked", False) or out.get("decision") == "Blocked":
        st.markdown(
            """<div style="background:linear-gradient(135deg,#f8d7da,#f5c6cb);
                           border-left:5px solid #dc3545; padding:24px; border-radius:8px; margin:16px 0;">
                <div style="font-size:12px; font-weight:600; color:#721c24; text-transform:uppercase; letter-spacing:1px;">
                    Domain Scope Guardrail — Request Blocked</div>
                <div style="font-size:22px; font-weight:800; color:#721c24; margin-top:8px;">
                    OUT OF DOMAIN — ACCESS DENIED</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card"><div class="card-header">GUARDRAIL REASON</div>'
            f'<div class="reasoning-box">'
            f'{out.get("reason", agent_output.get("domain_block_reason", "Blocked."))}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="disclaimer-box"><b>System Notice:</b> '
            f'{out.get("disclaimer", "Restricted to credit risk assessment only.")}</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Decision Banner ───────────────────────────────────────────────────────
    decision = out.get("decision", "Unknown")
    score_str = out.get("risk_score", "0.0")
    try:
        score_val = float(score_str)
    except Exception:
        score_val = 0.0

    st.markdown(
        f'<div class="{_decision_css(decision)}">'
        f'<div class="label">Agent Lending Recommendation</div>'
        f'<div class="value">{decision}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Borrower Snapshot KPIs ────────────────────────────────────────────────
    st.markdown(
        '<div class="card"><div class="card-header">'
        'BORROWER SNAPSHOT</div>',
        unsafe_allow_html=True,
    )
    if original_input:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Annual Income", f"${original_input.get('person_income', 0):,.0f}")
        k2.metric("Loan Amount",   f"${original_input.get('loan_amnt', 0):,.0f}")
        lti_d = original_input.get("loan_amnt", 0) / max(original_input.get("person_income", 1), 1) * 100
        k3.metric("Loan-to-Income", f"{lti_d:.1f}%")
        k4.metric("Employment",    f"{original_input.get('person_emp_length', 0)} yrs")
    st.markdown(
        f'<div style="font-size:13px; color:#000000; padding-top:8px;">'
        f'{out.get("profile_summary", "")}</div></div>',
        unsafe_allow_html=True,
    )

    # ── Risk Score Gauge ──────────────────────────────────────────────────────
    st.markdown(
        f'<div class="card"><div class="card-header">RISK SCORE</div>'
        f'<div style="font-size:42px; font-weight:800; color:#1a1a2e; margin-bottom:4px;">'
        f'{score_val * 100:.1f}%'
        f'<span style="font-size:14px; font-weight:400; color:#000000;"> probability of default</span></div>'
        f'<div style="font-size:14px; font-weight:600; color:#000000; margin-bottom:10px;">'
        f'Risk Class: {out.get("risk_class", "")}</div>',
        unsafe_allow_html=True,
    )
    _risk_bar(score_val)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Feature Contribution Bars ─────────────────────────────────────────────
    _feature_bars(out.get("feature_contributions", []))

    # Static Story Mode component removed to make way for conversational Chat Interface in app.py

    # ── What-If Analysis ──────────────────────────────────────────────────────
    if original_input:
        _what_if_section(original_input, score_val, decision)

    # ── Disclaimer + Export ───────────────────────────────────────────────────
    st.markdown(
        f'<div class="disclaimer-box"><b>Confidence:</b> {out.get("confidence", "N/A")}<br>'
        f'<b>Disclaimer:</b> {out.get("disclaimer", "AI-generated recommendation.")}</div>',
        unsafe_allow_html=True,
    )

    if original_input:
        report = _download_report(out, original_input)
        st.download_button(
            label="Download Decision Report (.txt)",
            data=report,
            file_name="credit_risk_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
