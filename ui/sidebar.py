from __future__ import annotations
import streamlit as st
import json

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Borrower Profile Input Panel
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict | None:
    with open("encoders.json") as f:
        encoders = json.load(f)

    st.markdown(
        '<div class="card"><div class="card-header">'
        'BORROWER PROFILE INPUT</div></div>',
        unsafe_allow_html=True,
    )

    # Optional name for personalized narrative
    st.markdown('<div class="field-label">Borrower Name (optional)</div>', unsafe_allow_html=True)
    borrower_name = st.text_input(
        "Borrower Name",
        placeholder="e.g. Rahul Sharma",
        key="bname",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Personal Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="field-label">Age</div>', unsafe_allow_html=True)
    person_age = st.slider("Age", 18, 70, 30, key="age", label_visibility="collapsed")

    st.markdown('<div class="field-label">Annual Income ($)</div>', unsafe_allow_html=True)
    person_income = st.slider(
        "Annual Income ($)", 10000, 500000, 55000, step=1000, key="income", label_visibility="collapsed"
    )

    st.markdown('<div class="field-label">Employment Length (years)</div>', unsafe_allow_html=True)
    person_emp_length = st.slider(
        "Employment Length (years)", 0, 40, 5, key="emp", label_visibility="collapsed"
    )

    st.markdown('<div class="field-label">Home Ownership</div>', unsafe_allow_html=True)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"],
        key="home",
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">Loan Details</div>', unsafe_allow_html=True)

    st.markdown('<div class="field-label">Loan Amount ($)</div>', unsafe_allow_html=True)
    loan_amnt = st.slider(
        "Loan Amount ($)", 1000, 50000, 10000, step=500, key="loan", label_visibility="collapsed"
    )

    st.markdown('<div class="field-label">Interest Rate (%)</div>', unsafe_allow_html=True)
    loan_int_rate = st.slider(
        "Interest Rate (%)", 5.0, 25.0, 10.0, step=0.25, key="rate", label_visibility="collapsed"
    )

    st.markdown('<div class="field-label">Loan Purpose</div>', unsafe_allow_html=True)
    loan_intent_options = list(encoders.get("loan_intent", {}).keys()) or [
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]
    loan_intent = st.selectbox(
        "Loan Purpose", loan_intent_options, key="intent", label_visibility="collapsed"
    )

    st.markdown('<div class="field-label">Loan Grade</div>', unsafe_allow_html=True)
    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"],
        key="grade",
        label_visibility="collapsed",
    )

    # Live loan-to-income ratio indicator
    lti  = loan_amnt / max(person_income, 1)
    lti_pct = lti * 100
    lti_color = "#28a745" if lti_pct < 25 else ("#ffc107" if lti_pct < 42 else "#dc3545")
    st.markdown(
        f"""<div style="background:#F8F9FA; border-radius:8px; padding:12px 16px;
                        margin:14px 0 4px 0; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:13px; font-weight:500; color:#000000;">Live Loan-to-Income Ratio</span>
            <span style="font-size:18px; font-weight:700; color:{lti_color};">{lti_pct:.1f}%</span>
        </div>""",
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "Analyze Borrower", use_container_width=True, type="primary"
    )

    if analyze_clicked:
        return {
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "borrower_name": borrower_name,
        }
    return None
