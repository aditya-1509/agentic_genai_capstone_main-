import os
import json
import pickle
import pandas as pd


GRADE_RISK   = {'A': 0.08, 'B': 0.22, 'C': 0.42, 'D': 0.65, 'E': 0.82, 'F': 0.92, 'G': 0.97}
OWN_RISK     = {'OWN': 0.18, 'MORTGAGE': 0.32, 'RENT': 0.54, 'OTHER': 0.72}
INTENT_RISK  = {
    'EDUCATION': 0.32, 'HOMEIMPROVEMENT': 0.38, 'MEDICAL': 0.40,
    'DEBTCONSOLIDATION': 0.50, 'PERSONAL': 0.56, 'VENTURE': 0.68
}


class RiskPredictor:
    def __init__(self, artifact_dir="."):
        with open(os.path.join(artifact_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(artifact_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(artifact_dir, 'encoders.json'), 'r') as f:
            self.encoders = json.load(f)
        with open(os.path.join(artifact_dir, 'model_columns.pkl'), 'rb') as f:
            self.model_columns = pickle.load(f)
        with open(os.path.join(artifact_dir, 'model_insights.json'), 'r') as f:
            insights = json.load(f)

        self.feature_importance = dict(
            zip(insights["feature_names"], insights["feature_importance"])
        )
        self.cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        self.num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                         'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

    def _preprocess(self, input_data: dict) -> pd.DataFrame:
        df = pd.DataFrame([input_data])

        if 'loan_percent_income' not in df.columns or pd.isna(df['loan_percent_income'].iloc[0]):
            df['loan_percent_income'] = df['loan_amnt'] / df['person_income'].clip(lower=1)

        if 'cb_person_default_on_file' not in df.columns:
            df['cb_person_default_on_file'] = "N"
        if 'cb_person_cred_hist_length' not in df.columns:
            df['cb_person_cred_hist_length'] = 5

        for col in self.cat_cols:
            val = str(df[col].iloc[0]) if col in df.columns else "N"
            df[col] = self.encoders.get(col, {}).get(val, 0)

        for col in self.num_cols:
            if col not in df.columns:
                df[col] = 0.0

        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df[self.model_columns]

    def score(self, input_data: dict) -> float:
        df = self._preprocess(input_data)
        return float(self.model.predict_proba(df)[0][1])

    def feature_contributions(self, input_data: dict) -> list:
        """
        Compute weighted feature contributions for this borrower
        using model feature importance + domain-driven risk directionality.
        Returns a list of dicts sorted by impact (highest first).
        """
        inc = max(input_data.get('person_income', 1), 1)
        lti = input_data.get('loan_amnt', 0) / inc

        grade   = input_data.get('loan_grade', 'C')
        own     = input_data.get('person_home_ownership', 'RENT')
        intent  = input_data.get('loan_intent', 'PERSONAL')
        emp     = input_data.get('person_emp_length', 5)
        income  = input_data.get('person_income', 50000)
        rate    = input_data.get('loan_int_rate', 10.0)

        contribs = [
            {
                "feature": "Loan Grade",
                "importance": self.feature_importance.get("loan_grade", 0),
                "risk_level": GRADE_RISK.get(grade, 0.45),
                "value": grade,
                "direction": "risk" if GRADE_RISK.get(grade, 0.45) > 0.40 else "safe",
            },
            {
                "feature": "Loan-to-Income Ratio",
                "importance": self.feature_importance.get("loan_percent_income", 0),
                "risk_level": min(lti * 1.8, 1.0),
                "value": f"{lti * 100:.1f}%",
                "direction": "risk" if lti > 0.30 else "safe",
            },
            {
                "feature": "Home Ownership",
                "importance": self.feature_importance.get("person_home_ownership", 0),
                "risk_level": OWN_RISK.get(own, 0.50),
                "value": own,
                "direction": "risk" if OWN_RISK.get(own, 0.50) > 0.40 else "safe",
            },
            {
                "feature": "Loan Purpose",
                "importance": self.feature_importance.get("loan_intent", 0),
                "risk_level": INTENT_RISK.get(intent, 0.50),
                "value": intent,
                "direction": "risk" if INTENT_RISK.get(intent, 0.50) > 0.50 else "safe",
            },
            {
                "feature": "Annual Income",
                "importance": self.feature_importance.get("person_income", 0),
                "risk_level": max(0.0, 1.0 - min(income / 150000.0, 1.0)),
                "value": f"${income:,.0f}",
                "direction": "safe" if income > 60000 else "risk",
            },
            {
                "feature": "Employment Length",
                "importance": self.feature_importance.get("person_emp_length", 0),
                "risk_level": max(0.0, 1.0 - min(emp / 20.0, 1.0)),
                "value": f"{emp} yrs",
                "direction": "safe" if emp >= 5 else "risk",
            },
            {
                "feature": "Interest Rate",
                "importance": self.feature_importance.get("loan_int_rate", 0),
                "risk_level": min(rate / 25.0, 1.0),
                "value": f"{rate}%",
                "direction": "risk" if rate > 15 else "safe",
            },
        ]

        # Sort by composite impact: importance * risk_level
        contribs.sort(key=lambda x: x["importance"] * x["risk_level"], reverse=True)
        return contribs


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — loaded once, reused across all requests
# ─────────────────────────────────────────────────────────────────────────────
_predictor = None


def _get_predictor() -> RiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RiskPredictor()
    return _predictor


def predict_risk(input_data: dict) -> dict:
    """
    Primary callable used throughout the agentic system.
    Returns risk_score, risk_class, and ranked feature_contributions.
    """
    pred = _get_predictor()
    prob = pred.score(input_data)
    contribs = pred.feature_contributions(input_data)

    if prob < 0.40:
        risk_class = "Low"
    elif prob <= 0.70:
        risk_class = "Medium"
    else:
        risk_class = "High"

    return {
        "risk_score": prob,
        "risk_class": risk_class,
        "feature_contributions": contribs,
    }
