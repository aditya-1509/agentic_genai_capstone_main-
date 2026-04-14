import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix

def main():
    print("Loading Dataset...")
    df = pd.read_csv('credit_risk_dataset-1.csv')
    
    # Missing value imputation matching notebook logic
    print("Imputing missing values...")
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    print("Dropping duplicates...")
    df.drop_duplicates(inplace=True)
    
    print("Encoding categorical features...")
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        # Save mapping for potential UI dropdowns
        encoders[col] = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        
    print("Saving encoders mapping...")
    with open('encoders.json', 'w') as f:
        json.dump(encoders, f)
        
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling numerical features...")
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    
    print("Saving scaler...")
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Training XGBoost Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = xgb_model.predict(X_val)
    y_score = xgb_model.predict_proba(X_val)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    
    roc_auc = roc_auc_score(y_val, y_score)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    print("Saving model and feature insights...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
    # Save column order for prediction consistency
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # Save confusion matrix & ROC data for the Insights page
    cm = confusion_matrix(y_val, y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_score)
    from sklearn.metrics import auc
    roc_auc_val = auc(fpr, tpr)
    
    insights = {
        'confusion_matrix': cm.tolist(),
        'feature_importance': xgb_model.feature_importances_.tolist(),
        'feature_names': X.columns.tolist(),
        'roc': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': roc_auc_val
        }
    }
    with open('model_insights.json', 'w') as f:
        json.dump(insights, f)
        
    print("Training Complete! Artifacts saved.")

if __name__ == "__main__":
    main()
