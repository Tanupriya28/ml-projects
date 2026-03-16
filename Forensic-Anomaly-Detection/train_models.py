import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import joblib

def generate_features(df):
    df = df.copy()
    def to_dt(x):
        try:
            return pd.to_datetime(x)
        except:
            return pd.NaT

    df["incident_dt"] = df["incident_date"].apply(to_dt)
    df["pm_dt"]       = df["pm_date"].apply(to_dt)
    df["fir_dt"]      = df["fir_date"].apply(to_dt)

    df["incident_ts"] = df["incident_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df["pm_ts"]       = df["pm_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df["fir_ts"]      = df["fir_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)

    df["pm_gap_hours"] = (df["pm_ts"] - df["incident_ts"])/3600
    df["fir_gap_days"] = (df["fir_ts"] - df["incident_ts"])/(3600*24)

    cat_cols = ["victim_gender", "district", "officer_id"]
    for c in cat_cols:
        df[c + "_enc"] = pd.factorize(df[c].astype(str))[0]

    keywords = ["missing", "delay", "inconsistent", "contradict", "tamper",
                "not available", "discrepancy", "unclear", "revised"]
    df["narrative_kw_score"] = df["narrative"].apply(lambda t: sum(str(t).lower().count(k) for k in keywords))
    df["severity_score"] = (df["delay_hours"].fillna(0)/100) + (df["narrative_kw_score"]/5)

    feature_cols = [
        "victim_age","pm_present","modifications","access_count","delay_hours",
        "tampered_flag","incident_ts","pm_ts","pm_gap_hours","fir_gap_days",
        "victim_gender_enc","district_enc","officer_id_enc","narrative_kw_score","severity_score"
    ]
    X = df[feature_cols].fillna(0)
    return df, X

# Load raw CSV
df = pd.read_csv("cases.csv")
df_features, X = generate_features(df)

# Train Isolation Forest

iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_model.fit(X)
iso_pred = iso_model.predict(X)
y = np.where(iso_pred==-1,1,0)  

# Train XGBoost

xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X, y)

# Save models
joblib.dump(iso_model, "model_isolation_forest_15feat.pkl")
joblib.dump(xgb_model, "model_xgboost_15feat.pkl")

print("Models trained and saved successfully!")
