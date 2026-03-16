from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import logging
import shap

# --------------------------------
# Initialize Flask
# --------------------------------
app = Flask(__name__, template_folder="../templates")

# --------------------------------
# Logging
# --------------------------------
logging.basicConfig(level=logging.INFO)

# --------------------------------
# Load model paths
# --------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "fraud_model_xgb.pkl")
feature_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(feature_path)
explainer = shap.TreeExplainer(model)

logging.info("Model and feature columns loaded successfully")


# --------------------------------
# Home page (UI)
# --------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# UI form prediction
# --------------------------------
@app.route("/predict-ui", methods=["POST"])
def predict_ui():

    try:

        type_value = request.form["type"]
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # One-hot encoding like training
        type_data = {
            "type_CASH_OUT": 1 if type_value == "CASH_OUT" else 0,
            "type_DEBIT": 1 if type_value == "DEBIT" else 0,
            "type_PAYMENT": 1 if type_value == "PAYMENT" else 0,
            "type_TRANSFER": 1 if type_value == "TRANSFER" else 0
        }

        data = {
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
        }

        data.update(type_data)

        df = pd.DataFrame([data])

        df = df.reindex(columns=feature_columns, fill_value=0)

        fraud_prob = model.predict_proba(df)[0][1]
        
        shap_values = explainer.shap_values(df)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_array = shap_values[1][0]
        else:
            shap_array = shap_values[0]

        feature_importance = pd.Series(
            shap_array,
            index=feature_columns
        ).sort_values(ascending=False)

        top_features = feature_importance.head(3).index.tolist()

        explanation_html = "<br>".join(top_features)
        
        if fraud_prob > 0.7:
            decision = "BLOCK TRANSACTION"
            risk = "HIGH"
        elif fraud_prob > 0.3:
            decision = "MANUAL REVIEW"
            risk = "MEDIUM"
        else:
            decision = "APPROVE"
            risk = "LOW"

        return f"""
        <html>

        <style>

        body {{
            font-family: Arial;
            background:#f1f5f9;
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
        }}

        .card {{
            background:white;
            padding:40px;
            border-radius:12px;
            width:420px;
            box-shadow:0 10px 25px rgba(0,0,0,0.1);
            text-align:center;
        }}

        .progress {{
            background:#e5e7eb;
            border-radius:10px;
            height:20px;
            margin-top:15px;
        }}

        .progress-bar {{
            height:100%;
            border-radius:10px;
            width:{fraud_prob*100}%;
            background:#ef4444;
        }}

        .low {{
            color:#16a34a;
        }}

        .medium {{
            color:#d97706;
        }}

        .high {{
            color:#dc2626;
        }}

        button {{
            margin-top:20px;
            padding:10px 20px;
            background:#2563eb;
            border:none;
            color:white;
            border-radius:6px;
            cursor:pointer;
        }}

        button:hover {{
            background:#1d4ed8;
        }}

        </style>

        <div class="card">

        <h2>Fraud Analysis Result</h2>

        <p><b>Fraud Probability:</b> {fraud_prob:.4f}</p>

        <div class="progress">
        <div class="progress-bar"></div>
        </div>

        <br>

        <p><b>Decision:</b> {decision}</p>

        <p class="{risk.lower()}"><b>Risk Level:</b> {risk}</p>

        <a href="/"><button>Analyze Another Transaction</button></a>

        </div>

        </html>
        """

    except Exception as e:

        return f"""
        <h3>Error occurred</h3>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """
# --------------------------------
# API health check
# --------------------------------
@app.route("/health")
def health():
    return jsonify({"status": "API running"})


# --------------------------------
# API prediction endpoint
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_columns, fill_value=0)

        fraud_prob = model.predict_proba(df)[0][1]

        if fraud_prob > 0.7:
            decision = "BLOCK TRANSACTION"
            risk_level = "HIGH"
        elif fraud_prob > 0.3:
            decision = "MANUAL REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "APPROVE"
            risk_level = "LOW"

        logging.info(f"Fraud probability: {fraud_prob}")

        return jsonify({
            "fraud_probability": float(fraud_prob),
            "decision": decision,
            "risk_level": risk_level
        })

    except Exception as e:

        logging.error(f"API Prediction error: {str(e)}")

        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# --------------------------------
# Run the app
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True)