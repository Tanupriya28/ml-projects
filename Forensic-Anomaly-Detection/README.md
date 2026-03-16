#  Forensic Document Integrity & Anomaly Detection

##  Project Overview
This project is an end-to-end forensic analytics and anomaly detection system designed to identify suspicious patterns, inconsistencies, and potential tampering in case-related documents and records.

The application combines machine learning, feature engineering, and an interactive dashboard to help analysts quickly flag high-risk cases and understand why a case was marked suspicious.

It is built as a Streamlit web application with explainable insights and visual analytics.

---

##  Problem Statement
In forensic and legal datasets, cases may contain:
- Delays in reporting (Incident, PM, FIR)
- Repeated document modifications
- Suspicious or inconsistent narrative text
- Potential data integrity issues

Manual review of such cases is time-consuming and error-prone.  
This project aims to automate anomaly detection while still providing human-readable explanations.

---

##  Key Features
- CSV file upload for batch case analysis
- Hybrid ML approach:
  - Isolation Forest (unsupervised anomaly detection)
  - XGBoost (supervised classification)
- Interactive visual insights:
  - Delay distribution
  - Delay vs severity analysis
  - District-wise comparison
  - Keyword frequency and heatmaps
- Explainable results showing why a case was flagged
- Downloadable outputs:
  - Filtered CSV
  - PDF summary report
- Dark-themed, professional forensic dashboard UI

---

##  Machine Learning Approach

### Models Used
- Isolation Forest  
  Detects unusual patterns without labeled data.
- XGBoost  
  Learns complex relationships from engineered features.

### Final Decision Logic
A case is flagged as suspicious if at least one of the models identifies it as anomalous.

---

##  Feature Engineering
Key engineered features include:
- Time gaps between:
  - Incident and Postmortem
  - Incident and FIR
- Delay-based severity scoring
- Narrative keyword frequency analysis
- Categorical encoding (district, officer ID, gender)
- Combined severity score using delay and narrative risk

---

##  Dashboard Screenshots
Dashboard screenshots are available in the `screenshots/` folder and include:
- Overall dashboard overview
- Delay distribution
- Delay vs severity scatter plot
- District-wise analysis
- Keyword frequency heatmap
- Suspicious case explanation view

---

##  Tech Stack
- Python
- Streamlit
- Pandas, NumPy
- Plotly
- Scikit-learn
- XGBoost
- Joblib
- FPDF

---

##  How to Run the Project

### 1. Clone the repository
git clone https://github.com/your-username/Forensic-Anomaly-Detection.git
cd Forensic-Anomaly-Detection

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Application
streamlit run app.py

### 4. Upload a CSV File
Use the provided cases.csv file or upload any dataset with a similar structure.

##  Project Structure

Forensic-Anomaly-Detection/
│

├── app.py

├── train_models.py

├── cases.csv

├── model_isolation_forest_15feat.pkl

├── model_xgboost_15feat.pkl

├── screenshots/


└── README.md

## ⚠️ Notes
- The dataset used is synthetic / sample data
- Model accuracy is not the primary focus

Emphasis is on:
- End-to-end workflow
- Explainability
- Interactive visualizations
- Built as a beginner-friendly portfolio project
