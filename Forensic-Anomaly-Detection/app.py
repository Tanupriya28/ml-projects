import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
import plotly.express as px
import plotly.io as pio
import joblib

st.set_page_config(page_title="Forensic Data Integrity Checker", layout="wide")
pio.templates.default = "plotly_dark"


# ---------------------------
# FEATURE ENGINEERING, PDF, EXPLAIN, LOAD MODELS
# ---------------------------
def generate_features(df):
    df = df.copy()
    def to_dt(x):
        try: return pd.to_datetime(x)
        except: return pd.NaT
    df["incident_dt"] = df["incident_date"].apply(to_dt)
    df["pm_dt"] = df["pm_date"].apply(to_dt)
    df["fir_dt"] = df["fir_date"].apply(to_dt)
    df["incident_ts"] = df["incident_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df["pm_ts"] = df["pm_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df["fir_ts"] = df["fir_dt"].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    df["pm_gap_hours"] = (df["pm_ts"] - df["incident_ts"])/3600
    df["fir_gap_days"] = (df["fir_ts"] - df["incident_ts"])/(3600*24)
    cat_cols = ["victim_gender", "district", "officer_id"]
    for c in cat_cols:
        df[c + "_enc"] = pd.factorize(df[c].astype(str))[0]
    keywords = ["missing", "delay", "inconsistent", "contradict", "tamper",
                "not available", "discrepancy", "unclear", "revised"]
    df["narrative_kw_score"] = df["narrative"].apply(lambda t: sum(str(t).lower().count(k) for k in keywords))
    all_text = " ".join(df["narrative"].astype(str))
    df.attrs["top_keywords"] = {k: all_text.lower().count(k) for k in keywords}
    df["severity_score"] = (df["delay_hours"].fillna(0)/100) + (df["narrative_kw_score"]/5)
    feature_cols = [
        "victim_age","pm_present","modifications","access_count","delay_hours",
        "tampered_flag","incident_ts","pm_ts","pm_gap_hours","fir_gap_days",
        "victim_gender_enc","district_enc","officer_id_enc","narrative_kw_score","severity_score"
    ]
    X = df[feature_cols].fillna(0)
    return df, X

def create_pdf(df, top_keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Forensic Case Analysis Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(0, 8, f"Total cases: {len(df)}", ln=True)
    pdf.cell(0, 8, f"Suspicious cases: {df['final_anomaly'].sum()}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 8, "Top Keywords in Narratives:", ln=True)
    for k, v in top_keywords.items():
        pdf.cell(0, 6, f"{k}: {v}", ln=True)
    flagged = df[df['final_anomaly']==1][["case_id","victim_age","district","delay_hours","tampered_flag","narrative_kw_score","severity_score"]]
    pdf.set_font("Arial", "", 10)
    pdf.ln(10)
    for i, row in flagged.iterrows():
        pdf.multi_cell(0, 5, f"{row.to_dict()}")
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

def explain_case(row, medians):
    reasons = []

    # Delay explanation
    if row["delay_hours"] > medians["delay_hours"] * 3:
        reasons.append(f"Unusually long delay: {row['delay_hours']} hours")

    # PM timing explanation
    if pd.notnull(row["pm_gap_hours"]):
        if row["pm_gap_hours"] < 0:
            reasons.append(f"PM conducted before incident (data error): {row['pm_gap_hours']:.1f} hrs")
        elif row["pm_gap_hours"] > medians["pm_gap_hours"] * 3:
            reasons.append(f"PM conducted very late: {row['pm_gap_hours']:.1f} hrs")

    # FIR timing explanation
    if pd.notnull(row["fir_gap_days"]):
        if row["fir_gap_days"] < 0:
            reasons.append(f"FIR filed before incident (data error): {row['fir_gap_days']:.1f} days")
        elif row["fir_gap_days"] > medians["fir_gap_days"] * 3:
            reasons.append(f"FIR filed unusually late: {row['fir_gap_days']:.1f} days")
        else:
            reasons.append(f"FIR filed within expected timeframe: {row['fir_gap_days']:.1f} days")

    # Other reasons
    if row["narrative_kw_score"] > 0:
        reasons.append("Narrative contains suspicious keywords")
    if row["modifications"] > medians["modifications"] * 2:
        reasons.append(f"High modifications: {row['modifications']}")
    if row["tampered_flag"] == 1:
        reasons.append("Marked tampered by officer")
    if row["severity_score"] > medians["severity_score"] * 2:
        reasons.append(f"High severity score: {row['severity_score']:.2f}")
    if row["iso_pred"] == 1:
        reasons.append("Isolation Forest flagged anomaly")
    if row["xgb_pred"] == 1:
        reasons.append("XGBoost flagged anomaly")

    return reasons if reasons else ["ML model flagged as suspicious without clear reason"]

@st.cache_resource
def load_models():
    iso_model = joblib.load("model_isolation_forest_15feat.pkl")
    xgb_model = joblib.load("model_xgboost_15feat.pkl")
    return iso_model, xgb_model

def run_anomaly_detection(X, iso_model, xgb_model):
    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred==-1,1,0)
    xgb_pred = xgb_model.predict(X)
    final_anomaly = np.where((iso_pred + xgb_pred)>=1,1,0)
    return iso_pred, xgb_pred, final_anomaly

def fix_plotly_visibility(fig):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        title_font=dict(color="black", size=18),
        legend=dict(font=dict(color="black")),
        xaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        )
    )
    return fig


# ---------------------------
# MODERN UI
# ---------------------------
st.markdown("""
<style>

/* ============================================
   GLOBAL BACKGROUND – Deep Cyber Black
============================================ */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #05070b !important; /* deep crime-dark */
    color: #ffffff !important; /* ALL TEXT WHITE */
    font-family: 'Inter', sans-serif;
}

/* ============================================
   PAGE HEADER – Electric Blue Glow
============================================ */
h1 {
    font-size: 46px !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg,#2da7ff,#7dd2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 25px rgba(60,150,255,0.55);
}
h2, h3 {
    color: #ffffff !important;
    text-shadow: 0 0 10px rgba(0,150,255,0.3);
}

/* ============================================
   SIDEBAR – Dark Panels + Orange Highlight
============================================ */
section[data-testid="stSidebar"] {
    background-color: #0b0d11 !important;
    border-right: 1px solid rgba(255,128,0,0.35);
}
/* Sidebar text (labels only, not inputs) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}


/* Radio buttons */
.stRadio label {
    color: #ffffff !important;
}

/* ============================================
   FILE UPLOADER (Fixed)
============================================ */
[data-testid="stFileUploader"] > div:first-child {
    background-color: #0f1115 !important;
    border: 1px solid #1f232a !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] button {
    background: #1e90ff !important;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
}
[data-testid="stFileUploader"] button:hover {
    background: #1c7de0 !important;
}

/* ============================================
   SELECTBOX (District)
============================================ */
.stSelectbox div[data-baseweb="select"] {
    background-color: #0f1115 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
ul[role="listbox"] {
    background-color: #0f1115 !important;
    border: 1px solid #1f232a !important;
}
ul[role="listbox"] li {
    color: #ffffff !important;
}
ul[role="listbox"] li:hover {
    background-color: #1c1f25 !important;
}

/* ============================================
   SLIDER – Blue Cyber Bar
============================================ */
.stSlider > div > div {
    background: #1b1f25 !important;
}
.stSlider [role="slider"] {
    background: #2da7ff !important;
}
.stSlider label {
    color: #ffffff !important;
}

/* ============================================
   KPI CARDS – Cyber Panels + Orange Borders
============================================ */
.kpi-card {
    background: #0f1115 !important;
    border-radius: 18px;
    padding: 26px;
    border: 1px solid rgba(255,128,0,0.35);
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    color: #ffffff !important;
}
.kpi-title {
    color: #d0d0d0 !important;
}
.kpi-value {
    color: #2da7ff !important; /* cyber blue */
}

/* ============================================
   DATAFRAME TABLE
============================================ */
.dataframe thead th {
    background: #16191f !important;
    color: #2da7ff !important; /* bright blue */
}
.dataframe tbody td {
    background: #0c0e12 !important;
    color: #ffffff !important;
}
.dataframe tbody tr:hover td {
    background: #1c1f25 !important;
}

/* ============================================
   EXPANDERS
============================================ */
.streamlit-expanderHeader {
    background: #13161c !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,128,0,0.35);
    border-radius: 10px;
}
.streamlit-expanderContent {
    background: #0c0e12 !important;
    color: #ffffff !important;
}

/* ============================================
   DOWNLOAD BUTTONS
============================================ */
.stDownloadButton > button {
    background: linear-gradient(90deg,#ff7e29,#ff5500) !important; /* crime orange */
    color: #ffffff !important;
    font-weight: 700;
    border-radius: 10px;
}
.stDownloadButton > button:hover {
    background: linear-gradient(90deg,#ff5500,#ff7e29) !important;
}

/* ============================================
   PLOTLY DARK MODE FIX
============================================ */
.js-plotly-plot .plotly {
    background: #0f1115 !important;
    border-radius: 12px;
}

g.grid line {
    stroke: #333 !important;
}
g .xaxis path, g .yaxis path {
    stroke: #666 !important;
}

/* ============================================
   INPUT TEXT FIX (FINAL WORKING)
============================================ */

/* Text input (District) */
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Selectbox selected value (North / South / etc.) */
div[data-baseweb="select"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* File uploader text */
div[data-testid="stFileUploader"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

</style>


""", unsafe_allow_html=True)

st.title("Forensic Document Integrity & Anomaly Detection")
st.markdown("Upload CSV data to analyze cases, flag anomalies, and generate insights.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_features, X = generate_features(df_raw)
    iso_model, xgb_model = load_models()
    iso_pred, xgb_pred, final_anomaly = run_anomaly_detection(X, iso_model, xgb_model)
    df_features["iso_pred"] = iso_pred
    df_features["xgb_pred"] = xgb_pred
    df_features["final_anomaly"] = final_anomaly

    # --- KPI Cards ---
    total_cases = len(df_features)
    suspicious_cases = df_features["final_anomaly"].sum()
    avg_severity = df_features["severity_score"].mean()
    avg_delay = df_features["delay_hours"].mean()
    max_delay = df_features["delay_hours"].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='kpi-card'>🗂<div class='kpi-title'>Total Cases</div><div class='kpi-value'>{total_cases}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card'>⚠️<div class='kpi-title'>Suspicious Cases</div><div class='kpi-value'>{suspicious_cases} <span class='badge'>ALERT</span></div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card'>📊<div class='kpi-title'>Avg Severity</div><div class='kpi-value'>{avg_severity:.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card'>⏱<div class='kpi-title'>Avg Delay Hours</div><div class='kpi-value'>{avg_delay:.1f}</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='kpi-card'>⏳<div class='kpi-title'>Max Delay Hours</div><div class='kpi-value'>{max_delay:.1f}</div></div>", unsafe_allow_html=True)

    # Sidebar filters
    page = st.sidebar.radio("Navigate", ["All Cases", "Suspicious Cases"])
    st.sidebar.header("Filters")
    districts = ["All"] + df_features["district"].dropna().unique().tolist()
    selected_district = st.sidebar.selectbox("District", districts)
    severity_min, severity_max = st.sidebar.slider("Severity Range", 0.0, df_features["severity_score"].max(), (0.0, df_features["severity_score"].max()))

    df_filtered = df_features.copy()
    if selected_district != "All":
        df_filtered = df_filtered[df_filtered["district"]==selected_district]
    df_filtered = df_filtered[(df_filtered["severity_score"]>=severity_min)&(df_filtered["severity_score"]<=severity_max)]

    # --- All Cases ---
    if page == "All Cases":
        st.subheader("Filtered Dataset Preview")
        st.dataframe(df_filtered.style.highlight_max(subset=['severity_score','delay_hours'], color='#fde68a'))

        st.subheader("Visual Insights")

        # Delay Distribution
        fig1 = px.histogram(df_filtered, x="delay_hours", nbins=30, hover_data=["district","victim_age"], color_discrete_sequence=['#4f46e5'], title="Delay Hours Distribution")
        fig1 = fix_plotly_visibility(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Analysis:** Most cases cluster between low delay hours (0–50 hours). Peaks in higher ranges indicate a small number of cases with unusually long delays, which could warrant further investigation.")

        # Delay vs Severity
        fig2 = px.scatter(df_filtered, x="delay_hours", y="severity_score", color="final_anomaly", hover_data=["district","victim_age"], color_discrete_sequence=['#6366f1','#f43f5e'], title="Delay vs Severity")
        fig2 = fix_plotly_visibility(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Analysis:** The scatter shows that cases with higher delay tend to have higher severity. Blue points represent flagged anomalies—these typically have extreme delay and severity values.")

        # Top Keywords Frequency
        fig3 = px.bar(x=list(df_features.attrs["top_keywords"].keys()), y=list(df_features.attrs["top_keywords"].values()), text=list(df_features.attrs["top_keywords"].values()), color=list(df_features.attrs["top_keywords"].values()), color_continuous_scale=px.colors.sequential.Purples, title="Top Keywords Frequency")
        fig3.update_traces(textfont=dict(color="black", size=14))
        fig3 = fix_plotly_visibility(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        top_kw_sorted = sorted(df_features.attrs["top_keywords"].items(), key=lambda x: x[1], reverse=True)
        most_freq_kw = top_kw_sorted[0][0] if top_kw_sorted else "None"
        st.markdown(f"**Analysis:** The most frequent keyword is **'{most_freq_kw}'**, indicating that this issue appears most commonly in case narratives. Bars directly reflect keyword frequency across all cases.")

        # Avg Delay & Severity per District
        district_stats = df_filtered.groupby("district").agg({"delay_hours":"mean","severity_score":"mean","final_anomaly":"sum"}).reset_index()
        fig_avg = px.bar(district_stats, x="district", y=["delay_hours","severity_score"], barmode='group', color_discrete_sequence=['#4f46e5','#f43f5e'], title="Avg Delay & Severity per District")
        fig_avg = fix_plotly_visibility(fig_avg)
        st.plotly_chart(fig_avg, use_container_width=True)
        st.markdown("**Analysis:** Some districts consistently show higher average delay and severity scores. Taller bars highlight districts needing closer monitoring for anomalies.")

        # Top Officers
        top_officers = df_features[df_features["final_anomaly"]==1]["officer_id"].value_counts().head(10)
        if not top_officers.empty:
            fig_officers = px.bar(x=top_officers.index, y=top_officers.values, text=top_officers.values, color=top_officers.values, color_continuous_scale=px.colors.sequential.Viridis, title="Top Officers by Suspicious Cases")
            fig_officers.update_traces(textfont=dict(color="black", size=14))  
            fig_officers = fix_plotly_visibility(fig_officers)  
            st.plotly_chart(fig_officers, use_container_width=True)
            st.markdown("**Analysis:** Officers with higher bars are repeatedly handling suspicious cases. This may indicate areas to review for procedural consistency or anomalies.")

        # Severity Breakdown
        severity_bins = [0,0.5,1.5,df_features["severity_score"].max()+1]
        severity_labels = ["Low","Medium","High"]
        df_filtered["severity_level"] = pd.cut(df_filtered["severity_score"], bins=severity_bins, labels=severity_labels)
        severity_counts = df_filtered["severity_level"].value_counts()
        fig_severity = px.pie(values=severity_counts.values, names=severity_counts.index, color_discrete_sequence=['#4f46e5','#6366f1','#f43f5e'], title="Severity Level Breakdown")
        fig_severity.update_traces(textfont=dict(color="black", size=16))  
        fig_severity = fix_plotly_visibility(fig_severity) 
        st.plotly_chart(fig_severity, use_container_width=True)
        st.markdown("**Analysis:** Most cases fall under Low or Medium severity. High severity cases, although fewer, are critical for attention and investigation.")

        # Keyword Heatmap
        keyword_matrix = pd.DataFrame([{**{"district": row["district"]}, **{k: str(row["narrative"]).lower().count(k) for k in df_features.attrs["top_keywords"].keys()}} for idx, row in df_filtered.iterrows()])
        keyword_heat = keyword_matrix.groupby("district").sum()
        fig_heat = px.imshow(keyword_heat, text_auto=True, aspect="auto", color_continuous_scale=px.colors.sequential.Purples, title="Keyword Frequency Heatmap by District")
        fig_heat.update_traces(textfont=dict(color="black", size=12))  
        fig_heat = fix_plotly_visibility(fig_heat) 
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("**Analysis:** Darker cells indicate higher counts of specific keywords per district. This helps identify regions where particular issues (e.g., 'not available') appear more frequently.")

        # Download buttons
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv, "filtered_results.csv", "text/csv")
        pdf_bytes = create_pdf(df_filtered, df_features.attrs["top_keywords"])
        st.download_button("Download PDF Summary", pdf_bytes, "summary.pdf", "application/pdf")

    # --- Suspicious Cases ---
    elif page == "Suspicious Cases":
        st.subheader("Suspicious Cases Details")
        suspicious_df = df_filtered[df_filtered["final_anomaly"]==1]
        medians = df_features.median(numeric_only=True)
        if suspicious_df.empty:
            st.info("No suspicious cases found with current filters.")
        else:
            def highlight_severity(row):
                if row.severity_score > medians['severity_score'] * 2:
                    return ['background-color:#fca5a5']*len(row)
                elif row.severity_score > medians['severity_score'] * 1.5:
                    return ['background-color:#fde68a']*len(row)
                else:
                    return ['']*len(row)
            st.dataframe(suspicious_df.style.apply(highlight_severity, axis=1))

            for idx, row in suspicious_df.iterrows():
                with st.expander(f"Case ID: {row['case_id']} ⚠️"):
                    st.write(row)
                    reasons = explain_case(row, medians)
                    st.write("**Reasons flagged:**")
                    for r in reasons:
                        st.write("- " + r)

            csv_susp = suspicious_df.to_csv(index=False).encode("utf-8")
            st.download_button("Save Suspicious CSV", csv_susp, "suspicious_cases.csv", "text/csv")
            pdf_bytes = create_pdf(suspicious_df, df_features.attrs["top_keywords"])
            st.download_button("Save Suspicious PDF", pdf_bytes, "suspicious_cases.pdf", "application/pdf")
else:
    st.info("Please upload a CSV file to begin.")


