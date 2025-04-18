import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
import requests
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np

# --- Page configuration must be first ---
st.set_page_config(
    page_title="Talent Lens",
    layout="wide",
    page_icon="🧠"  # Emoji used for favicon
)

# --- Load logo from GitHub (raw image link) ---
@st.cache_data
def load_logo(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

logo_url = "https://raw.githubusercontent.com/Jadav-Gajanand-19/TalentLens---See-Beyond-Resume/main/TalenLens%20Logo.png"
logo = load_logo(logo_url)

# --- Load models safely ---
try:
    clf = pickle.load(open("classifier_model.pkl", "rb"))
    reg = pickle.load(open("regression_model.pkl", "rb"))
except FileNotFoundError as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# --- Custom styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .title-section {
            text-align: center;
        }
        .tagline {
            font-size: 18px;
            color: #555;
        }
        .brand-blurb {
            font-size: 15px;
            color: #444;
            margin-top: -10px;
        }
        .stButton>button {
            background-color: #6a0dad;
            color: white;
        }
        .stSelectbox>div>div>div>div,
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSlider>div>div>div {
            background-color: white !important;
            color: #333 !important;
        }
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image(logo, width=150)
st.sidebar.title("Talent Lens")
st.sidebar.markdown("See Beyond the Resume")
st.sidebar.markdown("Empowering HR with smart insights into employee attrition and performance.")

# Time-based greeting
hour = datetime.now().hour
greeting = "🌞 Good Morning" if hour < 12 else "🌇 Good Evening" if hour > 17 else "🌤 Good Afternoon"
st.sidebar.markdown(f"### {greeting}, HR 👋")

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    normalize = st.checkbox("Normalize Inputs")
    show_proba = st.checkbox("Show Prediction Confidence")
    use_top_features = st.checkbox("Use Only Top 10 Features")
    show_confidence = st.checkbox("Display Confidence %")

# Export section
st.sidebar.markdown("### 📄 Export Predictions")
if "attrition_predictions" in st.session_state and not st.session_state["attrition_predictions"].empty:
    st.sidebar.download_button(
        "Download Attrition Predictions",
        data=st.session_state["attrition_predictions"].to_csv(index=False).encode(),
        file_name="attrition_predictions.csv",
        mime="text/csv"
    )
if "performance_predictions" in st.session_state and not st.session_state["performance_predictions"].empty:
    st.sidebar.download_button(
        "Download Performance Predictions",
        data=st.session_state["performance_predictions"].to_csv(index=False).encode(),
        file_name="performance_predictions.csv",
        mime="text/csv"
    )

# Mini leaderboard
st.sidebar.markdown("### 🏆 Top Departments (Performance)")
st.sidebar.markdown("- R&D: ⭐ 4.5\n- Sales: ⭐ 4.3\n- HR: ⭐ 4.1")

# Help section
st.sidebar.markdown("### 📘 How to Use Talent Lens")
st.sidebar.markdown("""
**1. Attrition Prediction**  
➡️ Enter employee details to check the likelihood of them leaving.  
➡️ Use confidence toggle for prediction probability.  

**2. Performance Analysis**  
➡️ Predict future performance rating based on key HR metrics.  

**3. Visualize Trends**  
➡️ Upload your HR CSV to view patterns in income, attrition, and performance.  
➡️ Explore charts to gain actionable insights.  

**💾 Tip:** Use the export buttons to save predictions for your records.
""")

# --- Navigation ---
section = st.sidebar.radio("Navigate", ["Attrition Prediction", "Performance Analysis", "Visualize Trends", "Upload & Predict Batch"])

# --- HR Data Visualization ---
if section == "Visualize Trends":
    st.title("📊 HR Data Visualization")
    uploaded_file = st.file_uploader("Upload your HR Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            st.subheader("Attrition Distribution")
            fig1 = px.histogram(df, x="Attrition", color="Attrition", title="Attrition Count")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Monthly Income by Department")
            fig2 = px.box(df, x="Department", y="MonthlyIncome", color="Department", title="Monthly Income by Department")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Average Performance by Job Role")
            fig3 = px.bar(df.groupby("JobRole")["PerformanceRating"].mean().reset_index(), x="JobRole", y="PerformanceRating", title="Performance by Job Role")
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Job Satisfaction vs. Environment Satisfaction")
            fig4 = px.scatter(df, x="JobSatisfaction", y="EnvironmentSatisfaction", color="Attrition", title="Satisfaction Comparison")
            st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# --- Attrition Prediction ---
if section == "Attrition Prediction":
    st.title("🔢 Attrition Prediction")
    st.markdown("Fill in employee details below to predict attrition likelihood.")

    age = st.slider("Age", 18, 60, 30)
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    over_time = st.selectbox("OverTime", ["Yes", "No"])
    distance_from_home = st.slider("Distance from Home (km)", 1, 30, 5)
    monthly_income = st.number_input("Monthly Income", min_value=1000, value=5000)

    if st.button("Predict Attrition"):
        input_data = pd.DataFrame([{
            "Age": age,
            "YearsAtCompany": years_at_company,
            "JobSatisfaction": job_satisfaction,
            "OverTime": 1 if over_time == "Yes" else 0,
            "DistanceFromHome": distance_from_home,
            "MonthlyIncome": monthly_income
        }])

        if normalize:
            input_data = (input_data - input_data.mean()) / input_data.std()

        prediction = clf.predict(input_data)[0]
        proba = clf.predict_proba(input_data)[0][1]

        result = "Yes 🚫" if prediction == 1 else "No 🚀"
        st.metric("Attrition Prediction", result)
        if show_confidence:
            st.metric("Confidence", f"{proba*100:.2f}%")

        st.session_state["attrition_predictions"] = input_data.copy()
        st.session_state["attrition_predictions"]["Prediction"] = result
        if show_confidence:
            st.session_state["attrition_predictions"]["Confidence"] = f"{proba*100:.2f}%"

# --- Performance Analysis ---
if section == "Performance Analysis":
    st.title("🏋️ Performance Analysis")
    st.markdown("Predict future employee performance rating.")

    age = st.slider("Age", 18, 60, 30, key="age_perf")
    training_times = st.slider("Training Times Last Year", 0, 6, 2)
    years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 3)
    work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    years_in_current_role = st.slider("Years in Current Role", 0, 18, 4)

    if st.button("Predict Performance"):
        perf_input = pd.DataFrame([{
            "Age": age,
            "TrainingTimesLastYear": training_times,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "WorkLifeBalance": work_life_balance,
            "YearsInCurrentRole": years_in_current_role
        }])

        if normalize:
            perf_input = (perf_input - perf_input.mean()) / perf_input.std()

        perf_rating = reg.predict(perf_input)[0]
        st.metric("Predicted Performance Rating", f"{perf_rating:.2f} / 5")

        st.session_state["performance_predictions"] = perf_input.copy()
        st.session_state["performance_predictions"]["Predicted Rating"] = perf_rating

# --- Batch Prediction ---
if section == "Upload & Predict Batch":
    st.title("📁 Batch Prediction")
    st.markdown("Upload a CSV to predict attrition and performance for multiple employees.")

    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"], key="batch_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data", df.head())

        try:
            if use_top_features:
                attrition_cols = ["Age", "YearsAtCompany", "JobSatisfaction", "OverTime", "DistanceFromHome", "MonthlyIncome"]
                df_attrition = df[attrition_cols].copy()
                df_attrition["OverTime"] = df_attrition["OverTime"].apply(lambda x: 1 if x == "Yes" else 0)
            else:
                df_attrition = df.copy()

            if normalize:
                df_attrition = (df_attrition - df_attrition.mean()) / df_attrition.std()

            df["AttritionPrediction"] = clf.predict(df_attrition)
            df["AttritionPrediction"] = df["AttritionPrediction"].map({0: "No", 1: "Yes"})
            if show_proba:
                df["AttritionConfidence"] = clf.predict_proba(df_attrition)[:, 1]

            perf_cols = ["Age", "TrainingTimesLastYear", "YearsSinceLastPromotion", "WorkLifeBalance", "YearsInCurrentRole"]
            df_perf = df[perf_cols].copy()
            if normalize:
                df_perf = (df_perf - df_perf.mean()) / df_perf.std()

            df["PerformancePrediction"] = reg.predict(df_perf)

            st.write("### Predictions", df[["AttritionPrediction", "PerformancePrediction"] + (["AttritionConfidence"] if show_proba else [])])

            st.download_button(
                "Download Full Prediction Results",
                data=df.to_csv(index=False).encode(),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
