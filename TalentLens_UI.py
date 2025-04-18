import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# --- Load logo from GitHub (raw image link) ---
logo_url = "https://raw.githubusercontent.com/Jadav-Gajanand-19/TalentLens---See-Beyond-Resume/main/TalenLens%20Logo.png"
response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))

# --- Load models ---
clf = pickle.load(open("classifier_model.pkl", "rb"))
reg = pickle.load(open("regression_model.pkl", "rb"))

# --- Page configuration ---
st.set_page_config(
    page_title="Talent Lens",
    layout="wide",
    page_icon="📊"
)

# --- Custom styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: white;
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

# Recent prediction history
if 'history' not in st.session_state:
    st.session_state['history'] = []
st.sidebar.markdown("### 🕒 Recent Predictions")
for event in st.session_state['history'][-3:]:
    st.sidebar.markdown(f"- {event}")

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    normalize = st.checkbox("Normalize Inputs")
    show_proba = st.checkbox("Show Prediction Confidence")

# Mini leaderboard
st.sidebar.markdown("### 🏆 Top Departments (Performance)")
st.sidebar.markdown("- R&D: ⭐ 4.5\n- Sales: ⭐ 4.3\n- HR: ⭐ 4.1")

section = st.sidebar.radio("Navigate", ["Attrition Prediction", "Performance Analysis", "Visualize Trends"])

# --- Helper: Encode categorical inputs to match training ---
def encode_inputs(df, model_features):
    df_encoded = pd.get_dummies(df)
    for col in model_features:
        if col not in df_encoded:
            df_encoded[col] = 0
    return df_encoded[model_features]

if section == "Attrition Prediction":
    st.subheader("👥 Predict Employee Attrition")
    with st.form("attrition_form"):
        st.markdown("#### Enter Employee Details")
        attrition_inputs = {
            'Age': st.slider("Age", 18, 60, 30),
            'BusinessTravel': st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']),
            'Department': st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources']),
            'DistanceFromHome': st.slider("Distance From Home", 1, 50, 10),
            'Education': st.selectbox("Education", [1, 2, 3, 4, 5]),
            'EducationField': st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']),
            'EnvironmentSatisfaction': st.slider("Environment Satisfaction", 1, 4, 3),
            'Gender': st.selectbox("Gender", ['Male', 'Female']),
            'JobInvolvement': st.slider("Job Involvement", 1, 4, 3),
            'JobLevel': st.slider("Job Level", 1, 5, 2),
            'JobRole': st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']),
            'JobSatisfaction': st.slider("Job Satisfaction", 1, 4, 3),
            'MaritalStatus': st.selectbox("Marital Status", ['Single', 'Married', 'Divorced']),
            'MonthlyIncome': st.slider("Monthly Income", 1000, 20000, 5000),
            'NumCompaniesWorked': st.slider("Num Companies Worked", 0, 10, 3),
            'OverTime': st.selectbox("OverTime", ['Yes', 'No']),
            'PerformanceRating': st.slider("Performance Rating", 1, 4, 3),
            'RelationshipSatisfaction': st.slider("Relationship Satisfaction", 1, 4, 3),
            'StockOptionLevel': st.slider("Stock Option Level", 0, 3, 1),
            'TotalWorkingYears': st.slider("Total Working Years", 0, 40, 10),
            'WorkLifeBalance': st.slider("Work Life Balance", 1, 4, 3),
            'YearsAtCompany': st.slider("Years at Company", 0, 40, 5),
            'YearsInCurrentRole': st.slider("Years in Current Role", 0, 20, 5),
            'YearsWithCurrManager': st.slider("Years with Current Manager", 0, 20, 4)
        }
        submitted1 = st.form_submit_button("Predict Attrition")

    if submitted1:
        input_data = pd.DataFrame([attrition_inputs])
        input_encoded = encode_inputs(input_data, clf.feature_names_in_)
        prediction = clf.predict(input_encoded)[0]
        st.session_state['history'].append(f"Attrition: {'Yes' if prediction == 1 else 'No'} @ {pd.Timestamp.now().strftime('%H:%M:%S')}")
        st.success(f"Attrition Prediction: {'Yes' if prediction == 1 else 'No'}")

elif section == "Performance Analysis":
    st.subheader("📈 Performance Rating Predictor")
    st.markdown("#### Enter Employee Metrics for Performance Analysis")

    with st.form("performance_form"):
        perf_inputs = {
            'Age': st.slider("Age", 18, 60, 30),
            'BusinessTravel': st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']),
            'Department': st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources']),
            'DistanceFromHome': st.slider("Distance From Home", 1, 50, 10),
            'Education': st.selectbox("Education", [1, 2, 3, 4, 5]),
            'EducationField': st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other']),
            'EnvironmentSatisfaction': st.slider("Environment Satisfaction", 1, 4, 3),
            'Gender': st.selectbox("Gender", ['Male', 'Female']),
            'JobInvolvement': st.slider("Job Involvement", 1, 4, 3),
            'JobLevel': st.slider("Job Level", 1, 5, 2),
            'JobRole': st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']),
            'JobSatisfaction': st.slider("Job Satisfaction", 1, 4, 3),
            'MaritalStatus': st.selectbox("Marital Status", ['Single', 'Married', 'Divorced']),
            'MonthlyIncome': st.slider("Monthly Income", 1000, 20000, 5000),
            'NumCompaniesWorked': st.slider("Num Companies Worked", 0, 10, 3),
            'OverTime': st.selectbox("OverTime", ['Yes', 'No']),
            'PerformanceRating': st.slider("Performance Rating", 1, 4, 3),
            'RelationshipSatisfaction': st.slider("Relationship Satisfaction", 1, 4, 3),
            'StockOptionLevel': st.slider("Stock Option Level", 0, 3, 1),
            'TotalWorkingYears': st.slider("Total Working Years", 0, 40, 10),
            'WorkLifeBalance': st.slider("Work Life Balance", 1, 4, 3),
            'YearsAtCompany': st.slider("Years at Company", 0, 40, 5),
            'YearsInCurrentRole': st.slider("Years in Current Role", 0, 20, 5),
            'YearsWithCurrManager': st.slider("Years with Current Manager", 0, 20, 4)
        }
        submitted2 = st.form_submit_button("Predict Performance")

    if submitted2:
        input_data = pd.DataFrame([perf_inputs])
        input_encoded = encode_inputs(input_data, reg.feature_names_in_)
        performance = reg.predict(input_encoded)[0]
        st.session_state['history'].append(f"Performance: {performance:.2f}⭐ @ {pd.Timestamp.now().strftime('%H:%M:%S')}")
        st.markdown("### Predicted Performance Rating")
        stars = "⭐" * round(performance)
        st.markdown(f"## {stars} ({performance:.2f} / 5)")
        st.markdown("#### AI-Powered Rating Predictor")
