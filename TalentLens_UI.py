import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
import requests
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

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
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image(logo, width=150)
st.sidebar.title("Talent Lens")
st.sidebar.markdown("See Beyond the Resume")
st.sidebar.markdown("Empowering HR with smart insights into employee attrition and performance.")

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
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
        department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        over_time = st.selectbox("OverTime", ['Yes', 'No'])
        daily_rate = st.slider("Daily Rate", 100, 1500, 800)
        distance_from_home = st.slider("Distance From Home", 1, 50, 10)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 60)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        job_level = st.slider("Job Level", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        percent_salary_hike = st.slider("Percent Salary Hike", 10, 25, 15)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        training_times = st.slider("Training Times Last Year", 0, 6, 2)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 20, 5)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 3)
        years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 4)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        monthly_rate = st.slider("Monthly Rate", 1000, 25000, 10000)
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])

        submitted1 = st.form_submit_button("Predict Attrition")

    if submitted1:
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BusinessTravel': business_travel,
            'Department': department,
            'Education': education,
            'EducationField': education_field,
            'JobRole': job_role,
            'MaritalStatus': marital_status,
            'OverTime': over_time,
            'DailyRate': daily_rate,
            'DistanceFromHome': distance_from_home,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'RelationshipSatisfaction': relationship_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'PercentSalaryHike': percent_salary_hike,
            'StockOptionLevel': stock_option_level,
            'TrainingTimesLastYear': training_times,
            'TotalWorkingYears': total_working_years,
            'NumCompaniesWorked': num_companies_worked,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'PerformanceRating': performance_rating
        }])

        input_encoded = encode_inputs(input_data, clf.feature_names_in_)
        prediction = clf.predict(input_encoded)[0]
        st.success(f"Attrition Prediction: {'Yes' if prediction == 1 else 'No'}")

# --- Keep other sections (Performance Analysis, Visualize Trends) unchanged for now ---
