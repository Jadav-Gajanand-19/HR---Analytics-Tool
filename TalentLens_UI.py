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
            color: #333 !important;
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
        education_field = st.selectbox("Education Field", ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
        job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        over_time = st.selectbox("OverTime", ['Yes', 'No'])
        distance_from_home = st.slider("Distance From Home", 1, 50, 10)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        job_level = st.slider("Job Level", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_current_role = st.slider("Years in Current Role", 0, 20, 5)
        years_with_curr_manager = st.slider("Years with Current Manager", 0, 20, 4)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)

        submitted1 = st.form_submit_button("Predict Attrition")

    if submitted1:
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'BusinessTravel': business_travel,
            'Department': department,
            'EducationField': education_field,
            'JobRole': job_role,
            'MaritalStatus': marital_status,
            'OverTime': over_time,
            'DistanceFromHome': distance_from_home,
            'JobInvolvement': job_involvement,
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'EnvironmentSatisfaction': environment_satisfaction,
            'RelationshipSatisfaction': relationship_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsWithCurrManager': years_with_curr_manager,
            'MonthlyIncome': monthly_income
        }])

        input_encoded = encode_inputs(input_data, clf.feature_names_in_)
        prediction = clf.predict(input_encoded)[0]
        st.success(f"Attrition Prediction: {'Yes' if prediction == 1 else 'No'}")

elif section == "Performance Analysis":
    st.subheader("📈 Performance Rating Predictor")
    st.markdown("#### Enter Employee Metrics for Performance Analysis")

    with st.form("performance_form"):
        age = st.slider("Age", 18, 60, 30)
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        job_level = st.slider("Job Level", 1, 5, 2)
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
        years_at_company = st.slider("Years at Company", 0, 40, 5)

        submitted2 = st.form_submit_button("Predict Performance")

    if submitted2:
        input_data = pd.DataFrame([[age, education, job_level, job_involvement, job_satisfaction,
                                    work_life_balance, total_working_years, years_at_company]],
                                  columns=['Age', 'Education', 'JobLevel', 'JobInvolvement', 'JobSatisfaction',
                                           'WorkLifeBalance', 'TotalWorkingYears', 'YearsAtCompany'])

        performance = reg.predict(input_data)[0]

        st.markdown("### Predicted Performance Rating")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=performance,
            title={'text': "Performance Rating", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 2], 'color': '#ffcccc'},
                    {'range': [2, 3], 'color': '#ffe0b3'},
                    {'range': [3, 5], 'color': '#ccffcc'}
                ]
            }
        ))
        st.plotly_chart(gauge)
        st.markdown("#### AI-Powered Rating Predictor")
