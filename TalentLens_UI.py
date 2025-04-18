import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
import requests

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

# --- Custom background color with CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f0eb;
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
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.image(logo, width=180)
st.markdown("<div class='title-section'><h1>Talent Lens</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>See Beyond the Resume</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-blurb'>Empowering HR with smart insights into employee attrition and performance. Predictive intelligence to retain talent and enhance potential.</div>", unsafe_allow_html=True)

# --- Mappings ---
bt_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
dept_map = {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2}
edu_field_map = {'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Human Resources': 4, 'Other': 5}
job_role_map = {
    'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 2,
    'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
    'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8
}
marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
gender_map = {'Male': 1, 'Female': 0}
overtime_map = {'Yes': 1, 'No': 0}
over18_map = {'Y': 1}
attrition_map = {'Yes': 1, 'No': 0}

# --- Inputs filtering ---
clf_cols = set(clf.feature_names_in_)
reg_cols = set(reg.feature_names_in_)

# --- Common Input Generator ---
def get_filtered_inputs(selected_features):
    inputs = {}
    if 'Age' in selected_features: inputs["age"] = st.slider("Age", 18, 60, 30)
    if 'Gender' in selected_features: inputs["gender"] = st.selectbox("Gender", list(gender_map.keys()))
    if 'BusinessTravel' in selected_features: inputs["business_travel"] = st.selectbox("Business Travel", list(bt_map.keys()))
    if 'Department' in selected_features: inputs["department"] = st.selectbox("Department", list(dept_map.keys()))
    if 'Education' in selected_features: inputs["education"] = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    if 'EducationField' in selected_features: inputs["education_field"] = st.selectbox("Education Field", list(edu_field_map.keys()))
    if 'JobRole' in selected_features: inputs["job_role"] = st.selectbox("Job Role", list(job_role_map.keys()))
    if 'MaritalStatus' in selected_features: inputs["marital_status"] = st.selectbox("Marital Status", list(marital_map.keys()))
    if 'OverTime' in selected_features: inputs["over_time"] = st.selectbox("OverTime", list(overtime_map.keys()))
    if 'Over18' in selected_features: inputs["over_18"] = st.selectbox("Over 18", ['Y'])
    if 'DailyRate' in selected_features: inputs["daily_rate"] = st.slider("Daily Rate", 100, 1500, 800)
    if 'DistanceFromHome' in selected_features: inputs["distance_from_home"] = st.slider("Distance From Home", 1, 50, 10)
    if 'HourlyRate' in selected_features: inputs["hourly_rate"] = st.slider("Hourly Rate", 30, 100, 60)
    if 'JobInvolvement' in selected_features: inputs["job_involvement"] = st.slider("Job Involvement", 1, 4, 3)
    if 'JobLevel' in selected_features: inputs["job_level"] = st.slider("Job Level", 1, 5, 2)
    if 'JobSatisfaction' in selected_features: inputs["job_satisfaction"] = st.slider("Job Satisfaction", 1, 4, 3)
    if 'EnvironmentSatisfaction' in selected_features: inputs["environment_satisfaction"] = st.slider("Environment Satisfaction", 1, 4, 3)
    if 'RelationshipSatisfaction' in selected_features: inputs["relationship_satisfaction"] = st.slider("Relationship Satisfaction", 1, 4, 3)
    if 'WorkLifeBalance' in selected_features: inputs["work_life_balance"] = st.slider("Work Life Balance", 1, 4, 3)
    if 'PercentSalaryHike' in selected_features: inputs["percent_salary_hike"] = st.slider("Percent Salary Hike", 10, 25, 15)
    if 'StockOptionLevel' in selected_features: inputs["stock_option_level"] = st.slider("Stock Option Level", 0, 3, 1)
    if 'TrainingTimesLastYear' in selected_features: inputs["training_times"] = st.slider("Training Times Last Year", 0, 6, 2)
    if 'TotalWorkingYears' in selected_features: inputs["total_working_years"] = st.slider("Total Working Years", 0, 40, 10)
    if 'NumCompaniesWorked' in selected_features: inputs["num_companies_worked"] = st.slider("Number of Companies Worked", 0, 10, 2)
    if 'YearsAtCompany' in selected_features: inputs["years_at_company"] = st.slider("Years at Company", 0, 40, 5)
    if 'YearsInCurrentRole' in selected_features: inputs["years_in_current_role"] = st.slider("Years in Current Role", 0, 20, 5)
    if 'YearsSinceLastPromotion' in selected_features: inputs["years_since_last_promotion"] = st.slider("Years Since Last Promotion", 0, 15, 3)
    if 'YearsWithCurrManager' in selected_features: inputs["years_with_curr_manager"] = st.slider("Years with Current Manager", 0, 20, 4)
    if 'MonthlyIncome' in selected_features: inputs["monthly_income"] = st.slider("Monthly Income", 1000, 20000, 5000)
    if 'MonthlyRate' in selected_features: inputs["monthly_rate"] = st.slider("Monthly Rate", 1000, 25000, 10000)
    return inputs

# --- Tabs for Classifier and Regressor ---
tabs = st.tabs(["👥 Attrition Prediction", "📈 Performance Analysis"])

with tabs[0]:
    st.subheader("👥 Predict Employee Attrition")
    with st.form("attrition_form"):
        data = get_filtered_inputs(clf.feature_names_in_)
        if 'PerformanceRating' in clf.feature_names_in_:
            performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        submitted1 = st.form_submit_button("Predict Attrition")

    if submitted1:
        data_map = {
            "BusinessTravel": bt_map.get(data.get("business_travel")),
            "Department": dept_map.get(data.get("department")),
            "EducationField": edu_field_map.get(data.get("education_field")),
            "Gender": gender_map.get(data.get("gender")),
            "JobRole": job_role_map.get(data.get("job_role")),
            "MaritalStatus": marital_map.get(data.get("marital_status")),
            "OverTime": overtime_map.get(data.get("over_time")),
            "Over18": over18_map.get(data.get("over_18")),
            "PerformanceRating": performance_rating
        }
        features = [data_map.get(col, data.get(col.lower())) for col in clf.feature_names_in_]
        input_df = pd.DataFrame([features], columns=clf.feature_names_in_)
        prediction = clf.predict(input_df)[0]
        st.success(f"Attrition Prediction: **{'Yes' if prediction == 1 else 'No'}**")

with tabs[1]:
    st.subheader("📈 Analyze Predicted Performance")
    with st.form("regression_form"):
        data = get_filtered_inputs(reg.feature_names_in_)
        if 'Attrition' in reg.feature_names_in_:
            attrition_input = st.selectbox("Has the employee left?", ["No", "Yes"])
        submitted2 = st.form_submit_button("Predict Performance")

    if submitted2:
        data_map = {
            "BusinessTravel": bt_map.get(data.get("business_travel")),
            "Department": dept_map.get(data.get("department")),
            "EducationField": edu_field_map.get(data.get("education_field")),
            "Gender": gender_map.get(data.get("gender")),
            "JobRole": job_role_map.get(data.get("job_role")),
            "MaritalStatus": marital_map.get(data.get("marital_status")),
            "OverTime": overtime_map.get(data.get("over_time")),
            "Over18": over18_map.get(data.get("over_18")),
            "Attrition": attrition_map.get(attrition_input)
        }
        features = [data_map.get(col, data.get(col.lower())) for col in reg.feature_names_in_]
        input_df = pd.DataFrame([features], columns=reg.feature_names_in_)
        prediction = reg.predict(input_df)[0]
        st.info(f"Predicted Performance Score: **{prediction:.2f}**")
