import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Talent Lens", layout="wide", page_icon="📊")

# --- Branding ---
logo = Image.open("https://github.com/Jadav-Gajanand-19/TalentLens---See-Beyond-Resume/blob/main/TalenLens%20Logo.png")
st.image(logo, width=160)
st.markdown("""
# 🎯 Talent Lens
### *See Beyond the Resume*
Unlock data-driven insights into employee potential and retention.
""")

# --- Load Models ---
clf = pickle.load(open("classifier_model.pkl", "rb"))
reg = pickle.load(open("regression_model.pkl", "rb"))

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
attrition_map = {'Yes': 1, 'No': 0}

# --- Form Input UI ---
def get_inputs():
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        age = col1.slider("Age", 18, 60, 30)
        gender = col2.selectbox("Gender", list(gender_map.keys()))
        business_travel = col3.selectbox("Business Travel", list(bt_map.keys()))

        dept = col1.selectbox("Department", list(dept_map.keys()))
        education = col2.selectbox("Education Level", [1, 2, 3, 4, 5])
        education_field = col3.selectbox("Education Field", list(edu_field_map.keys()))

        job_role = col1.selectbox("Job Role", list(job_role_map.keys()))
        job_level = col2.slider("Job Level", 1, 5, 2)
        job_involvement = col3.slider("Job Involvement", 1, 4, 3)

        job_satisfaction = col1.slider("Job Satisfaction", 1, 4, 3)
        environment_satisfaction = col2.slider("Environment Satisfaction", 1, 4, 3)
        relationship_satisfaction = col3.slider("Relationship Satisfaction", 1, 4, 3)

        work_life_balance = col1.slider("Work Life Balance", 1, 4, 3)
        over_time = col2.selectbox("OverTime", list(overtime_map.keys()))
        marital_status = col3.selectbox("Marital Status", list(marital_map.keys()))

        monthly_income = col1.slider("Monthly Income", 1000, 20000, 5000)
        total_working_years = col2.slider("Total Working Years", 0, 40, 10)
        years_at_company = col3.slider("Years at Company", 0, 40, 5)

        years_in_current_role = col1.slider("Years in Current Role", 0, 20, 5)
        years_since_last_promotion = col2.slider("Years Since Last Promotion", 0, 15, 3)
        years_with_curr_manager = col3.slider("Years with Current Manager", 0, 20, 4)

        num_companies_worked = col1.slider("Number of Companies Worked", 0, 10, 2)
        training_times = col2.slider("Training Times Last Year", 0, 6, 2)
        stock_option_level = col3.slider("Stock Option Level", 0, 3, 1)

        percent_salary_hike = col1.slider("Percent Salary Hike", 10, 25, 15)
        attrition = col2.selectbox("Has the employee left?", list(attrition_map.keys()))
        performance_rating = col3.selectbox("Performance Rating (if known)", [1, 2, 3, 4])

        return [
            age, attrition_map[attrition], bt_map[business_travel], dept_map[dept],
            distance := 10, education, edu_field_map[education_field], environment_satisfaction,
            gender_map[gender], job_involvement, job_level, job_role_map[job_role],
            job_satisfaction, marital_map[marital_status], monthly_income, num_companies_worked,
            overtime_map[over_time], percent_salary_hike, performance_rating, relationship_satisfaction,
            stock_option_level, total_working_years, training_times, work_life_balance,
            years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager
        ]

# --- Tabs ---
tabs = st.tabs(["🔍 Predict Attrition", "📈 Predict Performance"])

with tabs[0]:
    st.header("🔍 Will This Employee Leave?")
    st.caption("Use the form below to predict employee attrition.")
    with st.form("predict_attrition"):
        input_row = get_inputs()
        submitted1 = st.form_submit_button("Predict Attrition")
        if submitted1:
            input_df = pd.DataFrame([input_row], columns=clf.feature_names_in_)
            prediction = clf.predict(input_df)[0]
            st.success(f"Attrition Prediction: **{'Yes' if prediction == 1 else 'No'}**")

with tabs[1]:
    st.header("📈 Estimate Performance Rating")
    st.caption("Predict an employee's performance score based on current data.")
    with st.form("predict_performance"):
        input_row = get_inputs()
        submitted2 = st.form_submit_button("Predict Performance")
        if submitted2:
            input_df = pd.DataFrame([input_row], columns=reg.feature_names_in_)
            score = reg.predict(input_df)[0]
            st.info(f"Predicted Performance Score: **{score:.2f}**")

# --- Custom Styling ---
st.markdown("""
    <style>
        body { background-color: #f8f3f0; }
        .stApp { background-color: #f3eaf4; }
        .css-1d391kg { background-color: #fff; border-radius: 10px; padding: 20px; }
    </style>
""", unsafe_allow_html=True)
