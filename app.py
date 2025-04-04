import streamlit as st
import pickle
import pandas as pd

# Load models
clf = pickle.load(open("classifier_model.pkl", "rb"))
reg = pickle.load(open("regression_model.pkl", "rb"))

st.set_page_config(page_title="HR Analytics Tool", layout="centered")
st.title("📊 HR Analytics Dashboard")

tabs = st.tabs(["👥 Attrition Prediction", "📈 Performance Analysis"])

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

# --- Reusable Inputs ---
def get_common_inputs():
    return {
        "age": st.slider("Age", 18, 60, 30),
        "gender": st.selectbox("Gender", ['Male', 'Female']),
        "business_travel": st.selectbox("Business Travel", list(bt_map.keys())),
        "department": st.selectbox("Department", list(dept_map.keys())),
        "education": st.selectbox("Education Level", [1, 2, 3, 4, 5]),
        "education_field": st.selectbox("Education Field", list(edu_field_map.keys())),
        "job_role": st.selectbox("Job Role", list(job_role_map.keys())),
        "marital_status": st.selectbox("Marital Status", list(marital_map.keys())),
        "over_time": st.selectbox("OverTime", list(overtime_map.keys())),
        "over_18": st.selectbox("Over 18", ['Y']),
        "daily_rate": st.slider("Daily Rate", 100, 1500, 800),
        "distance_from_home": st.slider("Distance From Home", 1, 50, 10),
        "hourly_rate": st.slider("Hourly Rate", 30, 100, 60),
        "job_involvement": st.slider("Job Involvement", 1, 4, 3),
        "job_level": st.slider("Job Level", 1, 5, 2),
        "job_satisfaction": st.slider("Job Satisfaction", 1, 4, 3),
        "environment_satisfaction": st.slider("Environment Satisfaction", 1, 4, 3),
        "relationship_satisfaction": st.slider("Relationship Satisfaction", 1, 4, 3),
        "work_life_balance": st.slider("Work Life Balance", 1, 4, 3),
        "percent_salary_hike": st.slider("Percent Salary Hike", 10, 25, 15),
        "stock_option_level": st.slider("Stock Option Level", 0, 3, 1),
        "training_times": st.slider("Training Times Last Year", 0, 6, 2),
        "total_working_years": st.slider("Total Working Years", 0, 40, 10),
        "num_companies_worked": st.slider("Number of Companies Worked", 0, 10, 2),
        "years_at_company": st.slider("Years at Company", 0, 40, 5),
        "years_in_current_role": st.slider("Years in Current Role", 0, 20, 5),
        "years_since_last_promotion": st.slider("Years Since Last Promotion", 0, 15, 3),
        "years_with_curr_manager": st.slider("Years with Current Manager", 0, 20, 4),
        "monthly_income": st.slider("Monthly Income", 1000, 20000, 5000),
        "monthly_rate": st.slider("Monthly Rate", 1000, 25000, 10000)
    }

# --- Tab 1: Classifier ---
with tabs[0]:
    st.subheader("👥 Predict Employee Attrition")

    with st.form("attrition_form"):
        data = get_common_inputs()
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        submitted1 = st.form_submit_button("Predict Attrition")

    if submitted1:
        row = [
            data["age"], bt_map[data["business_travel"]], data["daily_rate"], dept_map[data["department"]],
            data["distance_from_home"], data["education"], edu_field_map[data["education_field"]], 1, 999999,
            data["environment_satisfaction"], gender_map[data["gender"]], data["hourly_rate"], data["job_involvement"],
            data["job_level"], job_role_map[data["job_role"]], data["job_satisfaction"], marital_map[data["marital_status"]],
            data["monthly_income"], data["monthly_rate"], data["num_companies_worked"], over18_map[data["over_18"]],
            overtime_map[data["over_time"]], data["percent_salary_hike"], performance_rating,
            data["relationship_satisfaction"], 8, data["stock_option_level"], data["total_working_years"],
            data["training_times"], data["work_life_balance"], data["years_at_company"],
            data["years_in_current_role"], data["years_since_last_promotion"], data["years_with_curr_manager"]
        ]

        clf_columns = clf.feature_names_in_
        input_df = pd.DataFrame([row], columns=clf_columns)

        prediction = clf.predict(input_df)[0]
        st.success(f"Attrition Prediction: **{'Yes' if prediction == 1 else 'No'}**")

# --- Tab 2: Regressor ---
with tabs[1]:
    st.subheader("📈 Analyze Predicted Performance")

    with st.form("regression_form"):
        data = get_common_inputs()
        attrition_input = st.selectbox("Has the employee left?", ["No", "Yes"])
        submitted2 = st.form_submit_button("Predict Performance")

    if submitted2:
        row = [
            data["age"], attrition_map[attrition_input], bt_map[data["business_travel"]],
            data["daily_rate"], dept_map[data["department"]], data["distance_from_home"], data["education"],
            edu_field_map[data["education_field"]], 1, 999999, data["environment_satisfaction"],
            gender_map[data["gender"]], data["hourly_rate"], data["job_involvement"], data["job_level"],
            job_role_map[data["job_role"]], data["job_satisfaction"], marital_map[data["marital_status"]],
            data["monthly_income"], data["monthly_rate"], data["num_companies_worked"], over18_map[data["over_18"]],
            overtime_map[data["over_time"]], data["percent_salary_hike"], data["relationship_satisfaction"], 8,
            data["stock_option_level"], data["total_working_years"], data["training_times"], data["work_life_balance"],
            data["years_at_company"], data["years_in_current_role"], data["years_since_last_promotion"],
            data["years_with_curr_manager"]
        ]

        reg_columns = reg.feature_names_in_
        input_df = pd.DataFrame([row], columns=reg_columns)

        prediction = reg.predict(input_df)[0]
        st.info(f"Predicted Performance Score: **{prediction:.2f}**")
