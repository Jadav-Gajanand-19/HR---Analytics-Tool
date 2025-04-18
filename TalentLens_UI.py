import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from io import BytesIO
import requests
import plotly.graph_objects as go
import plotly.express as px

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
        .stSelectbox>div>div>div>div {
            background-color: black !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image(logo, width=150)
st.sidebar.title("Talent Lens")
st.sidebar.markdown("See Beyond the Resume")
st.sidebar.markdown("Empowering HR with smart insights into employee attrition and performance.")

section = st.sidebar.radio("Navigate", ["Attrition Prediction", "Performance Analysis", "Visualize Trends"])

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
        input_data = pd.DataFrame([[
            age, business_travel, daily_rate, department, distance_from_home, education, education_field, environment_satisfaction,
            gender, hourly_rate, job_involvement, job_level, job_role, job_satisfaction, marital_status, monthly_income,
            monthly_rate, num_companies_worked, over_time, percent_salary_hike, performance_rating,
            relationship_satisfaction, stock_option_level, total_working_years, training_times, work_life_balance,
            years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager
        ]], columns=clf.feature_names_in_)

        prediction = clf.predict(input_data)[0]
        st.success(f"Attrition Prediction: {'Yes' if prediction == 1 else 'No'}")

elif section == "Performance Analysis":
    st.subheader("📈 Analyze Predicted Performance")
    with st.form("regression_form"):
        st.markdown("#### Enter Employee Details")
        attrition_input = st.selectbox("Has the employee left?", ['Yes', 'No'])
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

        submitted2 = st.form_submit_button("Predict Performance")

    if submitted2:
        attrition_map = {'Yes': 1, 'No': 0}
        input_data = pd.DataFrame([[
            age, attrition_map.get(attrition_input), business_travel, daily_rate, department, distance_from_home, education,
            education_field, environment_satisfaction, gender, hourly_rate, job_involvement, job_level, job_role,
            job_satisfaction, marital_status, monthly_income, monthly_rate, num_companies_worked, over_time,
            percent_salary_hike, relationship_satisfaction, stock_option_level, total_working_years, training_times,
            work_life_balance, years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager
        ]], columns=reg.feature_names_in_)

        prediction = reg.predict(input_data)[0]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted Performance Rating"},
            gauge={'axis': {'range': [1, 4]}, 'bar': {'color': "purple"}}
        ))
        st.plotly_chart(gauge, use_container_width=True)

elif section == "Visualize Trends":
    st.header("\U0001F4CA Visualize Trends")
    uploaded_file = st.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

        st.subheader("\U0001F50D Filter Data")
        department_filter = st.multiselect("Department", df['Department'].unique(), default=df['Department'].unique())
        gender_filter = st.multiselect("Gender", df['Gender'].unique(), default=df['Gender'].unique())
        education_filter = st.multiselect("Education Field", df['EducationField'].unique(), default=df['EducationField'].unique())

        filtered_df = df[(df['Department'].isin(department_filter)) &
                         (df['Gender'].isin(gender_filter)) &
                         (df['EducationField'].isin(education_filter))]

        # Bar Chart: Attrition by Department
        attrition_chart = px.bar(
            filtered_df.groupby(['Department', 'Attrition']).size().reset_index(name='Count'),
            x="Department",
            y="Count",
            color="Attrition",
            barmode="group",
            title="Attrition by Department",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(attrition_chart, use_container_width=True)

        # Pie Chart: Attrition Ratio
        attr_pie = filtered_df['Attrition'].value_counts().reset_index()
        attr_pie.columns = ['Attrition', 'Count']
        pie_chart = px.pie(attr_pie, values='Count', names='Attrition', title='Overall Attrition Distribution')
        st.plotly_chart(pie_chart, use_container_width=True)

        # Line Chart: Monthly Income by Job Role
        if 'MonthlyIncome' in df.columns and 'JobRole' in df.columns:
            line_chart = px.line(
                df.groupby("JobRole")["MonthlyIncome"].mean().reset_index(),
                x="JobRole",
                y="MonthlyIncome",
                markers=True,
                title="Average Monthly Income by Job Role"
            )
            st.plotly_chart(line_chart, use_container_width=True)
