# HR-Analytics-Tool
A Streamlit web app for HR analytics with two key features: predicting employee attrition (classification) and analyzing performance ratings (regression) using pre-trained ML models.

# 🧠 HR Analytics Streamlit App

A simple and interactive Streamlit web application for HR data analysis. This app helps HR teams **predict employee attrition** and **analyze performance** using pre-trained machine learning models.

## 🚀 Features

- 🔍 **Attrition Prediction** (Classification)
- 📈 **Performance Analysis** (Regression)
- 🧾 Separate input forms for each prediction type
- ⚙️ Easy-to-use Streamlit interface
- 🧠 Powered by pre-trained Random Forest models

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Pickle (for loading ML models)

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/hr-analytics-app.git
cd hr-analytics-app
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
📁 Files
app.py – Main Streamlit application

classifier_model.pkl – Trained classifier for attrition prediction

regression_model.pkl – Trained regressor for performance rating

requirements.txt – List of Python dependencies

📊 Model Inputs
Classification Model
Predicts whether an employee will leave.

Input features:
Age, BusinessTravel, DailyRate, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate, JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked, Over18, OverTime, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager

Regression Model
Predicts performance rating.

Input features:
Same as classification, but replaces PerformanceRating with Attrition.

🙌 Contributing
Feel free to fork this repo and submit pull requests to add new features or improve the UI.

📄 License
MIT License
