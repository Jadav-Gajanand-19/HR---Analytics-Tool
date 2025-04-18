
Talent Lens – See Beyond the Resume 👁️‍🗨️
Welcome to Talent Lens, a smart HR analytics dashboard designed to help you predict employee attrition, evaluate performance, and visualize trends from your workforce data—all in one place.



🚀 Features
🎯 Attrition Prediction
Input employee details to predict whether they are at risk of leaving.

Optionally view the confidence level of the prediction.

Supports normalized inputs and top feature selection.

🌟 Performance Analysis
Predict an employee’s performance rating on a 5-star scale.

Input role, department, experience, satisfaction, and more.

Great for performance review preparation and workforce optimization.

📊 Visualize Trends
Upload your own HR dataset (.csv) to explore insightful trends.

Interactive Plotly charts: histograms, box plots, scatter plots, bar charts, pie charts.

Understand relationships between income, job roles, attrition, satisfaction, and more.

🧠 Models Used
Classifier: Trained to predict whether an employee is likely to leave.

Regressor: Trained to estimate employee performance on a scale of 1 to 5.

Models are loaded from pickled .pkl files.

🛠️ Tech Stack
Python 3.8+

Streamlit

scikit-learn

Pandas, NumPy

Plotly, Seaborn, Matplotlib

Pillow (Image processing)

Pickle (for model serialization)

📂 Project Structure
bash
Copy
Edit
📁 TalentLens/
├── app.py                     # Main Streamlit app
├── classifier_model.pkl       # Trained classifier model
├── regression_model.pkl       # Trained regression model
├── requirements.txt           # Required Python packages
📦 Installation & Running
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Jadav-Gajanand-19/TalentLens---See-Beyond-Resume.git
cd TalentLens---See-Beyond-Resume
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
📈 Sample Dataset
For the Visualize Trends section, upload a CSV with columns like:

Age, Department, MonthlyIncome, Attrition, JobSatisfaction, WorkLifeBalance, etc.

🎨 UI Highlights
Responsive layout with dark-text-on-light theme.

Custom sidebar with logo, greetings, and leaderboard.

Interactive widgets for a personalized experience.

👥 Built For
HR Analysts

Talent Management Teams

Organizational Leaders

Data-Driven Recruiters

❤️ Credits
Built with love by Team Talent Lens
GitHub Repo: TalentLens - See Beyond the Resume
