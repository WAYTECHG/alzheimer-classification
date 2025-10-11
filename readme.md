# 🧠 Alzheimer Prediction System

A web-based **Alzheimer’s Disease Prediction System** built using **Streamlit** and **Machine Learning models**.
This system predicts Alzheimer’s disease stages based on **clinical data** or **MRI data**, and also includes an integrated **chatbot** and **data analysis visualizations**.

---

## 🚀 Features

### 🏠 Web Application (Streamlit)

The application includes several interactive pages:

- **Home Page** — Introduction and overview of the system.
- **Chatbot** — Simple conversational assistant for user inquiries.
- **Alzheimer Prediction Data Info** — Predict Alzheimer’s diagnosis using structured data.
- **Alzheimer Prediction MRI** — Predict Alzheimer’s diagnosis from MRI data (image-based).
- **About Us** — Team and project information.

### 🧩 Machine Learning Models

The system compares and evaluates several models for Alzheimer’s prediction:

- **Linear Discriminant Analysis (LDA)**
- **Logistic Regression (Base & Tuned)**
- **Random Forest (Base & Tuned)**
- **Ridge Classifier (Base & Tuned)**
- **Gradient Boosting (Base & Tuned)**
- **MLP Neural Network (Base & Tuned)**
- **XGBoost (Base & Tuned)**
- **SVM (Linear & Polynomial, Base & Tuned)**
- **Stochastic Gradient Descent (Base & Tuned)**

All models are evaluated using metrics like:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix Visualization

---

## 📂 Project Structure

├── app.py                             # Main Streamlit app

├── config.py                          # Configuration settings

├── function.py                        # Helper functions (e.g., model evaluation)

├── models/

│   └── alzheimer_info_model.pkl       # Saved ML model (Logistic Regression)

├── data/

│   └── alzheimer_dataset.csv          # Dataset used for model training

├── web_pages/

│   ├── _home_page.py

│   ├── _chatbot_page.py

│   ├── _predict_page.py

│   ├── _mri_predict_page.py

│   └── _about_us.py

└── README.md                          # Project documentation

## ▶️ Running the Application

streamlit run app.py

## 📊 Dataset Information

The dataset (`alzheimer_dataset.csv`) includes clinical features such as:

* **Age**
* **Gender**
* **MMSE (Mini-Mental State Examination)**
* **Education_Year**
* **Ethnicity**
* **APOE4_Status**
* **APOE_Genotype**
* and others...

The target variable is  **DX.bl** , representing the Alzheimer’s diagnosis stage.

## 🧪 Model Training Overview

1. **Data Cleaning** — Missing values are removed.
2. **Feature Encoding** — Categorical features converted using one-hot encoding.
3. **Balancing** — Handled using **SMOTE** to mitigate class imbalance.
4. **Splitting** — 80% training / 20% testing.
5. **Model Training & Tuning** — Using **GridSearchCV** for hyperparameter optimization.
6. **Evaluation** — Metrics and visualizations using Seaborn and Matplotlib.
7. **Saving Model** — Best performing model saved as:
   ./models/alzheimer_info_model.pkl


## 🧠 Prediction Disclaimer

> ⚠️ **Disclaimer:**
>
> The predictions from this system are  **not intended for medical use** .
>
> This tool is designed for  **informational and educational purposes only** .
>
