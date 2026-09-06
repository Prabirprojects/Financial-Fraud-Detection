# 💳 Financial Fraud Detection System

A Machine Learning based system that detects potentially fraudulent credit card transactions.

## 📌 Project Overview

Financial fraud is a major problem in digital transactions. This project uses Machine Learning to classify credit card transactions as either **Legitimate** or **Fraudulent**.

The project follows a complete Machine Learning workflow:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Model Training
* Hyperparameter Tuning
* Model Saving
* Streamlit Deployment

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

* 284,807 transactions
* 492 fraudulent transactions
* 30 input features
* 1 target variable (`Class`)

### Target Variable

| Class | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

The dataset is highly imbalanced because fraudulent transactions represent only a very small percentage of all transactions.

## 🤖 Machine Learning Models

Two classification models were explored:

1. Logistic Regression
2. Random Forest Classifier

Random Forest was selected as the final model after comparing the models and performing hyperparameter tuning.

### Final Random Forest Parameters

```text
n_estimators = 200
max_depth = 20
min_samples_leaf = 2
class_weight = balanced
```

## 🧠 Hyperparameter Tuning

GridSearchCV was used to find better Random Forest parameters.

Best parameters obtained:

```text
max_depth: 20
min_samples_leaf: 2
n_estimators: 200
```

Best cross-validation F1-score:

```text
0.8232
```

## 🖥️ Streamlit Application

The project includes a Streamlit web application where users can enter transaction features and receive a prediction.

The application provides:

* Fraud / Legitimate prediction
* Fraud probability
* Risk level
* Transaction summary

## 📁 Project Structure

```text
Financial-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   └── fraud_detection_model.pkl
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Create and activate a virtual environment:

```bash
python -m venv ccenv
```

Activate it on Windows:

```bash
ccenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

From the project root directory:

```bash
streamlit run app/app.py
```

The Streamlit application will open in your browser.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Jupyter Notebook

## 🚀 Future Improvements

* Improve fraud detection using advanced models such as XGBoost
* Perform more extensive feature engineering
* Optimize prediction thresholds
* Add real-time transaction monitoring
* Deploy the application online
* Add a transaction history/dashboard

## 👨‍💻 Author

**Prabir kumar Pattanayak**

Machine Learning Project — Financial Fraud Detection System
