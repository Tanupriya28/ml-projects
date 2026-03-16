#  Customer Churn Prediction — End-to-End Machine Learning Pipeline

## Project Overview

This project develops a complete end-to-end machine learning system to predict customer churn using structured telecom customer data. The objective is to identify customers likely to leave the service and uncover the key factors driving churn behavior.

The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, class imbalance handling, hyperparameter tuning, and business insight extraction.

---

##  Business Problem

Customer churn significantly impacts company revenue and customer acquisition costs.  

The objectives of this project are to:

- Predict whether a customer will churn (`Exited`)
- Improve detection of minority churn cases
- Identify the most influential churn drivers
- Provide actionable insights for customer retention strategies

---

##  Dataset Description

Telecom customer dataset containing demographic and account-level information.

### Key Features

- Credit Score  
- Geography  
- Gender  
- Age  
- Tenure  
- Balance  
- Number of Products  
- Estimated Salary  

**Target Variable:**  
- `Exited` (1 = Churned, 0 = Retained)

---

##  Data Preprocessing

- Removed non-informative identifier columns
- Handled missing and invalid values
- Encoded categorical variables
- Applied feature scaling using `StandardScaler`
- Performed stratified train-test split (70%-30%)
- Addressed class imbalance using class-weighted Random Forest

---

##  Exploratory Data Analysis (EDA)

Key observations:

- Dataset exhibited class imbalance in churn distribution
- Higher churn probability among:
  - Older customers
  - Customers with high account balance
  - Customers with low tenure
- Geographic distribution influenced churn behavior

---

##  Models Implemented

### Baseline Models
- Logistic Regression
- Random Forest

### Deep Learning Model
- Artificial Neural Network (MLPClassifier)

### Model Optimization
- GridSearchCV for ANN hyperparameter tuning
- Class-weight balancing for Random Forest

---

##  Model Evaluation

Models were evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve Analysis

###  ROC-AUC Comparison

| Model | ROC-AUC |
|--------|--------|
| Logistic Regression | 0.63 |
| Random Forest | 0.73 |
| ANN | 0.72 |
| Improved Random Forest | **0.84** |

The improved Random Forest achieved the highest ROC-AUC score of **0.84**, demonstrating strong discriminative capability between churned and retained customers.

---

##  Key Insights from Feature Importance

Top churn-driving factors:

- Age  
- Account Balance  
- Estimated Salary  
- Tenure  
- Credit Score  

### Business Interpretation

- Older customers with high balances showed increased churn probability.
- Low-tenure customers are more likely to churn early.
- High-value customers require proactive retention strategies.

These insights support targeted marketing campaigns and data-driven retention planning.

---

##  Technical Highlights

- End-to-end ML pipeline development
- Multiple model comparison
- Imbalance handling strategy
- Hyperparameter tuning
- Probability-based ROC-AUC evaluation
- Visual diagnostics (ROC Curve & Confusion Matrix)
- Business-focused interpretation of results

---

##  Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Seaborn  
- Matplotlib  

---

##  Future Improvements

- SHAP-based model explainability
- Threshold optimization for recall enhancement
- Deployment as a REST API
- Real-time churn prediction dashboard

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow from data preprocessing to model optimization and business insight generation. By addressing class imbalance and comparing multiple models, the improved Random Forest model achieved a strong ROC-AUC score of 0.84.

The analysis provides meaningful insights into customer behavior, enabling organizations to implement proactive and targeted churn reduction strategies.
