# Enhancing Employee Attrition Prediction Using Hybrid Machine Learning Models

## 📌 Project Overview

Employee attrition is a significant challenge for organizations as employee turnover can lead to increased recruitment costs, loss of experienced employees, reduced productivity, and operational disruptions.

This project presents a hybrid machine learning approach for predicting employee attrition using HR analytics data. The proposed framework combines data preprocessing, class imbalance handling, Genetic Algorithm-based feature selection, ensemble machine learning, and model explainability.

The proposed system integrates **SMOTE**, **Genetic Algorithm (GA)**, **LightGBM**, **CatBoost**, and **SHAP** to build a predictive and interpretable employee attrition prediction framework.

The primary goal is to identify employees who may be at risk of leaving an organization and provide insights into the factors that influence the model's predictions.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Predict employee attrition using machine learning techniques.
- Analyze employee-related factors associated with attrition.
- Handle class imbalance using SMOTE.
- Select relevant features using a Genetic Algorithm.
- Develop a hybrid ensemble model using LightGBM and CatBoost.
- Compare the proposed model with existing and baseline machine learning approaches.
- Evaluate model performance using multiple classification metrics.
- Improve model transparency using SHAP-based explainability.
- Provide insights that can support data-driven employee retention strategies.

---

## ❓ Problem Statement

Employee attrition prediction is challenging because HR datasets often contain a large number of features, imbalanced target classes, and complex relationships between employee characteristics.

Traditional single-model approaches may not effectively capture these complex relationships. In addition, class imbalance can cause machine learning models to favor the majority class, making it difficult to correctly identify employees who are likely to leave.

Another challenge is model interpretability. High-performing machine learning models may provide accurate predictions but can be difficult to understand from a decision-making perspective.

This project addresses these challenges through a unified pipeline that combines:

- Data preprocessing
- SMOTE-based class balancing
- Genetic Algorithm-based feature selection
- LightGBM
- CatBoost
- Soft voting ensemble learning
- SHAP-based explainability

---

## 💡 Proposed Solution

The proposed framework follows a sequential machine learning pipeline:

```text
                    HR Analytics Dataset
                            │
                            ▼
                  Data Preprocessing
                            │
                            ▼
             Encoding & Feature Scaling
                            │
                            ▼
                   SMOTE Oversampling
                            │
                            ▼
              Genetic Algorithm (GA)
                 Feature Selection
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
          LightGBM                    CatBoost
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                  Soft Voting Ensemble
                            │
                            ▼
                Attrition Prediction
                            │
                            ▼
                  Model Evaluation
                            │
                            ▼
                   SHAP Explainability
```

---

## 🔬 Methodology

### 1. Data Preprocessing

The raw HR dataset is first cleaned and prepared for machine learning.

The preprocessing stage includes:

- Removing duplicate records
- Handling missing values
- Removing unnecessary attributes
- Encoding categorical variables
- Normalizing numerical features
- Feature scaling
- Preparing the target variable for binary classification

The target variable represents whether an employee is likely to stay or leave the organization.

### 2. Handling Class Imbalance Using SMOTE

Employee attrition datasets generally contain fewer employees who leave compared to employees who remain in the organization.

This creates a class imbalance problem that can affect model performance.

To address this issue, SMOTE (Synthetic Minority Oversampling Technique) is used to generate synthetic samples for the minority class.

This helps the model learn patterns associated with attrition cases more effectively.

### 3. Genetic Algorithm-Based Feature Selection

A Genetic Algorithm is used to identify a relevant subset of features.

Each possible feature combination is represented as a chromosome. The algorithm evaluates different feature combinations using a fitness function and iteratively searches for a better feature subset.

The process involves:

- Population initialization
- Fitness evaluation
- Selection
- Mutation
- Iterative optimization

The objective is to reduce redundant features while retaining features that contribute to prediction performance.

### 4. LightGBM

LightGBM (Light Gradient Boosting Machine) is used as one of the base learners in the proposed ensemble.

LightGBM is a gradient boosting algorithm that is effective for structured datasets and can capture nonlinear relationships between features.

In this project, LightGBM contributes to the ensemble by learning complex patterns in the selected employee features.

### 5. CatBoost

CatBoost is used as the second base learner.

CatBoost is a gradient boosting algorithm designed to work effectively with structured and categorical data.

Using CatBoost together with LightGBM provides model diversity, allowing the ensemble to learn different patterns from the same feature space.

### 6. Hybrid Ensemble Using Soft Voting

The outputs of LightGBM and CatBoost are combined using a soft voting ensemble.

Instead of relying on a single model, the ensemble combines the predicted probabilities produced by both models.

The proposed implementation uses:

```text
LightGBM + CatBoost
        ↓
   Soft Voting
        ↓
 Final Prediction
```

This approach is designed to improve prediction stability and take advantage of the complementary strengths of the two models.

The implemented ensemble uses a voting configuration with model weights of 2 for LightGBM and 1 for CatBoost.

### 7. Employee Attrition Prediction

The trained ensemble predicts whether an employee is likely to:

- Stay
- Leave

The model also produces probability scores that can be used to represent the estimated likelihood of attrition.

This allows employees to be viewed according to their predicted attrition risk rather than relying only on a binary prediction.

### 8. Model Explainability Using SHAP

Machine learning predictions are more useful when the reasons behind the predictions can also be understood.

This project uses SHAP (SHapley Additive exPlanations) to analyze the contribution of individual features to model predictions.

SHAP provides:

- Global feature importance
- Feature contribution analysis
- Better understanding of model behavior
- Greater transparency of predictions

This makes the model more interpretable for HR analytics applications.

---

## 📊 Dataset

The project uses an HR Analytics dataset containing employee-related attributes.

The dataset initially contains:

- **1,473 records**
- **30 features**

The dataset includes employee characteristics related to areas such as:

- Demographics
- Job role
- Job satisfaction
- Compensation
- Work experience
- Work environment
- Work-life balance
- Performance-related factors

The target variable is Attrition, representing whether an employee leaves the organization.

The project report identifies class imbalance between employees who remain and employees who leave.

---

## 📈 Model Evaluation

The proposed model is evaluated using several classification metrics.

### Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Measures the overall percentage of correct predictions |
| Precision | Measures how many predicted attrition cases were actually attrition cases |
| Recall | Measures how effectively the model identifies actual attrition cases |
| F1-Score | Provides a balance between precision and recall |
| ROC-AUC | Measures the model's ability to distinguish between classes |
| Confusion Matrix | Shows correct and incorrect classification outcomes |

These metrics provide a broader view of model performance, particularly because employee attrition involves an imbalanced classification problem.

---

