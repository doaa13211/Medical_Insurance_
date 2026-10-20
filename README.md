# Medical_Insurance_
Medical insurance cost prediction using linear and polynomial regression
Medical Insurance Cost Prediction

Project Overview

This project predicts individual medical insurance charges using the Medical Cost Personal Dataset.

Dataset

- Source: "Kaggle - Medical Cost Personal Dataset" (https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Number of records: 1,338
- Features: age, sex, BMI, children, smoker, region
- Target variable: charges

Models

1. Linear Regression
2. Polynomial Regression (Degree 2)

Methodology

- Data loading and inspection
- Categorical feature encoding using One-Hot Encoding
- Train-test split (80% training, 20% testing)
- Model training and prediction
- Evaluation using MAE, RMSE, and R² Score
- Actual vs. predicted values visualization

Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

How to Run

Install the required libraries:

pip install -r requirements.txt

Run the project:

python main.py

Results

The two regression models are compared using MAE, RMSE, and R² Score. The model with the lower RMSE is selected as the best-performing model on the test set.

Author

Doaa Mohamed
