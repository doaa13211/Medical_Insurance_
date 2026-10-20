import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

X = df.drop("charges", axis=1)
y = df["charges"]

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

polynomial_model = Pipeline([
    ("preprocessor", preprocessor),
    ("polynomial", PolynomialFeatures(
        degree=2, include_bias=False
    )),
    ("model", LinearRegression())
])

polynomial_model.fit(X_train, y_train)
polynomial_predictions = polynomial_model.predict(X_test)

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

results = pd.DataFrame([
    evaluate_model("Linear Regression", y_test, linear_predictions),
    evaluate_model("Polynomial Regression", y_test, polynomial_predictions)
])

print("\nModel Comparison:")
print(results.round(3).to_string(index=False))

plt.figure(figsize=(10, 5))

plt.scatter(
    y_test, linear_predictions,
    alpha=0.6, label="Linear Regression"
)

plt.scatter(
    y_test, polynomial_predictions,
    alpha=0.6, label="Polynomial Regression"
)

minimum = min(y_test.min(), linear_predictions.min(),
              polynomial_predictions.min())
maximum = max(y_test.max(), linear_predictions.max(),
              polynomial_predictions.max())

plt.plot([minimum, maximum], [minimum, maximum],
         "k--", label="Perfect Prediction")

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.legend()
plt.tight_layout()
plt.show()

best_model = results.loc[results["RMSE"].idxmin(), "Model"]
print("\nBest model based on RMSE:", best_model)