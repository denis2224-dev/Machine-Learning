import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('students.csv')
X = dataset.iloc[:, 0:1].to_numpy() #feature
y = dataset.iloc[:, 1].to_numpy() #target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)

model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)  # fit = “learn everything needed from the training data.”
y_pred = model.predict(X_test)  # Given new inputs, what scores does the model predict?

print("Intercept (b0):", model.named_steps["regressor"].intercept_)  # Value of y when x = 0 (after scaling)
print("Slope (b1):", model.named_steps["regressor"].coef_[0])  # How much y changes for a 1-unit change in scaled X
print("MSE:", mean_squared_error(y_test, y_pred))  # Measures average squared prediction error
print("R²:", r2_score(y_test, y_pred))  # R² = 0.85 → model explains 85% of variance

# Training set - data the model learns from
plt.figure()
plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train))
plt.title("Training Set: Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Test set - data the model is examined on
plt.figure()
plt.scatter(X_test, y_test)
plt.plot(X_train, model.predict(X_train))
plt.title("Test Set: Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

'''
Training set → data the model learns from
Test set → data the model is examined on
'''