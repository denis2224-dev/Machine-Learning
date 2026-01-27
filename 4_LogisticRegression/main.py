import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("Social_Network_Ads.csv")

X = df.drop(columns=["User ID", "Purchased"])
y = df["Purchased"]

categorical_cols = ["Gender"]
numerical_cols = ["Age", "EstimatedSalary"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ],
    remainder="drop"
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model = pipeline.named_steps["regressor"]
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]

# feature names after preprocessing
feature_names = list(ohe.get_feature_names_out(categorical_cols)) + numerical_cols

print("\nIntercept (b):", model.intercept_[0])
print("Weights (w):")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"  {name}: {coef}")


# Choose gender to visualize (0 = Female, 1 = Male)
gender_value = "Female"  # change for Male

# Extract trained pipeline parts
preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["regressor"]

# Create grid for Age and Salary
age_min, age_max = X["Age"].min() - 2, X["Age"].max() + 2
sal_min, sal_max = X["EstimatedSalary"].min() - 5000, X["EstimatedSalary"].max() + 5000

age_grid, sal_grid = np.meshgrid(
    np.linspace(age_min, age_max, 300), # Creates 300 evenly spaced Age values
    np.linspace(sal_min, sal_max, 300)
)

# Build grid dataframe
grid_df = pd.DataFrame({
    "Gender": gender_value,
    "Age": age_grid.ravel(), # Shape goes from (300, 300) → (90000,)
    "EstimatedSalary": sal_grid.ravel()
})

# Predict probabilities on grid
grid_proba = pipeline.predict_proba(grid_df)[:, 1]
grid_pred = grid_proba.reshape(age_grid.shape)

# Plot decision surface
plt.figure(figsize=(10, 7))
plt.contourf(age_grid, sal_grid, grid_pred, levels=20, alpha=0.7)
plt.contour(age_grid, sal_grid, grid_pred, levels=[0.5])

# Plot actual data points
mask = X["Gender"] == gender_value
plt.scatter(
    X.loc[mask, "Age"],
    X.loc[mask, "EstimatedSalary"],
    c=y[mask],
    edgecolor="black",
    s=50
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title(f"Logistic Regression Decision Boundary (Gender = {gender_value})")
plt.colorbar(label="P(Purchased = 1)")
plt.show()