# 1. Imports
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 2. Load data
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1]   # Features: Country, Age, Salary
y = dataset.iloc[:, -1]    # Target: Purchased


# 3. Define column types
categorical_cols = ["Country"]
numeric_cols = ["Age", "Salary"]


# 4. Numeric preprocessing 
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# 5. Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# 6. Combine preprocessing 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)


# 7. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)


# 8. Apply preprocessing 
X_train = preprocessor.fit_transform(X_train)
X_test  = preprocessor.transform(X_test)


# 9. Train model
model = LogisticRegression()
model.fit(X_train, y_train)


# 10. Predictions & evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("y_test:", y_test.tolist())
print("y_pred:", y_pred.tolist())
#print(classification_report(y_test, y_pred))
