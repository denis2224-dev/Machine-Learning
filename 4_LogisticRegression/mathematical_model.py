import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def roc_auc_score_binary(y_true, y_score):
    """
    ROC AUC computed via rank statistic (Mann–Whitney U).
    Interpretable as: P(score(pos) > score(neg)).
    Handles ties by assigning average ranks.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0  # undefined; choose safe output

    # Sort scores and align labels
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    sorted_y = y_true[order]

    # Compute average ranks with tie handling
    ranks = np.empty_like(sorted_scores, dtype=float)
    i = 0
    rank = 1  # ranks start at 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        ranks[i:j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[sorted_y == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])

def precision_recall_f1_from_cm(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    # Class 1
    prec1 = safe_div(tp, tp + fp)
    rec1  = safe_div(tp, tp + fn)
    f11   = safe_div(2 * prec1 * rec1, prec1 + rec1)

    # Class 0 (treat 0 as positive)
    prec0 = safe_div(tn, tn + fn)
    rec0  = safe_div(tn, tn + fp)
    f10   = safe_div(2 * prec0 * rec0, prec0 + rec0)

    return (prec0, rec0, f10), (prec1, rec1, f11)

def train_test_split_stratified(X_df, y, test_size=0.35, random_state=42):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(len(idx0) * test_size)
    n1_test = int(len(idx1) * test_size)

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    X_train = X_df.iloc[train_idx].reset_index(drop=True)
    X_test  = X_df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test  = y[test_idx]
    return X_train, X_test, y_train, y_test

# Preprocessing (matches your sklearn pipeline)
# Gender -> Gender_Male (drop first)
# Standardize Age, EstimatedSalary using train stats only
def preprocess_fit_transform(X_train_df): # OneHotEncoder.fit_transform and StandardScaler.fit_transform
    X = X_train_df.copy()

    # One-hot (drop-first): Female baseline -> Gender_Male in {0,1}
    gender_male = (X["Gender"] == "Male").astype(float).to_numpy().reshape(-1, 1)

    age = X["Age"].astype(float).to_numpy().reshape(-1, 1)
    sal = X["EstimatedSalary"].astype(float).to_numpy().reshape(-1, 1)

    mu_age = float(age.mean())
    std_age = float(age.std(ddof=0)) if float(age.std(ddof=0)) != 0.0 else 1.0

    mu_sal = float(sal.mean())
    std_sal = float(sal.std(ddof=0)) if float(sal.std(ddof=0)) != 0.0 else 1.0

    age_s = (age - mu_age) / std_age
    sal_s = (sal - mu_sal) / std_sal

    X_num = np.hstack([gender_male, age_s, sal_s])  # [Gender_Male, Age, EstimatedSalary]
    stats = {"mu_age": mu_age, "std_age": std_age, "mu_sal": mu_sal, "std_sal": std_sal}
    feature_names = ["Gender_Male", "Age", "EstimatedSalary"]
    return X_num, stats, feature_names

def preprocess_transform(X_df, stats):
    X = X_df.copy()

    gender_male = (X["Gender"] == "Male").astype(float).to_numpy().reshape(-1, 1)
    age = X["Age"].astype(float).to_numpy().reshape(-1, 1)
    sal = X["EstimatedSalary"].astype(float).to_numpy().reshape(-1, 1)

    age_s = (age - stats["mu_age"]) / stats["std_age"]
    sal_s = (sal - stats["mu_sal"]) / stats["std_sal"]

    return np.hstack([gender_male, age_s, sal_s])

def sigmoid(z):
    z = np.clip(z, -500, 500)  # stability
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg_gd(X, y, lr=0.15, epochs=8000, l2_lambda=0.0):
    """
    X: (m, n) standardized numeric matrix
    y: (m,) in {0,1}
    l2_lambda: L2 strength (bias not regularized)
    """
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0.0
    y = np.asarray(y).reshape(-1, 1).astype(float)

    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)

        dz = (p - y)
        dw = (X.T @ dz) / m
        db = float(np.sum(dz) / m)

        if l2_lambda > 0:
            dw += (l2_lambda / m) * w

        w -= lr * dw
        b -= lr * db

    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b).ravel()

def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)



df = pd.read_csv("Social_Network_Ads.csv")

X_df = df.drop(columns=["User ID", "Purchased"])
y = df["Purchased"].astype(int).to_numpy()

X_train_df, X_test_df, y_train, y_test = train_test_split_stratified(
    X_df, y, test_size=0.35, random_state=42
)

X_train, stats, feature_names = preprocess_fit_transform(X_train_df)
X_test = preprocess_transform(X_test_df, stats)

w, b = train_logreg_gd(X_train, y_train, lr=0.15, epochs=8000, l2_lambda=0.0)

y_pred = predict(X_test, w, b, threshold=0.5)
y_prob = predict_proba(X_test, w, b)

acc = float(np.mean(y_pred == y_test))
auc = roc_auc_score_binary(y_test, y_prob)
cm = confusion_matrix_binary(y_test, y_pred)
(c0_p, c0_r, c0_f1), (c1_p, c1_r, c1_f1) = precision_recall_f1_from_cm(cm)

print("Accuracy:", acc)
print("ROC AUC: ", auc)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report (binary):")
print("              precision    recall  f1-score")
print(f"           0       {c0_p:0.2f}      {c0_r:0.2f}      {c0_f1:0.2f}")
print(f"           1       {c1_p:0.2f}      {c1_r:0.2f}      {c1_f1:0.2f}")

print("\nIntercept (b):", float(b))
print("Weights (w):")
for name, coef in zip(feature_names, w.ravel()):
    print(f"  {name}: {float(coef)}")


gender_value = "Female"  # change to "Male"

age_min, age_max = X_df["Age"].min() - 2, X_df["Age"].max() + 2
sal_min, sal_max = X_df["EstimatedSalary"].min() - 5000, X_df["EstimatedSalary"].max() + 5000

age_grid, sal_grid = np.meshgrid(
    np.linspace(age_min, age_max, 300),
    np.linspace(sal_min, sal_max, 300)
)

grid_df = pd.DataFrame({
    "Gender": [gender_value] * age_grid.size,
    "Age": age_grid.ravel(),
    "EstimatedSalary": sal_grid.ravel()
})

grid_X = preprocess_transform(grid_df, stats)
grid_proba = predict_proba(grid_X, w, b).reshape(age_grid.shape)

plt.figure(figsize=(10, 7))
plt.contourf(age_grid, sal_grid, grid_proba, levels=20, alpha=0.7)
plt.contour(age_grid, sal_grid, grid_proba, levels=[0.5])

mask = X_df["Gender"] == gender_value
plt.scatter(
    X_df.loc[mask, "Age"],
    X_df.loc[mask, "EstimatedSalary"],
    c=y[mask],
    edgecolor="black",
    s=50
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title(f"From-scratch Logistic Regression (Gender = {gender_value})")
plt.colorbar(label="P(Purchased = 1)")
plt.show()