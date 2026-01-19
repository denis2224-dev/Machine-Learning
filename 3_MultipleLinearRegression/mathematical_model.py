import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size = 0.2, random_state = 0):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def impute_mean_fit(X): # SimpleImputer(strategy="mean").fit()
    means = np.nanmean(X, axis=0) # mean for each column (d,)
    return means

def impute_mean_transform(X, means): # SimpleImputer(...).transform()
    X_out = X.copy()
    nan_mask = np.isnan(X_out) # Checks which values are NaN. Boolean mask (n,d)
    X_out[nan_mask] = np.take(means, np.where(nan_mask)[1]) # Replace NaNs with the column mean
    return X_out

def standard_scaler_fit(X): # StandardScaler().fit()
    mu = X.mean(axis=0) # column means (d,)
    sigma = X.std(axis=0, ddof=0) # population std (d,)
    sigma[sigma == 0] = 1.0 # avoid divide by zero if a column is constant
    return mu, sigma

def standard_scaler_transform(X, mu, sigma): # StandardScaler().transform()
    return (X - mu) / sigma

def one_hot_fit(categories):
    return np.unique(categories) # uniques from TRAIN only

def one_hot_transform(categories, uniques):
    """
    categories: (n,)
    uniques: (k,)
    returns one-hot (n, k-1) using drop_first to avoid dummy trap
    """
    uniques = list(uniques)
    # drop first category
    base = uniques[0]
    used = uniques[1:]

    n = len(categories)
    k = len(used)
    out = np.zeros((n, k), dtype=float)

    cat_to_col = {c: j for j, c in enumerate(used)}
    for i, c in enumerate(categories):
        if c in cat_to_col:
            out[i, cat_to_col[c]] = 1.0
        # if it's base or unseen -> all zeros (base-like)
    return out, used, base

def add_intercept(X):
    """
    X: (n, d) -> (n, d+1) with first col = 1
    """
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X]) # adds the intercept column

def fit_multiple_linear_regression_normal_eq(X, y):
    """
    X: (n, d) features WITHOUT intercept col
    y: (n,)
    returns beta: (d+1,) including intercept
    """
    Xb = add_intercept(X)  # (n, d+1)
    # Use pseudo-inverse for numerical stability
    beta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)
    return beta

def predict_multiple_linear(X, beta):
    Xb = add_intercept(X)
    return Xb @ beta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


dataset = pd.read_csv("50_Startups.csv")

# Separate features/target
y = dataset["Profit"].to_numpy(dtype=float) # target (n,)

# Numeric features
X_num_all = dataset[["R&D Spend", "Administration", "Marketing Spend"]].to_numpy(dtype=float) # (n,3)

# Categorical feature
X_cat_all = dataset["State"].to_numpy(dtype=str) # (n,)

# Combine into a single index-based split (so rows stay aligned)
idx_all = np.arange(len(dataset))
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    idx_all, y, test_size=0.2, random_state=42
)

X_num_train = X_num_all[X_train_idx]
X_num_test  = X_num_all[X_test_idx]

X_cat_train = X_cat_all[X_train_idx]
X_cat_test  = X_cat_all[X_test_idx]


# Fit preprocessing on TRAIN only
means = impute_mean_fit(X_num_train)
X_num_train_imp = impute_mean_transform(X_num_train, means)
X_num_test_imp  = impute_mean_transform(X_num_test,  means)

mu, sigma = standard_scaler_fit(X_num_train_imp)
X_num_train_scaled = standard_scaler_transform(X_num_train_imp, mu, sigma)
X_num_test_scaled  = standard_scaler_transform(X_num_test_imp,  mu, sigma)

uniques = one_hot_fit(X_cat_train)
X_cat_train_oh, used_cats, base_cat = one_hot_transform(X_cat_train, uniques)
X_cat_test_oh,  _,        _        = one_hot_transform(X_cat_test,  uniques)

# Final X = [scaled numeric | one-hot state]
X_train = np.hstack([X_num_train_scaled, X_cat_train_oh])
X_test  = np.hstack([X_num_test_scaled,  X_cat_test_oh])

feature_names = ["R&D Spend", "Administration", "Marketing Spend"] + [f"State_{c}" for c in used_cats]


# Fit + predict
beta = fit_multiple_linear_regression_normal_eq(X_train, y_train)

y_train_pred = predict_multiple_linear(X_train, beta)
y_test_pred  = predict_multiple_linear(X_test,  beta)

print("Intercept (b0):", beta[0])
print("Coefficients:")
for name, coef in zip(feature_names, beta[1:]):
    print(f"  {name}: {coef}")

print("\nTrain MSE:", mse(y_train, y_train_pred))
print("Train R²:", r2(y_train, y_train_pred))
print("Test  MSE:", mse(y_test, y_test_pred))
print("Test  R²:", r2(y_test, y_test_pred))


# Matplotlib results (multiple regression-friendly)
# Actual vs Predicted (TEST)
plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Multiple Linear Regression (No sklearn): Actual vs Predicted (Test)")

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val]) # perfect prediction line y = x
plt.show()

# Residuals vs Predicted (TEST)
residuals = y_test - y_test_pred
plt.figure()
plt.scatter(y_test_pred, residuals)
plt.axhline(0) # zero residual line
plt.xlabel("Predicted Profit")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted (Test)")
plt.show()