import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size = 0.2, random_state = 0):
    rng = np.random.default_rng(random_state) # Creates a random number generator
    n = len(X)
    idx = np.arange(n) # Creates an index array
    rng.shuffle(idx) # shuffles the indeces

    n_test = int(round(n * test_size)) # Computes how many samples go into the test set
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def impute_mean_fit(X): # SimpleImputer(strategy="mean").fit()
    mean = np.nanmean(X[:, 0])
    return mean

def impute_mean_transform(X, mean): # SimpleImputer(...).transform()
    X_out = X.copy()
    nan_mask = np.isnan(X_out[:, 0]) # Checks which values are NaN. Returns a boolean mask like: [False, True, False, False]
    X_out[nan_mask, 0] = mean # Replaces only missing entries with the mean
    return X_out 

def standard_scaler_fit(X): # StandardScaler().fit()
    mu = X[:, 0].mean()
    sigma = X[:, 0].std(ddof = 0) # Computes the population standard deviation: σ = sqrt((1/n)​∑(xi​−μ)^2). ddof=0 means: divide by n
    if sigma == 0:
        sigma = 1.0 # avoid divide by zero if all X identical
    return mu, sigma

def standard_scaler_transform(X, mu, sigma): # StandardScaler().transform()
    X_out = X.copy()
    X_out[:, 0] = (X_out[:, 0] - mu) / sigma
    return X_out

def fit_linear_regression(X, y):
    """
    Fits y = b0 + b1 * x (where X is (n,1)).
    Returns b0, b1.
    """
    x = X[:, 0]
    x_bar = x.mean() # xˉ
    y_bar = y.mean() # yˉ​

    numerator = np.sum((x - x_bar) * (y - y_bar)) # covariance
    denominator = np.sum((x - x_bar) ** 2) # variance

    if denominator == 0:
        # all x are the same - can't learn a slope
        b1 = 0.0
        b0 = y_bar
        return b0, b1

    b1 = numerator / denominator
    b0 = y_bar - b1 * x_bar
    return b0, b1

def predict_linear(X, b0, b1):
    return b0 + b1 * X[:, 0]

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2) # SSE
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) #SST
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0



dataset = pd.read_csv("students.csv")

X = dataset.iloc[:, 0:1].to_numpy(dtype=float)  # feature (n,1)
y = dataset.iloc[:, 1].to_numpy(dtype=float)    # target  (n,)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Fit preprocessing on TRAIN only
x_mean = impute_mean_fit(X_train)
X_train_imp = impute_mean_transform(X_train, x_mean)
X_test_imp  = impute_mean_transform(X_test,  x_mean)

mu, sigma = standard_scaler_fit(X_train_imp)
X_train_scaled = standard_scaler_transform(X_train_imp, mu, sigma)
X_test_scaled  = standard_scaler_transform(X_test_imp,  mu, sigma)

# Fit Linear Regression on scaled TRAIN
b0, b1 = fit_linear_regression(X_train_scaled, y_train)


# Predict on scaled TEST
y_pred = predict_linear(X_test_scaled, b0, b1)

print("Intercept (b0):", b0)
print("Slope (b1):", b1)
print("MSE:", mse(y_test, y_pred))
print("R²:", r2(y_test, y_pred))

# For plotting a clean line: sort by X to make the line not zig-zag
train_order = np.argsort(X_train[:, 0])
test_order = np.argsort(X_test[:, 0])

# Training plot
plt.figure()
plt.scatter(X_train[:, 0], y_train)
plt.plot(
    X_train[train_order, 0],
    predict_linear(X_train_scaled[train_order], b0, b1)
)
plt.title("Training Set: Linear Regression (no sklearn)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Test plot
plt.figure()
plt.scatter(X_test[:, 0], y_test)
plt.plot(
    X_train[train_order, 0],
    predict_linear(X_train_scaled[train_order], b0, b1)
)
plt.title("Test Set: Linear Regression (no sklearn)")
plt.xlabel("X")
plt.ylabel("y")
plt.show()