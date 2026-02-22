import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.mu_ = None
        self.sigma_ = None

    def fit(self, X: np.ndarray):
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_ = np.where(self.sigma_ < self.eps, 1.0, self.sigma_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("Scaler not fitted.")
        return (X - self.mu_) / self.sigma_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LinearSVMHinge:
    """
    Linear soft-margin SVM in the primal with hinge loss, trained with SGD.

    Objective:
        J(w,b) = 0.5||w||^2 + C * sum_i max(0, 1 - y_i (w^T x_i + b))

    Labels y must be in {-1, +1}.
    """

    def __init__(self, C=1.0, lr=1e-3, epochs=2000, random_state=42):
        self.C = float(C)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.rng = np.random.default_rng(random_state)
        self.w_ = None
        self.b_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n, d = X.shape
        self.w_ = np.zeros(d, dtype=float)
        self.b_ = 0.0

        for _ in range(self.epochs):
            i = self.rng.integers(0, n)
            xi = X[i]
            yi = y[i]

            margin = yi * (np.dot(self.w_, xi) + self.b_)

            if margin >= 1.0:
                grad_w = self.w_
                grad_b = 0.0
            else:
                grad_w = self.w_ - self.C * yi * xi
                grad_b = -self.C * yi

            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)


def confusion_matrix_binary(y_true01: np.ndarray, y_pred01: np.ndarray):
    tn = int(np.sum((y_true01 == 0) & (y_pred01 == 0)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)

def accuracy(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    return float(np.mean(y_true01 == y_pred01))

def precision_recall_f1(y_true01: np.ndarray, y_pred01: np.ndarray):
    cm = confusion_matrix_binary(y_true01, y_pred01)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, cm


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified split for binary labels y in {0,1}.
    Works with X as a numpy array.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(round(len(idx0) * test_size))
    n1_test = int(round(len(idx1) * test_size))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def main():
    df = pd.read_csv("data/wdbc_data.csv")

    # Target: diagnosis (M/B) -> {1,0}
    y01 = df["diagnosis"].map({"M": 1, "B": 0}).astype(int).to_numpy()

    # Features: drop id + diagnosis
    X = df.drop(columns=["id", "diagnosis"], errors="ignore").to_numpy(dtype=float)

    # Split
    X_train, X_test, y_train01, y_test01 = train_test_split(
        X, y01, test_size=0.2, random_state=42
    )

    # Scale (fit on train only)
    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_test = scaler.transform(X_test)

    # Convert labels {0,1} -> {-1,+1}
    y_train_pm = np.where(y_train01 == 1, 1, -1)

    # Train linear SVM (math)
    svm = LinearSVMHinge(C=1.0, lr=1e-3, epochs=50000, random_state=42)
    svm.fit(Z_train, y_train_pm)

    # Predict
    y_pred_pm = svm.predict(Z_test)
    y_pred01 = np.where(y_pred_pm == 1, 1, 0)

    # Evaluate
    acc = accuracy(y_test01, y_pred01)
    prec, rec, f1, cm = precision_recall_f1(y_test01, y_pred01)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()