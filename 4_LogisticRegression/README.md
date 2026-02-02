# Logistic Regression - From Mathematics to Classification Models

This module is part of the **Machine-Learning** repository, which documents my structured learning path in machine learning - from **mathematical foundations** to **production-style ML pipelines**.

This project presents an **end-to-end implementation of Logistic Regression**, developed in two parallel ways:
- using a **clean, production-style classification pipeline** with `scikit-learn` (preprocessing, pipelines, probabilistic outputs, and evaluation), and
- **from scratch**, using only **linear algebra, calculus, and NumPy** (sigmoid function, cross-entropy loss, gradient descent, and manual preprocessing).

The objective is not just to *apply* logistic regression, but to **fully understand and implement the mathematical and engineering pipeline behind binary classification models**.

---

## 📌 What Problem Does Logistic Regression Solve?

**Logistic Regression solves binary classification problems**, where the target variable takes values in:

$$
y \in \{0, 1\}
$$

Given a feature vector:

$$
\mathbf{x} = [x_1, x_2, \dots, x_d]
$$

the model estimates the **probability** that the output belongs to the positive class:

$$
P(y = 1 \mid \mathbf{x})
$$

using a linear model combined with a nonlinear activation.

---

## 🧮 Model Definition

The linear predictor (logit) is:

$$
z = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_d x_d
$$

This is converted into a probability using the **sigmoid function**:

$$
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The predicted class is obtained via thresholding:

$$
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 📊 Example Use Cases

- Predicting whether a **user will purchase a product**
- Classifying **spam vs non-spam emails**
- Fraud detection (fraud / no fraud)
- Medical diagnosis (disease present / absent)
- Credit risk assessment (default / no default)

The model learns a **decision boundary** that separates the two classes while maximizing probabilistic separation.

---

## 🎯 What This Module Demonstrates

- Translating **probabilistic models into working ML systems**
- Implementing classification **without black-box libraries**
- Understanding **log-odds, probabilities, and decision thresholds**
- Handling **categorical + numerical features**
- Correct **train/test separation** with stratification
- Pipeline-based **feature preprocessing**
- Gradient-based optimization (no closed-form solution)
- Interpreting coefficients via **odds ratios**
- Comparing **sklearn vs from-scratch implementations**

This module builds directly on:
- Multiple Linear Regression
- Vectorized linear algebra
- Feature scaling and encoding

and prepares for:
- Regularized logistic regression
- Neural networks
- Maximum likelihood estimation
- Generalized linear models (GLMs)

---

## 📈 Dataset Overview

- **Source:** Social Network Ads dataset  
- **Samples:** approximately 400  

- **Numerical Features:**
  - Age ($x_1$)
  - Estimated Salary ($x_2$)

- **Categorical Feature:**
  - Gender ($x_3$) → One-hot encoded (drop-first)

- **Target:**
  - Purchased ($y \in \{0,1\}$)


### Modeling Assumptions
- Binary outcome variable
- Independent and identically distributed samples (i.i.d.)
- Log-odds are a linear function of features
- No perfect multicollinearity
- Features standardized for gradient descent stability

---

## 🧠 Concepts Covered

### Machine Learning Engineering
- Column-based preprocessing pipelines
- One-hot encoding for categorical variables
- Feature standardization (Z-score normalization)
- Stratified train/test splitting
- Probabilistic prediction (`predict_proba`)
- Decision thresholds
- Confusion matrix–based evaluation
- Decision boundary visualization

### Mathematical Foundations
- Logistic model (logit function)
- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent optimization
- Vectorized gradient computation
- Odds ratios and coefficient interpretation
- ROC AUC as a ranking metric

---

## 🧮 Mathematical Model Overview

Let:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$ be the feature matrix (with intercept)
- $\mathbf{y} \in \{0,1\}^n$ be the target vector
- $\beta \in \mathbb{R}^{d+1}$ be the parameter vector

### Model

$$
z = \mathbf{X}\beta
\quad \Rightarrow \quad
\hat{\mathbf{p}} = \sigma(z)
$$

where the sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Loss Function (Binary Cross-Entropy)

$$
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(\hat{p}_i)+(1 - y_i)\log(1 - \hat{p}_i)\right]
$$

### Gradient

$$
\nabla_{\beta} J =
\frac{1}{n}\mathbf{X}^\top(\hat{\mathbf{p}} - \mathbf{y})
$$

### Optimization

Since no closed-form solution exists, parameters are learned via gradient descent:

$$
\beta^{(t+1)} =
\beta^{(t)} -
\alpha \nabla_{\beta} J
$$

All steps are implemented explicitly in the from-scratch version using **NumPy only**.

---

## 🧪 Experimental Results

| Metric | sklearn Pipeline | From-Scratch Model |
|------|------------------|--------------------|
| Accuracy | ≈ 0.85           | ≈ 0.827            |
| ROC AUC | ≈ 0.913          | ≈ 0.919            |
| Decision Boundary | ✔                | ✔                  |

Both implementations produce **consistent probabilistic predictions and rankings**, validating:
- correctness of the gradient derivation
- proper preprocessing alignment
- numerical stability of the implementation

---

## 🔍 Error Analysis

Observed sources of classification error:
- Overlap between classes in feature space
- Linear decision boundary limitation
- Sensitivity to threshold choice
- Class imbalance effects on recall

Planned improvements:
- L2 (Ridge) regularization
- Precision–Recall AUC analysis
- Threshold tuning
- Polynomial feature expansion
- Comparison with non-linear classifiers

---

## 🏗 Design Decisions

- **Sigmoid + cross-entropy:** ensures probabilistic interpretation
- **Stratified splitting:** preserves class proportions
- **Drop-first encoding:** avoids dummy variable trap
- **Feature standardization:** stabilizes gradient descent
- **Metric parity:** ensures sklearn and from-scratch comparability
- **Probability-first design:** separates scoring from decision policy

---

## 🧠 Mathematical Validation

To verify correctness, I compared:
- learned parameters $(w, b)$ from the from-scratch implementation
- coefficients from `scikit-learn`’s `LogisticRegression`

Both models:
- produce comparable probability estimates
- achieve similar ROC AUC scores
- induce nearly identical decision boundaries

This confirms the **correctness of the mathematical model, preprocessing steps, and optimization procedure**.

---

## 📊 Visualization Outputs

- **Decision Boundary (Age vs Salary)**  
  Visualizes the learned classification surface for each gender.

- **Probability Heatmap**  
  Shows how predicted purchase probability varies across feature space.

- **Confusion Matrix Metrics**  
  Precision, recall, and F1-score derived from raw predictions.

---

## 🚀 Learning Progression

This module prepares for advanced topics including:
- Regularized logistic regression
- Multiclass softmax regression
- Gradient-based optimization algorithms
- Neural networks and backpropagation
- Probabilistic ML models
- Model deployment with calibrated probabilities
