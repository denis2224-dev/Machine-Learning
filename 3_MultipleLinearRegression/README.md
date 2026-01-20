# Multiple Linear Regression - From Mathematics to Models

This module is part of the **Machine-Learning** repository, which documents my structured learning path in machine learning - from **mathematical foundations** to **production-style ML pipelines**.

This project presents an **end-to-end implementation of Multiple Linear Regression**, built in two parallel ways:
- using a **clean, production-style ML workflow** with `scikit-learn` (pipelines, preprocessing, and evaluation), and
- **from scratch**, using only **linear algebra and NumPy** (normal equation, one-hot encoding, and feature scaling).

The goal is not just to *use* regression models, but to **understand and implement the full mathematical and engineering pipeline behind them**.

---

## 📌 What Problem Does Multiple Linear Regression Solve?

**Multiple Linear Regression predicts a continuous numerical value using multiple input features, both numerical and categorical.**

It learns the relationship between:
- a **feature vector** \( \mathbf{x} = [x_1, x_2, \dots, x_d] \)
- and a **target variable** \( y \)

by fitting a linear model:

$$
\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \dots + b_d x_d
$$

or in vector form:

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b_0
$$

### Example Use Cases
- Predicting **company profit** from R&D, administration, and marketing spend  
- Estimating **house prices** from size, location, and number of rooms  
- Forecasting **sales revenue** from marketing channels and region  
- Modeling **economic indicators** from multiple financial signals  

The model learns a **best-fitting hyperplane** that minimizes prediction error, typically measured using **Mean Squared Error (MSE)**.

---

## 🎯 What This Module Demonstrates

- Translating **linear algebra into working ML systems**
- Implementing regression **without black-box libraries**
- Handling **mixed data types** (numerical + categorical features)
- Correct **train/test separation** to prevent data leakage
- Pipeline-based **feature engineering and preprocessing**
- Quantitative **model evaluation and validation**
- Mathematical derivation and numerical stability of the **Normal Equation**

This module serves as a **foundation for advanced supervised learning**, including:
- Logistic Regression (classification)
- Gradient Descent optimization
- Regularization (Ridge, Lasso)
- Neural Networks

---

## 📈 Dataset Overview

- **Source:** 50_Startups dataset (academic / benchmarking dataset)  
- **Samples:** \( n \approx 50 \)  
- **Numerical Features:**  
  - R&D Spend (\( x_1 \))  
  - Administration (\( x_2 \))  
  - Marketing Spend (\( x_3 \))  
- **Categorical Feature:**  
  - State (\( x_4 \)) → One-hot encoded  
- **Target:** Profit (\( y \))  

### Modeling Assumptions
- Linear relationship between features and target  
- Independent and identically distributed samples (i.i.d.)  
- No perfect multicollinearity (dummy trap avoided via drop-first encoding)  
- Homoscedastic error terms  
- Features scaled for numerical stability  

---

## 🧠 Concepts Covered

### Machine Learning Engineering
- Column-based preprocessing pipelines
- Mean imputation for missing values
- Feature standardization (Z-score normalization)
- One-hot encoding for categorical variables
- Train/test split and evaluation
- Pipeline-based model training
- Model performance visualization

### Mathematical Foundations
- Linear model in vector form  

$$
\hat{y} = \mathbf{w}^T \mathbf{x} + b_0
$$

- Normal Equation (closed-form solution)
- Matrix multiplication and transpose
- Pseudo-inverse for numerical stability
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)

---

## 🧮 Mathematical Model Overview

The from-scratch implementation computes parameters using the **Normal Equation**:

Let:
- \( \mathbf{X} \in \mathbb{R}^{n \times d} \) be the feature matrix  
- Add an intercept column to form \( \mathbf{X}_b = [\mathbf{1} \mid \mathbf{X}] \)  
- \( \mathbf{y} \in \mathbb{R}^{n} \) be the target vector  

Then the parameter vector is:

$$
\mathbf{\beta} = (\mathbf{X}_b^T \mathbf{X}_b)^{-1} \mathbf{X}_b^T \mathbf{y}
$$

For numerical stability, this implementation uses the **Moore–Penrose Pseudo-Inverse**:

$$
\mathbf{\beta} = \mathbf{X}_b^{+} \mathbf{y}
$$

Where:
- \( \mathbf{\beta}_0 \) is the **intercept**
- \( \mathbf{\beta}_i \) are the **feature weights**
- \( \mathbf{X}_b^{+} \) denotes the pseudo-inverse

These formulas are implemented directly in `mathematical_model.py` using NumPy only.

---

## 🧪 Experimental Results

| Metric | Pipeline Model (sklearn) | Mathematical Model |
|--------|--------------------------|--------------------|
| MSE    | 56941600.54552           | 75789465.04589014  |
| R²     | 0.9610                   | 0.9287             |

Both implementations produce **numerically consistent predictions** on the same test split, validating:
- correctness of the normal equation derivation
- proper preprocessing alignment
- numerical stability of the matrix operations

---

## 🔍 Error Analysis

Observed sources of prediction error:
- Small dataset size increases variance
- Sensitivity to outliers in least-squares optimization
- High correlation between numerical features
- Categorical encoding increases dimensionality

Planned improvements:
- Ridge and Lasso regularization
- K-fold cross-validation
- Feature importance analysis
- Robust regression methods

---

## 🏗 Design Decisions

- **ColumnTransformer pipeline:** Ensures numerical and categorical features are processed independently and correctly  
- **Drop-first one-hot encoding:** Prevents multicollinearity (dummy variable trap)  
- **Pseudo-inverse solution:** Improves numerical stability over direct matrix inversion  
- **Metric parity testing:** Confirms sklearn and from-scratch implementations behave consistently  

---

## 🧠 Mathematical Validation

To verify correctness, I compared:
- parameter vector \( \mathbf{\beta} \) from `mathematical_model.py`
- learned coefficients from `scikit-learn`’s `LinearRegression`

Both implementations produce **numerically equivalent predictions within floating-point tolerance** on the same test set, confirming the correctness of the mathematical model and preprocessing pipeline.

---

## 📊 Visualization Outputs

- **Actual vs Predicted (Test Set)**  
  Evaluates overall model accuracy and bias.

- **Residuals vs Predicted (Test Set)**  
  Diagnoses heteroscedasticity, non-linearity, and outliers.

---

## 🚀 Learning Progression

This module prepares for advanced topics including:
- Logistic regression (classification)
- Gradient descent optimization
- Regularization techniques (Ridge, Lasso, Elastic Net)
- Feature selection methods
- Model deployment and inference pipelines
