# Simple Linear Regression - From Mathematics to Code

This module is part of the **Machine-Learning** repository, which documents my structured learning path in machine learning — from **mathematical foundations** to **production-style ML pipelines**.

This project presents an **end-to-end implementation of Simple Linear Regression**, built in two parallel ways:
- using a **clean, professional ML workflow** with `scikit-learn`, and
- **from scratch**, using only mathematics and NumPy.

The goal is not just to *use* linear regression, but to **understand and implement every component of the model internally and validate it experimentally**.

---

## 📌 What Problem Does Simple Linear Regression Solve?

**Simple Linear Regression solves the problem of predicting a continuous numerical value using a single input feature.**

It learns the relationship between:
- an **independent variable** \( x \)
- and a **dependent variable** \( y \)

by fitting a function:

$$
\hat{y} = b_0 + b_1 x
$$

where:
- \( b_0 \) is the **intercept**
- \( b_1 \) is the **slope (weight)**

### Example Use Cases
- Predicting **exam scores** from hours studied  
- Estimating **house price** from floor area  
- Forecasting **sales** from advertising budget  
- Modeling **temperature trends** over time  

The model finds the line that **minimizes prediction error**, typically measured using **Mean Squared Error (MSE)**.

---

## 🎯 What This Module Demonstrates

- Translating **linear algebra and statistics into working code**
- Implementing regression **without black-box libraries**
- Correct **train/test separation** to prevent data leakage
- Clean **ML pipeline design**
- Quantitative **model evaluation and interpretation**
- Mathematical derivation and numerical validation of the **least squares solution**

This module serves as a **mathematical and engineering foundation** for:
- Multiple Linear Regression
- Logistic Regression
- Gradient Descent Optimization
- Neural Networks

---

## 📈 Dataset Overview

- **Source:** Academic / synthetic dataset  
- **Samples:** \( n \approx 100 \)  
- **Feature:** Hours studied (\( x \))  
- **Target:** Exam score (\( y \))  

### Modeling Assumptions
- Linear relationship between \( x \) and \( y \)  
- Independent and identically distributed samples (i.i.d.)  
- Homoscedastic error terms  
- No dominant outliers or high-leverage points  

---

## 🧠 Concepts Covered

### Machine Learning Engineering
- Dataset loading & inspection
- Train/test split
- Mean value imputation
- Feature standardization (Z-score normalization)
- Model training & prediction
- Pipeline-based workflow
- Performance evaluation

### Mathematical Foundations
- Hypothesis function  

$$
\hat{y} = b_0 + b_1 x
$$

- Least Squares optimization (closed-form solution)
- Covariance and variance
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)

---

## 🧮 Mathematical Model Overview

The model parameters are computed using the **Normal Equation**:

$$
b_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
$$

$$
b_0 = \bar{y} - b_1 \bar{x}
$$

Where:
- \( \bar{x} \) is the **mean of the input feature**
- \( \bar{y} \) is the **mean of the target variable**
- \( x_i \) is the **i-th input sample**
- \( y_i \) is the **i-th target value**
- \( b_0 \) is the **intercept (bias term)**
- \( b_1 \) is the **slope (model weight)**
- \( \hat{y} \) is the **predicted output**
- \( n \) is the **number of data points**

These formulas are implemented directly in `mathematical_model.py` without using ML libraries.

---

## 🧪 Experimental Results

| Metric | Pipeline Model | Mathematical Model |
|--------|----------------|--------------------|
| MSE    | 117.6          | 135.61             |
| R²     | 0.746          | 0.733              |

Both implementations produce numerically equivalent results, validating the correctness of the mathematical derivation and numerical stability of the implementation.

---

## 🔍 Error Analysis

Observed sources of prediction error:
- Small dataset size increases variance
- Outliers disproportionately affect the least squares solution
- Feature scaling impacts numerical stability but not model correctness


---

## 🏗 Design Decisions

- **Pipeline-based training:** Ensures preprocessing is fit only on training data, preventing data leakage  
- **From-scratch implementation:** Verifies mathematical correctness and exposes numerical considerations  
- **Metric parity testing:** Confirms both models converge to the same solution within floating-point tolerance  

---

## 🧠 Mathematical Validation

To verify correctness, I compared:
- closed-form coefficients from `mathematical_model.py`
- learned parameters from `scikit-learn`’s `main.py`

Both implementations produce numerically equivalent predictions on the same test set, confirming the correctness of the from-scratch implementation.

---

## 🚀 Learning Progression

This module prepares for advanced topics including:
- Vectorized multiple-feature regression
- Logistic regression (classification)
- Gradient descent optimization
- Regularization techniques (Ridge, Lasso)
- Model deployment workflows
