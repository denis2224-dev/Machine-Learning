# Multiple Linear Regression - From Mathematics to Code

An end-to-end implementation of **Multiple Linear Regression**, built:
- **entirely from scratch**, using only mathematics and NumPy, and  
- following a **clean, ML-correct workflow** (no sklearn, no black boxes).

The goal is not just to *run* a regression model, but to **understand every mathematical and algorithmic step behind it**.

---

## 📌 What This Project Demonstrates

- extending simple linear regression to **multiple input features**  
- handling **numeric and categorical data** manually  
- implementing preprocessing without sklearn  
- solving regression using the **normal equation**  
- preventing data leakage via correct train/test workflows  
- interpreting coefficients and evaluation metrics  
- visualizing training results correctly for multivariate models  

This project builds directly on simple linear regression and forms a strong foundation for:
- regularized regression (Ridge, Lasso)  
- gradient descent–based optimization  
- logistic regression  
- generalized linear models  

---

## 📁 Project Structure

3_MultipleLinearRegression/
- main.py # Production-style ML pipeline
- mathematical_model.py # Multiple linear regression from scratch
- 50_Startups.csv # Real-world dataset (numeric + categorical features)
- formulas.pdf # Math Summary
- README.md


---

## 🧠 Concepts Covered

### Machine Learning
- train/test split (manual implementation)  
- mean imputation (numeric features)  
- feature standardization (z-score)  
- one-hot encoding with dummy-variable trap avoidance  
- model training and prediction  
- evaluation using train vs test metrics  

### Mathematics
- hypothesis:  
  ŷ = b₀ + b₁x₁ + b₂x₂ + … + bₙxₙ  
- matrix formulation of linear regression  
- normal equation:  
  β = (XᵀX)⁻¹Xᵀy  
- pseudo-inverse for numerical stability  
- mean squared error (MSE)  
- coefficient of determination (R²)  

---

## 🧩 Dataset Description

The model is trained on a startup profit dataset containing:

- R&D Spend (numeric)  
- Administration (numeric)  
- Marketing Spend (numeric)  
- State (categorical)  
- Profit (target variable)  

The categorical feature is encoded manually using **one-hot encoding with a dropped base category** to ensure the design matrix is full-rank.

---

## 📊 Model Evaluation & Visualization

Because multiple linear regression cannot be represented as a single line, evaluation is done using:

- Actual vs Predicted plot (test set)  
- Residuals vs Predicted plot  

These plots allow inspection of:
- prediction accuracy  
- bias  
- variance patterns  
- potential violations of linear regression assumptions  

---

## 🎯 Key Takeaway

This project shows that **multiple linear regression is not a new algorithm**, but a natural generalization of simple linear regression once you:
- move from scalars to vectors  
- replace formulas with linear algebra  
- treat preprocessing as part of the model  

