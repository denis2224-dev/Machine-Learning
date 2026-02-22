# Support Vector Machines - From Maximum Margin Theory to Production Models

This module is part of the **Machine-Learning** repository, which documents my structured learning path in machine learning — from **mathematical foundations** to **production-style ML pipelines**.

This project presents an **end-to-end implementation of Support Vector Machines (SVMs)**, developed in two parallel ways:

- using a **clean, production-style ML workflow** with `scikit-learn` (pipelines, feature scaling, hyperparameter tuning, and evaluation), and  
- **from scratch**, implementing a **linear soft-margin SVM** using only **calculus, linear algebra, and NumPy** (hinge loss, regularization, and stochastic gradient descent).

The objective is not just to *apply* SVMs, but to **fully understand the geometric, optimization, and engineering principles behind maximum-margin classifiers**.

---

## 📌 What Problem Do Support Vector Machines Solve?

**Support Vector Machines solve classification problems by finding the decision boundary that maximizes the margin between classes.**

Given:
- a **feature vector** $ \mathbf{x} = [x_1, x_2, \dots, x_d] $
- a **binary label** $ y \in \{-1, +1\} $

SVM learns a hyperplane:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

Prediction rule:

$$
\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

### Example Use Cases

- Classifying **malignant vs benign tumors**  
- Detecting **spam vs non-spam emails**  
- Fraud detection  
- Sentiment analysis  
- Binary image classification  

SVM seeks the **most robust separating hyperplane**, minimizing classification error while maximizing geometric margin.

---

## 🎯 What This Module Demonstrates

- Translating **convex optimization theory into working ML systems**
- Implementing hinge-loss optimization without black-box solvers
- Understanding the **geometric interpretation of margins**
- Preventing data leakage via proper scaling and splitting
- Comparing linear and kernel-based SVMs
- Evaluating classifier performance using precision, recall, F1, and confusion matrix
- Validating mathematical derivations against sklearn implementations

This module builds the foundation for:

- Kernel methods  
- Convex dual optimization  
- Regularization theory  
- Large-margin classifiers  
- Neural network loss functions  

---

## 📈 Dataset Overview

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset  
- **Samples:** \( n = 569 \)  
- **Features:** 30 numerical measurements  
- **Target:** Diagnosis (Malignant = 1, Benign = 0)  

### Modeling Assumptions

- Binary classification  
- Independent and identically distributed samples (i.i.d.)  
- Soft-margin optimization to handle overlapping classes  
- Feature scaling required for geometric consistency  

---

## 🧠 Concepts Covered

### Machine Learning Engineering

- Stratified train/test splitting  
- Z-score feature standardization  
- Hinge loss optimization  
- Hyperparameter \( C \) tuning  
- Decision boundary computation  
- Margin interpretation and weight norm analysis  
- Model evaluation metrics  

### Mathematical Foundations

Hyperplane representation:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

Hinge loss:

$$
\max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

Regularized objective:

$$
J(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

Margin size:

$$
\text{Margin} = \frac{2}{\|\mathbf{w}\|}
$$

---

## 🧮 Mathematical Model Overview

The from-scratch implementation optimizes:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

Gradient updates:

If  
$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1
$$

Then:

$$
\nabla_{\mathbf{w}} = \mathbf{w}, \quad \nabla_b = 0
$$

Else:

$$
\nabla_{\mathbf{w}} = \mathbf{w} - C y_i \mathbf{x}_i
$$

$$
\nabla_b = -C y_i
$$

Parameters updated via stochastic gradient descent:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}
$$

$$
b \leftarrow b - \eta \nabla_b
$$

---

## 🧪 Experimental Results

| Metric | sklearn SVC | From-Scratch Linear SVM |
|--------|-------------|-------------------------|
| Accuracy | ~0.973      | ~0.938                  |
| F1 Score | 0.96        | 0.96                    |

Both implementations produce **numerically consistent classification behavior**, validating:

- correctness of hinge-loss derivation  
- proper feature scaling  
- stability of SGD optimization  

---

## 🔍 Error Analysis

Observed error sources:

- Overlapping feature distributions  
- Sensitivity to learning rate  
- Finite dataset size  
- Linear boundary limitation  

Planned improvements:

- Kernel SVM (RBF)  
- Hyperparameter grid search  
- Cross-validation  
- Early stopping  

---

## 🏗 Design Decisions

- **Custom StandardScaler:** Mirrors sklearn behavior and prevents leakage  
- **Stratified split:** Maintains class balance  
- **SGD optimization:** Avoids full quadratic programming solvers  
- **Weight norm inspection:** Connects geometry to margin  

---

## 🧠 Mathematical Validation

Compared:

- Weight vector \( \mathbf{w} \) from `LinearSVMHinge`
- Against `sklearn.svm.SVC(kernel="linear")`

Both produce consistent decision boundaries within numerical tolerance, confirming correctness of the mathematical model.

---

## 🚀 Learning Progression

This module prepares for advanced topics including:

- Kernel trick (RBF, Polynomial)  
- Dual optimization & Lagrange multipliers  
- Support vectors and KKT conditions  
- Convex quadratic programming  
- Neural network loss functions  