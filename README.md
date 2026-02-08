# Machine Learning - From Foundations to Practice

This repository documents my **structured learning path in Machine Learning**, focusing on understanding algorithms **from first principles** and implementing them in **production-style pipelines**.

The goal of this repo is not to present a single polished project, but to **build deep intuition** for how classical ML algorithms work mathematically, how they behave in practice, and how they are implemented using standard tools such as `scikit-learn`.

---

## 🎯 Purpose

This repository exists to answer one question:

> *How do machine-learning algorithms actually work, mathematically and in code, and how do they behave on real data?*

To answer this, each topic is explored through:
- mathematical intuition and formulas,
- from-scratch implementations (NumPy-based),
- comparison with `scikit-learn` implementations,
- evaluation and interpretation of results.

---

## 🧠 Topics Covered

The repository includes implementations and experiments with:

- Linear Regression
- Multiple Linear Regression
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Support Vector Machines (SVM)
- Core evaluation metrics (MSE, R², accuracy, confusion matrix, etc.)
- Train/test splits and cross-validation
- Feature scaling and preprocessing

(Some models are implemented both **from scratch** and using **scikit-learn** for comparison.)

---

## 🧪 Structure & Approach

Each topic typically follows this structure:

1. **Mathematical model**
   - equations
   - loss functions
   - optimization intuition

2. **From-scratch implementation**
   - NumPy-based code
   - explicit gradient updates where applicable

3. **Library-based implementation**
   - `scikit-learn` pipelines
   - standard preprocessing and evaluation

4. **Comparison & validation**
   - numerical comparison of parameters
   - metric comparison on the same dataset

This approach helps bridge the gap between **theory** and **real-world ML engineering**.

