# K-Nearest Neighbors (KNN)

This module is part of the **Machine-Learning** repository and documents my structured learning path in machine learning - from **classical supervised models** to **production-ready ML pipelines**.

This project focuses on the **K-Nearest Neighbors (KNN)** algorithm, implemented in two complementary ways:

- a **simple, educational implementation** (`main.py`) designed to clearly illustrate how KNN works step-by-step, and
- a **production-level implementation** (`production.py`) using modern ML best practices such as preprocessing pipelines, cross-validation, and hyperparameter tuning.

At this stage, the repository does **not yet include a full mathematical derivation** of KNN. The emphasis is on **algorithmic intuition, correct usage, and evaluation**. A dedicated mathematical model may be added later.

---

## 📌 What Problem Does KNN Solve?

**K-Nearest Neighbors solves classification and regression problems** by making predictions based on **similarity**.

Instead of learning explicit model parameters, KNN:
- stores the training data,
- measures distances between points,
- and predicts outcomes based on the *K most similar samples*.

Typical use cases:
- non-linear decision boundaries
- small to medium-sized datasets
- problems where similarity is meaningful

---

## 🧠 Core Idea

Given a new data point:

1. Compute its distance to **all training samples**
2. Select the **K closest neighbors**
3. Aggregate their outputs:
   - **Classification** → majority vote (or distance-weighted vote)
   - **Regression** → mean (or distance-weighted mean)

KNN is an example of a **lazy, non-parametric algorithm**:
- no training phase in the traditional sense
- computation happens at prediction time

---

## 🧪 Dataset

The project currently uses the **Iris dataset**, a standard benchmark for KNN:

- 150 samples
- 3 classes (`setosa`, `versicolor`, `virginica`)
- 4 numeric features

This dataset is ideal for:
- visualizing nearest neighbors
- understanding distance-based classification
- experimenting with different values of *K*

---

## ⚙️ `main.py` — Simple / Educational Version

The goal of `main.py` is **clarity**, not abstraction.

Characteristics:
- explicit `train_test_split`
- manual feature scaling with `StandardScaler`
- fixed value of `K`
- direct evaluation using accuracy, F1 score, confusion matrix

This version is suitable for:
- learning the algorithm
- exams and labs
- understanding how a single object is classified

---

## 🏗️ `production.py` — Production-Level Version

The production implementation introduces **engineering best practices**:

- preprocessing with `ColumnTransformer`
- leak-safe `Pipeline`
- hyperparameter tuning with `GridSearchCV`
- cross-validation with `StratifiedKFold`
- clean separation between training, tuning, and evaluation

This version answers:
> *“Which KNN configuration generalizes best to unseen data?”*

It is suitable for:
- real-world datasets
- fair model comparison
- reproducible experiments

---

## 🔁 Hyperparameters Tuned (Production)

The production model automatically searches over:
- number of neighbors (`n_neighbors`)
- distance metric (Euclidean vs Manhattan)
- voting strategy (`uniform` vs `distance`)

Hyperparameters are selected using **cross-validation**, without touching the test set.

---

## 📊 Evaluation Metrics

Depending on the script, evaluation includes:
- Accuracy
- F1 score (weighted for multiclass)
- Confusion matrix
- Classification report

These metrics provide both **overall performance** and **per-class diagnostics**.

---

## 🚧 Future Work

Planned extensions:
- mathematical formulation of KNN
- distance metric analysis
- curse of dimensionality experiments
- comparison with Logistic Regression and SVM
- visualization of nearest neighbors in feature space

---

## ✅ Key Takeaways

- KNN is simple but powerful
- performance depends heavily on:
  - feature scaling
  - choice of *K*
  - distance metric
- cross-validation is essential for fair evaluation
- production-ready KNN requires careful preprocessing

---

## 📎 Notes

This module complements previous work on:
- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression

and continues the progression toward **robust, well-evaluated ML models**.


