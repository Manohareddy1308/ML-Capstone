# 23CSE301 Machine Learning Capstone — Review 1

## Project Objective

This project applies supervised machine-learning algorithms to two supplied datasets:
- **Regression:** Wine Quality White, predicting `quality`.
- **Classification Part-A:** UCI Bank Marketing, predicting `y` (`yes`/`no`).

## Datasets

- `data/winequality-white.csv` — target: `quality`
- `data/bank-full.csv` — target: `y`

## Regression — 10 Required Algorithms

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. SVR
10. KNN Regressor

All ten regression notebooks use the same cleaned/engineered wine dataset and deterministic 80:20 split (`random_state=42`). Metrics: **R², RMSE, MAE**.

### Regression feature engineering

`total_acidity = fixed acidity + volatile acidity` is created before the train/test split and is used consistently by all ten regression models.

### Regression outlier and duplicate handling

Exact duplicate wine rows are removed before splitting. IQR screening identifies potential outliers; potential extreme chemical measurements are retained rather than blindly deleted because they may be plausible observations. No test-set-derived outlier treatment is fitted.

Ridge and Random Forest use **GridSearchCV**. The comparison notebook consolidates all ten results, performs 5-fold CV for the top two models by held-out R², and includes predicted-vs-actual, residual, and tree feature-importance diagnostics.

## Classification Part-A — 5 Required Algorithms

1. Logistic Regression
2. KNN Classifier
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. SVC

Classification uses an 80:20 **stratified** split with `random_state=42`, leakage-safe preprocessing, and reports **Accuracy, Precision, Recall, Weighted F1, ROC-AUC, and Confusion Matrix**.

The EDA includes target distribution, numerical histograms, categorical count plots, numerical correlation heatmap, duplicate checking, and IQR screening of numerical variables. Duplicate-looking bank records are retained because the dataset has no unique customer identifier and identical feature rows are not automatically proof of erroneous duplication.

## Preprocessing and Data Leakage

Preprocessing is implemented with scikit-learn pipelines. Imputation, scaling, polynomial expansion, and one-hot encoding are fitted through the training portion only. Test data is only transformed by fitted preprocessing objects.

Regression uses the same deterministic split across all ten algorithms. Classification uses `stratify=y`.

## Project Structure

```text
Review-1/
├── data/
│   ├── winequality-white.csv
│   └── bank-full.csv
├── notebooks/
│   ├── 00_regression_model_comparison.ipynb
│   ├── 01_linear_regression.ipynb
│   ├── 02_ridge_regression.ipynb
│   ├── 03_lasso_regression.ipynb
│   ├── 04_elasticnet_regression.ipynb
│   ├── 05_polynomial_regression.ipynb
│   ├── 06_decision_tree_regressor.ipynb
│   ├── 07_random_forest_regressor.ipynb
│   ├── 08_gradient_boosting_regressor.ipynb
│   ├── 09_svr.ipynb
│   ├── 10_knn_regressor.ipynb
│   ├── 00_classification_model_comparison.ipynb
│   ├── 01_logistic_regression_classification.ipynb
│   ├── 02_knn_classifier.ipynb
│   ├── 03_gaussian_naive_bayes.ipynb
│   ├── 04_decision_tree_classifier.ipynb
│   └── 05_svc.ipynb
├── models/
├── results/
├── src/
│   ├── preprocessing.py
│   └── classification_preprocessing.py
├── README.md
├── requirements.txt
└── AUDIT_REPORT.txt
```

## How to Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml-capstone --display-name "ML Capstone (.venv)"
```

Open `Review-1/notebooks` in VS Code/Jupyter and select the **ML Capstone (.venv)** kernel.

## How to Run

Run the ten individual regression notebooks first, then:
`00_regression_model_comparison.ipynb`

Run the five Classification Part-A notebooks first, then:
`00_classification_model_comparison.ipynb`

Run each notebook from the first cell to the last cell. Restart the kernel when changing source files.

## Student Observations

Important EDA and diagnostic visualizations include explicit **Student Observation** placeholders. The final observations and interpretations must be written by the student team after inspecting the actual plots and outputs.

## AI Assistance Disclosure

Generative AI tools were used for code scaffolding, debugging support, and documentation assistance. Dataset analysis, interpretation of results, observations, and final conclusions were reviewed/completed by the student team.
