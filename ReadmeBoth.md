# 23CSE301 Machine Learning Capstone

**Academic Year:** 2026–2027  
**Reviews Covered:** Review 1 and Review 2

## 1. Project Overview

This repository contains the Machine Learning Capstone project for course 23CSE301. It covers supervised learning, model evaluation, hyperparameter tuning, exploratory data analysis, and unsupervised customer segmentation.

The project is divided into the following areas:

- **Review 1 — Regression:** Compare ten regression algorithms using the Wine Quality White dataset.
- **Review 1 — Classification Part A:** Evaluate five classifiers using the UCI Bank Marketing dataset.
- **Review 2 — Classification Part B:** Evaluate five additional classifiers and compare all ten classifiers.
- **Review 2 — Clustering:** Apply K-Means and Agglomerative Hierarchical Clustering to Online Retail customer data.

The project uses Python, pandas, NumPy, Matplotlib, scikit-learn, and Jupyter notebooks.

## 2. Datasets

| Dataset | Repository path | Task | Target or features |
|---|---|---|---|
| Wine Quality White | `data/winequality-white.csv` | Regression | Predict `quality` |
| UCI Bank Marketing | `data/bank-full.csv` | Binary classification | Predict `y` (`yes`/`no`) |
| Online Retail | `data/Online Retail.xlsx` | Customer clustering | Recency, Frequency, Monetary value, and Quantity |

The datasets should be placed in the `data/` directory before running the notebooks.

## 3. Review 1 — Regression

### Objective

Predict wine quality using physicochemical measurements and compare ten regression algorithms.

### Algorithms

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regression (SVR)
10. K-Nearest Neighbors Regressor

### Evaluation

The regression models are evaluated using:

- **R²:** Proportion of target variance explained by the model.
- **RMSE:** Root mean squared prediction error.
- **MAE:** Mean absolute prediction error.

The workflow uses a deterministic 80:20 train/test split with `random_state=42`. The `total_acidity` feature is derived from fixed acidity and volatile acidity. Exact duplicate rows are removed, and IQR screening is used to investigate potential outliers without automatically deleting plausible extreme measurements.

Ridge Regression and Random Forest use GridSearchCV in the documented workflow. The regression comparison notebook consolidates results and provides additional diagnostics, including predicted-versus-actual plots, residual analysis, and feature importance where applicable.

**Notebooks:** `notebooks/regression/`

**Comparison notebook:** `notebooks/comparison/regression_model_comparison.ipynb`

## 4. Review 1 — Classification Part A

### Objective

Predict whether a bank client subscribes to a term deposit.

### Algorithms

1. Logistic Regression
2. K-Nearest Neighbors Classifier
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)

### Evaluation

The classification workflow reports:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- ROC-AUC
- Confusion matrix

The workflow uses an 80:20 stratified train/test split with `random_state=42`. Exploratory analysis includes target distribution, numerical and categorical feature distributions, correlations, duplicate checking, and IQR-based outlier screening.

**Notebooks:** `notebooks/classification/`

## 5. Review 2 — Classification Part B

Five additional classifiers are evaluated on the Bank Marketing dataset:

1. Random Forest
2. AdaBoost
3. Gradient Boosting
4. Bagging Classifier
5. Multi-Layer Perceptron (MLP)

Together with the five Review 1 classifiers, these form the ten-model classification comparison.

The comparison workflow consolidates individual model results and is intended to compare Accuracy, Precision, Recall, Weighted F1-score, and ROC-AUC. Confusion matrices and relevant visualizations support the evaluation.

**All-ten comparison notebook:** `notebooks/comparison/classification_all_10_comparison.ipynb`

**Additional comparison notebook:** `notebooks/comparison/classification_model_comparison.ipynb`

**Individual classifier results:** `results/classification/`

The final best-model selection should be based on the actual generated metrics and the requirements of the classification task. Model performance should not be assumed from the algorithm name alone.

## 6. Review 2 — Customer Clustering

### Objective

Segment Online Retail customers according to their purchasing behavior using unsupervised learning.

### Feature preparation

The clustering workflow cleans transaction records, excludes cancellation invoices and invalid non-positive quantities or prices, removes duplicate transactions, and aggregates the data into customer-level features.

The features include:

- **Recency:** Time since the customer's most recent purchase.
- **Frequency:** Number of distinct invoices.
- **Monetary value:** Total purchase value.
- **Quantity:** Total quantity purchased.

Skewed purchasing features are log-transformed, and the features are standardized before clustering.

### K-Means

K-Means partitions customers into a selected number of clusters.

The notebook generates an elbow curve, evaluation metrics, PCA visualization, t-SNE visualization, and customer cluster labels.

**Notebook:** `notebooks/clustering/01_kmeans_clustering.ipynb`

### Agglomerative Hierarchical Clustering

Agglomerative clustering builds a hierarchy of clusters. The implementation compares three linkage strategies:

- Ward
- Complete
- Average

The notebook generates a dendrogram, linkage comparison metrics, PCA plots, and customer cluster labels.

**Notebook:** `notebooks/clustering/02_agglomerative_clustering.ipynb`

### Evaluation metrics

| Metric | Preferred direction | Purpose |
|---|---|---|
| Silhouette Score | Higher | Measures cluster cohesion and separation |
| Davies–Bouldin Index | Lower | Measures similarity between clusters |
| Calinski–Harabasz Index | Higher | Compares between-cluster and within-cluster dispersion |

The metrics may favor different methods. Cluster sizes and visualizations should also be examined before interpreting the segments.

### Clustering comparison

**Notebook:** `notebooks/comparison/clustering_model_comparison.ipynb`

The comparison combines K-Means with the Ward, Complete, and Average Agglomerative results.

Generated artifacts include:

- `results/clustering/clustering_comparison.csv`
- `figures/clustering/comparison_silhouette.png`
- `figures/clustering/comparison_davies_bouldin.png`
- `figures/clustering/comparison_calinski_harabasz.png`

Additional clustering metrics, labels, and plots are stored under `results/clustering/` and `figures/clustering/`.

## 7. Preprocessing and Data Leakage Prevention

The supervised learning workflows use scikit-learn pipelines where appropriate.

The main principles are:

- Split supervised data before fitting preprocessing operations.
- Fit imputers, scalers, encoders, and other learned transformations on training data only.
- Apply the fitted transformations to test data without refitting.
- Use stratification for Bank Marketing classification.
- Keep random seeds consistent where applicable.
- Perform model selection and hyperparameter tuning within the appropriate training workflow.

Clustering uses customer-level features that are transformed and standardized before fitting the unsupervised algorithms. Cluster labels are produced by the algorithms; they are not ground-truth targets.

## 8. Repository Structure

```text
ML-Capstone/
├── data/
│   ├── winequality-white.csv
│   ├── bank-full.csv
│   └── Online Retail.xlsx
├── notebooks/
│   ├── classification/
│   │   ├── 01_logistic_regression_classification.ipynb
│   │   ├── 02_knn_classifier.ipynb
│   │   ├── 03_gaussian_naive_bayes.ipynb
│   │   ├── 04_decision_tree_classifier.ipynb
│   │   ├── 05_svc.ipynb
│   │   ├── 06_random_forest_classifier.ipynb
│   │   ├── 07_adaboost_classifier.ipynb
│   │   ├── 08_gradient_boosting_classifier.ipynb
│   │   ├── 09_bagging_classifier.ipynb
│   │   └── 10_mlp_classifier.ipynb
│   ├── regression/
│   │   ├── 01_linear_regression.ipynb
│   │   ├── 02_ridge_regression.ipynb
│   │   ├── 03_lasso_regression.ipynb
│   │   ├── 04_elasticnet_regression.ipynb
│   │   ├── 05_polynomial_regression.ipynb
│   │   ├── 06_decision_tree_regressor.ipynb
│   │   ├── 07_random_forest_regressor.ipynb
│   │   ├── 08_gradient_boosting_regressor.ipynb
│   │   ├── 09_svr.ipynb
│   │   └── 10_knn_regressor.ipynb
│   ├── clustering/
│   │   ├── 01_kmeans_clustering.ipynb
│   │   └── 02_agglomerative_clustering.ipynb
│   └── comparison/
│       ├── classification_model_comparison.ipynb
│       ├── classification_all_10_comparison.ipynb
│       ├── regression_model_comparison.ipynb
│       └── clustering_model_comparison.ipynb
├── results/
│   ├── classification/
│   ├── regression/
│   └── clustering/
├── figures/
│   └── clustering/
├── models/
├── src/
├── docs/
├── requirements.txt
└── README.md
```

## 9. Environment Setup

Python 3.11 or 3.12 is recommended, subject to dependency compatibility.

### Linux / Fedora

```bash
cd ML-Capstone

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user \
  --name ml-capstone \
  --display-name "ML Capstone (.venv)"
```

### Windows PowerShell

```powershell
cd ML-Capstone

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m ipykernel install --user --name ml-capstone --display-name "ML Capstone (.venv)"
```

Open the repository in VS Code, select the **ML Capstone (.venv)** notebook kernel, and confirm that the datasets are present under `data/`.

The main dependencies pinned in `requirements.txt` are NumPy, pandas, scikit-learn, Matplotlib, Jupyter Notebook, ipykernel, and joblib.

## 10. Recommended Execution Order

Run the individual notebooks before their corresponding comparison notebooks.

### Regression

1. Run all ten notebooks under `notebooks/regression/`.
2. Run `notebooks/comparison/regression_model_comparison.ipynb`.

### Classification

1. Run the five Review 1 classifiers under `notebooks/classification/`.
2. Run the five Review 2 classifiers under `notebooks/classification/`.
3. Run `notebooks/comparison/classification_model_comparison.ipynb` if required for the Review 1 workflow.
4. Run `notebooks/comparison/classification_all_10_comparison.ipynb`.

### Clustering

1. Run `notebooks/clustering/01_kmeans_clustering.ipynb`.
2. Run `notebooks/clustering/02_agglomerative_clustering.ipynb`.
3. Run `notebooks/comparison/clustering_model_comparison.ipynb`.

For each notebook, select the correct kernel, run cells from top to bottom, inspect any errors, and save the notebook after execution. Comparison notebooks depend on the individual model results being available at the expected paths.

## 11. Results and Reproducibility

Individual model metrics are stored under `results/classification/` and `results/regression/`. Clustering metrics and customer labels are stored under `results/clustering/`, while clustering figures are stored under `figures/clustering/`.

For reproducibility:

1. Use the documented environment and dependencies.
2. Ensure all required datasets are available.
3. Follow the recommended execution order.
4. Confirm that notebooks execute without errors.
5. Verify that the expected CSV files and figures are generated.
6. Review actual metrics before drawing conclusions or selecting a best model.

The random seed is `42` where applicable. Results may vary with library versions or implementation details, so the environment and preprocessing workflow should remain consistent when comparing experiments.

## 12. Student Observations and Interpretation

The generated metrics and plots should be used to write the final analysis. The student team should review and explain:

- Which regression models perform well on the chosen evaluation metrics.
- Whether hyperparameter tuning improves model performance.
- How prediction errors and residuals behave.
- Whether classification performance differs between positive and negative classes.
- Whether the clustering methods form balanced and interpretable customer segments.
- The limitations of the datasets and evaluation procedures.

Best-model claims and final conclusions should be supported by the actual outputs rather than assumptions about the algorithms.

## 13. AI Assistance Disclosure

Generative AI tools were used for code scaffolding, debugging support, explanations, and documentation assistance. The student team is responsible for verifying the implementation, executing experiments, reviewing results, and preparing the final interpretations and conclusions.

## 14. Academic Context

**Course:** 23CSE301 — Machine Learning Capstone  
**Academic Year:** 2026–2027  
**Reviews Covered:** Review 1 and Review 2
