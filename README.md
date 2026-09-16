# 23CSE301 Machine Learning Capstone — Review 1

## Project Objective

This project applies supervised machine-learning algorithms to two supplied datasets as part of Review 1 of the 23CSE301 Machine Learning course.

The project covers:

- **Regression:** Wine Quality White dataset, predicting `quality`.
- **Classification Part-A:** UCI Bank Marketing dataset, predicting `y` (`yes`/`no`).

The implementation follows a common preprocessing pipeline, deterministic train/test splitting, model evaluation, hyperparameter tuning, and comparison of the required machine-learning algorithms.

---

## Datasets

### 1. Wine Quality White — Regression

File:

```text
data/winequality-white.csv
```

Target variable:

```text
quality
```

The regression task predicts the wine quality score from the available physicochemical features.

### 2. Bank Marketing — Classification

File:

```text
data/bank-full.csv
```

Target variable:

```text
y
```

The classification task predicts whether the client subscribed to a term deposit (`yes`/`no`).

---

# Regression

## Required Algorithms

The project implements the following 10 regression algorithms:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regression (SVR)
10. KNN Regressor

All ten models use the same cleaned and engineered wine dataset and the same deterministic 80:20 train/test split with:

```text
random_state = 42
```

## Regression Evaluation Metrics

The regression models are evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

The models are compared using their performance on the same held-out test set.

Five-fold cross-validation is performed for the two best-performing regression models based on held-out test-set R².

Hyperparameter tuning is performed for at least two regression models using `GridSearchCV`.

---

## Regression Feature Engineering

The following engineered feature is created:

```text
total_acidity = fixed acidity + volatile acidity
```

The engineered feature is created before model training and is used consistently across the regression models.

Feature-engineering decisions are based on the characteristics of the wine-quality dataset.

---

## Regression Data Cleaning and Outlier Handling

The regression dataset is checked for duplicate records and potential outliers.

Exact duplicate wine rows are removed before the train/test split.

IQR-based screening is used to identify potential outliers in numerical features. Potential extreme chemical measurements are retained when they may represent plausible observations rather than being removed solely because they fall outside the IQR range.

No test-set-derived outlier treatment is fitted or applied to the training process.

---

## Regression Diagnostics

The regression comparison includes:

- Model performance comparison table
- Five-fold cross-validation for the top two models
- Predicted-vs-actual plot for the selected best-performing model
- Residual plot/diagnostic
- Tree-based feature-importance analysis

---

# Classification — Part A

## Required Algorithms

Classification Part-A implements the following five algorithms:

1. Logistic Regression
2. KNN Classifier
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)

The classification task uses an 80:20 stratified train/test split with:

```text
random_state = 42
```

Stratification is used to preserve the class distribution between the training and test sets.

---

## Classification Evaluation Metrics

The classification models are evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1 Score
- ROC-AUC
- Confusion Matrix

The models are compared using the same held-out test set.

---

## Classification Data Exploration

The classification dataset is explored using:

- Target-class distribution
- Numerical feature histograms
- Categorical feature count plots
- Numerical correlation heatmap
- Duplicate checking
- IQR-based screening of numerical variables

Two feature-target relationships are also examined through appropriate visualizations as part of the required EDA.

### Duplicate Handling

The bank dataset is checked for duplicate records.

Duplicate-looking records are not automatically removed because the dataset does not contain a unique customer identifier, and identical feature rows do not by themselves prove that the records are erroneous duplicates.

---

# Preprocessing and Data Leakage Prevention

Preprocessing is implemented using scikit-learn pipelines wherever appropriate.

The preprocessing workflow ensures that information from the test set is not used when fitting preprocessing steps.

Depending on the model, preprocessing includes:

- Missing-value handling where required
- Feature scaling
- One-hot encoding of categorical variables
- Polynomial feature expansion where required

Transformations are fitted using the training data and then applied to the test data.

For classification, the train/test split uses:

```text
stratify = y
```

For regression and classification, deterministic splitting uses:

```text
random_state = 42
```

This ensures reproducible experiments while maintaining separation between training and test data.

---

# Exploratory Data Analysis

EDA is performed separately for the regression and classification datasets.

The analysis includes:

- Feature-level exploration
- Target distribution
- Numerical feature distributions
- Categorical feature distributions
- Correlation analysis
- Duplicate checking
- Outlier screening
- Feature-target relationships

The observations from the EDA are based on the actual plots and outputs generated from the datasets.

---

# Student Observations

The notebooks contain sections for recording observations from the actual EDA plots, model results, and diagnostic visualizations.

The final observations and interpretations are written by the student team after inspecting the generated outputs.

Examples of observations may include:

- Patterns observed in feature distributions
- Relationships between important features and the target
- Effects of preprocessing
- Differences in model performance
- Error patterns visible in residual plots
- Important features identified by tree-based models
- Classification performance and confusion-matrix patterns

The final conclusions are based on the actual experimental results rather than predefined expectations.

---

# Project Structure

```text
Review-1/
│
├── data/
│   ├── winequality-white.csv
│   └── bank-full.csv
│
├── notebooks/
│   ├── regression/
│   │   ├── 00_regression_model_comparison.ipynb
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
│   │
│   └── classification/
│       ├── 00_classification_model_comparison.ipynb
│       ├── 01_logistic_regression_classification.ipynb
│       ├── 02_knn_classifier.ipynb
│       ├── 03_gaussian_naive_bayes.ipynb
│       ├── 04_decision_tree_classifier.ipynb
│       └── 05_svc.ipynb
│
├── results/
│   ├── regression/
│   │   ├── 01_linear_regression.csv
│   │   ├── 02_ridge_regression.csv
│   │   ├── 03_lasso_regression.csv
│   │   ├── 04_elasticnet_regression.csv
│   │   ├── 05_polynomial_regression.csv
│   │   ├── 06_decision_tree_regressor.csv
│   │   ├── 07_random_forest_regressor.csv
│   │   ├── 08_gradient_boosting_regressor.csv
│   │   ├── 09_svr.csv
│   │   ├── 10_knn_regressor.csv
│   │   ├── regression_model_comparison.csv
│   │   └── top_two_regression_5fold_cv.csv
│   │
│   └── classification/
│       ├── 01_logistic_regression_classification.csv
│       ├── 02_knn_classifier_classification.csv
│       ├── 03_gaussian_naive_bayes_classification.csv
│       ├── 04_decision_tree_classifier_classification.csv
│       └── 05_svc_classification.csv
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   └── classification_preprocessing.py
│
├── README.md
├── requirements.txt
└── AUDIT_REPORT.txt
```

> The project structure is populated progressively as individual models are implemented and committed to the repository.

---

# Installation

Create and activate the Python virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install the Jupyter kernel:

```bash
python -m ipykernel install --user --name ml-capstone --display-name "ML Capstone (.venv)"
```

Open the project in VS Code or Jupyter and select:

```text
ML Capstone (.venv)
```

---

# How to Run

## Regression

Run the individual regression notebooks as required:

```text
01_linear_regression.ipynb
02_ridge_regression.ipynb
03_lasso_regression.ipynb
04_elasticnet_regression.ipynb
05_polynomial_regression.ipynb
06_decision_tree_regressor.ipynb
07_random_forest_regressor.ipynb
08_gradient_boosting_regressor.ipynb
09_svr.ipynb
10_knn_regressor.ipynb
```

After generating the individual results, run:

```text
00_regression_model_comparison.ipynb
```

This notebook consolidates the regression results and performs the required comparisons and diagnostics.

---

## Classification

Run the five Classification Part-A notebooks:

```text
01_logistic_regression_classification.ipynb
02_knn_classifier.ipynb
03_gaussian_naive_bayes.ipynb
04_decision_tree_classifier.ipynb
05_svc.ipynb
```

After generating the individual results, run:

```text
00_classification_model_comparison.ipynb
```

This notebook consolidates the classification results and provides the required model comparison.

---

# Reproducibility

The experiments use deterministic random seeds where required:

```text
random_state = 42
```

Regression uses the same 80:20 train/test split for all ten algorithms.

Classification uses an 80:20 stratified train/test split.

Preprocessing is fitted only on the training data to prevent test-set information from influencing model training.

---

# Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git
- GitHub

---

# AI Assistance Disclosure

Generative AI tools were used for code scaffolding, debugging support, and documentation assistance.

Dataset analysis, feature-engineering decisions, interpretation of results, student observations, and final conclusions are reviewed and completed by the student team based on the actual experimental outputs.

AI assistance is used as a support tool and does not replace the team's responsibility for understanding, validating, and explaining the implementation and results.

---

# Academic Integrity

The project follows the Machine Learning Review 1 requirements provided for the course.

All model results, observations, interpretations, and conclusions presented in the final submission are based on the team's actual execution of the notebooks and analysis of the supplied datasets.
