from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "quality"

def load_wine_dataset(csv_path):
    """Load the supplied Wine Quality White CSV (semicolon separated)."""
    return pd.read_csv(csv_path, sep=";").copy()

def clean_wine_dataset(df):
    """Remove exact duplicates and create the required total_acidity feature."""
    data = df.drop_duplicates().reset_index(drop=True).copy()
    data["total_acidity"] = data["fixed acidity"] + data["volatile acidity"]
    return data

def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def column_lists(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical

def make_preprocessor(X, scale_numeric=True):
    numeric, categorical = column_lists(X)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

def make_polynomial_preprocessor(X, degree=2):
    numeric, categorical = column_lists(X)
    transformers = [("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler())
    ]), numeric)]
    if categorical:
        from sklearn.preprocessing import OneHotEncoder
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers, remainder="drop")

def regression_metrics(y_true, y_pred):
    return {"R2": float(r2_score(y_true, y_pred)), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))), "MAE": float(mean_absolute_error(y_true, y_pred))}
