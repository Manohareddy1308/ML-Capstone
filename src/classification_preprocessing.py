from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "bank-full.csv"

def load_bank_dataset(path=DATA_PATH):
    """Load the supplied UCI Bank Marketing bank-full.csv (semicolon separated)."""
    df = pd.read_csv(path, sep=";").copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"unknown": np.nan, "?": np.nan, "nan": np.nan})
    return df

def prepare_features_target(df):
    data=df.copy()
    y=data["y"].astype(str).str.strip().map({"no":0,"yes":1})
    valid=y.notna()
    X=data.loc[valid].drop(columns=["y"])
    y=y.loc[valid].astype(int)
    return X,y

def build_preprocessor(X, scale_numeric=True, dense=False):
    numeric=X.select_dtypes(include=np.number).columns.tolist()
    categorical=X.select_dtypes(exclude=np.number).columns.tolist()
    num=[("imputer",SimpleImputer(strategy="median"))]
    if scale_numeric: num.append(("scaler",StandardScaler()))
    cat=[("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))]
    return ColumnTransformer([("num",Pipeline(num),numeric),("cat",Pipeline(cat),categorical)],remainder="drop")

def save_result(result, filename):
    results_dir=BASE_DIR/"results"
    results_dir.mkdir(parents=True,exist_ok=True)
    output=results_dir/filename
    pd.DataFrame([result]).to_csv(output,index=False)
    return output
