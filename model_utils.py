from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _save_bundle(bundle, path):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def train_titanic_bundle():
    data_path = DATA_DIR / "sample-data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Expected the included dataset at {data_path}")
    df = pd.read_csv(data_path)
    features = ["PClass", "Gender", "Sibling", "Embarked"]
    df = df[features + ["Survived"]].dropna().copy()
    for c in features + ["Survived"]:
        df[c] = df[c].astype(str).str.strip()
    encoders = {}
    X = df[features].copy()
    for c in features:
        encoders[c] = LabelEncoder()
        X[c] = encoders[c].fit_transform(X[c])
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Survived"])
    model = SVC(gamma="auto", random_state=0)
    model.fit(X, y)
    bundle = {"model": model, "feature_columns": features, "feature_encoders": encoders,
              "target_encoder": target_encoder, "categorical_features": features,
              "dataset_file": "data/sample-data.csv"}
    _save_bundle(bundle, ROOT / "titanic_model.pkl")
    return bundle


def load_heart_dataframe():
    path = DATA_DIR / "heart_disease.csv"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    if path.exists():
        df = pd.read_csv(path, na_values="?")
        if set(columns).issubset(df.columns):
            return df
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    df = pd.read_csv(url, header=None, names=columns, na_values="?")
    df.to_csv(path, index=False)
    return df


def train_heart_bundle():
    df = load_heart_dataframe()
    features = ["age", "sex", "cp", "thalach"]
    for c in features + ["num"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    data = df[features + ["num"]].dropna().copy()
    data["sex"] = data["sex"].astype(int)
    data["cp"] = data["cp"].astype(int)
    data["num"] = data["num"].astype(int)
    data["DiseaseStatus"] = np.where(data["num"] == 0, "No disease", "Disease present")
    categorical = ["sex", "cp"]
    X = data[features].copy()
    encoders = {}
    for c in categorical:
        encoders[c] = LabelEncoder()
        X[c] = encoders[c].fit_transform(X[c].astype(int).astype(str))
    X["age"] = X["age"].astype(float)
    X["thalach"] = X["thalach"].astype(float)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(data["DiseaseStatus"].astype(str))
    model = SVC(gamma="auto", random_state=0)
    model.fit(X, y)
    bundle = {"model": model, "feature_columns": features, "feature_encoders": encoders,
              "target_encoder": target_encoder, "categorical_features": categorical,
              "dataset_file": "data/heart_disease.csv",
              "target_definition": "num == 0: No disease; num > 0: Disease present"}
    _save_bundle(bundle, ROOT / "heart_disease_model.pkl")
    return bundle


def ensure_models():
    titanic_path = ROOT / "titanic_model.pkl"
    heart_path = ROOT / "heart_disease_model.pkl"
    if titanic_path.exists():
        with open(titanic_path, "rb") as f:
            titanic = pickle.load(f)
    else:
        titanic = train_titanic_bundle()
    if heart_path.exists():
        with open(heart_path, "rb") as f:
            heart = pickle.load(f)
    else:
        heart = train_heart_bundle()
    return titanic, heart


def predict_bundle(bundle, values):
    """Encode one app input row and return its human-readable label."""
    features = bundle["feature_columns"]
    if set(values) != set(features):
        raise ValueError(f"Expected exactly these input fields: {features}")

    row = pd.DataFrame([{column: values[column] for column in features}], columns=features)
    for column in bundle.get("categorical_features", []):
        encoder = bundle["feature_encoders"][column]
        if column in ("sex", "cp"):
            row[column] = encoder.transform(row[column].astype(int).astype(str))
        else:
            row[column] = encoder.transform(row[column].astype(str))
    for column in features:
        if column not in bundle.get("categorical_features", []):
            row[column] = pd.to_numeric(row[column], errors="raise")

    predicted_code = bundle["model"].predict(row[features])[0]
    return bundle["target_encoder"].inverse_transform([predicted_code])[0]
