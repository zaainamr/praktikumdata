import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
DATA_PATH = BASE_DIR / "Diabetes_dataset_2026.csv"


def fix_health_data(row):
    for col in [
        "HbA1c",
        "Glucose",
        "BloodPressure",
        "LDL",
        "HDL",
        "Triglycerides",
        "WaistCircumference",
        "HipCircumference",
    ]:
        if col in row and row[col] > 300:
            row[col] = round(row[col] / 10, 1)

    if "BMI" in row and row["BMI"] > 100:
        row["BMI"] = round(row["BMI"] / 100, 2)

    if "WHR" in row and row["WHR"] > 2:
        row["WHR"] = round(row["WHR"] / 100, 2)

    return row


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.apply(fix_health_data, axis=1)

    cols = ["BMI", "Glucose", "BloodPressure", "HbA1c"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def export_models():
    df = load_and_prepare_data()
    df_numeric = df.select_dtypes(include=["number"]).dropna()

    # Regresi Glukosa
    X = df_numeric.drop(columns=["Glucose"])
    y = df_numeric["Glucose"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    scaler_lr = StandardScaler()
    scaler_lr.fit(X_train)

    with open(MODEL_DIR / "linear_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    with open(MODEL_DIR / "scaler_lr.pkl", "wb") as f:
        pickle.dump(scaler_lr, f)

    # Klasifikasi Diabetes
    X_clf = df_numeric.drop(columns=["Outcome"])
    y_clf = df_numeric["Outcome"]
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )

    scaler_nb = StandardScaler()
    X_train_scaled = scaler_nb.fit_transform(X_train_clf)
    X_test_scaled = scaler_nb.transform(X_test_clf)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_scaled, y_train_clf.round().astype(int)
    )

    nb_model = GaussianNB()
    nb_model.fit(X_train_smote, y_train_smote)

    with open(MODEL_DIR / "naive_bayes_model.pkl", "wb") as f:
        pickle.dump(nb_model, f)

    with open(MODEL_DIR / "scaler_nb.pkl", "wb") as f:
        pickle.dump(scaler_nb, f)

    print("✅ Model berhasil disimpan ke folder 'models' dengan nama:")
    print("   - linear_regression_model.pkl")
    print("   - scaler_lr.pkl")
    print("   - naive_bayes_model.pkl")
    print("   - scaler_nb.pkl")


if __name__ == "__main__":
    export_models()
