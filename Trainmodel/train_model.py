# backend/Trainmodel/train_model.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

print("=== Training Started ===")

# ---------------- PATH SETUP (NO ERROR GUARANTEE) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "pima_diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------------- PREPROCESS ----------------
imputer = KNNImputer(n_neighbors=5)
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
rf = RandomForestClassifier(n_estimators=150, random_state=42)
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=150,
    random_state=42
)
et = ExtraTreesClassifier(n_estimators=150, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
et.fit(X_train, y_train)

# ---------------- SAVE EVERYTHING ----------------
joblib.dump(
    {
        "rf": rf,
        "xgb": xgb,
        "et": et
    },
    os.path.join(MODEL_DIR, "tri_ensemble.pkl")
)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))

print("✅ Model trained & saved successfully")
print("📁 Saved at:", MODEL_DIR)
