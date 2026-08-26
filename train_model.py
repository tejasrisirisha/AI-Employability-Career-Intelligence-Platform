"""
train_model.py
---------------
Trains and evaluates the AI Employability / Placement Prediction model,
then serializes everything the Streamlit app needs:
  - models/placement_classifier.pkl   (predicts Placed 0/1 + probability)
  - models/package_regressor.pkl      (predicts expected package in LPA if placed)
  - models/scaler.pkl                 (StandardScaler for numeric features)
  - models/label_encoders.pkl         (encoders for categorical features)
  - models/feature_columns.pkl        (ordered list of model input columns)
  - models/metrics.json               (evaluation metrics for the app's "model card")
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "data", "student_employability_dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

CATEGORICAL = ["Gender", "Branch", "Placement_Training"]
NUMERIC = [
    "CGPA", "SSC_Marks", "HSC_Marks", "Internships", "Projects", "Certifications",
    "Backlogs", "Aptitude_Score", "Technical_Skill", "Coding_Skill",
    "Communication_Skill", "Soft_Skill", "Extracurricular_Score", "Leadership_Score",
    "Workshops_Attended", "Mock_Interview_Score", "LinkedIn_GitHub_Activity",
]
TARGET = "Placed"
REGRESSION_TARGET = "Package_LPA"

encoders = {}
df_enc = df.copy()
for col in CATEGORICAL:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

FEATURE_COLUMNS = NUMERIC + CATEGORICAL

X = df_enc[FEATURE_COLUMNS]
y = df_enc[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Model comparison ----------------
candidates = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
    ),
    "LogisticRegression": LogisticRegression(max_iter=2000),
}

results = {}
best_name, best_model, best_auc = None, None, -1
for name, model in candidates.items():
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()
    results[name] = {"auc": auc, "accuracy": acc, "f1": f1, "cv_auc": cv}
    print(f"{name:20s} | Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  CV-AUC={cv:.3f}")
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = model

print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")

final_preds = best_model.predict(X_test_scaled)
final_proba = best_model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "best_model": best_name,
    "accuracy": round(accuracy_score(y_test, final_preds), 4),
    "precision": round(precision_score(y_test, final_preds), 4),
    "recall": round(recall_score(y_test, final_preds), 4),
    "f1_score": round(f1_score(y_test, final_preds), 4),
    "roc_auc": round(roc_auc_score(y_test, final_proba), 4),
    "confusion_matrix": confusion_matrix(y_test, final_preds).tolist(),
    "all_models_compared": {k: {m: round(v2, 4) for m, v2 in v.items()} for k, v in results.items()},
    "n_train": len(X_train),
    "n_test": len(X_test),
    "feature_columns": FEATURE_COLUMNS,
}
print("\nClassification report:\n", classification_report(y_test, final_preds))

# Feature importance (for explainability fallback / global view)
if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(FEATURE_COLUMNS, best_model.feature_importances_.tolist()))
else:
    importances = dict(zip(FEATURE_COLUMNS, np.abs(best_model.coef_[0]).tolist()))
metrics["feature_importances"] = {k: round(v, 4) for k, v in
                                   sorted(importances.items(), key=lambda x: -x[1])}

# ---------------- Package (LPA) regressor, trained only on placed students ----------------
placed_df = df_enc[df_enc[TARGET] == 1]
Xr = placed_df[FEATURE_COLUMNS]
yr = placed_df[REGRESSION_TARGET]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
Xr_train_scaled = scaler.transform(Xr_train)
Xr_test_scaled = scaler.transform(Xr_test)

reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
reg.fit(Xr_train_scaled, yr_train)
r2 = reg.score(Xr_test_scaled, yr_test)
print(f"\nPackage regressor R^2 on held-out placed students: {r2:.3f}")
metrics["package_regressor_r2"] = round(r2, 4)

# ---------------- Save everything ----------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "placement_classifier.pkl"))
joblib.dump(reg, os.path.join(MODEL_DIR, "package_regressor.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll model artifacts saved to:", MODEL_DIR)
