import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("data/german_credit.csv")

# =========================
# 2. Target preprocessing
# UCI: 1 = good credit, 2 = bad credit
# Convert to binary: 1 = good, 0 = bad
# =========================
df["CreditRisk"] = df["CreditRisk"].map({1: 1, 2: 0})

X = df.drop("CreditRisk", axis=1)
y = df["CreditRisk"]

# =========================
# 3. Column separation
# =========================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# =========================
# 4. Preprocessing pipeline
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# =========================
# 5. Models
# =========================
log_reg = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

rf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 7. Train Logistic Regression
# =========================
print("\nTraining Logistic Regression...")
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, log_preds))

# =========================
# 8. Train Random Forest
# =========================
print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))

# =========================
# 9. Confusion Matrix
# =========================
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# =========================
# 10. Save final model
# =========================
joblib.dump(rf, "app/model.pkl")
print("\nFinal model saved to app/model.pkl")
