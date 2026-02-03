
# Explainable Credit Risk Prediction – Demo ML System

## Purpose of This Demo

This repository contains a **demo end-to-end machine learning inference system** built as part of an internship evaluation task.

The objective of this demo is to demonstrate: - end-to-end ML system design - clean separation between API and ML logic - explainable model 
predictions - ability to debug and handle real-world library edge cases

This is **not a research project or notebook-only prototype**, but a deployable inference service with explainability.

---

## Problem Context

In credit lending workflows, organizations must decide whether an applicant represents a **low-risk or high-risk borrower**.

While traditional ML models can predict outcomes, production systems require: - transparency in decisions - explainability for audits and 
compliance - reproducible and consistent inference behavior

This demo system addresses those requirements by combining **predictive modeling** with **per-instance explainability**.

---

## System Overview

The system exposes a REST API that: - predicts credit risk probability - returns a binary decision using a configurable threshold - explains the 
decision using SHAP values

The service is implemented using **FastAPI**, with all preprocessing and inference encapsulated inside a trained scikit-learn pipeline.

---

## Dataset Used

- **German Credit Dataset (Statlog / UCI)** - 1000 records - 20 input features (categorical + numerical) - Binary target:
  - `1` → Good credit risk - `0` → Bad credit risk

The dataset is used here strictly for demonstration purposes.

---

## Architecture

```

Client (Swagger / REST call)
|
v FastAPI Application
|
v Preprocessing Pipeline (ColumnTransformer + OneHotEncoder)
|
v Random Forest Model
|
+--> Prediction Endpoint (/predict)
|
+--> Explainability Endpoint (/explain)

````

---

## Model Pipeline

### Preprocessing
- Categorical features are encoded using OneHotEncoder - Numerical features are passed through unchanged - Preprocessing and model are bundled 
into a single pipeline to avoid train–serve skew

### Model
- Random Forest Classifier - Selected for:
  - strong tabular data performance - compatibility with SHAP TreeExplainer

### Decision Logic
- Outputs probability of good credit - Binary classification is derived using a configurable threshold (default = 0.5)

---

## Explainability Design

Explainability is implemented using **SHAP (SHapley Additive exPlanations)**.

### Key design choices
- SHAP values are computed per request - One-hot encoded SHAP values are **aggregated back to original feature groups** - Only the top 5 most 
influential features are returned - Output is human-readable and business-facing

This avoids exposing raw one-hot encoded features directly to end users.

---

## API Endpoints

### POST `/predict`
Returns credit risk prediction and probability.

**Response** ```json {
  "prediction": 1, "probability": 0.51
}
````

---

### POST `/explain`

Returns prediction along with feature-level explanation.

**Response**

```json {
  "prediction": 1, "probability": 0.51, "top_features": {
    "Credit_Amount": 0.33, "Credit_History": 0.11, "Savings_Account": -0.21, "Duration": 0.18, "Age": -0.12
  }
}
```

Positive values increase risk, negative values reduce risk.

---

### GET `/health`

Simple health check endpoint.

---

## Project Structure

``` app/ ├── api/ │ └── routes.py # API endpoints ├── core/ │ ├── model.py # Model loading and prediction logic │ └── explain.py # SHAP 
explainability logic ├── schemas/ │ └── schema.py # Request/response schemas ├── main.py # FastAPI application entry point ├── model.pkl # 
Trained ML pipeline artifact data/ ├── german_credit_raw.data ├── german_credit.csv ```

---

## How to Run Locally

```bash python3 -m venv venv source venv/bin/activate pip install -r requirements.txt python -m uvicorn app.main:app --reload ```

Swagger UI:

``` http://127.0.0.1:8000/docs ```

---

## What This Demo Demonstrates

* Ability to design an end-to-end ML inference system * Clean separation of API, model, and explainability logic * Handling of explainability 
edge cases in SHAP * Debugging and iteration under ambiguous library behavior * Production-oriented thinking rather than notebook-only 
experimentation

--

## Notes

This demo is intended solely as a **technical evaluation artifact** and not as a finalized production deployment.

````

---

