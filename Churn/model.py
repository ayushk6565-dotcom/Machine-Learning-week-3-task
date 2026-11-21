
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset
df = pd.read_csv("Churn\Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)

# 2. Data Cleaning
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)

# Encode Target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Separate Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Identify Numerical and Categorical Columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include="object").columns

# 6. Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 7. Random Forest Model
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

rf_params = {
    "model__n_estimators": [200],
    "model__max_depth": [None, 12],
    "model__min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_

# 8. XGBoost Model
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

weight = negative / positive

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        scale_pos_weight=weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ))
])

xgb_params = {
    "model__n_estimators": [200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1]
}

xgb_grid = GridSearchCV(
    xgb_pipeline,
    xgb_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

xgb_best = xgb_grid.best_estimator_

# 9. Predictions
rf_pred = rf_best.predict(X_test)
xgb_pred = xgb_best.predict(X_test)

rf_prob = rf_best.predict_proba(X_test)[:, 1]
xgb_prob = xgb_best.predict_proba(X_test)[:, 1]

# 10. Model Evaluation
models = {
    "Random Forest": (rf_pred, rf_prob),
    "XGBoost": (xgb_pred, xgb_prob)
}

results = []

for name, (pred, prob) in models.items():

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)

    results.append([
        name, accuracy, precision,
        recall, f1, auc
    ])

    print("\n", name)
    print(classification_report(
        y_test, pred,
        target_names=["No Churn", "Churn"]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

# 11. Comparison Table
results_df = pd.DataFrame(
    results,
    columns=[
        "Model", "Accuracy", "Precision",
        "Recall", "F1 Score", "ROC-AUC"
    ]
)

print("\nModel Comparison:")
print(results_df)

print("\nRandom Forest Best Parameters:")
print(rf_grid.best_params_)

print("\nXGBoost Best Parameters:")
print(xgb_grid.best_params_)

print("\nRandom Forest CV ROC-AUC:", rf_grid.best_score_)
print("XGBoost CV ROC-AUC:", xgb_grid.best_score_)

# 12. ROC-AUC Curve
plt.figure(figsize=(8, 6))

for name, (pred, prob) in models.items():

    fpr, tpr, thresholds = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.show()

# 13. Feature Importance
feature_names = rf_best.named_steps[
    "preprocessor"
].get_feature_names_out()

rf_importance = rf_best.named_steps["model"].feature_importances_
xgb_importance = xgb_best.named_steps["model"].feature_importances_

for name, importance in [
    ("Random Forest", rf_importance),
    ("XGBoost", xgb_importance)
]:

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    print(f"\nTop 10 Features - {name}")
    print(importance_df)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(f"Top 10 Feature Importance - {name}")
    plt.show()
