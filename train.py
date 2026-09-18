import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
classification_report,
confusion_matrix
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from pytorch_tabnet.tab_model import TabNetClassifier



RANDOM_STATE = 42

DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_DIR = "models"



def evaluate_model(model_name, y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)

    print("\n" + "=" * 60)
    print(f"{model_name} RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
"Model": model_name,
"Accuracy": accuracy,
"Precision": precision,
"Recall": recall,
"F1-Score": f1,
"ROC-AUC": roc_auc
}



def train_and_save_model():


    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
    f"'{DATA_PATH}' File not found"
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("LOADING DATASET...")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Shape: {df.shape}")


    X = df.drop(columns=["Class"])
    y = df["Class"]

    print("\nClass Distribution:")
    print(y.value_counts())


    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)


    print("\nSCALING FEATURES...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    results = []

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST...")
    print("=" * 60)

    rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"
    )

    rf_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    results.append(
    evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_proba
    )
    )


    print("\n" + "=" * 60)
    print("TRAINING XGBOOST...")
    print("=" * 60)

    xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
    )

    xgb_model.fit(X_train_scaled, y_train)

    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

    results.append(
    evaluate_model(
    "XGBoost",
    y_test,
    xgb_pred,
    xgb_proba
    )
    )


    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION...")
    print("=" * 60)

    lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
    )

    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    results.append(
    evaluate_model(
    "Logistic Regression",
    y_test,
    lr_pred,
    lr_proba
    )
    )


    print("\n" + "=" * 60)
    print("TRAINING CATBOOST...")
    print("=" * 60)

    cat_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False,
    thread_count=-1
    )

    cat_model.fit(X_train_scaled, y_train)

    cat_pred = cat_model.predict(X_test_scaled).astype(int).ravel()
    cat_proba = cat_model.predict_proba(X_test_scaled)[:, 1]

    results.append(
    evaluate_model(
    "CatBoost",
    y_test,
    cat_pred,
    cat_proba
    )
    )

    print("\n" + "=" * 60)
    print("TRAINING TABNET...")
    print("=" * 60)


    X_tab_train, X_tab_val, y_tab_train, y_tab_val = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=y_train
    )

    # TabNet requires float32 input
    X_tab_train = X_tab_train.astype(np.float32)
    X_tab_val = X_tab_val.astype(np.float32)
    X_test_tabnet = X_test_scaled.astype(np.float32)

    y_tab_train = y_tab_train.values
    y_tab_val = y_tab_val.values


    tabnet_model = TabNetClassifier(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    seed=RANDOM_STATE,
    verbose=10
    )


    tabnet_model.fit(
    X_tab_train,
    y_tab_train,

    eval_set=[
    (
    X_tab_val,
    y_tab_val
    )
    ],

    eval_name=["validation"],
    eval_metric=["auc"],

    max_epochs=50,
    patience=10,

    batch_size=1024,
    virtual_batch_size=128,

    num_workers=0,
    drop_last=False
    )


    tabnet_pred = tabnet_model.predict(
    X_test_tabnet
    ).astype(int).ravel()

    tabnet_proba = tabnet_model.predict_proba(
    X_test_tabnet
    )[:, 1]


    results.append(
    evaluate_model(
    "TabNet",
    y_test,
    tabnet_pred,
    tabnet_proba
    )
    )


    print("\n" + "=" * 60)
    print("TRAINING LIGHTGBM...")
    print("=" * 60)

    lgb_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
    )

    lgb_model.fit(
    X_train_scaled,
    y_train
    )

    lgb_pred = lgb_model.predict(
    X_test_scaled
    )

    lgb_proba = lgb_model.predict_proba(
    X_test_scaled
    )[:, 1]

    results.append(
    evaluate_model(
    "LightGBM",
    y_test,
    lgb_pred,
    lgb_proba
    )
    )


    print("\n" + "=" * 60)
    print("GENERATING FINAL HYBRID PREDICTION...")
    print("=" * 60)

    # Average probability from all six models

    hybrid_proba = (
    rf_proba +
    xgb_proba +
    lr_proba +
    cat_proba +
    tabnet_proba +
    lgb_proba
    ) / 6

    # Threshold = 0.5
    hybrid_pred = (
    hybrid_proba >= 0.5
    ).astype(int)

    results.append(
    evaluate_model(
    "Hybrid (6 Models)",
    y_test,
    hybrid_pred,
    hybrid_proba
    )
    )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
    results_df.to_string(
    index=False
    )
    )


    print("\n" + "=" * 60)
    print("SAVING MODELS...")
    print("=" * 60)

    joblib.dump(
    rf_model,
    os.path.join(
    MODEL_DIR,
    "random_forest.pkl"
    )
    )

    joblib.dump(
    xgb_model,
    os.path.join(
    MODEL_DIR,
    "xgboost.pkl"
    )
    )

    joblib.dump(
    lr_model,
    os.path.join(
    MODEL_DIR,
    "logistic_regression.pkl"
    )
    )

    joblib.dump(
    cat_model,
    os.path.join(
    MODEL_DIR,
    "catboost.pkl"
    )
    )

    # TabNet uses its own save format
    tabnet_model.save_model(
    os.path.join(
    MODEL_DIR,
    "tabnet_model"
    )
    )

    joblib.dump(
    lgb_model,
    os.path.join(
    MODEL_DIR,
    "lightgbm.pkl"
    )
    )

    joblib.dump(
    scaler,
    os.path.join(
    MODEL_DIR,
    "scaler.pkl"
    )
    )

    # Save hybrid information
    hybrid_config = {
    "models": [
    "random_forest",
    "xgboost",
    "logistic_regression",
    "catboost",
    "tabnet",
    "lightgbm"
    ],
    "weights": [
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6,
    1 / 6
    ],
    "threshold": 0.5
    }

    joblib.dump(
    hybrid_config,
    os.path.join(
    MODEL_DIR,
    "hybrid_config.pkl"
    )
    )

    # Save metrics
    results_df.to_csv(
    os.path.join(
    MODEL_DIR,
    "model_results.csv"
    ),
    index=False
    )

    print("\n" + "=" * 60)
    print("ALL MODELS SAVED SUCCESSFULLY!")
    print("=" * 60)

    print("\nSaved files:")

    print("✓ models/random_forest.pkl")
    print("✓ models/xgboost.pkl")
    print("✓ models/logistic_regression.pkl")
    print("✓ models/catboost.pkl")
    print("✓ models/tabnet_model.zip")
    print("✓ models/lightgbm.pkl")
    print("✓ models/scaler.pkl")
    print("✓ models/hybrid_config.pkl")
    print("✓ models/model_results.csv")


    print("✓ models/model_results.csv")

# MAIN

if __name__ == "__main__":
    train_and_save_model()