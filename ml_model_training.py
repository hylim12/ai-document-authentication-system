"""Train a leakage-aware RandomForest model with strict dataset split usage."""

import json
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_DATASET_PATH = "dataset_outputs/train.csv"
VAL_DATASET_PATH = "dataset_outputs/val.csv"
TEST_DATASET_PATH = "dataset_outputs/test.csv"
MODEL_DIR = Path("trained_models")
MODEL_PATH = MODEL_DIR / "forged_document_rf_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "feature_preprocessor.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
TARGET_ACCURACY = 0.80
RANDOM_SEED = 42

LEAKAGE_PREFIXES = ("Image_Name_",)
LEAKAGE_TOKENS = ("fake", "real", "forged", "authentic")


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that leak identity or pre-encoded country dimensions."""
    leakage_columns = []
    for col in df.columns:
        if col in {"Image_Name", "Document_ID", "Detection_Label"}:
            leakage_columns.append(col)
            continue
        if col.startswith(("Country_Code_", "Country_Name_")):
            leakage_columns.append(col)
            continue
        if col.startswith(LEAKAGE_PREFIXES):
            leakage_columns.append(col)

    if leakage_columns:
        print(f"[WARN] Dropping leakage/pre-encoded columns: {sorted(leakage_columns)}")
        df = df.drop(columns=sorted(set(leakage_columns)))
    return df


def _assert_no_label_tokens(columns):
    suspicious = [
        col for col in columns
        if any(tok in col.lower() for tok in LEAKAGE_TOKENS)
    ]
    if suspicious:
        raise ValueError(f"Potential leakage columns found (name-based): {suspicious}")


def _drop_overpowered_features(X_train: pd.DataFrame, y_train: pd.Series, threshold: float = 0.995):
    """Remove near-perfect single-feature separators discovered in training only."""
    dropped = []
    for col in list(X_train.columns):
        if col in {"Country_Code", "Country_Name"}:
            continue

        numeric_series = pd.to_numeric(X_train[col], errors="coerce")
        if numeric_series.notna().sum() < 10:
            continue

        filled = numeric_series.fillna(numeric_series.median())
        try:
            auc = roc_auc_score(y_train, filled)
        except Exception:
            continue

        auc = max(auc, 1.0 - auc)
        if auc >= threshold:
            dropped.append(col)

    if dropped:
        print(f"[WARN] Dropping overpowered features (AUC >= {threshold}): {sorted(dropped)}")
        X_train = X_train.drop(columns=dropped)
    return X_train, dropped


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Label", "Country_Code"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    optional_numeric_defaults = {
        "Font_Size_Variance": 0.0,
        "OCR_Confidence_Mean": 0.0,
        "Field_Blur_Variance": 0.0,
        "Risk_Score": 0.0,
        "Field_Count": 0,
        "Has_POB": 0,
        "Field_Completeness": 0.0,
        "OCR_Quality": 0.0,
        "Num_Anomalies": 0,
        "Num_Background_Anomalies": 0,
        "Num_OCR_Box_Anomalies": 0,
    }
    for col, default in optional_numeric_defaults.items():
        if col not in df.columns:
            df[col] = default

    if "Country_Code" in df.columns:
        df.loc[df["Country_Code"] == "SVK", "Has_POB"] = 0

    df = _drop_leakage_columns(df)
    _assert_no_label_tokens(df.columns)
    return df


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["Label", "Image_Name"], errors="ignore").copy()
    y = df["Label"].astype(int)
    _assert_no_label_tokens(X.columns)
    return X, y


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = _build_feature_matrix(train_df)
    X_test, y_test = _build_feature_matrix(test_df)

    X_train, dropped_overpowered = _drop_overpowered_features(X_train, y_train)
    X_test = X_test.drop(columns=dropped_overpowered, errors="ignore")

    missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
    for col in missing_in_test:
        X_test[col] = 0
    X_test = X_test[X_train.columns]

    categorical_features = [c for c in ["Country_Code", "Country_Name"] if c in X_train.columns]
    numeric_features = [c for c in X_train.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    specificity = tn / (tn + fp + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    roc_auc = roc_auc_score(y_test, pred_probs)

    print("Unique predictions:", np.unique(preds))

    return {
        "model": model,
        "accuracy": acc,
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "advanced_metrics": {
            "precision": float(precision),
            "recall_detection": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "false_positive_rate": float(fpr),
            "roc_auc": float(roc_auc),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "feature_columns": list(X_train.columns),
        "dropped_overpowered_features": dropped_overpowered,
    }


def main():
    train_df = load_dataset(TRAIN_DATASET_PATH)
    _ = load_dataset(VAL_DATASET_PATH)
    test_df = load_dataset(TEST_DATASET_PATH)
    result = train_and_evaluate(train_df, test_df)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(result["model"], f)

    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(result["model"].named_steps["preprocessor"], f)

    metrics = {
        "target_accuracy": TARGET_ACCURACY,
        "achieved_accuracy": float(result["accuracy"]),
        "meets_target": bool(result["accuracy"] >= TARGET_ACCURACY),
        "seed": RANDOM_SEED,
        "train_dataset": TRAIN_DATASET_PATH,
        "val_dataset": VAL_DATASET_PATH,
        "test_dataset": TEST_DATASET_PATH,
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "feature_columns": result["feature_columns"],
        "dropped_overpowered_features": result["dropped_overpowered_features"],
        "classification_report": result["report"],
        "advanced_metrics": result["advanced_metrics"],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Test accuracy: {result['accuracy']*100:.2f}%")
    print("\n===== ADVANCED AML METRICS =====")
    print(f"Precision            : {result['advanced_metrics']['precision']:.4f}")
    print(f"Recall (Detection)   : {result['advanced_metrics']['recall_detection']:.4f}")
    print(f"F1-score             : {result['advanced_metrics']['f1_score']:.4f}")
    print(f"Specificity          : {result['advanced_metrics']['specificity']:.4f}")
    print(f"False Positive Rate  : {result['advanced_metrics']['false_positive_rate']:.4f}")
    print(f"ROC-AUC              : {result['advanced_metrics']['roc_auc']:.4f}")
    print(f"[INFO] Model saved to {MODEL_PATH}")
    print(f"[INFO] Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
