"""
ml_core.py - helpers shared by train_models.py and app.py.

Keeping the metric code in one place guarantees that the numbers printed by the training script
and the numbers shown in the Streamlit app are computed in exactly the same way.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

# display name -> file in model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------- labels
def label_to_str(value) -> str:
    """Canonical text form of a class label, so 1, 1.0 and '1' all mean the same class."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)) and not np.isnan(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def order_classes(labels, positive=None):
    """Sort class labels (numbers numerically, text alphabetically).
    For a binary problem the LAST class is the 'positive' class used for precision/recall/F1;
    pass `positive` to choose it explicitly."""
    def key(s):
        try:
            return (0, float(s), s)
        except ValueError:
            return (1, 0.0, s)

    classes = sorted({label_to_str(v) for v in labels}, key=key)
    if positive is not None:
        positive = label_to_str(positive)
        if positive not in classes:
            raise ValueError(f"positive class '{positive}' is not one of {classes}")
        if len(classes) == 2:
            classes = [c for c in classes if c != positive] + [positive]
    return classes


# --------------------------------------------------------------------------- saved artefacts
def load_metadata(model_dir=MODEL_DIR) -> dict:
    with open(Path(model_dir) / "metadata.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_models(model_dir=MODEL_DIR) -> dict:
    return {name: joblib.load(Path(model_dir) / fname) for name, fname in MODEL_FILES.items()}


# --------------------------------------------------------------------------- uploaded data
def check_test_frame(df: pd.DataFrame, meta: dict) -> list:
    """Return a list of human-readable problems (empty list = the file can be scored)."""
    problems = []
    if df.empty:
        problems.append("The file has no rows.")
    missing = [c for c in meta["features"] if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
        problems.append(f"{len(missing)} feature column(s) are missing: {shown}.")
    if meta["target"] not in df.columns:
        problems.append(f"The target column '{meta['target']}' is missing.")
    else:
        seen = set(df[meta["target"]].map(label_to_str))
        unknown = sorted(seen - set(meta["classes"]))
        if unknown:
            problems.append("The target column has labels the models were not trained on: "
                            + ", ".join(unknown[:8]) + f". Expected: {', '.join(meta['classes'])}.")
    return problems


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Give text / boolean columns a plain `object` dtype with NaN for missing values.
    (pandas 3 uses a dedicated `str` dtype that scikit-learn's imputers reject; doing this in one place keeps
    training and the Streamlit app identical.) Numeric columns are left untouched."""
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            continue
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    return X


def split_xy(df: pd.DataFrame, meta: dict):
    """Feature frame (in training column order) and integer class indices (position in meta['classes'])."""
    X = clean_features(df[meta["features"]])
    index = {c: i for i, c in enumerate(meta["classes"])}
    y = df[meta["target"]].map(label_to_str).map(index).to_numpy(dtype=int)
    return X, y


# --------------------------------------------------------------------------- metrics
def _auc(y_true, proba, n_classes, average):
    """ROC AUC: binary -> probability of the positive class; multi-class -> one-vs-rest, averaged.
    Classes that do not appear in y_true carry zero weight; if AUC is undefined (fewer than two classes) -> NaN."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            if n_classes == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average=average,
                                       labels=list(range(n_classes))))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, proba, n_classes, average) -> dict:
    """Accuracy, AUC, Precision, Recall, F1 and MCC.
    Binary: precision/recall/F1 are for the positive class (index 1).
    Multi-class: they are averaged over classes with `average` ('weighted' = by class support)."""
    kw = dict(zero_division=0)
    if n_classes == 2:
        kw.update(average="binary", pos_label=1)
    else:
        kw.update(average=average)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)      # sklearn warns when the data holds a single class
        mcc = float(matthews_corrcoef(y_true, y_pred))
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": _auc(y_true, proba, n_classes, average),
        "Precision": float(precision_score(y_true, y_pred, **kw)),
        "Recall": float(recall_score(y_true, y_pred, **kw)),
        "F1": float(f1_score(y_true, y_pred, **kw)),
        "MCC": mcc,
    }


def _report_frame(y_true, y_pred, meta) -> pd.DataFrame:
    """Per-class precision/recall/F1/support plus accuracy and the macro / weighted averages."""
    labels = list(range(len(meta["classes"])))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=meta["classes"],
                                output_dict=True, zero_division=0)
    per_class = pd.DataFrame({k: rep[k] for k in meta["classes"]}).T
    averages = pd.DataFrame({k: rep[k] for k in ("macro avg", "weighted avg")}).T
    accuracy = pd.DataFrame({"precision": [np.nan], "recall": [np.nan],
                             "f1-score": [accuracy_score(y_true, y_pred)],
                             "support": [float(len(y_true))]}, index=["accuracy"])
    frame = pd.concat([per_class, accuracy, averages])[["precision", "recall", "f1-score", "support"]]
    frame["support"] = frame["support"].astype(int)
    return frame


def evaluate_model(model, X, y, meta) -> dict:
    """Score one saved model on (X, y). No fitting happens here."""
    n = len(meta["classes"])
    y_pred = np.asarray(model.predict(X)).astype(int)
    proba = np.asarray(model.predict_proba(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {
            "metrics": compute_metrics(y, y_pred, proba, n, meta["average"]),
            "y_pred": y_pred,
            "proba": proba,
            "confusion": confusion_matrix(y, y_pred, labels=list(range(n))),
            "report": _report_frame(y, y_pred, meta),
        }


def evaluate_all(models: dict, X, y, meta) -> dict:
    return {name: evaluate_model(model, X, y, meta) for name, model in models.items()}


def leaderboard(results: dict) -> pd.DataFrame:
    """One row per model, one column per metric."""
    return pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T[METRICS]


def mean_ranks(table: pd.DataFrame) -> pd.Series:
    """Average rank of every model over the six metrics (1 = best); lowest value wins overall."""
    ranks = table.rank(ascending=False, method="min", na_option="bottom")
    return ranks.mean(axis=1).sort_values()
