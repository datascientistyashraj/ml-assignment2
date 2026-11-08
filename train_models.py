#!/usr/bin/env python3
"""
train_models.py - train, tune and evaluate five classifiers, then save everything the Streamlit app needs.

    python train_models.py                                   # demo data: UCI handwritten digits (bundled with scikit-learn)
    python train_models.py --data my_data.csv --target label # your own CSV (Kaggle / UCI download)
    python train_models.py --data my_data.csv --target Churn --positive-class Yes --drop CustomerID

What it does
  1. loads the data, checks the assignment minimums (>= 12 features, >= 500 rows)
  2. makes a stratified 80/20 train/test split and writes the test rows to test_data.csv
  3. builds one preprocessing + classifier pipeline per model (imputing, scaling, one-hot encoding)
  4. tunes small hyper-parameter grids with 5-fold cross-validation on the TRAINING split only
  5. scores each tuned model once on the untouched test split (Accuracy, AUC, Precision, Recall, F1, MCC)
  6. saves model/*.joblib, model/metadata.json, results/metrics_test.csv and results/README_tables.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import ml_core as core

MIN_FEATURES, MIN_ROWS = 12, 500          # minimums required by the assignment brief


# --------------------------------------------------------------------------- data loading
def load_dataset(args):
    """Return (DataFrame, target column, dataset name, source text)."""
    key = args.data or "builtin:digits"
    if key.startswith("builtin:"):
        from sklearn import datasets
        which = key.split(":", 1)[1]
        if which == "digits":
            frame = datasets.load_digits(as_frame=True).frame.copy()
            return (frame, "target", args.name or "Handwritten digits",
                    args.source or "UCI Machine Learning Repository, Optical Recognition of Handwritten Digits "
                                   "(test portion, as bundled with scikit-learn)")
        if which == "breast_cancer":
            bunch = datasets.load_breast_cancer(as_frame=True)
            frame = bunch.frame.copy()
            frame["diagnosis"] = frame["target"].map(dict(enumerate(bunch.target_names)))
            return (frame.drop(columns="target"), "diagnosis", args.name or "Breast cancer diagnosis",
                    args.source or "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)")
        sys.exit(f"Unknown built-in dataset '{which}'. Use builtin:digits or builtin:breast_cancer.")
    path = Path(key)
    if not path.exists():
        sys.exit(f"Data file not found: {path}")
    if not args.target:
        sys.exit("Please say which column is the label: --target <column name>")
    frame = pd.read_csv(path)
    if args.target not in frame.columns:
        sys.exit(f"Column '{args.target}' not found. Columns are: {', '.join(map(str, frame.columns))}")
    return frame, args.target, args.name or path.stem, args.source or str(path)


# --------------------------------------------------------------------------- pipelines
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric columns: median-impute, drop constant columns, standardise.
    Text/categorical columns: most-frequent-impute, one-hot encode (unseen categories are ignored)."""
    numeric = [c for c in X.columns
               if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("drop_constant", VarianceThreshold()),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                      categorical))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def model_specs(seed: int) -> dict:
    """name -> (estimator, hyper-parameter grid). Grids are deliberately small."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=5000),
                                {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=seed),
                          {"clf__max_depth": [3, 5, 8, 12, None], "clf__min_samples_leaf": [1, 3, 5]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [1, 3, 5, 7, 9, 11], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (GaussianNB(),
                        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 0.3, 1.0]}),
        # probability=False while searching (faster); it is switched on for the final fit so AUC can be computed
        "Support Vector Machine": (SVC(kernel="rbf", probability=False, random_state=seed),
                                   {"clf__C": [0.5, 1, 5, 10, 50], "clf__gamma": ["scale", 0.001, 0.003, 0.01, 0.03]}),
    }


def to_py(obj):
    """Make numpy scalars JSON-friendly."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def md_table(df: pd.DataFrame, index_name="ML Model Name", digits=4) -> str:
    cols = [index_name] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
    for name, row in df.iterrows():
        lines.append("| " + " | ".join([str(name)] + ["n/a" if pd.isna(v) else f"{v:.{digits}f}" for v in row]) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="CSV file, or builtin:digits / builtin:breast_cancer (default: builtin:digits)")
    p.add_argument("--target", help="label column (required for CSV files)")
    p.add_argument("--name", help="dataset name shown in the app and README")
    p.add_argument("--source", help="where the data came from (shown in the app)")
    p.add_argument("--drop", default="", help="comma-separated columns to ignore (IDs, names, ...)")
    p.add_argument("--positive-class", help="binary problems: the class treated as 'positive'")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tune", action="store_true", help="skip the grid search and use default hyper-parameters")
    p.add_argument("--jobs", type=int, default=-1, help="parallel jobs for the grid search")
    p.add_argument("--out", default=str(core.ROOT), help="where to write model/, results/ and test_data.csv")
    args = p.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    out = Path(args.out).resolve()
    model_dir, results_dir = out / "model", out / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. data -----------------------------------------------------------------------------
    df, target, name, source = load_dataset(args)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    features = [c for c in df.columns if c != target and c not in drop]
    labels = df[target].map(core.label_to_str)
    classes = core.order_classes(labels, args.positive_class)
    if len(classes) < 2 or len(classes) > 50:
        sys.exit(f"The target column has {len(classes)} distinct values; this script expects a classification "
                 "label with 2 to 50 classes.")
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=int)
    X = core.clean_features(df[features])

    print(f"Dataset : {name}\nSource  : {source}")
    print(f"Rows    : {len(df)}   Features: {len(features)}   Classes: {len(classes)} {classes}")
    print("Class counts:", dict(zip(classes, np.bincount(y).tolist())))
    if len(features) < MIN_FEATURES:
        print(f"WARNING: the brief asks for at least {MIN_FEATURES} features; this data has {len(features)}.")
    if len(df) < MIN_ROWS:
        print(f"WARNING: the brief asks for at least {MIN_ROWS} rows; this data has {len(df)}.")

    # ---- 2. stratified train / test split ------------------------------------------------------
    counts = np.bincount(y)
    stratify = y if counts.min() >= 2 else None
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=args.test_size,
                                           stratify=stratify, random_state=args.seed)
    X_train, X_test, y_train, y_test = X.iloc[idx_train], X.iloc[idx_test], y[idx_train], y[idx_test]
    df.iloc[idx_test][features + [target]].to_csv(out / "test_data.csv", index=False)
    print(f"Split   : {len(idx_train)} training rows, {len(idx_test)} test rows -> {out / 'test_data.csv'}")

    meta = {
        "dataset_name": name, "source": source, "target": target, "features": features,
        "classes": classes, "n_classes": len(classes),
        "task": "binary" if len(classes) == 2 else "multiclass",
        "average": "binary" if len(classes) == 2 else "weighted",
        "positive_class": classes[1] if len(classes) == 2 else None,
        "n_rows": len(df), "n_train": len(idx_train), "n_test": len(idx_test),
        "test_size": args.test_size, "random_state": args.seed, "tuned": not args.no_tune,
        "sklearn_version": sklearn.__version__, "best_params": {},
    }

    # ---- 3-5. fit, tune, evaluate --------------------------------------------------------------
    n_splits = int(max(2, min(5, np.bincount(y_train).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    scoring = "f1_weighted"
    scores, tuning = {}, {}
    final_models = {}
    for model_name, (estimator, grid) in model_specs(args.seed).items():
        started = time.time()
        pipe = Pipeline([("prep", make_preprocessor(X_train)), ("clf", estimator)])
        if args.no_tune:
            best, params, cv_score = pipe.fit(X_train, y_train), {}, float("nan")
        else:
            search = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=args.jobs, refit=True)
            search.fit(X_train, y_train)
            best, cv_score = search.best_estimator_, float(search.best_score_)
            params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        if model_name == "Support Vector Machine":      # switch on probability estimates, then refit
            best = clone(best).set_params(clf__probability=True).fit(X_train, y_train)
        final_models[model_name] = best
        res = core.evaluate_model(best, X_test, y_test, meta)
        scores[model_name] = res["metrics"]
        meta["best_params"][model_name] = to_py(params)
        tuning[model_name] = {"best_params": to_py(params), f"cv_{scoring}": cv_score}
        print(f"  {model_name:24s} done in {time.time() - started:5.1f}s   params={params}")

    # ---- 6. save ------------------------------------------------------------------------------
    for model_name, fitted in final_models.items():
        joblib.dump(fitted, model_dir / core.MODEL_FILES[model_name], compress=3)
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(to_py(meta), fh, indent=2)
    with open(results_dir / "tuning.json", "w", encoding="utf-8") as fh:
        json.dump(to_py(tuning), fh, indent=2)

    table = pd.DataFrame(scores).T[core.METRICS]
    table.index.name = "ML Model Name"
    table.to_csv(results_dir / "metrics_test.csv")
    winner = core.mean_ranks(table)
    tables_md = (f"## Comparison table (test split, {len(idx_test)} rows)\n\n{md_table(table)}\n\n"
                 f"Best average rank over the six metrics: **{winner.index[0]}** (average rank {winner.iloc[0]:.2f}).\n")
    (results_dir / "README_tables.md").write_text(tables_md, encoding="utf-8")

    print("\n" + md_table(table))
    print(f"\nBest average rank over the six metrics: {winner.index[0]} (average rank {winner.iloc[0]:.2f})")
    note = ("Precision/Recall/F1 are for the positive class '" + str(meta['positive_class']) + "'"
            if len(classes) == 2 else "Precision/Recall/F1 are weighted averages over classes; AUC is one-vs-rest weighted")
    print(note)
    print(f"\nSaved: {model_dir} (5 models + metadata.json), {out / 'test_data.csv'}, {results_dir}")


if __name__ == "__main__":
    main()
