from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5

TASK_CONFIG = {
    "syll": "syll_",
    "mean": "mean_",
    "pseudo": "pseudo_",
}

EXCLUDED_BASE_FEATURES = {
    "trial_id",
    "n_rows_trial",
    "n_rows_aoi_used",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate specialist task-specific models using subject-wise cross-validation on the development set."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to subject-level wide features.csv",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Path to dev_subjects.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_models",
        help="Directory to save predictions and metrics",
    )
    parser.add_argument(
        "--subject-col",
        type=str,
        default="subject_id",
        help="Subject ID column name. Default: subject_id",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Numeric label column name. Default: label",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of CV folds. Default: 5",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def normalize_subject_id(value) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def load_features(csv_path: str, subject_col: str, label_col: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Features CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    missing = [c for c in [subject_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")

    df = df.copy()
    df[subject_col] = df[subject_col].apply(normalize_subject_id)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    if df[label_col].isna().any():
        raise ValueError("Found invalid labels in features file.")

    dup = df[df.duplicated(subset=[subject_col], keep=False)]
    if not dup.empty:
        raise ValueError("features.csv must have one row per subject.")

    return df


def load_dev_split(split_file: str, subject_col: str, label_col: str) -> pd.DataFrame:
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    missing = [c for c in [subject_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in split file: {missing}")

    df = df.copy()
    df[subject_col] = df[subject_col].apply(normalize_subject_id)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype(int)

    return df[[subject_col, label_col]].drop_duplicates()


def filter_to_dev_set(
    features_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    subject_col: str,
    label_col: str,
) -> pd.DataFrame:
    merged = features_df.merge(
        dev_df,
        on=subject_col,
        how="inner",
        suffixes=("", "_dev"),
    )

    if f"{label_col}_dev" in merged.columns:
        mismatch = merged[
            merged[label_col].astype(int) != merged[f"{label_col}_dev"].astype(int)
        ]
        if not mismatch.empty:
            raise ValueError("Label mismatch between features.csv and dev_subjects.csv")

        merged = merged.drop(columns=[f"{label_col}_dev"])

    return merged.sort_values(subject_col).reset_index(drop=True)


def extract_task_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    task_cols = [c for c in df.columns if c.startswith(prefix)]
    if not task_cols:
        raise ValueError(f"No columns found for prefix '{prefix}'")

    X = df[task_cols].copy()
    rename_map = {c: c[len(prefix):] for c in task_cols}
    X = X.rename(columns=rename_map)

    keep_cols = [c for c in X.columns if c not in EXCLUDED_BASE_FEATURES]
    X = X[keep_cols]

    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def select_valid_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid_cols = [c for c in X_train.columns if not X_train[c].isna().all()]
    if not valid_cols:
        raise ValueError("No valid columns remain after removing all-NaN features.")
    return X_train[valid_cols], X_test[valid_cols]


def build_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = np.nan
    return metrics


def evaluate_one_task(
    df: pd.DataFrame,
    task_name: str,
    prefix: str,
    subject_col: str,
    label_col: str,
    n_splits: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict]:
    X_all = extract_task_frame(df, prefix=prefix)
    y_all = df[label_col].astype(int).reset_index(drop=True)
    subject_ids = df[subject_col].astype(str).reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_predictions = []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_train = X_all.iloc[train_idx].reset_index(drop=True)
        X_test = X_all.iloc[test_idx].reset_index(drop=True)
        y_train = y_all.iloc[train_idx].reset_index(drop=True)
        y_test = y_all.iloc[test_idx].reset_index(drop=True)
        test_subjects = subject_ids.iloc[test_idx].reset_index(drop=True)

        X_train, X_test = select_valid_columns(X_train, X_test)

        model = build_model(random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(
            y_true=y_test.tolist(),
            y_pred=y_pred.tolist(),
            y_prob=y_prob.tolist(),
        )
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        for i in range(len(test_subjects)):
            fold_predictions.append(
                {
                    "task": task_name,
                    "fold": fold_idx,
                    "subject_id": test_subjects.iloc[i],
                    "true_label": int(y_test.iloc[i]),
                    "pred_label": int(y_pred[i]),
                    "pred_prob": float(y_prob[i]),
                }
            )

    pred_df = pd.DataFrame(fold_predictions)

    overall_metrics = compute_metrics(
        y_true=pred_df["true_label"].astype(int).tolist(),
        y_pred=pred_df["pred_label"].astype(int).tolist(),
        y_prob=pred_df["pred_prob"].astype(float).tolist(),
    )

    summary = {
        "task": task_name,
        "n_subjects_evaluated": int(len(pred_df)),
        "n_folds": n_splits,
        "fold_metrics": fold_metrics,
        "overall_metrics": overall_metrics,
        "n_features": int(X_all.shape[1]),
    }

    return pred_df, summary


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_df = load_features(
        csv_path=args.csv,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )
    dev_df = load_dev_split(
        split_file=args.split_file,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )
    eval_df = filter_to_dev_set(
        features_df=features_df,
        dev_df=dev_df,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )

    print(f"Development subjects used for model evaluation: {len(eval_df)}")

    all_summaries = {}
    for task_name, prefix in TASK_CONFIG.items():
        print(f"\n=== Evaluating task: {task_name} ({prefix}) ===")

        pred_df, summary = evaluate_one_task(
            df=eval_df,
            task_name=task_name,
            prefix=prefix,
            subject_col=args.subject_col,
            label_col=args.label_col,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        pred_path = out_dir / f"fold_predictions_{task_name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")

        all_summaries[task_name] = summary

        print(f"Saved predictions: {pred_path}")
        print("Overall metrics:")
        print(json.dumps(summary["overall_metrics"], indent=2))

    summary_path = out_dir / "metrics_summary.json"
    save_json(summary_path, all_summaries)

    print("\n=== Specialist model evaluation complete ===")
    print(f"Saved metrics summary: {summary_path}")


if __name__ == "__main__":
    main()