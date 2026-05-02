from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.config import (
    MEANINGFUL_PREFIX,
    PSEUDOTEXT_PREFIX,
    SYLLABLES_PREFIX,
)
from app.graph.workflow import get_compiled_graph


RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5

EXCLUDED_BASE_FEATURES = {
    "trial_id",
    "n_rows_trial",
    "n_rows_aoi_used",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full multi-agent dyslexia system using subject-wise CV on the development set."
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
        default="outputs/eval",
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
        help="Number of folds for StratifiedKFold. Default: 5",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--max-test-subjects-per-fold",
        type=int,
        default=None,
        help="Optional debugging cap on number of validation subjects per fold.",
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

    required = [subject_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features file: {missing}")

    df = df.copy()
    df[subject_col] = df[subject_col].apply(normalize_subject_id)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    if df[label_col].isna().any():
        raise ValueError(f"Found invalid labels in {csv_path}")

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

    required = [subject_col, label_col]
    missing = [c for c in required if c not in df.columns]
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

    # sanity check if both labels exist
    if f"{label_col}_dev" in merged.columns:
        mismatch = merged[merged[label_col].astype(int) != merged[f"{label_col}_dev"].astype(int)]
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

    # remove metadata-like columns that should not be learned
    keep_cols = [c for c in X.columns if c not in EXCLUDED_BASE_FEATURES]
    X = X[keep_cols]

    # numeric only
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def select_valid_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop columns that are fully missing in train fold.
    """
    valid_cols = [c for c in X_train.columns if not X_train[c].isna().all()]
    if not valid_cols:
        raise ValueError("No valid feature columns remain after filtering all-NaN columns.")

    return X_train[valid_cols], X_test[valid_cols]


def build_model() -> Pipeline:
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
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_one_task_model(
    train_df: pd.DataFrame,
    prefix: str,
    label_col: str,
    model_path: Path,
) -> None:
    X_train = extract_task_frame(train_df, prefix=prefix)
    y_train = train_df[label_col].astype(int)

    # align feature space to non-empty columns only
    valid_cols = [c for c in X_train.columns if not X_train[c].isna().all()]
    X_train = X_train[valid_cols]

    model = build_model()
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def final_risk_to_binary(final_risk_level: str) -> int:
    """
    Binary rule for full-system prediction.

    low_risk -> 0
    moderate_risk/high_risk -> 1
    """
    if final_risk_level == "low_risk":
        return 0
    if final_risk_level in {"moderate_risk", "high_risk"}:
        return 1
    raise ValueError(f"Unknown final_risk_level: {final_risk_level}")


def specialist_mean_score(final_report: Dict) -> float:
    """
    Continuous score proxy for ROC-AUC.

    We use the mean of the three specialist risk scores because the final
    board output is categorical, while AUC needs a continuous score.
    """
    specialist_reports = final_report["specialist_reports"]
    scores = [float(r["risk_score"]) for r in specialist_reports]
    return float(np.mean(scores))


def compute_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auc"] = np.nan

    return metrics


def run_subject_through_system(
    graph,
    csv_path: str,
    subject_id: str,
    model_paths: Dict[str, str],
) -> Dict:
    initial_state = {
        "case_csv": csv_path,
        "subject_id": subject_id,
        "mode": "cv",
        "syllables_model_path": model_paths["syllables"],
        "meaningful_model_path": model_paths["meaningful"],
        "pseudotext_model_path": model_paths["pseudotext"],
    }
    result = graph.invoke(initial_state)
    return result["final_report"]


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
    dev_subjects_df = load_dev_split(
        split_file=args.split_file,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )
    dev_df = filter_to_dev_set(
        features_df=features_df,
        dev_df=dev_subjects_df,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )

    print(f"Development subjects used for CV: {len(dev_df)}")

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    graph = get_compiled_graph()

    all_predictions = []
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(dev_df, dev_df[args.label_col].astype(int)),
        start=1,
    ):
        print(f"\n=== Fold {fold_idx}/{args.n_splits} ===")

        train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        test_df = dev_df.iloc[test_idx].reset_index(drop=True)

        if args.max_test_subjects_per_fold is not None:
            test_df = test_df.iloc[: args.max_test_subjects_per_fold].copy()

        fold_model_dir = Path("models") / "tmp_cv" / f"fold_{fold_idx}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        syll_model_path = fold_model_dir / "syllables_rf.joblib"
        mean_model_path = fold_model_dir / "meaningful_rf.joblib"
        pseudo_model_path = fold_model_dir / "pseudotext_rf.joblib"

        print("Training fold-specific task models...")
        train_one_task_model(
            train_df=train_df,
            prefix=SYLLABLES_PREFIX,
            label_col=args.label_col,
            model_path=syll_model_path,
        )
        train_one_task_model(
            train_df=train_df,
            prefix=MEANINGFUL_PREFIX,
            label_col=args.label_col,
            model_path=mean_model_path,
        )
        train_one_task_model(
            train_df=train_df,
            prefix=PSEUDOTEXT_PREFIX,
            label_col=args.label_col,
            model_path=pseudo_model_path,
        )

        fold_y_true = []
        fold_y_pred = []
        fold_y_score = []

        for _, row in test_df.iterrows():
            subject_id = normalize_subject_id(row[args.subject_col])
            true_label = int(row[args.label_col])

            final_report = run_subject_through_system(
                graph=graph,
                csv_path=args.csv,
                subject_id=subject_id,
                model_paths={
                    "syllables": str(syll_model_path),
                    "meaningful": str(mean_model_path),
                    "pseudotext": str(pseudo_model_path),
                },
            )

            final_risk_level = final_report["final_assessment"]["final_risk_level"]
            pred_label = final_risk_to_binary(final_risk_level)
            score = specialist_mean_score(final_report)

            fold_y_true.append(true_label)
            fold_y_pred.append(pred_label)
            fold_y_score.append(score)

            all_predictions.append(
                {
                    "fold": fold_idx,
                    "subject_id": subject_id,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "system_score": score,
                    "final_risk_level": final_risk_level,
                    "overall_confidence": final_report["final_assessment"]["overall_confidence"],
                    "agreement_level": final_report["final_assessment"]["agreement_level"],
                    "critic_verdict": final_report["critic_report"]["verdict"],
                }
            )

        metrics = compute_metrics(fold_y_true, fold_y_pred, fold_y_score)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        print(f"Fold {fold_idx} metrics:")
        print(json.dumps(metrics, indent=2))

    pred_df = pd.DataFrame(all_predictions)
    pred_path = out_dir / "dev_cv_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    overall_metrics = compute_metrics(
        y_true=pred_df["true_label"].astype(int).tolist(),
        y_pred=pred_df["pred_label"].astype(int).tolist(),
        y_score=pred_df["system_score"].astype(float).tolist(),
    )

    summary = {
        "n_subjects_evaluated": int(len(pred_df)),
        "n_folds": args.n_splits,
        "fold_metrics": fold_metrics,
        "overall_metrics": overall_metrics,
    }

    summary_path = out_dir / "dev_cv_metrics.json"
    save_json(summary_path, summary)

    print("\n=== Evaluation complete ===")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics    : {summary_path}")
    print("\nOverall metrics:")
    print(json.dumps(overall_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()