from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42

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
        description="Train final task-specific models on the development set only."
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
        default="models/final",
        help="Directory to save final trained models",
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
        raise ValueError("Found invalid labels in features.csv")

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


def train_and_save_one_task(
    df: pd.DataFrame,
    task_name: str,
    prefix: str,
    label_col: str,
    output_dir: Path,
    random_state: int,
) -> Dict:
    X = extract_task_frame(df, prefix=prefix)
    y = df[label_col].astype(int)

    valid_cols = [c for c in X.columns if not X[c].isna().all()]
    if not valid_cols:
        raise ValueError(f"No valid feature columns remain for task '{task_name}'")

    X = X[valid_cols]

    model = build_model(random_state=random_state)
    model.fit(X, y)

    model_path = output_dir / f"{task_name}_rf.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "task_name": task_name,
        "prefix": prefix,
        "n_subjects": int(len(df)),
        "n_features_used": int(len(valid_cols)),
        "feature_names": valid_cols,
        "model_path": str(model_path),
        "label_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
    }

    metadata_path = output_dir / f"{task_name}_rf.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_df = filter_to_dev_set(
        features_df=features_df,
        dev_df=dev_df,
        subject_col=args.subject_col,
        label_col=args.label_col,
    )

    print(f"Training final models on development subjects only: {len(train_df)}")

    all_metadata = {}
    for task_name, prefix in TASK_CONFIG.items():
        print(f"\n=== Training final model for task: {task_name} ===")
        metadata = train_and_save_one_task(
            df=train_df,
            task_name={
                "syll": "syllables",
                "mean": "meaningful",
                "pseudo": "pseudotext",
            }[task_name],
            prefix=prefix,
            label_col=args.label_col,
            output_dir=output_dir,
            random_state=args.random_state,
        )
        all_metadata[task_name] = metadata
        print(json.dumps(metadata, indent=2, ensure_ascii=False))

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(all_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Final model training complete ===")
    print(f"Saved models and metadata to: {output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()