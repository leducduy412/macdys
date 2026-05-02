from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split subject-level features into development and final hold-out sets."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to subject-level features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to save split files",
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
        "--label-text-col",
        type=str,
        default="label_text",
        help="Readable text label column name. Default: label_text",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Fraction of subjects reserved for final hold-out. Default: 0.2",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible split. Default: 42",
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


def load_features(
    csv_path: str,
    subject_col: str,
    label_col: str,
    label_text_col: str,
) -> pd.DataFrame:
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
        bad = df[df[label_col].isna()][[subject_col, label_col]]
        raise ValueError(
            f"Found missing/non-numeric labels in features file. Example rows:\n{bad.head()}"
        )

    if label_text_col not in df.columns:
        # create a readable label if missing
        df[label_text_col] = df[label_col].map({0: "non-dyslexic", 1: "dyslexic"})

    dup = df[df.duplicated(subset=[subject_col], keep=False)]
    if not dup.empty:
        raise ValueError(
            f"features.csv must have one row per subject. Duplicate subjects found, e.g.:\n"
            f"{dup[[subject_col]].head()}"
        )

    return df


def stratified_subject_split(
    df: pd.DataFrame,
    subject_col: str,
    label_col: str,
    label_text_col: str,
    holdout_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError("--holdout-ratio must be between 0 and 1.")

    split_df = df[[subject_col, label_col, label_text_col]].copy()

    dev_df, holdout_df = train_test_split(
        split_df,
        test_size=holdout_ratio,
        stratify=split_df[label_col],
        random_state=random_state,
    )

    dev_df = dev_df.sort_values(subject_col).reset_index(drop=True)
    holdout_df = holdout_df.sort_values(subject_col).reset_index(drop=True)

    # make label explicit integers
    dev_df[label_col] = dev_df[label_col].astype(int)
    holdout_df[label_col] = holdout_df[label_col].astype(int)

    return dev_df, holdout_df


def build_summary(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    label_col: str,
    holdout_ratio: float,
    random_state: int,
) -> dict:
    return {
        "random_state": random_state,
        "holdout_ratio": holdout_ratio,
        "n_dev_subjects": int(len(dev_df)),
        "n_holdout_subjects": int(len(holdout_df)),
        "dev_label_distribution": {
            str(k): int(v) for k, v in dev_df[label_col].value_counts().sort_index().items()
        },
        "holdout_label_distribution": {
            str(k): int(v) for k, v in holdout_df[label_col].value_counts().sort_index().items()
        },
    }


def save_outputs(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    summary: dict,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev_path = out / "dev_subjects.csv"
    holdout_path = out / "holdout_subjects.csv"
    summary_path = out / "split_summary.json"

    dev_df.to_csv(dev_path, index=False, encoding="utf-8", na_rep="NA")
    holdout_df.to_csv(holdout_path, index=False, encoding="utf-8", na_rep="NA")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved development split : {dev_path}")
    print(f"Saved hold-out split    : {holdout_path}")
    print(f"Saved summary           : {summary_path}")


def main() -> None:
    args = parse_args()

    df = load_features(
        csv_path=args.csv,
        subject_col=args.subject_col,
        label_col=args.label_col,
        label_text_col=args.label_text_col,
    )

    dev_df, holdout_df = stratified_subject_split(
        df=df,
        subject_col=args.subject_col,
        label_col=args.label_col,
        label_text_col=args.label_text_col,
        holdout_ratio=args.holdout_ratio,
        random_state=args.random_state,
    )

    summary = build_summary(
        dev_df=dev_df,
        holdout_df=holdout_df,
        label_col=args.label_col,
        holdout_ratio=args.holdout_ratio,
        random_state=args.random_state,
    )

    save_outputs(
        dev_df=dev_df,
        holdout_df=holdout_df,
        summary=summary,
        output_dir=args.output_dir,
    )

    print("\nDevelopment set preview:")
    print(dev_df.head().to_string(index=False))

    print("\nHold-out set preview:")
    print(holdout_df.head().to_string(index=False))

    print("\nSplit summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()