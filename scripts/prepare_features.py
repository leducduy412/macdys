from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


TRIAL_FEATURE_COLS = [
    "n_fix_trial",
    "sum_fix_dur_trial",
    "dwell_time_trial",
    "mean_fix_dur_trial",
    "n_sacc_trial",
    "sum_sacc_dur_trial",
    "mean_sacc_dur_trial",
    "mean_sacc_ampl_trial",
    "ratio_progress_regress_trial",
    "n_between_line_regress_trial",
    "n_within_line_regress_trial",
    "n_regress_trial",
    "n_progress_trial",
    "n_transit_trial",
]

AOI_FEATURE_COLS = [
    "dwell_time_aoi",
    "n_fix_aoi",
    "sum_fix_dur_aoi",
    "mean_fix_dur_aoi",
    "skipped_aoi",
    "n_fix_first_visit_aoi",
    "first_fix_dur_aoi",
    "first_fix_land_pos_aoi",
    "dwell_time_first_visit_aoi",
    "sum_fix_dur_first_visit_aoi",
    "sum_fix_dur_after_first_visit_aoi",
    "dwell_time_rereading_aoi",
    "n_revisits_aoi",
]

AGGREGATE_REQUIRED_COLS = {
    "sid",
    "task",
    "trialid",
    "n_fix_trial",
    "sum_fix_dur_trial",
    "mean_fix_dur_trial",
    "n_sacc_trial",
    "mean_sacc_ampl_trial",
    "aoi",
    "aoi_kind",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare subject-level wide features from eye-tracking aggregate CSV files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to one CSV file or a directory containing multiple CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save subject_task_features.csv and features.csv",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional CSV with columns: subject_id,class_id,label",
    )
    parser.add_argument(
        "--sid-col",
        type=str,
        default="sid",
        help="Subject ID column name in the raw aggregate files.",
    )
    parser.add_argument(
        "--task-col",
        type=str,
        default="task",
        help="Task column name in the raw aggregate files.",
    )
    parser.add_argument(
        "--trial-col",
        type=str,
        default="trialid",
        help="Trial ID column name in the raw aggregate files.",
    )
    return parser.parse_args()


def normalize_subject_id(value) -> str:
    if pd.isna(value):
        return "unknown_subject"

    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def normalize_task_name(task_value) -> str:
    if pd.isna(task_value):
        return "unknown"

    t = str(task_value).strip().lower()
    if t in {"", "nan", "none"}:
        return "unknown"

    if "syll" in t or "t1" in t:
        return "syll"
    if "meaning" in t or "meaningful" in t or "t4" in t or "t2" in t:
        return "mean"
    if "pseudo" in t or "t5" in t or "t3" in t or "nonword" in t:
        return "pseudo"

    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t if t else "unknown"


def collect_csv_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        return [path]

    files = sorted(path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under directory: {path}")
    return files


def is_aggregate_dataframe(df: pd.DataFrame) -> bool:
    return AGGREGATE_REQUIRED_COLS.issubset(set(df.columns))


def load_raw_data(files: List[Path]) -> pd.DataFrame:
    dfs = []
    skipped = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

            if not is_aggregate_dataframe(df):
                skipped.append(fp.name)
                continue

            df["__source_file__"] = fp.name
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not dfs:
        raise ValueError("No aggregate CSV files were loaded.")

    print(f"Loaded {len(dfs)} aggregate file(s).")
    if skipped:
        print(f"Skipped {len(skipped)} non-aggregate file(s).")

    return pd.concat(dfs, ignore_index=True)


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan


def safe_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.std()) if not s.empty else np.nan


def safe_max(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.max()) if not s.empty else np.nan


def pick_aoi_rows(group: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer subline rows because they are the most informative AOI-level units.
    Fallback: non-line rows, then any non-null AOI rows.
    """
    if "aoi_kind" in group.columns:
        sub = group[group["aoi_kind"].astype(str).str.lower() == "subline"].copy()
        if not sub.empty:
            return sub

        non_line = group[group["aoi_kind"].astype(str).str.lower() != "line"].copy()
        if not non_line.empty:
            return non_line

    if "aoi" in group.columns:
        non_null_aoi = group[group["aoi"].notna()].copy()
        if not non_null_aoi.empty:
            return non_null_aoi

    return group.copy()


def pick_trial_row(group: pd.DataFrame) -> pd.Series:
    """
    Choose one representative row that contains valid trial-level aggregate values.
    """
    available_trial_cols = [c for c in TRIAL_FEATURE_COLS if c in group.columns]
    if not available_trial_cols:
        return group.iloc[0]

    aggregate_rows = group.dropna(subset=available_trial_cols, how="all")
    if not aggregate_rows.empty:
        return aggregate_rows.iloc[0]

    return group.iloc[0]


def summarize_one_trial(
    group: pd.DataFrame,
    sid_col: str,
    task_col: str,
    trial_col: str,
) -> Dict[str, float]:
    trial_row = pick_trial_row(group)
    aoi_rows = pick_aoi_rows(group)

    result: Dict[str, float] = {
        "subject_id": normalize_subject_id(trial_row[sid_col]),
        "task_raw": str(trial_row[task_col]),
        "task_key": normalize_task_name(trial_row[task_col]),
        "trial_id": pd.to_numeric(
            pd.Series([trial_row[trial_col]]), errors="coerce"
        ).iloc[0] if trial_col in group.columns else np.nan,
        "n_rows_trial": float(len(group)),
        "n_rows_aoi_used": float(len(aoi_rows)),
    }

    for col in TRIAL_FEATURE_COLS:
        if col in group.columns:
            result[col] = pd.to_numeric(
                pd.Series([trial_row[col]]), errors="coerce"
            ).iloc[0]
        else:
            result[col] = np.nan

    for col in AOI_FEATURE_COLS:
        if col not in aoi_rows.columns:
            result[f"{col}_mean"] = np.nan
            if col in {"dwell_time_aoi", "n_fix_aoi", "mean_fix_dur_aoi", "n_revisits_aoi"}:
                result[f"{col}_std"] = np.nan
                result[f"{col}_max"] = np.nan
            continue

        s = pd.to_numeric(aoi_rows[col], errors="coerce")
        result[f"{col}_mean"] = safe_mean(s)

        if col in {"dwell_time_aoi", "n_fix_aoi", "mean_fix_dur_aoi", "n_revisits_aoi"}:
            result[f"{col}_std"] = safe_std(s)
            result[f"{col}_max"] = safe_max(s)

    if "skipped_aoi" in aoi_rows.columns:
        skipped = pd.to_numeric(aoi_rows["skipped_aoi"], errors="coerce")
        result["skipped_aoi_rate"] = safe_mean(skipped)
    else:
        result["skipped_aoi_rate"] = np.nan

    # if "content" in aoi_rows.columns:
    #     result["n_unique_content_units"] = float(aoi_rows["content"].astype(str).nunique())
    # else:
    #     result["n_unique_content_units"] = np.nan

    return result


def build_subject_task_features(
    raw_df: pd.DataFrame,
    sid_col: str,
    task_col: str,
    trial_col: str,
) -> pd.DataFrame:
    for required in [sid_col, task_col]:
        if required not in raw_df.columns:
            raise ValueError(f"Missing required column: {required}")

    if trial_col not in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df[trial_col] = 0

    trial_rows = []
    grouped = raw_df.groupby([sid_col, task_col, trial_col], dropna=False)

    for _, g in grouped:
        trial_rows.append(
            summarize_one_trial(
                g,
                sid_col=sid_col,
                task_col=task_col,
                trial_col=trial_col,
            )
        )

    trial_df = pd.DataFrame(trial_rows)
    trial_df = trial_df[trial_df["task_key"] != "unknown"].copy()

    numeric_cols = trial_df.select_dtypes(include=[np.number]).columns.tolist()

    subject_task_df = (
        trial_df.groupby(["subject_id", "task_key"], dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )

    raw_task_map = (
        trial_df.groupby(["subject_id", "task_key"], dropna=False)["task_raw"]
        .first()
        .reset_index()
    )

    subject_task_df = subject_task_df.merge(
        raw_task_map,
        on=["subject_id", "task_key"],
        how="left",
    )

    ordered_cols = ["subject_id", "task_key", "task_raw"] + [
        c for c in subject_task_df.columns if c not in {"subject_id", "task_key", "task_raw"}
    ]
    subject_task_df = subject_task_df[ordered_cols]

    return subject_task_df.sort_values(["subject_id", "task_key"]).reset_index(drop=True)


def merge_labels_to_subject_task_df(
    subject_task_df: pd.DataFrame,
    labels_csv: Optional[str],
) -> pd.DataFrame:
    if labels_csv is None:
        return subject_task_df

    labels = pd.read_csv(labels_csv)
    labels.columns = [str(c).strip().replace("\ufeff", "") for c in labels.columns]

    if "subject_id" not in labels.columns:
        if "sid" in labels.columns:
            labels = labels.rename(columns={"sid": "subject_id"})
        else:
            raise ValueError("labels_csv must contain either 'subject_id' or 'sid' column.")

    labels["subject_id"] = labels["subject_id"].apply(normalize_subject_id)
    subject_task_df["subject_id"] = subject_task_df["subject_id"].apply(normalize_subject_id)

    if "label" in labels.columns:
        labels["label_text"] = labels["label"].astype(str)
    else:
        labels["label_text"] = np.nan

    if "class_id" in labels.columns:
        labels["label"] = pd.to_numeric(labels["class_id"], errors="coerce")
    elif "label_text" in labels.columns:
        label_map = {
            "non-dyslexic": 0,
            "dyslexic": 1,
        }
        labels["label"] = (
            labels["label_text"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(label_map)
        )
    else:
        raise ValueError("labels_csv must contain either 'class_id' or 'label' column.")

    labels = labels[["subject_id", "label", "label_text"]].drop_duplicates()

    merged = subject_task_df.merge(labels, on="subject_id", how="left")

    cols = ["subject_id", "label", "label_text"] + [
        c for c in merged.columns if c not in {"subject_id", "label", "label_text"}
    ]
    return merged[cols]


def pivot_subject_task_to_wide(subject_task_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        c for c in subject_task_df.columns
        if c not in {"subject_id", "label", "label_text", "task_key", "task_raw"}
    ]

    task_keys = sorted(subject_task_df["task_key"].dropna().unique().tolist())
    if not task_keys:
        raise ValueError("No valid task keys found for pivoting.")

    wide_parts = []
    for task_key in task_keys:
        sub = subject_task_df[subject_task_df["task_key"] == task_key].copy()
        rename_map = {col: f"{task_key}_{col}" for col in feature_cols}
        sub = sub[["subject_id"] + feature_cols].rename(columns=rename_map)
        wide_parts.append(sub)

    wide_df = wide_parts[0]
    for part in wide_parts[1:]:
        wide_df = wide_df.merge(part, on="subject_id", how="outer")

    label_df = (
        subject_task_df[["subject_id", "label", "label_text"]]
        .drop_duplicates(subset=["subject_id"])
        .copy()
    )

    wide_df = wide_df.merge(label_df, on="subject_id", how="left")

    cols = ["subject_id", "label", "label_text"] + [
        c for c in wide_df.columns if c not in {"subject_id", "label", "label_text"}
    ]
    return wide_df[cols].sort_values("subject_id").reset_index(drop=True)


def save_outputs(
    subject_task_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    output_dir: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    subject_task_path = out / "subject_task_features.csv"
    wide_path = out / "features.csv"
    meta_path = out / "prepare_features_summary.json"

    subject_task_df.to_csv(subject_task_path, index=False, encoding="utf-8")
    wide_df.to_csv(wide_path, index=False, encoding="utf-8")

    summary = {
        "n_subject_task_rows": int(len(subject_task_df)),
        "n_subject_rows": int(len(wide_df)),
        "task_keys": sorted(subject_task_df["task_key"].dropna().unique().tolist()),
        "subject_task_file": str(subject_task_path),
        "wide_file": str(wide_path),
    }
    meta_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved subject-task features: {subject_task_path}")
    print(f"Saved wide features       : {wide_path}")
    print(f"Saved summary             : {meta_path}")


def main() -> None:
    args = parse_args()

    csv_files = collect_csv_files(args.input)
    print(f"Found {len(csv_files)} CSV file(s).")

    raw_df = load_raw_data(csv_files)
    print(f"Loaded raw rows: {len(raw_df)}")

    subject_task_df = build_subject_task_features(
        raw_df=raw_df,
        sid_col=args.sid_col,
        task_col=args.task_col,
        trial_col=args.trial_col,
    )

    subject_task_df = merge_labels_to_subject_task_df(
        subject_task_df=subject_task_df,
        labels_csv=args.labels_csv,
    )

    wide_df = pivot_subject_task_to_wide(subject_task_df)

    save_outputs(
        subject_task_df=subject_task_df,
        wide_df=wide_df,
        output_dir=args.output_dir,
    )

    print("\nDone.")
    print("\nSubject-task preview:")
    print(subject_task_df.head(5).to_string(index=False))
    print("\nWide preview:")
    print(wide_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()