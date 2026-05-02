from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

VOWELS = set("aeiouyáéěíóúůýAEIOUYÁÉĚÍÓÚŮÝ")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare subject-level eye-tracking + linguistic features using "
            "fixation files, optional saccade files, optional precomputed metric files, "
            "and AOI layout files. "
            "Output label convention: 0 = non-dyslexic, 1 = dyslexic."
        )
    )

    parser.add_argument(
        "--fixation-dir",
        type=str,
        required=True,
        help=(
            "Directory containing fixation-level CSV files. "
            "Expected columns: task, sid, stimfile, duration_ms, fix_x, fix_y, "
            "aoi_subline, and aoi_line."
        ),
    )

    parser.add_argument(
        "--saccade-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing saccade-level CSV files. "
            "Common columns include sid/subject_id, task, stimfile, duration_ms, "
            "ampl, avg_vel, peak_vel, start_x, start_y, end_x, end_y."
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing precomputed metric CSV files. "
            "These may include reading time, fixation counts, trial duration, "
            "global task metrics, or other ETDD70-derived measures."
        ),
    )

    parser.add_argument(
        "--aoi-dir",
        type=str,
        required=True,
        help=(
            "Directory containing AOI layout CSV files. "
            "Expected columns: stimfile, content, kind, x, y, width, height, "
            "line, and part."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/features.csv",
        help="Output subject-level feature CSV.",
    )

    parser.add_argument(
        "--aoi-profile-json",
        type=str,
        default=None,
        help=(
            "Optional output JSON storing subject-task AOI difficulty profiles. "
            "If not provided, it will be saved next to output CSV as aoi_profiles.json."
        ),
    )

    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV file containing subject_id,label. "
            "If not provided, labels are inferred from file/folder names."
        ),
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Label column name in labels CSV. Default: label.",
    )

    parser.add_argument(
        "--subject-col",
        type=str,
        default="subject_id",
        help="Subject ID column name in labels CSV. Default: subject_id.",
    )

    parser.add_argument(
        "--top-k-aoi",
        type=int,
        default=8,
        help="Number of top difficult AOI/textual units to store per subject-task.",
    )

    return parser.parse_args()


# =============================================================================
# CSV helpers
# =============================================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Read CSV with several possible encodings.
    """
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path)


def validate_fixation_file(df: pd.DataFrame, path: Path) -> None:
    """
    Validate fixation-level file.

    Expected fixation columns:
    id, task, sid, eye, stimfile, trialid, start_ms, end_ms, duration_ms,
    fix_x, fix_y, orig_fix_x, orig_fix_y, disp_x, disp_y,
    aoi_subline, aoi_line
    """
    required = {
        "task",
        "sid",
        "stimfile",
        "trialid",
        "start_ms",
        "end_ms",
        "duration_ms",
        "fix_x",
        "fix_y",
        "aoi_subline",
        "aoi_line",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is not a valid fixation file. Missing columns: {missing}"
        )


def validate_aoi_file(df: pd.DataFrame, path: Path) -> None:
    """
    Validate AOI layout file.
    """
    required = {
        "stimfile",
        "content",
        "kind",
        "x",
        "y",
        "width",
        "height",
        "line",
        "part",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is not a valid AOI layout file. Missing columns: {missing}"
        )


# =============================================================================
# General column helpers
# =============================================================================

def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first existing column from candidate names.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def sanitize_feature_name(name: str) -> str:
    """
    Make column names safe and consistent for feature output.
    """
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def convert_possible_numeric_columns(
    df: pd.DataFrame,
    excluded_cols: set[str],
) -> pd.DataFrame:
    """
    Convert columns to numeric when possible.

    We only replace a column if conversion yields at least one non-null numeric value.
    """
    df = df.copy()

    for col in df.columns:
        if col in excluded_cols:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")

        if converted.notna().sum() > 0:
            df[col] = converted

    return df


# =============================================================================
# Label helpers
# =============================================================================

def normalize_label(value) -> int:
    """
    Convert different label formats into:
    0 = non-dyslexic
    1 = dyslexic
    """
    text = str(value).strip().lower()

    if text in ["0", "0.0"]:
        return 0

    if text in ["1", "1.0"]:
        return 1

    non_dyslexic_values = {
        "non-dyslexic",
        "non dyslexic",
        "nondyslexic",
        "non_dyslexic",
        "non-dyslexia",
        "non dyslexia",
        "control",
        "controls",
        "typical",
        "typically developing",
        "normal",
        "healthy",
        "non",
    }

    dyslexic_values = {
        "dyslexic",
        "dyslexia",
        "dys",
        "developmental dyslexia",
        "reading_disorder",
        "reading disorder",
    }

    if text in non_dyslexic_values:
        return 0

    if text in dyslexic_values:
        return 1

    raise ValueError(f"Unknown label value: {value}")


def infer_label_from_path(path: Path) -> Optional[int]:
    """
    Infer label from file/folder name if labels.csv is not provided.
    """
    text = " ".join([p.lower() for p in path.parts])

    if any(k in text for k in [
        "non-dyslexic",
        "non_dyslexic",
        "nondyslexic",
        "non dyslexic",
        "control",
        "typical",
        "normal",
        "healthy",
    ]):
        return 0

    if any(k in text for k in [
        "dyslexic",
        "dyslexia",
        " dys ",
        "_dys_",
        "-dys-",
    ]):
        return 1

    for part in path.parts:
        if part == "0":
            return 0
        if part == "1":
            return 1

    stem = path.stem.lower()

    patterns = [
        r"(?:sbj|subject|sub|sid)[_-]?\d+[^\d]+([01])(?:\D|$)",
        r"(?:sbj|subject|sub|sid)[_-]?([01])[^\d]+\d+",
        r"\d+[^\d]+([01])(?:\D|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return int(match.group(1))

    return None


def load_labels_csv(
    labels_csv: Optional[Path],
    subject_col: str,
    label_col: str,
) -> Optional[pd.DataFrame]:
    """
    Load optional labels CSV and normalize labels to 0/1.
    """
    if labels_csv is None:
        return None

    labels = pd.read_csv(labels_csv)

    if subject_col not in labels.columns:
        raise ValueError(
            f"labels_csv must contain subject column '{subject_col}'."
        )

    if label_col not in labels.columns:
        raise ValueError(
            f"labels_csv must contain label column '{label_col}'."
        )

    labels = labels[[subject_col, label_col]].copy()
    labels = labels.rename(columns={subject_col: "subject_id", label_col: "label"})

    labels["subject_id"] = labels["subject_id"].astype(str)
    labels["label"] = labels["label"].apply(normalize_label)

    invalid_labels = sorted(set(labels["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(
            f"Labels must be binary 0/1. Found invalid labels: {invalid_labels}"
        )

    return labels


# =============================================================================
# Task helpers
# =============================================================================

def normalize_task_name(value: str) -> str:
    """
    Convert task names into internal prefixes:
    - T1_Syllables      -> syll
    - T2_MeaningfulText -> mean
    - T3_PseudoText     -> pseudo

    Some ETDD70 stimulus files may use t4/t5 naming for text/pseudo-text.
    """
    text = str(value).strip().lower()

    if "syll" in text or "t1" in text:
        return "syll"

    if "pseudo" in text or "t3" in text or "t5" in text:
        return "pseudo"

    if "meaning" in text or "text" in text or "t2" in text or "t4" in text:
        return "mean"

    raise ValueError(f"Cannot infer task prefix from task value: {value}")


def infer_task_from_stimfile(stimfile: str) -> str:
    """
    Fallback task inference from stimulus filename.
    """
    text = str(stimfile).lower()

    if "t1" in text or "syll" in text:
        return "syll"

    if "t2" in text or "t4" in text or "meaning" in text or "text" in text:
        return "mean"

    if "t3" in text or "t5" in text or "pseudo" in text:
        return "pseudo"

    raise ValueError(f"Cannot infer task prefix from stimfile: {stimfile}")


def infer_task_from_path(path: Path) -> str:
    """
    Fallback task inference from file path.
    """
    text = str(path).lower()

    if "syll" in text or "t1" in text:
        return "syll"

    if "pseudo" in text or "t3" in text or "t5" in text:
        return "pseudo"

    if "meaning" in text or "text" in text or "t2" in text or "t4" in text:
        return "mean"

    raise ValueError(f"Cannot infer task prefix from path: {path}")


# =============================================================================
# AOI preparation
# =============================================================================

def make_aoi_key_from_line_part(line, part) -> Optional[str]:
    """
    Convert AOI line/part into fixation AOI format.

    Example:
    line = 1, part = 1
    -> line_001-part_001
    """
    if pd.isna(line) or pd.isna(part):
        return None

    return f"line_{int(float(line)):03d}-part_{int(float(part)):03d}"


def prepare_aoi_table(aoi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare AOI table for merging with fixation files.
    """
    aoi_df = aoi_df.copy()

    aoi_df = aoi_df[
        aoi_df["kind"]
        .astype(str)
        .str.lower()
        .isin(["sub-line", "word", "token", "syllable"])
    ]

    aoi_df["stimfile"] = aoi_df["stimfile"].astype(str)

    aoi_df["aoi_subline"] = aoi_df.apply(
        lambda row: make_aoi_key_from_line_part(row["line"], row["part"]),
        axis=1,
    )

    aoi_df = aoi_df.dropna(subset=["aoi_subline"])

    keep_cols = [
        "stimfile",
        "aoi_subline",
        "content",
        "kind",
        "name",
        "line",
        "part",
        "column",
        "x",
        "y",
        "width",
        "height",
    ]

    existing_cols = [col for col in keep_cols if col in aoi_df.columns]

    aoi_df = aoi_df[existing_cols].drop_duplicates(
        subset=["stimfile", "aoi_subline"]
    )

    return aoi_df


def load_aoi_tables(aoi_dir: Path) -> pd.DataFrame:
    """
    Load all AOI layout CSV files.
    """
    csv_files = sorted(aoi_dir.rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No AOI CSV files found in: {aoi_dir}")

    tables: List[pd.DataFrame] = []

    for path in csv_files:
        df = read_csv_safely(path)
        validate_aoi_file(df, path)

        prepared = prepare_aoi_table(df)

        if prepared.empty:
            print(f"Warning: no textual AOIs found in {path.name}")
            continue

        tables.append(prepared)

        print(
            f"Loaded AOI file: {path.name}, textual AOIs={len(prepared)}"
        )

    if not tables:
        raise ValueError("No valid textual AOI rows found.")

    return pd.concat(tables, ignore_index=True)


# =============================================================================
# Linguistic features
# =============================================================================

def count_vowels(text: str) -> int:
    return sum(1 for ch in text if ch in VOWELS)


def count_consonants(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ch not in VOWELS)


def has_diacritic(text: str) -> int:
    return int(any(ord(ch) > 127 for ch in text))


def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple linguistic features from AOI content.
    """
    df = df.copy()

    df["content"] = df["content"].fillna("").astype(str)

    df["content_length"] = df["content"].apply(len)
    df["num_vowels"] = df["content"].apply(count_vowels)
    df["num_consonants"] = df["content"].apply(count_consonants)
    df["has_diacritic"] = df["content"].apply(has_diacritic)
    df["vowel_ratio"] = df["num_vowels"] / df["content_length"].replace(0, 1)

    return df


# =============================================================================
# Fixation cleaning and merging
# =============================================================================

def clean_fixation_df(
    df: pd.DataFrame,
    path: Path,
    inferred_label: Optional[int],
) -> pd.DataFrame:
    """
    Clean one fixation dataframe.
    """
    validate_fixation_file(df, path)

    df = df.copy()

    df["sid"] = df["sid"].astype(str)
    df["subject_id"] = df["sid"]
    df["stimfile"] = df["stimfile"].astype(str)

    if "task" in df.columns and not df["task"].dropna().empty:
        df["task_prefix"] = df["task"].apply(normalize_task_name)
    else:
        df["task_prefix"] = df["stimfile"].apply(infer_task_from_stimfile)

    df["aoi_subline"] = df["aoi_subline"].astype(str)

    if inferred_label is not None:
        df["label_from_path"] = int(inferred_label)

    numeric_cols = [
        "start_ms",
        "end_ms",
        "duration_ms",
        "fix_x",
        "fix_y",
        "orig_fix_x",
        "orig_fix_y",
        "disp_x",
        "disp_y",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_fixations_with_aoi_content(
    fix_df: pd.DataFrame,
    aoi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge fixation events with AOI content using:
    - stimfile
    - aoi_subline
    """
    fix_df = fix_df.copy()

    valid_fix = fix_df[
        fix_df["aoi_subline"].notna()
        & (fix_df["aoi_subline"] != "")
        & (fix_df["aoi_subline"] != "nan")
    ].copy()

    merged = valid_fix.merge(
        aoi_df,
        on=["stimfile", "aoi_subline"],
        how="left",
        suffixes=("", "_aoi"),
    )

    merged = merged.dropna(subset=["content"]).copy()

    if merged.empty:
        return merged

    merged = add_linguistic_features(merged)

    return merged


# =============================================================================
# Saccade cleaning
# =============================================================================

def clean_saccade_df(
    df: pd.DataFrame,
    path: Path,
    inferred_label: Optional[int],
) -> pd.DataFrame:
    """
    Clean one saccade dataframe.

    The function is tolerant to different column names.
    """
    df = df.copy()

    sid_col = find_first_column(
        df,
        ["sid", "subject_id", "subj_id", "participant_id", "participant"],
    )
    if sid_col is None:
        raise ValueError(f"{path} is not a valid saccade file. Missing sid/subject_id.")

    stim_col = find_first_column(df, ["stimfile", "stimulus", "stim_file"])
    if stim_col is None:
        raise ValueError(f"{path} is not a valid saccade file. Missing stimfile.")

    df["sid"] = df[sid_col].astype(str)
    df["subject_id"] = df["sid"]
    df["stimfile"] = df[stim_col].astype(str)

    if "task" in df.columns and not df["task"].dropna().empty:
        df["task_prefix"] = df["task"].apply(normalize_task_name)
    else:
        df["task_prefix"] = df["stimfile"].apply(infer_task_from_stimfile)

    if inferred_label is not None:
        df["label_from_path"] = int(inferred_label)

    rename_candidates = {
        "start_ms": ["start_ms", "start", "start_time", "start_time_ms"],
        "end_ms": ["end_ms", "end", "end_time", "end_time_ms"],
        "duration_ms": ["duration_ms", "dur_ms", "duration", "sacc_dur", "sacc_duration"],
        "ampl": ["ampl", "amplitude", "sacc_ampl", "saccade_ampl"],
        "avg_vel": ["avg_vel", "average_velocity", "mean_vel", "sacc_avg_vel"],
        "peak_vel": ["peak_vel", "peak_velocity", "max_vel", "sacc_peak_vel"],
        "start_x": ["start_x", "x_start", "sacc_start_x", "orig_start_x"],
        "start_y": ["start_y", "y_start", "sacc_start_y", "orig_start_y"],
        "end_x": ["end_x", "x_end", "sacc_end_x", "orig_end_x"],
        "end_y": ["end_y", "y_end", "sacc_end_y", "orig_end_y"],
    }

    for target_col, candidates in rename_candidates.items():
        if target_col not in df.columns:
            source_col = find_first_column(df, candidates)
            if source_col is not None:
                df[target_col] = df[source_col]

    numeric_cols = [
        "start_ms",
        "end_ms",
        "duration_ms",
        "ampl",
        "avg_vel",
        "peak_vel",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "start_x" in df.columns and "end_x" in df.columns:
        df["sacc_dx"] = df["end_x"] - df["start_x"]
        df["sacc_abs_dx"] = df["sacc_dx"].abs()
        df["is_regression"] = (df["sacc_dx"] < 0).astype(int)
        df["is_progression"] = (df["sacc_dx"] > 0).astype(int)

    if "start_y" in df.columns and "end_y" in df.columns:
        df["sacc_dy"] = df["end_y"] - df["start_y"]
        df["sacc_abs_dy"] = df["sacc_dy"].abs()

    if "sacc_abs_dx" in df.columns and "sacc_abs_dy" in df.columns:
        df["is_vertical_saccade"] = (df["sacc_abs_dy"] > df["sacc_abs_dx"]).astype(int)

    return df


def load_saccade_files(
    saccade_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Load all saccade files if provided.
    """
    if saccade_dir is None:
        return pd.DataFrame()

    if not saccade_dir.exists():
        print(f"Warning: saccade directory does not exist: {saccade_dir}")
        return pd.DataFrame()

    csv_files = sorted(saccade_dir.rglob("*.csv"))

    if not csv_files:
        print(f"Warning: no saccade CSV files found in: {saccade_dir}")
        return pd.DataFrame()

    tables: List[pd.DataFrame] = []

    for path in csv_files:
        label_from_path = infer_label_from_path(path)

        try:
            raw_df = read_csv_safely(path)
            sacc_df = clean_saccade_df(
                raw_df,
                path=path,
                inferred_label=label_from_path,
            )
        except Exception as exc:
            print(f"Warning: skipping saccade file {path.name}: {exc}")
            continue

        tables.append(sacc_df)

        subject_preview = sacc_df["subject_id"].dropna().iloc[0]
        task_preview = sacc_df["task_prefix"].dropna().iloc[0]
        stim_preview = sacc_df["stimfile"].dropna().iloc[0]

        print(
            f"Loaded saccade file: {path.name} | "
            f"subject={subject_preview}, task={task_preview}, "
            f"stimfile={stim_preview}, saccades={len(sacc_df)}"
        )

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


# =============================================================================
# Metric cleaning
# =============================================================================

def clean_metric_df(
    df: pd.DataFrame,
    path: Path,
    inferred_label: Optional[int],
) -> pd.DataFrame:
    """
    Clean one precomputed metrics dataframe.

    This is intentionally flexible because metrics files can vary.

    It expects at least:
    - sid / subject_id / participant_id
    - task or stimfile or task information in filename
    """
    df = df.copy()

    sid_col = find_first_column(
        df,
        ["sid", "subject_id", "subj_id", "participant_id", "participant"],
    )

    if sid_col is None:
        raise ValueError(f"{path} is not a valid metric file. Missing sid/subject_id.")

    df["sid"] = df[sid_col].astype(str)
    df["subject_id"] = df["sid"]

    stim_col = find_first_column(df, ["stimfile", "stimulus", "stim_file"])

    if stim_col is not None:
        df["stimfile"] = df[stim_col].astype(str)
    else:
        df["stimfile"] = path.stem

    if "task" in df.columns and not df["task"].dropna().empty:
        df["task_prefix"] = df["task"].apply(normalize_task_name)
    elif stim_col is not None:
        df["task_prefix"] = df["stimfile"].apply(infer_task_from_stimfile)
    else:
        df["task_prefix"] = infer_task_from_path(path)

    if inferred_label is not None:
        df["label_from_path"] = int(inferred_label)

    excluded_cols = {
        "sid",
        "subject_id",
        "task",
        "task_prefix",
        "stimfile",
        "stimulus",
        "stim_file",
        "label",
        "label_from_path",
    }

    df = convert_possible_numeric_columns(df, excluded_cols=excluded_cols)

    return df


def load_metric_files(
    metrics_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Load all precomputed metric files if provided.
    """
    if metrics_dir is None:
        return pd.DataFrame()

    if not metrics_dir.exists():
        print(f"Warning: metrics directory does not exist: {metrics_dir}")
        return pd.DataFrame()

    csv_files = sorted(metrics_dir.rglob("*.csv"))

    if not csv_files:
        print(f"Warning: no metric CSV files found in: {metrics_dir}")
        return pd.DataFrame()

    tables: List[pd.DataFrame] = []

    for path in csv_files:
        label_from_path = infer_label_from_path(path)

        try:
            raw_df = read_csv_safely(path)
            metric_df = clean_metric_df(
                raw_df,
                path=path,
                inferred_label=label_from_path,
            )
        except Exception as exc:
            print(f"Warning: skipping metric file {path.name}: {exc}")
            continue

        metric_df["metric_source_file"] = path.name
        tables.append(metric_df)

        subject_preview = metric_df["subject_id"].dropna().iloc[0]
        task_preview = metric_df["task_prefix"].dropna().iloc[0]

        print(
            f"Loaded metric file: {path.name} | "
            f"subject={subject_preview}, task={task_preview}, rows={len(metric_df)}"
        )

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables, ignore_index=True)


# =============================================================================
# Aggregation helpers
# =============================================================================

def safe_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return 0.0
    return float(df[col].mean())


def safe_std(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return 0.0
    return float(df[col].std(ddof=0))


def safe_median(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return 0.0
    return float(df[col].median())


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return 0.0
    return float(df[col].sum())


def normalize_series(s: pd.Series) -> pd.Series:
    """
    Min-max normalize a numeric series safely.
    """
    if s.empty:
        return s

    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or math.isclose(min_val, max_val):
        return pd.Series([0.0] * len(s), index=s.index)

    return (s - min_val) / (max_val - min_val)


def classify_unit(content: str, content_length: int, has_diacritic_value: int) -> str:
    """
    Classify textual AOI into interpretable difficulty categories.
    """
    if content_length <= 3:
        return "short_unit"

    if content_length >= 6 or has_diacritic_value == 1:
        return "complex_or_long_unit"

    return "regular_unit"


def build_aoi_stats(matched_fix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build AOI-level statistics for one subject-task.

    Used both for feature aggregation and explanation profiles.
    """
    if matched_fix_df.empty:
        return pd.DataFrame()

    group_cols = ["stimfile", "aoi_subline", "content"]

    if "name" in matched_fix_df.columns:
        group_cols.append("name")

    aoi_stats = (
        matched_fix_df.groupby(group_cols, dropna=False)
        .agg(
            n_fix_aoi=("duration_ms", "count"),
            sum_fix_dur_aoi=("duration_ms", "sum"),
            mean_fix_dur_aoi=("duration_ms", "mean"),
            first_fix_dur_aoi=("duration_ms", "first"),
            content_length=("content_length", "first"),
            num_vowels=("num_vowels", "first"),
            num_consonants=("num_consonants", "first"),
            has_diacritic=("has_diacritic", "first"),
            vowel_ratio=("vowel_ratio", "first"),
        )
        .reset_index()
    )

    aoi_stats["unit_category"] = aoi_stats.apply(
        lambda row: classify_unit(
            content=row["content"],
            content_length=int(row["content_length"]),
            has_diacritic_value=int(row["has_diacritic"]),
        ),
        axis=1,
    )

    # Explanation-only score, not a clinical score.
    aoi_stats["score_mean_fix"] = normalize_series(aoi_stats["mean_fix_dur_aoi"])
    aoi_stats["score_sum_fix"] = normalize_series(aoi_stats["sum_fix_dur_aoi"])
    aoi_stats["score_n_fix"] = normalize_series(aoi_stats["n_fix_aoi"])

    aoi_stats["difficulty_score"] = (
        0.4 * aoi_stats["score_mean_fix"]
        + 0.4 * aoi_stats["score_sum_fix"]
        + 0.2 * aoi_stats["score_n_fix"]
    )

    return aoi_stats


def estimate_reading_time_ms(
    original_fix_df: pd.DataFrame,
    saccade_df: pd.DataFrame,
    metric_df: pd.DataFrame,
) -> float:
    """
    Estimate total reading time from fixation, saccade, or metric timestamps.

    If precomputed metrics contain an obvious reading time/duration column, that
    value is not directly trusted here because naming can vary. Instead, timestamp
    range is used where possible. Precomputed metrics are still added as features.
    """
    start_values: List[float] = []
    end_values: List[float] = []

    for df in [original_fix_df, saccade_df, metric_df]:
        if df.empty:
            continue

        if "start_ms" in df.columns and not df["start_ms"].dropna().empty:
            start_values.append(float(df["start_ms"].min()))

        if "end_ms" in df.columns and not df["end_ms"].dropna().empty:
            end_values.append(float(df["end_ms"].max()))

    if not start_values or not end_values:
        return 0.0

    return max(0.0, max(end_values) - min(start_values))


def summarize_saccades(
    saccade_df: pd.DataFrame,
    total_reading_time_ms: float,
) -> Dict[str, float]:
    """
    Summarize saccade-level features for one subject-task.
    """
    features: Dict[str, float] = {}

    features["n_sacc_total"] = float(len(saccade_df))

    total_time_sec = total_reading_time_ms / 1000.0 if total_reading_time_ms > 0 else 0.0
    features["saccade_rate_per_sec"] = (
        float(len(saccade_df) / total_time_sec) if total_time_sec > 0 else 0.0
    )

    if saccade_df.empty:
        return features

    numeric_summary_cols = {
        "duration_ms": "sacc_duration_ms",
        "ampl": "sacc_ampl",
        "avg_vel": "sacc_avg_vel",
        "peak_vel": "sacc_peak_vel",
        "sacc_dx": "sacc_dx",
        "sacc_dy": "sacc_dy",
        "sacc_abs_dx": "sacc_abs_dx",
        "sacc_abs_dy": "sacc_abs_dy",
    }

    for col, feature_base in numeric_summary_cols.items():
        if col in saccade_df.columns:
            features[f"{feature_base}_mean"] = safe_mean(saccade_df, col)
            features[f"{feature_base}_std"] = safe_std(saccade_df, col)
            features[f"{feature_base}_median"] = safe_median(saccade_df, col)

    if "ampl" in saccade_df.columns:
        features["scanpath_length"] = safe_sum(saccade_df, "ampl")
    elif "sacc_abs_dx" in saccade_df.columns and "sacc_abs_dy" in saccade_df.columns:
        temp = saccade_df.copy()
        temp["approx_step"] = (temp["sacc_abs_dx"] ** 2 + temp["sacc_abs_dy"] ** 2) ** 0.5
        features["scanpath_length"] = safe_sum(temp, "approx_step")
    else:
        features["scanpath_length"] = 0.0

    if "is_regression" in saccade_df.columns:
        features["regression_rate"] = safe_mean(saccade_df, "is_regression")
        regressions = saccade_df[saccade_df["is_regression"] == 1]

        if "ampl" in regressions.columns:
            features["mean_regression_amplitude"] = safe_mean(regressions, "ampl")
        elif "sacc_abs_dx" in regressions.columns:
            features["mean_regression_amplitude"] = safe_mean(regressions, "sacc_abs_dx")
        else:
            features["mean_regression_amplitude"] = 0.0
    else:
        features["regression_rate"] = 0.0
        features["mean_regression_amplitude"] = 0.0

    if "is_progression" in saccade_df.columns:
        features["progression_rate"] = safe_mean(saccade_df, "is_progression")
        progressions = saccade_df[saccade_df["is_progression"] == 1]

        if "ampl" in progressions.columns:
            features["mean_progression_amplitude"] = safe_mean(progressions, "ampl")
        elif "sacc_abs_dx" in progressions.columns:
            features["mean_progression_amplitude"] = safe_mean(progressions, "sacc_abs_dx")
        else:
            features["mean_progression_amplitude"] = 0.0
    else:
        features["progression_rate"] = 0.0
        features["mean_progression_amplitude"] = 0.0

    if "is_vertical_saccade" in saccade_df.columns:
        features["vertical_saccade_rate"] = safe_mean(saccade_df, "is_vertical_saccade")
    else:
        features["vertical_saccade_rate"] = 0.0

    return features


def summarize_precomputed_metrics(
    metric_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Summarize precomputed metrics for one subject-task.

    Supports metrics files that may be:
    - one row per subject-task,
    - one row per trial,
    - one row per stimulus,
    - one row per segment.

    Numeric columns are aggregated as:
    - mean
    - std
    - median
    - sum for time/count/duration-like columns
    """
    features: Dict[str, float] = {}

    if metric_df.empty:
        return features

    excluded_cols = {
        "id",
        "sid",
        "subject_id",
        "participant_id",
        "participant",
        "label",
        "label_from_path",
        "task",
        "task_prefix",
        "stimfile",
        "stimulus",
        "stim_file",
        "trialid",
        "trial_id",
        "line",
        "part",
        "column",
        "metric_source_file",
    }

    numeric_cols = []

    for col in metric_df.columns:
        if col in excluded_cols:
            continue

        if pd.api.types.is_numeric_dtype(metric_df[col]):
            numeric_cols.append(col)

    for col in numeric_cols:
        clean_col = sanitize_feature_name(col)
        feature_base = f"metric_{clean_col}"

        features[f"{feature_base}_mean"] = safe_mean(metric_df, col)
        features[f"{feature_base}_std"] = safe_std(metric_df, col)
        features[f"{feature_base}_median"] = safe_median(metric_df, col)

        lower = col.lower()

        if any(k in lower for k in [
            "time",
            "duration",
            "dur",
            "count",
            "num",
            "n_",
            "nfix",
            "nsacc",
            "fix",
            "sacc",
            "total",
            "sum",
        ]):
            features[f"{feature_base}_sum"] = safe_sum(metric_df, col)

    features["metric_rows"] = float(len(metric_df))

    return features


def summarize_subject_task(
    original_fix_df: pd.DataFrame,
    matched_fix_df: pd.DataFrame,
    saccade_df: Optional[pd.DataFrame] = None,
    metric_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Create one feature vector for one subject-task.

    Includes:
    1. Global fixation features
    2. Precomputed metrics
    3. Optional saccade + scanpath/global reading metrics
    4. AOI-level fixation features
    5. Linguistic features
    6. Eye-language interaction features
    """
    if saccade_df is None:
        saccade_df = pd.DataFrame()

    if metric_df is None:
        metric_df = pd.DataFrame()

    features: Dict[str, float] = {}

    total_reading_time_ms = estimate_reading_time_ms(
        original_fix_df=original_fix_df,
        saccade_df=saccade_df,
        metric_df=metric_df,
    )

    total_reading_time_sec = (
        total_reading_time_ms / 1000.0 if total_reading_time_ms > 0 else 0.0
    )

    # -------------------------------------------------------------------------
    # 1. Global fixation and reading-time features
    # -------------------------------------------------------------------------
    features["total_reading_time_ms"] = float(total_reading_time_ms)
    features["total_reading_time_sec"] = float(total_reading_time_sec)

    features["n_fix_total"] = float(len(original_fix_df))
    features["n_fix_matched_aoi"] = float(len(matched_fix_df))
    features["aoi_match_rate"] = (
        float(len(matched_fix_df) / len(original_fix_df))
        if len(original_fix_df) > 0
        else 0.0
    )

    features["fixation_rate_per_sec"] = (
        float(len(original_fix_df) / total_reading_time_sec)
        if total_reading_time_sec > 0
        else 0.0
    )

    features["total_fixation_duration"] = safe_sum(original_fix_df, "duration_ms")
    features["fixation_time_ratio"] = (
        features["total_fixation_duration"] / total_reading_time_ms
        if total_reading_time_ms > 0
        else 0.0
    )

    features["fix_duration_mean"] = safe_mean(original_fix_df, "duration_ms")
    features["fix_duration_std"] = safe_std(original_fix_df, "duration_ms")
    features["fix_duration_median"] = safe_median(original_fix_df, "duration_ms")

    features["fix_x_mean"] = safe_mean(original_fix_df, "fix_x")
    features["fix_x_std"] = safe_std(original_fix_df, "fix_x")
    features["fix_y_mean"] = safe_mean(original_fix_df, "fix_y")
    features["fix_y_std"] = safe_std(original_fix_df, "fix_y")

    if "disp_x" in original_fix_df.columns:
        features["disp_x_mean"] = safe_mean(original_fix_df, "disp_x")
        features["disp_x_std"] = safe_std(original_fix_df, "disp_x")

    if "disp_y" in original_fix_df.columns:
        features["disp_y_mean"] = safe_mean(original_fix_df, "disp_y")
        features["disp_y_std"] = safe_std(original_fix_df, "disp_y")

    # -------------------------------------------------------------------------
    # 2. Precomputed metrics
    # -------------------------------------------------------------------------
    metric_features = summarize_precomputed_metrics(metric_df)
    features.update(metric_features)

    # -------------------------------------------------------------------------
    # 3. Optional saccade / scanpath metrics
    # -------------------------------------------------------------------------
    saccade_features = summarize_saccades(
        saccade_df=saccade_df,
        total_reading_time_ms=total_reading_time_ms,
    )
    features.update(saccade_features)

    if matched_fix_df.empty:
        return features

    # -------------------------------------------------------------------------
    # 4. AOI-level fixation features
    # -------------------------------------------------------------------------
    aoi_stats = build_aoi_stats(matched_fix_df)

    if aoi_stats.empty:
        return features

    features["n_unique_aoi_fixated"] = float(len(aoi_stats))
    features["n_fix_aoi_mean"] = safe_mean(aoi_stats, "n_fix_aoi")
    features["n_fix_aoi_std"] = safe_std(aoi_stats, "n_fix_aoi")

    features["sum_fix_dur_aoi_mean"] = safe_mean(aoi_stats, "sum_fix_dur_aoi")
    features["sum_fix_dur_aoi_std"] = safe_std(aoi_stats, "sum_fix_dur_aoi")

    features["mean_fix_dur_aoi_mean"] = safe_mean(aoi_stats, "mean_fix_dur_aoi")
    features["mean_fix_dur_aoi_std"] = safe_std(aoi_stats, "mean_fix_dur_aoi")

    features["first_fix_dur_aoi_mean"] = safe_mean(aoi_stats, "first_fix_dur_aoi")
    features["first_fix_dur_aoi_std"] = safe_std(aoi_stats, "first_fix_dur_aoi")

    # -------------------------------------------------------------------------
    # 5. Linguistic features
    # -------------------------------------------------------------------------
    linguistic_cols = [
        "content_length",
        "num_vowels",
        "num_consonants",
        "has_diacritic",
        "vowel_ratio",
    ]

    for col in linguistic_cols:
        features[f"{col}_mean"] = safe_mean(aoi_stats, col)
        features[f"{col}_std"] = safe_std(aoi_stats, col)

    features["long_unit_rate"] = float((aoi_stats["content_length"] >= 6).mean())
    features["short_unit_rate"] = float((aoi_stats["content_length"] <= 3).mean())
    features["diacritic_unit_rate"] = float(aoi_stats["has_diacritic"].mean())

    # -------------------------------------------------------------------------
    # 6. Eye-language interaction features
    # -------------------------------------------------------------------------
    long_units = aoi_stats[aoi_stats["content_length"] >= 6]
    short_units = aoi_stats[aoi_stats["content_length"] <= 3]
    complex_units = aoi_stats[
        (aoi_stats["content_length"] >= 6) | (aoi_stats["has_diacritic"] == 1)
    ]

    features["mean_fix_dur_long_units"] = safe_mean(
        long_units, "mean_fix_dur_aoi"
    )
    features["mean_fix_dur_short_units"] = safe_mean(
        short_units, "mean_fix_dur_aoi"
    )
    features["mean_fix_dur_complex_units"] = safe_mean(
        complex_units, "mean_fix_dur_aoi"
    )

    features["sum_fix_dur_long_units"] = safe_mean(
        long_units, "sum_fix_dur_aoi"
    )
    features["sum_fix_dur_short_units"] = safe_mean(
        short_units, "sum_fix_dur_aoi"
    )
    features["sum_fix_dur_complex_units"] = safe_mean(
        complex_units, "sum_fix_dur_aoi"
    )

    features["n_fix_long_units"] = safe_mean(long_units, "n_fix_aoi")
    features["n_fix_short_units"] = safe_mean(short_units, "n_fix_aoi")
    features["n_fix_complex_units"] = safe_mean(complex_units, "n_fix_aoi")

    # Reading-load metrics: useful especially for MeaningfulText.
    features["complex_unit_fixation_load"] = safe_sum(complex_units, "sum_fix_dur_aoi")
    features["long_unit_fixation_load"] = safe_sum(long_units, "sum_fix_dur_aoi")
    features["short_unit_fixation_load"] = safe_sum(short_units, "sum_fix_dur_aoi")

    if features["total_fixation_duration"] > 0:
        features["complex_unit_fixation_load_ratio"] = (
            features["complex_unit_fixation_load"] / features["total_fixation_duration"]
        )
        features["long_unit_fixation_load_ratio"] = (
            features["long_unit_fixation_load"] / features["total_fixation_duration"]
        )
        features["short_unit_fixation_load_ratio"] = (
            features["short_unit_fixation_load"] / features["total_fixation_duration"]
        )
    else:
        features["complex_unit_fixation_load_ratio"] = 0.0
        features["long_unit_fixation_load_ratio"] = 0.0
        features["short_unit_fixation_load_ratio"] = 0.0

    return features


def build_specific_difficulty_profile(
    subject_id: str,
    task_prefix: str,
    matched_fix_df: pd.DataFrame,
    top_k: int = 8,
) -> Dict:
    """
    Build explanation-only AOI profile for one subject-task.

    This is not used for model training. It is used later by specialist agents
    to say which concrete textual units/AOIs showed effortful reading behavior.
    """
    aoi_stats = build_aoi_stats(matched_fix_df)

    if aoi_stats.empty:
        return {
            "subject_id": subject_id,
            "task_prefix": task_prefix,
            "top_difficult_units": [],
            "difficulty_by_category": {},
        }

    top_units = (
        aoi_stats.sort_values("difficulty_score", ascending=False)
        .head(top_k)
        .copy()
    )

    difficulty_by_category = {}

    for category, g in aoi_stats.groupby("unit_category"):
        difficulty_by_category[category] = {
            "n_units": int(len(g)),
            "mean_fix_dur_aoi": float(g["mean_fix_dur_aoi"].mean()),
            "sum_fix_dur_aoi": float(g["sum_fix_dur_aoi"].mean()),
            "n_fix_aoi": float(g["n_fix_aoi"].mean()),
            "mean_difficulty_score": float(g["difficulty_score"].mean()),
        }

    records = []

    for _, row in top_units.iterrows():
        records.append(
            {
                "content": str(row["content"]),
                "aoi_subline": str(row["aoi_subline"]),
                "stimfile": str(row["stimfile"]),
                "unit_category": str(row["unit_category"]),
                "content_length": int(row["content_length"]),
                "has_diacritic": int(row["has_diacritic"]),
                "n_fix_aoi": int(row["n_fix_aoi"]),
                "sum_fix_dur_aoi": float(row["sum_fix_dur_aoi"]),
                "mean_fix_dur_aoi": float(row["mean_fix_dur_aoi"]),
                "first_fix_dur_aoi": float(row["first_fix_dur_aoi"]),
                "difficulty_score": float(row["difficulty_score"]),
            }
        )

    return {
        "subject_id": subject_id,
        "task_prefix": task_prefix,
        "top_difficult_units": records,
        "difficulty_by_category": difficulty_by_category,
    }


def prefix_features(features: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in features.items()}


# =============================================================================
# Main feature-building logic
# =============================================================================

def load_and_merge_fixation_files(
    fixation_dir: Path,
    aoi_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load all fixation files, clean them, merge with AOI, and infer labels if possible.
    """
    fixation_files = sorted(fixation_dir.rglob("*.csv"))

    if not fixation_files:
        raise ValueError(f"No fixation CSV files found in: {fixation_dir}")

    all_fixations: List[pd.DataFrame] = []
    all_matched: List[pd.DataFrame] = []
    inferred_labels: Dict[str, int] = {}

    for path in fixation_files:
        label_from_path = infer_label_from_path(path)

        raw_df = read_csv_safely(path)
        fix_df = clean_fixation_df(
            raw_df,
            path=path,
            inferred_label=label_from_path,
        )

        subject_id = str(fix_df["subject_id"].dropna().iloc[0])

        if label_from_path is not None:
            if (
                subject_id in inferred_labels
                and inferred_labels[subject_id] != label_from_path
            ):
                raise ValueError(
                    f"Conflicting labels inferred for subject {subject_id}: "
                    f"{inferred_labels[subject_id]} vs {label_from_path}"
                )
            inferred_labels[subject_id] = label_from_path

        matched_df = merge_fixations_with_aoi_content(
            fix_df=fix_df,
            aoi_df=aoi_df,
        )

        all_fixations.append(fix_df)

        if not matched_df.empty:
            all_matched.append(matched_df)

        task_preview = fix_df["task_prefix"].dropna().iloc[0]
        stim_preview = fix_df["stimfile"].dropna().iloc[0]

        print(
            f"Processed fixation file: {path.name} | "
            f"subject={subject_id}, label_from_path={label_from_path}, "
            f"task={task_preview}, stimfile={stim_preview}, "
            f"fixations={len(fix_df)}, matched_aoi_fixations={len(matched_df)}"
        )

    all_fixations_df = pd.concat(all_fixations, ignore_index=True)

    if all_matched:
        all_matched_df = pd.concat(all_matched, ignore_index=True)
    else:
        all_matched_df = pd.DataFrame()

    return all_fixations_df, all_matched_df, inferred_labels


def build_feature_table(
    fixation_dir: Path,
    saccade_dir: Optional[Path],
    metrics_dir: Optional[Path],
    aoi_dir: Path,
    labels_csv: Optional[Path],
    subject_col: str,
    label_col: str,
    top_k_aoi: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build final subject-level feature table.

    Output feature CSV format:
    subject_id,label,syll_*,mean_*,pseudo_*

    Output AOI profile JSON format:
    {
      "1065": {
        "syll": {...},
        "mean": {...},
        "pseudo": {...}
      }
    }
    """
    labels_df = load_labels_csv(
        labels_csv=labels_csv,
        subject_col=subject_col,
        label_col=label_col,
    )

    aoi_df = load_aoi_tables(aoi_dir)

    all_fixations_df, all_matched_df, inferred_labels = load_and_merge_fixation_files(
        fixation_dir=fixation_dir,
        aoi_df=aoi_df,
    )

    all_saccades_df = load_saccade_files(saccade_dir)
    all_metrics_df = load_metric_files(metrics_dir)

    subject_rows: Dict[str, Dict[str, float]] = {}
    aoi_profiles: Dict[str, Dict] = {}

    for (subject_id, task_prefix), original_group in all_fixations_df.groupby(
        ["subject_id", "task_prefix"]
    ):
        if all_matched_df.empty:
            matched_group = pd.DataFrame()
        else:
            matched_group = all_matched_df[
                (all_matched_df["subject_id"] == subject_id)
                & (all_matched_df["task_prefix"] == task_prefix)
            ]

        if all_saccades_df.empty:
            saccade_group = pd.DataFrame()
        else:
            saccade_group = all_saccades_df[
                (all_saccades_df["subject_id"] == subject_id)
                & (all_saccades_df["task_prefix"] == task_prefix)
            ]

        if all_metrics_df.empty:
            metric_group = pd.DataFrame()
        else:
            metric_group = all_metrics_df[
                (all_metrics_df["subject_id"] == subject_id)
                & (all_metrics_df["task_prefix"] == task_prefix)
            ]

        features = summarize_subject_task(
            original_fix_df=original_group,
            matched_fix_df=matched_group,
            saccade_df=saccade_group,
            metric_df=metric_group,
        )

        profile = build_specific_difficulty_profile(
            subject_id=subject_id,
            task_prefix=task_prefix,
            matched_fix_df=matched_group,
            top_k=top_k_aoi,
        )

        if subject_id not in aoi_profiles:
            aoi_profiles[subject_id] = {}

        aoi_profiles[subject_id][task_prefix] = profile

        features = prefix_features(features, task_prefix)

        if subject_id not in subject_rows:
            subject_rows[subject_id] = {"subject_id": subject_id}

        subject_rows[subject_id].update(features)

    feature_df = pd.DataFrame(subject_rows.values())
    feature_df["subject_id"] = feature_df["subject_id"].astype(str)

    if labels_df is not None:
        final_df = labels_df.merge(feature_df, on="subject_id", how="inner")
    else:
        if not inferred_labels:
            raise ValueError(
                "No labels_csv was provided and no labels could be inferred from paths. "
                "Please either provide --labels-csv or encode 0/1/non-dyslexic/"
                "dyslexic in filenames or folders."
            )

        label_rows = [
            {"subject_id": sid, "label": label}
            for sid, label in inferred_labels.items()
        ]

        labels_df = pd.DataFrame(label_rows)
        labels_df["subject_id"] = labels_df["subject_id"].astype(str)
        labels_df["label"] = labels_df["label"].astype(int)

        missing_subjects = sorted(
            set(feature_df["subject_id"]) - set(labels_df["subject_id"])
        )

        if missing_subjects:
            raise ValueError(
                "Some subjects do not have inferred labels. "
                f"Missing labels for subjects: {missing_subjects[:10]}"
            )

        final_df = labels_df.merge(feature_df, on="subject_id", how="inner")

    cols = list(final_df.columns)
    ordered_cols = ["subject_id", "label"] + [
        col for col in cols if col not in ["subject_id", "label"]
    ]
    final_df = final_df[ordered_cols]

    return final_df, aoi_profiles


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    fixation_dir = Path(args.fixation_dir)
    saccade_dir = Path(args.saccade_dir) if args.saccade_dir else None
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else None
    aoi_dir = Path(args.aoi_dir)
    output_csv = Path(args.output_csv)
    labels_csv = Path(args.labels_csv) if args.labels_csv else None

    if args.aoi_profile_json:
        aoi_profile_json = Path(args.aoi_profile_json)
    else:
        aoi_profile_json = output_csv.with_name("aoi_profiles.json")

    final_df, aoi_profiles = build_feature_table(
        fixation_dir=fixation_dir,
        saccade_dir=saccade_dir,
        metrics_dir=metrics_dir,
        aoi_dir=aoi_dir,
        labels_csv=labels_csv,
        subject_col=args.subject_col,
        label_col=args.label_col,
        top_k_aoi=args.top_k_aoi,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    aoi_profile_json.parent.mkdir(parents=True, exist_ok=True)
    aoi_profile_json.write_text(
        json.dumps(aoi_profiles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Saved feature table to: {output_csv}")
    print(f"Saved AOI difficulty profiles to: {aoi_profile_json}")
    print(f"Shape: {final_df.shape}")
    print(f"Number of subjects: {final_df['subject_id'].nunique()}")

    if final_df.empty:
        print(
            "Warning: output is empty. Check whether labels.csv subject_id values "
            "match sid values in fixation files."
        )
        return

    print("\nLabel distribution:")
    print(final_df["label"].value_counts().sort_index())

    feature_cols = [
        col for col in final_df.columns
        if col not in ["subject_id", "label"]
    ]

    n_linguistic = sum(
        any(k in col for k in [
            "content_length",
            "num_vowels",
            "num_consonants",
            "vowel_ratio",
            "has_diacritic",
            "long_unit_rate",
            "short_unit_rate",
            "diacritic_unit_rate",
        ])
        for col in feature_cols
    )

    n_interaction = sum(
        any(k in col for k in [
            "long_units",
            "short_units",
            "complex_units",
            "fixation_load",
        ])
        for col in feature_cols
    )

    n_saccade = sum(
        any(k in col for k in [
            "sacc",
            "scanpath",
            "regression",
            "progression",
            "vertical_saccade",
        ])
        for col in feature_cols
    )

    n_precomputed_metric = sum(
        "_metric_" in col or col.endswith("_metric_rows")
        for col in feature_cols
    )

    n_global_metric = sum(
        any(k in col for k in [
            "total_reading_time",
            "fixation_rate",
            "saccade_rate",
            "fixation_time_ratio",
            "total_fixation_duration",
        ])
        for col in feature_cols
    )

    n_eye = len(feature_cols) - n_linguistic - n_interaction

    print("\nFeature summary:")
    print(f"All feature columns          : {len(feature_cols)}")
    print(f"Eye-tracking features        : {n_eye}")
    print(f"Global reading metrics       : {n_global_metric}")
    print(f"Saccade/scanpath features    : {n_saccade}")
    print(f"Precomputed metric features  : {n_precomputed_metric}")
    print(f"Linguistic features          : {n_linguistic}")
    print(f"Interaction features         : {n_interaction}")


if __name__ == "__main__":
    main()