from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd

from app.config import TOP_K_EVIDENCE
from app.schemas import EvidenceItem

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def _unwrap_estimator(model):
    """
    If model is a sklearn Pipeline, return the final classifier step.
    Otherwise return the model itself.
    """
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        return model.named_steps["classifier"]
    return model


def _transform_input_if_needed(model, X: pd.DataFrame):
    """
    If model is a Pipeline, transform X using all steps before the classifier.
    Otherwise return X unchanged.

    This is needed because SHAP should receive the same transformed feature space
    that the final classifier actually sees.
    """
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        preprocessor_steps = [
            step for name, step in model.named_steps.items()
            if name != "classifier"
        ]

        X_transformed = X.copy()
        for step in preprocessor_steps:
            X_transformed = step.transform(X_transformed)

        return X_transformed

    return X


def _compute_shap_importance(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP signed values and absolute importances for one sample.

    Returns:
        signed_values: np.ndarray
        importances: np.ndarray
    """
    classifier = _unwrap_estimator(model)
    X_for_explainer = _transform_input_if_needed(model, X)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_for_explainer)

    if isinstance(shap_values, list):
        signed_values = np.array(shap_values[1])[0]
    else:
        arr = np.array(shap_values)
        if arr.ndim == 3:
            signed_values = arr[1][0]
        elif arr.ndim == 2:
            signed_values = arr[0]
        else:
            raise ValueError("Unexpected SHAP output shape.")

    importances = np.abs(signed_values)
    return signed_values, importances


def _compute_fallback_importance(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback when SHAP is unavailable or fails.

    Preference:
    - classifier.feature_importances_
    - otherwise uniform importance
    """
    classifier = _unwrap_estimator(model)
    n_features = X.shape[1]

    if hasattr(classifier, "feature_importances_"):
        signed_values = np.array(classifier.feature_importances_, dtype=float)
        importances = np.abs(signed_values)
    else:
        signed_values = np.ones(n_features, dtype=float)
        importances = np.ones(n_features, dtype=float)

    return signed_values, importances


def compute_top_evidence(
    model,
    X: pd.DataFrame,
    top_k: int = TOP_K_EVIDENCE,
) -> List[EvidenceItem]:
    """
    Compute the top-k evidence items for one case.

    Returns:
        A list of EvidenceItem objects.
    """
    feature_names = list(X.columns)
    feature_values = X.iloc[0].to_dict()

    if HAS_SHAP:
        try:
            signed_values, importances = _compute_shap_importance(model, X)
        except Exception:
            signed_values, importances = _compute_fallback_importance(model, X)
    else:
        signed_values, importances = _compute_fallback_importance(model, X)

    ranked_indices = np.argsort(importances)[::-1][:top_k]

    evidence_items: List[EvidenceItem] = []
    for idx in ranked_indices:
        feature_name = feature_names[idx]
        signed_val = float(signed_values[idx])

        if signed_val > 0:
            direction = "higher"
        elif signed_val < 0:
            direction = "lower"
        else:
            direction = "mixed"

        evidence_type = infer_evidence_type(feature_name)

        evidence_items.append(
            EvidenceItem(
                feature=feature_name,
                value=float(feature_values[feature_name]),
                importance=float(importances[idx]),
                evidence_type=evidence_type,
                direction=direction,
                note=make_feature_note(feature_name, evidence_type),
            )
        )

    return evidence_items


def infer_evidence_type(feature_name: str) -> str:
    """
    Infer evidence type from feature name.

    Version 2 evidence groups:
    - fixation: fixation/AOI-local eye-tracking features
    - saccade: saccade, scanpath, regression/progression features
    - global_metric: precomputed/global ET metrics such as dwell time, transit, first visit
    - linguistic: pure linguistic/content features
    - interaction: eye-tracking x linguistic-unit interaction features

    Backward-compatible:
    - eye_tracking is kept as generic fallback.
    """
    name = str(feature_name).lower()

    # ------------------------------------------------------------------
    # 1. Eye-language interaction features
    # Check first because these names often contain "fix".
    # Examples:
    # - n_fix_complex_units
    # - sum_fix_dur_short_units
    # - complex_unit_fixation_load_ratio
    # ------------------------------------------------------------------
    interaction_keywords = [
        "complex_unit",
        "complex_units",
        "long_unit",
        "long_units",
        "short_unit",
        "short_units",
        "fixation_load",
        "unit_fixation_load",
        "mean_fix_dur_long_units",
        "mean_fix_dur_short_units",
        "mean_fix_dur_complex_units",
        "sum_fix_dur_long_units",
        "sum_fix_dur_short_units",
        "sum_fix_dur_complex_units",
        "n_fix_long_units",
        "n_fix_short_units",
        "n_fix_complex_units",
    ]

    if any(keyword in name for keyword in interaction_keywords):
        return "interaction"

    # ------------------------------------------------------------------
    # 2. Linguistic/content features
    # ------------------------------------------------------------------
    linguistic_keywords = [
        "content",
        "content_length",
        "num_vowels",
        "num_consonants",
        "vowel_ratio",
        "has_diacritic",
        "diacritic",
        "long_unit_rate",
        "short_unit_rate",
        "diacritic_unit_rate",
        "linguistic",
    ]

    if any(keyword in name for keyword in linguistic_keywords):
        return "linguistic"

    # ------------------------------------------------------------------
    # 3. Saccade / scanpath features
    # ------------------------------------------------------------------
    saccade_keywords = [
        "sacc",
        "saccade",
        "scanpath",
        "regression",
        "progression",
        "vertical_saccade",
        "ampl",
        "amplitude",
        "avg_vel",
        "peak_vel",
        "velocity",
        "sacc_dx",
        "sacc_dy",
    ]

    if any(keyword in name for keyword in saccade_keywords):
        return "saccade"

    # ------------------------------------------------------------------
    # 4. Global / precomputed ET metrics
    # Check before fixation because metric names may contain "aoi" or "fix".
    # Examples:
    # - metric_n_transit_trial_median
    # - metric_dwell_time_first_visit_aoi_mean
    # - metric_sum_fix_dur_first_visit_aoi_std
    # ------------------------------------------------------------------
    global_metric_keywords = [
        "metric_",
        "dwell",
        "first_visit",
        "first visit",
        "visit",
        "transit",
        "transition",
        "revisit",
        "trial",
        "reading_time",
        "total_reading_time",
        "fixation_rate",
        "saccade_rate",
        "fixation_time_ratio",
        "total_fixation_duration",
        "duration_total",
    ]

    if any(keyword in name for keyword in global_metric_keywords):
        return "global_metric"

    # ------------------------------------------------------------------
    # 5. Fixation / AOI-local features
    # ------------------------------------------------------------------
    fixation_keywords = [
        "fix",
        "fixation",
        "aoi",
        "duration",
        "dur",
        "disp_x",
        "disp_y",
        "fix_x",
        "fix_y",
        "n_fix_total",
        "n_fix_matched_aoi",
        "aoi_match_rate",
    ]

    if any(keyword in name for keyword in fixation_keywords):
        return "fixation"

    # Generic fallback for old ET-derived features.
    return "eye_tracking"

def split_evidence_by_type(evidence_items: List[EvidenceItem]) -> Dict[str, List[EvidenceItem]]:
    """
    Split EvidenceItem objects by evidence_type.

    Backward compatibility:
    - "eye_tracking" includes generic eye_tracking plus fixation, saccade,
      and global_metric evidence.
    - New Version 2 code should use:
      fixation / saccade / global_metric / linguistic / interaction.
    """
    return {
        "eye_tracking": [
            item for item in evidence_items
            if item.evidence_type in {
                "eye_tracking",
                "fixation",
                "saccade",
                "global_metric",
            }
        ],
        "fixation": [
            item for item in evidence_items
            if item.evidence_type == "fixation"
        ],
        "saccade": [
            item for item in evidence_items
            if item.evidence_type == "saccade"
        ],
        "global_metric": [
            item for item in evidence_items
            if item.evidence_type == "global_metric"
        ],
        "linguistic": [
            item for item in evidence_items
            if item.evidence_type == "linguistic"
        ],
        "interaction": [
            item for item in evidence_items
            if item.evidence_type == "interaction"
        ],
    }

def make_feature_note(feature_name: str, evidence_type: str) -> str:
    """
    Create a human-readable note for an evidence feature.
    """
    name = str(feature_name)

    if evidence_type == "fixation":
        return (
            f"'{name}' reflects fixation or AOI-level processing and may indicate "
            "local reading effort."
        )

    if evidence_type == "saccade":
        return (
            f"'{name}' reflects saccade, scanpath, regression, progression, "
            "or reading-flow behavior."
        )

    if evidence_type == "global_metric":
        return (
            f"'{name}' reflects a global eye-tracking metric such as dwell time, "
            "first visit, transit, transition, visit behavior, or task-level reading flow."
        )

    if evidence_type == "linguistic":
        return (
            f"'{name}' reflects textual or linguistic properties of the stimulus."
        )

    if evidence_type == "interaction":
        return (
            f"'{name}' reflects interaction between eye-tracking behavior and "
            "linguistic unit properties."
        )

    return f"'{name}' is among the strongest contributors in this task."