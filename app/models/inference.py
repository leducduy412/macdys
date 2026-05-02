from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from app.config import RISK_THRESHOLD
from app.models.explainability import compute_top_evidence
from app.utils.utils import probability_to_confidence


def load_model(model_path: str | Path):
    """
    Load a trained ML model from disk.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please train and save the model first."
        )
    return joblib.load(model_path)


def _get_expected_feature_order(model) -> Optional[List[str]]:
    """
    Return the expected feature order for inference.

    For sklearn Pipeline, prefer pipeline.feature_names_in_ if available.
    If unavailable, try classifier.feature_names_in_.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "feature_names_in_"):
            return list(classifier.feature_names_in_)

    return None


def select_task_features(
    case_row: Dict[str, Any],
    prefix: str,
    expected_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract task-specific features from a single case row using a prefix.

    Example:
    - prefix='syll_' extracts columns like:
      syll_mean_fixation_duration -> mean_fixation_duration
      syll_regression_ratio -> regression_ratio

    Returns:
        A single-row DataFrame ready for model inference.
    """
    feature_dict: Dict[str, float] = {}

    for key, value in case_row.items():
        if key.startswith(prefix):
            feature_name = key[len(prefix):]
            feature_dict[feature_name] = float(value)

    if not feature_dict:
        raise ValueError(f"No features found for prefix '{prefix}'.")

    X = pd.DataFrame([feature_dict])

    if expected_order is not None:
        missing = [col for col in expected_order if col not in X.columns]
        if missing:
            raise ValueError(
                f"Missing expected features for prefix '{prefix}': {missing}"
            )
        X = X[expected_order]

    return X


def predict_risk(model, X: pd.DataFrame) -> Dict[str, Any]:
    """
    Run model inference on one case.

    Returns:
        {
            'risk_score': float,
            'predicted_label': 'low_risk' | 'high_risk',
            'confidence': float
        }
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError("The loaded model does not support predict_proba().")

    risk_score = float(model.predict_proba(X)[0, 1])
    predicted_label = "high_risk" if risk_score >= RISK_THRESHOLD else "low_risk"
    confidence = probability_to_confidence(risk_score)

    return {
        "risk_score": risk_score,
        "predicted_label": predicted_label,
        "confidence": confidence,
    }


def run_task_inference(
    case_row: Dict[str, Any],
    prefix: str,
    model_path: str | Path,
):
    """
    End-to-end inference helper for one task:
    - load model
    - select task features
    - predict risk
    - compute top evidence
    """
    model = load_model(model_path)

    expected_order = _get_expected_feature_order(model)
    X = select_task_features(case_row, prefix=prefix, expected_order=expected_order)

    prediction = predict_risk(model, X)
    top_evidence = compute_top_evidence(model, X)

    return {
        "X": X,
        **prediction,
        "top_evidence": top_evidence,
    }