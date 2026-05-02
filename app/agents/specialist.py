from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    MEANINGFUL_MODEL_PATH,
    MEANINGFUL_PREFIX,
    PSEUDOTEXT_MODEL_PATH,
    PSEUDOTEXT_PREFIX,
    SYLLABLES_MODEL_PATH,
    SYLLABLES_PREFIX,
    TASK_MEANINGFUL,
    TASK_PSEUDOTEXT,
    TASK_SYLLABLES,
)
from app.models.explainability import split_evidence_by_type
from app.models.inference import run_task_inference
from app.schemas import DifficultyItem, SpecialistReport
from app.state import GraphState


# =============================================================================
# AOI profile loading
# =============================================================================

def normalize_prefix_for_profile(prefix: str) -> str:
    """
    Convert model feature prefix into AOI profile key.

    Examples:
    - syll_   -> syll
    - mean_   -> mean
    - pseudo_ -> pseudo
    """
    return prefix.replace("_", "")


def load_task_aoi_profile(
    aoi_profile_json: Optional[str],
    subject_id: str,
    task_prefix: str,
) -> Dict[str, Any]:
    """
    Load AOI difficulty profile for one subject-task.

    Expected JSON structure:
    {
      "1065": {
        "syll": {...},
        "mean": {...},
        "pseudo": {...}
      }
    }
    """
    if not aoi_profile_json:
        return {}

    path = Path(aoi_profile_json)

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data.get(str(subject_id), {}).get(task_prefix, {})


# =============================================================================
# Difficulty interpretation helpers
# =============================================================================

def severity_from_score(score: float) -> str:
    """
    Convert normalized AOI difficulty score into a relative severity label.

    This is NOT a clinical severity score.
    It only ranks effortful AOI/textual units within the same subject-task.
    """
    if score >= 0.67:
        return "high"

    if score >= 0.34:
        return "medium"

    return "low"


def normalize_unit_category(category: str) -> str:
    """
    Ensure unit category is compatible with DifficultyItem schema.
    """
    allowed = {
        "short_unit",
        "regular_unit",
        "complex_or_long_unit",
        "unknown",
    }

    if category in allowed:
        return category

    return "unknown"


def build_specific_difficulties(
    aoi_profile: Dict[str, Any],
    max_items: int = 3,
) -> List[DifficultyItem]:
    """
    Build concrete AOI/textual-unit difficulties for a specialist report.

    Important:
    This function does NOT generate intervention suggestions.
    Specialist agents only report observed difficulty evidence.
    Board Agent is responsible for converting these observations into
    expert-facing intervention_suggestions.
    """
    units = aoi_profile.get("top_difficult_units", [])[:max_items]

    difficulties: List[DifficultyItem] = []

    for unit in units:
        content = str(unit.get("content", ""))
        category = normalize_unit_category(str(unit.get("unit_category", "unknown")))
        location = str(unit.get("aoi_subline", "unknown"))
        stimfile = str(unit.get("stimfile", ""))

        difficulty_score = float(unit.get("difficulty_score", 0.0))
        mean_fix = float(unit.get("mean_fix_dur_aoi", 0.0))
        sum_fix = float(unit.get("sum_fix_dur_aoi", 0.0))
        n_fix = int(unit.get("n_fix_aoi", 0))
        content_length = int(unit.get("content_length", 0))
        has_diacritic = int(unit.get("has_diacritic", 0))

        evidence = (
            f"AOI '{content}' at {location}"
            f"{f' ({stimfile})' if stimfile else ''}: "
            f"mean fixation = {mean_fix:.1f} ms, "
            f"total fixation = {sum_fix:.1f} ms, "
            f"fixations = {n_fix}, "
            f"length = {content_length}, "
            f"diacritic = {has_diacritic}."
        )

        difficulties.append(
            DifficultyItem(
                unit=content,
                unit_category=category,
                location=location,
                evidence=evidence,
                severity=severity_from_score(difficulty_score),
            )
        )

    return difficulties


def count_difficulty_categories(
    specific_difficulties: List[DifficultyItem],
) -> Dict[str, int]:
    counts = {
        "short_unit": 0,
        "regular_unit": 0,
        "complex_or_long_unit": 0,
        "unknown": 0,
    }

    for item in specific_difficulties:
        counts[item.unit_category] = counts.get(item.unit_category, 0) + 1

    return counts


def infer_difficulty_pattern(
    task_name: str,
    interaction_evidence,
    linguistic_evidence,
    specific_difficulties: List[DifficultyItem],
) -> str:
    """
    Infer a concise, task-level difficulty pattern.

    This combines:
    - feature-level evidence,
    - AOI-specific difficulty categories.
    """
    feature_names = [
        item.feature
        for item in list(interaction_evidence) + list(linguistic_evidence)
    ]

    joined_features = " ".join(feature_names)
    category_counts = count_difficulty_categories(specific_difficulties)

    has_short_feature = "short_units" in joined_features
    has_complex_feature = (
        "complex_units" in joined_features
        or "long_units" in joined_features
    )

    dominant_category = max(
        category_counts,
        key=lambda key: category_counts.get(key, 0),
    )

    has_short = has_short_feature or dominant_category == "short_unit"
    has_complex = has_complex_feature or dominant_category == "complex_or_long_unit"

    if task_name == "Syllables":
        if has_complex:
            return "possible difficulty with longer or orthographically complex syllabic units"
        if has_short:
            return "possible weakness in automatic recognition of short syllabic units"
        return "possible general syllable-level fluency or decoding difficulty"

    if task_name == "MeaningfulText":
        if has_complex:
            return "possible word-level fluency difficulty for longer or complex words in connected text"
        if has_short:
            return "possible inefficient processing of short textual units in meaningful context"
        return "possible connected-text reading fluency difficulty"

    if task_name == "PseudoText":
        if has_complex:
            return "possible grapheme-to-phoneme decoding difficulty for longer or complex pseudo-units"
        if has_short:
            return "possible difficulty with basic pseudo-unit decoding under reduced semantic support"
        return "possible pseudo-text decoding difficulty"

    return "unspecified reading difficulty pattern"


# =============================================================================
# Specialist explanation helpers
# =============================================================================

def format_evidence_items(items, max_items: int = 2) -> str:
    """
    Format model-level evidence items for natural-language explanation.
    """
    if not items:
        return "none"

    return ", ".join(
        [f"{item.feature} ({item.direction})" for item in items[:max_items]]
    )


def format_specific_units(
    difficulties: List[DifficultyItem],
    max_items: int = 3,
) -> str:
    """
    Format concrete AOI/textual units for a short specialist explanation.
    """
    if not difficulties:
        return "none available"

    return ", ".join(
        [
            f"'{item.unit}' at {item.location} ({item.severity})"
            for item in difficulties[:max_items]
        ]
    )


def build_specialist_explanation(
    task_name: str,
    risk_score: float,
    predicted_label: str,
    fixation_evidence,
    saccade_evidence,
    metric_evidence,
    linguistic_evidence,
    interaction_evidence,
    specific_difficulties: List[DifficultyItem],
    difficulty_pattern: str,
    global_reading_patterns: List[str],
) -> str:
    """
    Build a concise task-level specialist explanation.

    Version 2:
    - separates fixation/AOI evidence from saccade and global metric evidence
    - includes global reading behavior patterns
    - does not recommend intervention
    """
    fixation_text = format_evidence_items(fixation_evidence)
    saccade_text = format_evidence_items(saccade_evidence)
    metric_text = format_evidence_items(metric_evidence)
    ling_text = format_evidence_items(linguistic_evidence)
    interaction_text = format_evidence_items(interaction_evidence)

    unit_text = format_specific_units(specific_difficulties)

    if global_reading_patterns:
        pattern_text = " ".join(global_reading_patterns)
    else:
        pattern_text = "No clear global saccade or metric pattern was highlighted."

    # Risk-gated wording.
    if predicted_label == "low_risk" or risk_score < 0.35:
        return (
            f"{task_name} predicts low_risk (risk={risk_score:.2f}). "
            f"The model does not indicate an elevated task-level dyslexia-risk pattern. "
            f"Local effortful AOIs: {unit_text}. "
            f"Global reading behavior: {pattern_text} "
            f"Feature evidence: fixation={fixation_text}; "
            f"saccade/scanpath={saccade_text}; "
            f"global metrics={metric_text}; "
            f"linguistic={ling_text}; "
            f"eye-language interaction={interaction_text}. "
            f"These are task-specific assessment-support observations, not a diagnosis."
        )

    return (
        f"{task_name} predicts {predicted_label} "
        f"(risk={risk_score:.2f}). "
        f"Task pattern: {difficulty_pattern}. "
        f"Global reading behavior: {pattern_text} "
        f"Feature evidence: fixation={fixation_text}; "
        f"saccade/scanpath={saccade_text}; "
        f"global metrics={metric_text}; "
        f"linguistic={ling_text}; "
        f"eye-language interaction={interaction_text}. "
        f"Effortful AOIs: {unit_text}. "
        f"This is task-specific assessment-support evidence, not a diagnosis."
    )

# =============================================================================
# Specialist runner
# =============================================================================

def run_specialist(
    task_name: str,
    prefix: str,
    model_path,
    case_row: Dict[str, Any],
    subject_id: str,
    aoi_profile_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic specialist runner.

    Version 2 responsibilities:
    - task-specific model inference
    - feature-level evidence extraction
    - split evidence into fixation / saccade / global_metric / linguistic / interaction
    - AOI-specific difficulty profile loading
    - global reading pattern interpretation
    - concise specialist explanation generation
    """
    result = run_task_inference(
        case_row=case_row,
        prefix=prefix,
        model_path=model_path,
    )

    evidence_by_type = split_evidence_by_type(result["top_evidence"])

    # Backward-compatible general ET evidence.
    eye_tracking_evidence = evidence_by_type["eye_tracking"]

    # Version 2 evidence groups.
    fixation_evidence = evidence_by_type["fixation"]
    saccade_evidence = evidence_by_type["saccade"]
    metric_evidence = evidence_by_type["global_metric"]

    linguistic_evidence = evidence_by_type["linguistic"]
    interaction_evidence = evidence_by_type["interaction"]

    task_profile_key = normalize_prefix_for_profile(prefix)

    aoi_profile = load_task_aoi_profile(
        aoi_profile_json=aoi_profile_json,
        subject_id=subject_id,
        task_prefix=task_profile_key,
    )

    specific_difficulties = build_specific_difficulties(
        aoi_profile=aoi_profile,
        max_items=3,
    )

    is_low_risk_case = (
        result["predicted_label"] == "low_risk"
        or result["risk_score"] < 0.35
    )

    if is_low_risk_case:
        difficulty_pattern = (
            "no elevated task-level dyslexia-risk pattern; "
            "some locally effortful AOIs may still be observed"
        )
    else:
        difficulty_pattern = infer_difficulty_pattern(
            task_name=task_name,
            interaction_evidence=interaction_evidence,
            linguistic_evidence=linguistic_evidence,
            specific_difficulties=specific_difficulties,
        )

    global_reading_patterns = infer_global_reading_patterns(
        task_name=task_name,
        fixation_evidence=fixation_evidence,
        saccade_evidence=saccade_evidence,
        metric_evidence=metric_evidence,
        interaction_evidence=interaction_evidence,
        max_patterns=3,
    )

    explanation = build_specialist_explanation(
        task_name=task_name,
        risk_score=result["risk_score"],
        predicted_label=result["predicted_label"],
        fixation_evidence=fixation_evidence,
        saccade_evidence=saccade_evidence,
        metric_evidence=metric_evidence,
        linguistic_evidence=linguistic_evidence,
        interaction_evidence=interaction_evidence,
        specific_difficulties=specific_difficulties,
        difficulty_pattern=difficulty_pattern,
        global_reading_patterns=global_reading_patterns,
    )

    report = SpecialistReport(
        task_name=task_name,
        predicted_label=result["predicted_label"],
        risk_score=result["risk_score"],
        confidence=result["confidence"],

        # Backward-compatible field.
        top_eye_tracking_evidence=eye_tracking_evidence,

        # Version 2 evidence groups.
        top_fixation_evidence=fixation_evidence,
        top_saccade_evidence=saccade_evidence,
        top_metric_evidence=metric_evidence,

        top_linguistic_evidence=linguistic_evidence,
        top_interaction_evidence=interaction_evidence,

        difficulty_pattern=difficulty_pattern,
        global_reading_patterns=global_reading_patterns,
        specific_difficulties=specific_difficulties,
        explanation=explanation,
    )

    return report.model_dump()


# =============================================================================
# LangGraph nodes
# =============================================================================

def syllables_agent_node(state: GraphState) -> GraphState:
    """
    LangGraph node for the Syllables Agent.
    """
    model_path = state.get("syllables_model_path", SYLLABLES_MODEL_PATH)

    report = run_specialist(
        task_name=TASK_SYLLABLES,
        prefix=SYLLABLES_PREFIX,
        model_path=model_path,
        case_row=state["case_row"],
        subject_id=state["subject_id"],
        aoi_profile_json=state.get("aoi_profile_json"),
    )

    return {"syllables_report": report}


def meaningful_agent_node(state: GraphState) -> GraphState:
    """
    LangGraph node for the MeaningfulText Agent.
    """
    model_path = state.get("meaningful_model_path", MEANINGFUL_MODEL_PATH)

    report = run_specialist(
        task_name=TASK_MEANINGFUL,
        prefix=MEANINGFUL_PREFIX,
        model_path=model_path,
        case_row=state["case_row"],
        subject_id=state["subject_id"],
        aoi_profile_json=state.get("aoi_profile_json"),
    )

    return {"meaningful_report": report}


def pseudotext_agent_node(state: GraphState) -> GraphState:
    """
    LangGraph node for the PseudoText Agent.
    """
    model_path = state.get("pseudotext_model_path", PSEUDOTEXT_MODEL_PATH)

    report = run_specialist(
        task_name=TASK_PSEUDOTEXT,
        prefix=PSEUDOTEXT_PREFIX,
        model_path=model_path,
        case_row=state["case_row"],
        subject_id=state["subject_id"],
        aoi_profile_json=state.get("aoi_profile_json"),
    )

    return {"pseudotext_report": report}


def infer_global_reading_patterns(
    task_name: str,
    fixation_evidence,
    saccade_evidence,
    metric_evidence,
    interaction_evidence,
    max_patterns: int = 3,
) -> List[str]:
    """
    Infer task-level reading behavior patterns from Version 2 evidence groups.

    This is different from specific_difficulties:
    - specific_difficulties = local AOI/textual units
    - global_reading_patterns = broader reading behavior from metrics/saccades
    """
    patterns: List[str] = []

    all_features = (
        list(fixation_evidence)
        + list(saccade_evidence)
        + list(metric_evidence)
        + list(interaction_evidence)
    )

    feature_names = " ".join(
        str(item.feature).lower()
        for item in all_features
    )

    # ------------------------------------------------------------------
    # Global/precomputed metrics
    # ------------------------------------------------------------------
    if any(k in feature_names for k in ["dwell", "first_visit", "first visit"]):
        patterns.append(
            f"{task_name}: dwell-time or first-visit metrics suggest uneven early processing of textual units."
        )

    if any(k in feature_names for k in ["transit", "transition"]):
        patterns.append(
            f"{task_name}: transition/transit metrics suggest less efficient movement through the stimulus."
        )

    if any(k in feature_names for k in ["reading_time", "trial", "total_reading_time"]):
        patterns.append(
            f"{task_name}: task-level time metrics suggest slower or more effortful reading behavior."
        )

    if any(k in feature_names for k in ["revisit", "visit"]):
        patterns.append(
            f"{task_name}: visit/revisit metrics suggest possible rereading or repeated processing."
        )

    # ------------------------------------------------------------------
    # Saccade / scanpath metrics
    # ------------------------------------------------------------------
    if any(k in feature_names for k in ["regression"]):
        patterns.append(
            f"{task_name}: regression-related evidence suggests possible rereading or backward eye movements."
        )

    if any(k in feature_names for k in ["scanpath"]):
        patterns.append(
            f"{task_name}: scanpath evidence suggests altered reading-flow organization."
        )

    if any(k in feature_names for k in ["sacc", "ampl", "velocity", "avg_vel", "peak_vel"]):
        patterns.append(
            f"{task_name}: saccade features suggest altered eye-movement dynamics during reading."
        )

    # ------------------------------------------------------------------
    # Fixation / interaction global load
    # ------------------------------------------------------------------
    if any(k in feature_names for k in ["fixation_load", "complex_units", "long_units"]):
        patterns.append(
            f"{task_name}: eye-language interaction features suggest increased processing load on complex textual units."
        )

    if not patterns and fixation_evidence:
        patterns.append(
            f"{task_name}: fixation/AOI evidence suggests local reading effort, but no clear global saccade or metric pattern was highlighted."
        )

    # Remove duplicates while preserving order.
    unique_patterns: List[str] = []
    seen = set()

    for pattern in patterns:
        if pattern in seen:
            continue

        seen.add(pattern)
        unique_patterns.append(pattern)

    return unique_patterns[:max_patterns]