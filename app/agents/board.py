from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from app.schemas import BoardReport, InterventionSuggestion
from app.state import GraphState


# =============================================================================
# LLM configuration
# =============================================================================

try:
    from app.config import OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_API_URL_HERE")
    OPENAI_MODEL = os.getenv("LLM_MODEL", "YOUR_MODEL_HERE")


BOARD_SYSTEM_PROMPT = """
You are the Assessment Board Agent in a multi-agent dyslexia assessment-support system.

Your role is NOT to diagnose dyslexia.
Your role is to synthesize evidence from three specialist agents and produce a concise,
expert-facing assessment-support report.

Specialist agents provide:
- risk_score
- predicted_label
- confidence
- fixation/AOI evidence
- saccade/scanpath evidence
- global reading metric evidence
- linguistic evidence
- eye-language interaction evidence
- difficulty_pattern
- global_reading_patterns
- specific_difficulties

Important role boundary:
- Specialist agents do NOT provide intervention suggestions.
- You are the only agent responsible for intervention_suggestions.
- Your suggestions must be linked to observed specialist evidence.
- Do not overclaim. Use language such as "may suggest", "is consistent with",
  "should be reviewed by an expert", and "assessment-support evidence".
- Never claim that the system diagnoses dyslexia.

Version 2 synthesis requirement:
You must synthesize BOTH:
1. Local AOI/textual-unit difficulties
   - specific words, syllables, pseudo-units, AOI locations
   - fixation/AOI-level effort
   - eye-language interaction on short, long, or complex units

2. Global reading behavior
   - saccade/scanpath behavior
   - regression/progression indicators
   - dwell-time or first-visit metrics
   - transit/transition metrics
   - visit/revisit behavior
   - task-level reading time or reading-flow efficiency

Do not reduce the explanation to fixation-only evidence.
If saccade or global metric evidence is present, mention how it affects reading flow,
rereading, transition efficiency, dwell time, first-visit processing, or scanpath behavior.

Conciseness rules:
- integrated_rationale: maximum 4 sentences.
- specific_difficulty_summary: maximum 3 strings.
- intervention_suggestions: maximum 3 items.
- recommendation: maximum 5 sentences.
- Do not list every AOI or every unit.
- Summarize patterns and include only the most informative examples.

Return ONLY valid JSON with exactly this structure:

{
  "final_risk_level": "low_risk | high_risk",
  "overall_confidence": 0.0,
  "agreement_level": "low | medium | high",
  "integrated_rationale": "...",
  "specific_difficulty_summary": [
    "..."
  ],
  "intervention_suggestions": [
    {
      "focus_area": "...",
      "reason": "...",
      "suggested_action": "...",
      "priority": "low | medium | high"
    }
  ],
  "uncertainty_note": "...",
  "recommendation": "..."
}

Risk-level guidance:
- low_risk: evidence does not sufficiently support elevated dyslexia-related risk.
- high_risk: evidence supports elevated dyslexia-related risk.
- If evidence is mixed or confidence is limited, still choose either low_risk or high_risk, but explain uncertainty clearly in uncertainty_note.

Intervention guidance:
- For low_risk: keep intervention_suggestions minimal, usually monitoring/contextual review.
- For high_risk: provide concrete expert-facing focus areas.
- Suggestions should be useful for a dyslexia expert, educational psychologist, reading specialist, or clinician.
- Suggestions should focus on assessment/intervention planning, not diagnosis.
"""


# =============================================================================
# Basic helpers
# =============================================================================

def get_specialist_reports(state: GraphState) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []

    for key in [
        "syllables_report",
        "meaningful_report",
        "pseudotext_report",
    ]:
        report = state.get(key)
        if report:
            reports.append(report)

    return reports


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def predicted_is_high(report: Dict[str, Any]) -> bool:
    return report.get("predicted_label") == "high_risk"


def clamp01(value: Any, default: float = 0.0) -> float:
    x = safe_float(value, default)
    return max(0.0, min(1.0, x))


# =============================================================================
# Rule-based baseline calculations
# =============================================================================

def compute_agreement_level(reports: List[Dict[str, Any]]) -> str:
    if len(reports) < 3:
        return "low"

    labels = [report.get("predicted_label") for report in reports]
    counts = Counter(labels)

    if len(counts) == 1:
        return "high"

    if counts.most_common(1)[0][1] == 2:
        return "medium"

    return "low"


def compute_final_risk_level(reports: List[Dict[str, Any]]) -> str:
    """
    Binary assessment-support aggregation.
    Output only:
    - low_risk
    - high_risk
    """
    if not reports:
        return "low_risk"

    risk_scores = [
        safe_float(report.get("risk_score"))
        for report in reports
    ]

    mean_risk = sum(risk_scores) / len(risk_scores)

    high_count = sum(
        1 for report in reports
        if report.get("predicted_label") == "high_risk"
    )

    if high_count >= 2 and mean_risk >= 0.50:
        return "high_risk"

    return "low_risk"


def compute_overall_confidence(
    reports: List[Dict[str, Any]],
    agreement_level: str,
) -> float:
    """
    System confidence, not clinical confidence.
    """
    if not reports:
        return 0.0

    confidences = [
        safe_float(report.get("confidence"))
        for report in reports
    ]

    mean_conf = sum(confidences) / len(confidences)

    agreement_bonus = {
        "high": 0.15,
        "medium": 0.00,
        "low": -0.15,
    }.get(agreement_level, 0.0)

    return clamp01(mean_conf + agreement_bonus)


# =============================================================================
# Difficulty extraction
# =============================================================================

def collect_specific_difficulties(
    reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Flatten specialist specific_difficulties while preserving task name.
    """
    all_items: List[Dict[str, Any]] = []

    severity_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    for report in reports:
        task_name = report.get("task_name", "UnknownTask")

        for item in report.get("specific_difficulties", []) or []:
            enriched = dict(item)
            enriched["task_name"] = task_name
            enriched["_severity_rank"] = severity_rank.get(
                str(item.get("severity", "low")),
                1,
            )
            all_items.append(enriched)

    all_items.sort(
        key=lambda item: (
            item.get("_severity_rank", 0),
            str(item.get("task_name", "")),
        ),
        reverse=True,
    )

    return all_items


def build_compact_specialist_reports(
    reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reduce specialist reports before sending to LLM.

    Version 2:
    - keep backward-compatible top_eye_tracking_evidence
    - add separated fixation / saccade / global metric evidence
    - add global_reading_patterns
    """
    compact_reports: List[Dict[str, Any]] = []

    for report in reports:
        compact_reports.append(
            {
                "task_name": report.get("task_name"),
                "predicted_label": report.get("predicted_label"),
                "risk_score": report.get("risk_score"),
                "confidence": report.get("confidence"),

                "difficulty_pattern": report.get("difficulty_pattern"),
                "global_reading_patterns": report.get("global_reading_patterns", [])[:3],

                # Backward-compatible field.
                "top_eye_tracking_evidence": report.get("top_eye_tracking_evidence", [])[:2],

                "top_fixation_evidence": report.get("top_fixation_evidence", [])[:2],
                "top_saccade_evidence": report.get("top_saccade_evidence", [])[:2],
                "top_metric_evidence": report.get("top_metric_evidence", [])[:2],

                "top_linguistic_evidence": report.get("top_linguistic_evidence", [])[:2],
                "top_interaction_evidence": report.get("top_interaction_evidence", [])[:2],

                "specific_difficulties": report.get("specific_difficulties", [])[:3],
                "explanation": report.get("explanation"),
            }
        )

    return compact_reports


def build_compact_difficulty_context(
    reports: List[Dict[str, Any]],
    max_items_per_task: int = 3,
) -> Dict[str, Any]:
    """
    Build compact task-wise context for the LLM.

    Version 2:
    This includes both:
    - local AOI/textual-unit difficulty context
    - global reading behavior context
    """
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in collect_specific_difficulties(reports):
        by_task[item.get("task_name", "UnknownTask")].append(item)

    context: Dict[str, Any] = {}

    for report in reports:
        task_name = report.get("task_name", "UnknownTask")
        items = by_task.get(task_name, [])[:max_items_per_task]

        context[task_name] = {
            "predicted_label": report.get("predicted_label"),
            "risk_score": report.get("risk_score"),
            "confidence": report.get("confidence"),
            "difficulty_pattern": report.get("difficulty_pattern"),

            "global_reading_patterns": report.get("global_reading_patterns", [])[:3],

            "evidence_groups": {
                "fixation": report.get("top_fixation_evidence", [])[:2],
                "saccade": report.get("top_saccade_evidence", [])[:2],
                "global_metric": report.get("top_metric_evidence", [])[:2],
                "linguistic": report.get("top_linguistic_evidence", [])[:2],
                "interaction": report.get("top_interaction_evidence", [])[:2],
            },

            "top_specific_difficulties": [
                {
                    "unit": item.get("unit"),
                    "unit_category": item.get("unit_category"),
                    "location": item.get("location"),
                    "severity": item.get("severity"),
                    "evidence": item.get("evidence"),
                }
                for item in items
            ],
        }

    return context


def build_rule_based_specific_summary(
    reports: List[Dict[str, Any]],
    max_total_items: int = 3,
) -> List[str]:
    """
    Create concise baseline summaries.

    Version 2:
    Each task summary may include:
    - risk/confidence
    - local AOI examples
    - global reading behavior patterns from metrics/saccades
    """
    summaries: List[str] = []

    for report in reports:
        task_name = report.get("task_name", "UnknownTask")
        predicted_label = report.get("predicted_label", "unknown")
        risk_score = safe_float(report.get("risk_score"))
        confidence = safe_float(report.get("confidence"))
        pattern = report.get("difficulty_pattern", "unspecified pattern")

        difficulties = report.get("specific_difficulties", []) or []
        global_patterns = report.get("global_reading_patterns", []) or []

        examples = ", ".join(
            [
                f"'{item.get('unit', '')}'"
                for item in difficulties[:2]
                if item.get("unit")
            ]
        )

        if global_patterns:
            global_text = " Global behavior: " + " ".join(global_patterns[:2])
        else:
            global_text = ""

        if examples:
            summary = (
                f"{task_name}: {predicted_label} "
                f"(risk={risk_score:.2f}, confidence={confidence:.2f}); "
                f"{pattern}; local examples: {examples}."
                f"{global_text}"
            )
        else:
            summary = (
                f"{task_name}: {predicted_label} "
                f"(risk={risk_score:.2f}, confidence={confidence:.2f}); "
                f"{pattern}; no concrete AOI units available."
                f"{global_text}"
            )

        summaries.append(summary)

    return summaries[:max_total_items]


# =============================================================================
# Rule-based intervention fallback
# =============================================================================

def intervention_focus_from_task_category(
    task_name: str,
    category: str,
) -> Tuple[str, str]:
    if task_name == "Syllables":
        if category == "complex_or_long_unit":
            return (
                "Complex syllable decoding",
                (
                    "Expert may examine decoding of longer or orthographically complex "
                    "syllabic units, including items with consonant clusters or diacritics."
                ),
            )

        if category == "short_unit":
            return (
                "Syllable decoding automaticity",
                (
                    "Expert may examine rapid recognition of short syllabic units and "
                    "phonological decoding speed."
                ),
            )

        return (
            "Syllable-level fluency",
            (
                "Expert may assess syllable-level reading fluency, decoding consistency, "
                "and signs of slow serial decoding."
            ),
        )

    if task_name == "MeaningfulText":
        if category == "complex_or_long_unit":
            return (
                "Word-level fluency in connected text",
                (
                    "Expert may examine word recognition, rereading, and fluency for "
                    "longer or orthographically complex words in connected text."
                ),
            )

        if category == "short_unit":
            return (
                "Automatic processing in connected text",
                (
                    "Expert may check whether brief textual units are processed less "
                    "automatically than expected during meaningful reading."
                ),
            )

        return (
            "Connected-text reading fluency",
            (
                "Expert may examine natural text reading fluency, word recognition, "
                "and rereading behavior."
            ),
        )

    if task_name == "PseudoText":
        if category == "complex_or_long_unit":
            return (
                "Grapheme-to-phoneme decoding for complex pseudo-units",
                (
                    "Expert may assess decoding of longer or complex pseudo-units, "
                    "focusing on grapheme-to-phoneme mapping without semantic support."
                ),
            )

        if category == "short_unit":
            return (
                "Basic pseudo-unit decoding",
                (
                    "Expert may examine non-lexical decoding of short pseudo-units where "
                    "semantic and lexical support are reduced."
                ),
            )

        return (
            "Pseudo-word decoding strategy",
            (
                "Expert may examine pseudo-word decoding strategy, phonological assembly, "
                "and decoding accuracy under reduced lexical support."
            ),
        )

    return (
        "Reading pattern follow-up",
        "Expert may review highlighted textual units alongside broader reading assessment results.",
    )


def priority_from_evidence(
    final_risk_level: str,
    severity_counts: Counter,
) -> str:
    if severity_counts.get("high", 0) >= 2:
        return "high"

    if final_risk_level == "high_risk" and severity_counts.get("high", 0) >= 1:
        return "high"

    if severity_counts.get("high", 0) >= 1 or severity_counts.get("medium", 0) >= 1:
        return "medium"

    return "low"


def build_rule_based_intervention_suggestions(
    reports: List[Dict[str, Any]],
    final_risk_level: str,
    max_suggestions: int = 3,
) -> List[InterventionSuggestion]:
    """
    Build concise fallback intervention suggestions.
    Board is the only agent that creates these suggestions.
    """
    if final_risk_level == "low_risk":
        return [
            InterventionSuggestion(
                focus_area="Monitoring and contextual review",
                reason=(
                    "Integrated risk is low; targeted intervention should not be inferred "
                    "from the model output alone."
                ),
                suggested_action=(
                    "Expert may monitor reading development and review broader educational "
                    "or clinical context if concerns persist."
                ),
                priority="low",
            )
        ]

    difficulties = collect_specific_difficulties(reports)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for item in difficulties:
        task_name = item.get("task_name", "UnknownTask")
        category = item.get("unit_category", "unknown")
        grouped[(task_name, category)].append(item)

    suggestions: List[InterventionSuggestion] = []

    for (task_name, category), items in grouped.items():
        focus_area, suggested_action = intervention_focus_from_task_category(
            task_name=task_name,
            category=category,
        )

        severity_counts = Counter(str(item.get("severity", "low")) for item in items)

        examples = [
            str(item.get("unit", ""))
            for item in items[:3]
            if item.get("unit")
        ]
        example_text = ", ".join([f"'{x}'" for x in examples])

        if example_text:
            reason = (
                f"{task_name} shows effortful processing for "
                f"{category.replace('_', ' ')} items, including {example_text}."
            )
        else:
            reason = (
                f"{task_name} shows a repeated pattern involving "
                f"{category.replace('_', ' ')} items."
            )

        suggestions.append(
            InterventionSuggestion(
                focus_area=focus_area,
                reason=reason,
                suggested_action=suggested_action,
                priority=priority_from_evidence(
                    final_risk_level=final_risk_level,
                    severity_counts=severity_counts,
                ),
            )
        )

    if not suggestions:
        high_tasks = [
            report.get("task_name", "UnknownTask")
            for report in reports
            if report.get("predicted_label") == "high_risk"
        ]

        suggestions.append(
            InterventionSuggestion(
                focus_area="Comprehensive reading assessment follow-up",
                reason=(
                    "Integrated risk is moderate/high, but concrete AOI-level difficulty "
                    "units are limited. High-risk specialist tasks: "
                    f"{', '.join(high_tasks) if high_tasks else 'none'}."
                ),
                suggested_action=(
                    "Expert should review phonological awareness, decoding fluency, "
                    "word-level reading, and pseudo-word decoding using standardized tools."
                ),
                priority="high",
            )
        )

    priority_order = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    suggestions.sort(
        key=lambda item: priority_order.get(item.priority, 0),
        reverse=True,
    )

    return suggestions[:max_suggestions]


# =============================================================================
# Prompt construction
# =============================================================================

def build_board_user_prompt(
    reports: List[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> str:
    payload = {
        "baseline_rule_based_assessment": baseline,
        "specialist_reports": build_compact_specialist_reports(reports),
        "specific_difficulty_context": build_compact_difficulty_context(reports),
        "version_2_evidence_requirements": {
            "must_use_local_aoi_difficulties": True,
            "must_use_global_reading_patterns": True,
            "must_consider_fixation_evidence": True,
            "must_consider_saccade_scanpath_evidence": True,
            "must_consider_global_metric_evidence": True,
            "must_not_reduce_explanation_to_fixation_only": True,
        },
        "output_constraints": {
            "integrated_rationale": "maximum 4 sentences",
            "specific_difficulty_summary": "maximum 3 strings",
            "intervention_suggestions": "maximum 3 items",
            "recommendation": "maximum 5 sentences",
            "do_not_repeat_all_units": True,
            "board_only_intervention": True,
        },
        "instructions": [
            "Use the baseline as guidance, but you may adjust final_risk_level if the specialist evidence justifies it.",
            "Do not overstate or diagnose dyslexia.",
            "Summarize concrete difficulties as patterns, not as a full list of AOIs.",
            "Use at most 1-2 example units per summary item.",
            "Create intervention_suggestions only at Board level.",
            "Make intervention_suggestions expert-facing and tied to observed evidence.",
            "Synthesize both local AOI/textual-unit difficulties and global reading behavior.",
            "If saccade/scanpath evidence is present, discuss reading flow, regression, progression, or scanpath organization.",
            "If global metric evidence is present, discuss dwell time, first visit, transit, transition, visit/revisit, or task-level reading behavior.",
            "Do not write a fixation-only explanation when metric or saccade evidence is present.",
            "Return only valid JSON matching the required schema.",
        ],
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


# =============================================================================
# LLM invocation
# =============================================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from an LLM response.
    Supports plain JSON or fenced ```json blocks.
    """
    cleaned = text.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start:end + 1]

    return json.loads(cleaned)


def normalize_suggestion(item: Dict[str, Any]) -> Dict[str, Any]:
    priority = str(item.get("priority", "medium"))

    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    return {
        "focus_area": str(item.get("focus_area", "Reading assessment follow-up")),
        "reason": str(item.get("reason", "Suggested based on specialist evidence.")),
        "suggested_action": str(item.get("suggested_action", "Expert should review this area.")),
        "priority": priority,
    }


def normalize_board_payload(
    payload: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Normalize LLM output before BoardReport validation.
    Also enforces concise list lengths.
    """
    risk = payload.get(
        "final_risk_level",
        baseline.get("final_risk_level", "low_risk"),
    )

    # Binary system: map any moderate_risk back to the binary baseline.
    if risk == "moderate_risk":
        risk = baseline.get("final_risk_level", "low_risk")

    if risk not in {"low_risk", "high_risk"}:
        risk = baseline.get("final_risk_level", "low_risk")

    payload["final_risk_level"] = risk

    agreement = payload.get("agreement_level", baseline.get("agreement_level", "low"))
    if agreement not in {"low", "medium", "high"}:
        agreement = baseline.get("agreement_level", "low")
    payload["agreement_level"] = agreement

    payload["overall_confidence"] = clamp01(
        payload.get("overall_confidence", baseline.get("overall_confidence", 0.0))
    )

    for key in [
        "integrated_rationale",
        "uncertainty_note",
        "recommendation",
    ]:
        if key not in payload or payload[key] is None:
            payload[key] = ""

    summary = payload.get("specific_difficulty_summary", [])
    if summary is None:
        summary = []
    if not isinstance(summary, list):
        summary = [str(summary)]
    payload["specific_difficulty_summary"] = [str(x) for x in summary[:3]]

    suggestions = payload.get("intervention_suggestions", [])
    if suggestions is None:
        suggestions = []
    if not isinstance(suggestions, list):
        suggestions = []

    payload["intervention_suggestions"] = [
        normalize_suggestion(item)
        for item in suggestions[:3]
        if isinstance(item, dict)
    ]

    return payload


def call_board_llm(
    reports: List[Dict[str, Any]],
    baseline: Dict[str, Any],
) -> BoardReport:
    """
    Call LLM to produce BoardReport.

    Board Agent is LLM-based by default.
    If LLM fails, raise an error unless MACDYS_ALLOW_LLM_FALLBACK=1.
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        msg = (
            "[Board LLM] langchain_openai import failed. "
            "Install it with: pip install langchain-openai"
        )

        if allow_llm_fallback():
            print(f"{msg} Falling back because MACDYS_ALLOW_LLM_FALLBACK=1.")
            raise RuntimeError(msg) from exc

        raise RuntimeError(msg) from exc

    try:
        print(f"[Board LLM] Calling model: {OPENAI_MODEL}")

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.2,
        )

        messages = [
            ("system", BOARD_SYSTEM_PROMPT),
            ("user", build_board_user_prompt(reports, baseline)),
        ]

        response = llm.invoke(messages)
        content = getattr(response, "content", response)

        print("[Board LLM] Response received.")

        parsed = extract_json_object(str(content))
        parsed = normalize_board_payload(parsed, baseline)

        board_report = BoardReport.model_validate(parsed)

        print("[Board LLM] JSON parsed and validated.")

        return board_report

    except Exception as exc:
        msg = f"[Board LLM] Failed: {exc}"

        if allow_llm_fallback():
            print(f"{msg} Falling back because MACDYS_ALLOW_LLM_FALLBACK=1.")
            raise RuntimeError(msg) from exc

        raise RuntimeError(
            msg
            + "\nBoard Agent is configured to use LLM by default. "
            "Set MACDYS_ALLOW_LLM_FALLBACK=1 only if you explicitly want fallback."
        ) from exc


# =============================================================================
# Deterministic fallback
# =============================================================================

def build_fallback_integrated_rationale(
    reports: List[Dict[str, Any]],
    final_risk_level: str,
    agreement_level: str,
) -> str:
    high_tasks = [
        report.get("task_name", "UnknownTask")
        for report in reports
        if report.get("predicted_label") == "high_risk"
    ]

    low_tasks = [
        report.get("task_name", "UnknownTask")
        for report in reports
        if report.get("predicted_label") == "low_risk"
    ]

    if final_risk_level == "high_risk":
        risk_text = (
            "The Board assigns high assessment-support risk because elevated evidence "
            "is supported by the integrated specialist pattern."
        )
    else:
        risk_text = (
            "The Board assigns low assessment-support risk because the integrated "
            "evidence does not sufficiently support elevated risk."
        )

    task_text = (
        f"High-risk tasks: {', '.join(high_tasks) if high_tasks else 'none'}; "
        f"low-risk tasks: {', '.join(low_tasks) if low_tasks else 'none'}."
    )

    return (
        f"{risk_text} Agreement is {agreement_level}. {task_text} "
        f"This output supports expert review and does not diagnose dyslexia."
    )


def build_fallback_uncertainty_note(
    reports: List[Dict[str, Any]],
    agreement_level: str,
    overall_confidence: float,
) -> str:
    confidences = [
        safe_float(report.get("confidence"))
        for report in reports
    ]

    if agreement_level in {"medium", "low"}:
        disagreement_text = "Specialists show mixed evidence or partial disagreement."
    else:
        disagreement_text = "Specialists show broad agreement."

    return (
        f"{disagreement_text} Overall system confidence is {overall_confidence:.2f}; "
        f"specialist confidence values are "
        f"{', '.join([f'{c:.2f}' for c in confidences])}. "
        f"Expert interpretation should consider educational and clinical context."
    )


def build_fallback_recommendation(
    final_risk_level: str,
    intervention_suggestions: List[InterventionSuggestion],
) -> str:
    if final_risk_level == "low_risk":
        return (
            "No dyslexia-related conclusion should be made from the model alone. "
            "Continue monitoring and consider expert review if broader concerns persist."
        )

    focus_areas = ", ".join(
        suggestion.focus_area
        for suggestion in intervention_suggestions[:3]
    )

    if focus_areas:
        return (
            "Refer the child for expert dyslexia screening or educational assessment. "
            f"Expert review should prioritize: {focus_areas}."
        )

    return (
        "Refer the child for expert dyslexia screening or educational assessment. "
        "Expert review should examine decoding, fluency, and connected-text reading."
    )


def build_fallback_board_report(
    reports: List[Dict[str, Any]],
    final_risk_level: str,
    agreement_level: str,
    overall_confidence: float,
) -> BoardReport:
    difficulty_summary = build_rule_based_specific_summary(reports)

    intervention_suggestions = build_rule_based_intervention_suggestions(
        reports=reports,
        final_risk_level=final_risk_level,
    )

    return BoardReport(
        final_risk_level=final_risk_level,
        overall_confidence=overall_confidence,
        agreement_level=agreement_level,
        integrated_rationale=build_fallback_integrated_rationale(
            reports=reports,
            final_risk_level=final_risk_level,
            agreement_level=agreement_level,
        ),
        specific_difficulty_summary=difficulty_summary,
        intervention_suggestions=intervention_suggestions,
        uncertainty_note=build_fallback_uncertainty_note(
            reports=reports,
            agreement_level=agreement_level,
            overall_confidence=overall_confidence,
        ),
        recommendation=build_fallback_recommendation(
            final_risk_level=final_risk_level,
            intervention_suggestions=intervention_suggestions,
        ),
    )


# =============================================================================
# Board node
# =============================================================================

def assessment_board_node(state: GraphState) -> GraphState:
    """
    LLM-based Assessment Board Agent.

    Flow:
    1. Collect specialist reports.
    2. Compute rule-based baseline values.
    3. Ask LLM Board Agent to synthesize concise final report.
    4. Validate against BoardReport schema.
    5. Fall back to deterministic BoardReport if LLM fails.
    """
    reports = get_specialist_reports(state)

    agreement_level = compute_agreement_level(reports)
    final_risk_level = compute_final_risk_level(reports)
    overall_confidence = compute_overall_confidence(
        reports=reports,
        agreement_level=agreement_level,
    )

    baseline_suggestions = build_rule_based_intervention_suggestions(
        reports=reports,
        final_risk_level=final_risk_level,
    )

    baseline = {
        "final_risk_level": final_risk_level,
        "overall_confidence": overall_confidence,
        "agreement_level": agreement_level,
        "rule_based_specific_difficulty_summary": build_rule_based_specific_summary(reports),
        "rule_based_intervention_suggestions": [
            suggestion.model_dump()
            for suggestion in baseline_suggestions
        ],
    }

    try:
        board_report = call_board_llm(
            reports=reports,
            baseline=baseline,
        )
        print("[Board] Using LLM-generated board report.")

    except Exception as exc:
        if not allow_llm_fallback():
            raise

        print("[Board] Using deterministic fallback board report.")
        board_report = build_fallback_board_report(
            reports=reports,
            final_risk_level=final_risk_level,
            agreement_level=agreement_level,
            overall_confidence=overall_confidence,
        )

    return {
        "board_report": board_report.model_dump(),
        "board_draft_report": board_report.model_dump(),
    }


# Backward-compatible alias for workflow.py
def board_agent_node(state: GraphState) -> GraphState:
    return assessment_board_node(state)


def allow_llm_fallback() -> bool:
    """
    By default, Board Agent MUST use LLM.

    Only allow deterministic fallback when explicitly enabled:
    export MACDYS_ALLOW_LLM_FALLBACK=1
    """
    return os.getenv("MACDYS_ALLOW_LLM_FALLBACK", "0") == "1"


# =============================================================================
# Board final revision after Critic
# =============================================================================

BOARD_FINAL_SYSTEM_PROMPT = """
You are the Final Assessment Board Agent.

You receive:
1. specialist_reports
2. the initial Board Draft report
3. the Critic Agent review

Your job is to produce the final BoardReport.

Important:
- The Critic does NOT decide the final risk level.
- The Critic does NOT write the final recommendation.
- The Critic only identifies problems in the Board Draft.
- You, the Final Board Agent, must make the final assessment decision.
- You must consider the Critic issues, but you must not blindly copy them.
- If the Critic verdict is "revise", you must address every issue.
- If the Critic identifies low agreement or low confidence, your final report must be cautious.
- Do not diagnose dyslexia.
- Use assessment-support language.
- The final risk output is binary: choose either low_risk or high_risk.
- Do not output moderate_risk.
- If evidence is mixed, choose the better-supported binary label and explain the uncertainty clearly.

You must synthesize BOTH:
1. Local AOI/textual-unit difficulties
   - specific words, syllables, pseudo-units, AOI locations
   - fixation/AOI-level effort
   - eye-language interaction on short, long, or complex units

2. Global reading behavior
   - saccade/scanpath behavior
   - regression/progression indicators
   - dwell-time or first-visit metrics
   - transit/transition metrics
   - visit/revisit behavior
   - task-level reading time or reading-flow efficiency

Do not reduce the final assessment to fixation-only evidence.
If the Critic says the Board Draft underused saccade/global metric evidence,
you must address that issue explicitly.

Conciseness rules:
- integrated_rationale: maximum 4 sentences.
- specific_difficulty_summary: maximum 3 strings.
- intervention_suggestions: maximum 3 items.
- recommendation: maximum 5 sentences.

Intervention suggestion rules:
- If final_risk_level is high_risk, you should provide 1-3 intervention_suggestions.
- These suggestions must be expert-facing and evidence-linked.
- If the Critic flags intervention_suggestions, revise and calibrate them.
- Do NOT remove all intervention_suggestions unless final_risk_level is low_risk or the evidence is insufficient.
- Suggestions should support expert assessment/intervention planning, not diagnosis.

Return ONLY valid JSON with exactly this structure:

{
  "final_risk_level": "low_risk | high_risk",
  "overall_confidence": 0.0,
  "agreement_level": "low | medium | high",
  "integrated_rationale": "...",
  "specific_difficulty_summary": [
    "..."
  ],
  "intervention_suggestions": [
    {
      "focus_area": "...",
      "reason": "...",
      "suggested_action": "...",
      "priority": "low | medium | high"
    }
  ],
  "uncertainty_note": "...",
  "recommendation": "..."
}
"""


def build_board_final_user_prompt(
    reports: List[Dict[str, Any]],
    board_draft_report: Dict[str, Any],
    critic_report: Dict[str, Any],
    baseline: Dict[str, Any],
) -> str:
    """
    Build prompt for the final Board pass after Critic review.

    Board Final receives Critic issues but must make its own final decision.
    Version 2 requires the Board to use both local AOI evidence and global
    saccade/metric evidence.
    """
    payload = {
        "specialist_reports": build_compact_specialist_reports(reports),
        "board_draft_report": board_draft_report,
        "critic_review": {
            "verdict": critic_report.get("verdict"),
            "issues": critic_report.get("issues", []),
            "overall_comment": critic_report.get("overall_comment", ""),
        },
        "baseline_rule_based_assessment": baseline,
        "specific_difficulty_context": build_compact_difficulty_context(reports),
        "version_2_evidence_requirements": {
            "must_use_local_aoi_difficulties": True,
            "must_use_global_reading_patterns": True,
            "must_consider_fixation_evidence": True,
            "must_consider_saccade_scanpath_evidence": True,
            "must_consider_global_metric_evidence": True,
            "must_not_reduce_explanation_to_fixation_only": True,
        },
        "role_constraints": {
            "critic_does_not_decide_final_risk": True,
            "board_final_must_decide_final_risk": True,
            "board_final_must_address_critic_issues": True,
            "do_not_diagnose": True,
        },
        "instructions": [
            "Read the Critic issues carefully.",
            "If Critic verdict is revise, address every issue in the final assessment.",
            "Do not blindly copy Critic wording.",
            "Do not treat Critic as the decision maker.",
            "Use specialist predictions, confidence, evidence, Board Draft, and Critic issues together.",
            "If agreement/confidence is limited, make the final assessment cautious.",
            "Synthesize both local AOI/textual-unit difficulties and global reading behavior.",
            "If saccade/scanpath evidence is present, discuss reading flow, regression, progression, or scanpath organization.",
            "If global metric evidence is present, discuss dwell time, first visit, transit, transition, visit/revisit, or task-level reading behavior.",
            "Do not write a fixation-only final assessment when metric or saccade evidence is present.",
            "Return only valid JSON matching the BoardReport schema.",
        ],
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


def call_board_final_llm(
    reports: List[Dict[str, Any]],
    board_draft_report: Dict[str, Any],
    critic_report: Dict[str, Any],
    baseline: Dict[str, Any],
) -> BoardReport:
    """
    Final Board LLM call after Critic review.

    Board Final is the final decision maker.
    Critic issues are feedback, not direct replacements.
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        msg = (
            "[Board Final LLM] langchain_openai import failed. "
            "Install it with: pip install langchain-openai"
        )
        raise RuntimeError(msg) from exc

    try:
        print(f"[Board Final LLM] Calling model: {OPENAI_MODEL}")

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.2,
        )

        messages = [
            ("system", BOARD_FINAL_SYSTEM_PROMPT),
            (
                "user",
                build_board_final_user_prompt(
                    reports=reports,
                    board_draft_report=board_draft_report,
                    critic_report=critic_report,
                    baseline=baseline,
                ),
            ),
        ]

        response = llm.invoke(messages)
        content = getattr(response, "content", response)

        print("[Board Final LLM] Response received.")

        parsed = extract_json_object(str(content))
        parsed = normalize_board_payload(parsed, baseline)

        final_board_report = BoardReport.model_validate(parsed)

        print("[Board Final LLM] JSON parsed and validated.")

        return final_board_report

    except Exception as exc:
        raise RuntimeError(f"[Board Final LLM] Failed: {exc}") from exc


def board_final_agent_node(state: GraphState) -> GraphState:
    """
    Final Board Agent node.

    This node runs AFTER Critic.

    It reads:
    - specialist reports
    - initial Board Draft report
    - Critic review

    Then it produces the final BoardReport.
    """
    reports = get_specialist_reports(state)

    board_draft_report = state.get("board_draft_report") or state.get("board_report", {})
    critic_report = state.get("critic_report", {})

    agreement_level = compute_agreement_level(reports)
    final_risk_level = compute_final_risk_level(reports)
    overall_confidence = compute_overall_confidence(
        reports=reports,
        agreement_level=agreement_level,
    )

    baseline_suggestions = build_rule_based_intervention_suggestions(
        reports=reports,
        final_risk_level=final_risk_level,
    )

    baseline = {
        "final_risk_level": final_risk_level,
        "overall_confidence": overall_confidence,
        "agreement_level": agreement_level,
        "rule_based_specific_difficulty_summary": build_rule_based_specific_summary(reports),
        "rule_based_intervention_suggestions": [
            suggestion.model_dump()
            for suggestion in baseline_suggestions
        ],
    }

    try:
        final_board_report = call_board_final_llm(
            reports=reports,
            board_draft_report=board_draft_report,
            critic_report=critic_report,
            baseline=baseline,
        )

        print("[Board Final] Using LLM-generated final board report.")

    except Exception:
        if not allow_llm_fallback():
            raise

        print("[Board Final] Using deterministic fallback final board report.")
        final_board_report = build_fallback_board_report(
            reports=reports,
            final_risk_level=final_risk_level,
            agreement_level=agreement_level,
            overall_confidence=overall_confidence,
        )

    return {
        "board_draft_report": board_draft_report,
        "board_report": final_board_report.model_dump(),
        "board_final_report": final_board_report.model_dump(),
    }