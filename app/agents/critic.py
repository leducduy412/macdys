from __future__ import annotations

import json
import os
import re

from app.state import GraphState
from collections import Counter
from typing import Any, Dict, List

from app.schemas import CriticIssue, CriticReport
MIN_PASS_OVERALL_CONFIDENCE = 0.50
MIN_PASS_SPECIALIST_CONFIDENCE = 0.50
REQUIRE_HIGH_AGREEMENT_FOR_PASS = True

# =============================================================================
# LLM configuration
# =============================================================================

try:
    from app.config import OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_API_URL_HERE")
    OPENAI_MODEL = os.getenv("LLM_MODEL", "YOUR_MODEL_HERE")


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent dyslexia assessment-support system.

Your role is to REVIEW the Board Draft report.

You must identify:
1. Calibration problems
2. Agreement/confidence problems
3. Inconsistency between specialist evidence and Board reasoning
4. Safety or diagnostic overclaiming
5. Missing or vague discussion of concrete AOI/textual difficulties
6. Overly strong intervention recommendations
7. Verbosity or unclear expert-facing language
8. Missing use of saccade, scanpath, or global reading metric evidence

Important role boundary:
- You are NOT the final decision maker.
- You must NOT choose final_risk_level.
- You must NOT provide revised_final_risk_level.
- You must NOT write the final recommendation.
- You must NOT write intervention_suggestions.
- Your job is to point out what is problematic or insufficient in the Board Draft.
- The Final Board Agent will read your critique and produce the final assessment.

Binary risk rule:
- The system only allows low_risk or high_risk.
- Board Draft must not output moderate_risk.
- If Board Draft outputs moderate_risk, verdict must be revise.
- Critic must not choose the final risk level; it only flags the invalid label.

Version 2 evidence requirements:
- Specialist reports may include fixation/AOI evidence, saccade/scanpath evidence,
  global metric evidence, linguistic evidence, and eye-language interaction evidence.
- If saccade/scanpath evidence is present, the Board Draft should mention reading flow,
  regression/progression, scanpath organization, or saccade dynamics when relevant.
- If global metric evidence is present, the Board Draft should mention dwell time,
  first visit, transit/transition, visit/revisit, reading time, or task-level reading behavior.
- If the Board Draft only discusses fixation/AOI difficulties while ignoring present
  saccade/global metric evidence, verdict must be "revise".

Hard pass criteria:
- You may return "pass" ONLY if specialist agreement is high.
- You may return "pass" ONLY if overall_confidence is sufficiently high.
- You may return "pass" ONLY if no specialist confidence is too low.
- If agreement is medium or low, verdict must be "revise".
- If confidence is low or uneven across specialists, verdict must be "revise".
- A report can be safe and useful but still require revision because confidence/agreement is insufficient.

Return ONLY valid JSON with exactly this structure:

{
  "verdict": "pass | revise",
  "issues": [
    {
      "issue_type": "calibration | agreement | confidence | evidence_consistency | safety | specificity | conciseness | intervention",
      "severity": "minor | major | blocking",
      "message": "...",
      "evidence": "...",
      "required_board_action": "..."
    }
  ],
  "overall_comment": "..."
}

Guidance:
- If verdict is "pass", issues should usually be empty or only minor.
- If verdict is "revise", include clear required_board_action for each issue.
- Do not diagnose dyslexia.
- Use assessment-support language.
- Do not provide final risk, final recommendation, or intervention suggestions.
"""


# =============================================================================
# Basic helpers
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


def contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


# =============================================================================
# Rule-based checks
# =============================================================================

def build_rule_based_issues(
    specialist_reports: List[Dict[str, Any]],
    board_report: Dict[str, Any],
) -> List[CriticIssue]:
    """
    Rule-based critic checks.

    Important:
    These checks only create review issues.
    They do NOT decide the final risk level.
    Board Final Agent will read these issues and decide the final assessment.
    """
    issues: List[CriticIssue] = []

    board_agreement = str(board_report.get("agreement_level", "low"))
    board_confidence = safe_float(board_report.get("overall_confidence", 0.0))
    final_risk_level = str(board_report.get("final_risk_level", "low_risk"))
    if final_risk_level == "moderate_risk":
        issues.append(
            make_critic_issue(
                issue_type="calibration",
                severity="blocking",
                message=(
                    "Board Draft uses moderate_risk, but the system is configured "
                    "for binary risk output."
                ),
                evidence="final_risk_level=moderate_risk",
                required_board_action=(
                    "Board Final must choose either low_risk or high_risk and explain "
                    "uncertainty separately in uncertainty_note."
                ),
            )
        )

    integrated_rationale = str(board_report.get("integrated_rationale", ""))
    uncertainty_note = str(board_report.get("uncertainty_note", ""))
    recommendation = str(board_report.get("recommendation", ""))

    expected_agreement = compute_expected_agreement_level(specialist_reports)

    # ------------------------------------------------------------------
    # Hard gate: agreement
    # ------------------------------------------------------------------
    if board_agreement != expected_agreement:
        issues.append(
            make_critic_issue(
                issue_type="agreement",
                severity="blocking",
                message="Board agreement_level is inconsistent with specialist predictions.",
                evidence=(
                    f"Expected agreement from specialist predictions is "
                    f"{expected_agreement}, but Board reported {board_agreement}."
                ),
                required_board_action=(
                    "Board Final must recompute or justify agreement level using "
                    "specialist predicted labels, and avoid overstating certainty."
                ),
            )
        )

    if REQUIRE_HIGH_AGREEMENT_FOR_PASS and expected_agreement != "high":
        issues.append(
            make_critic_issue(
                issue_type="agreement",
                severity="blocking",
                message="Critic pass is not allowed because specialist agreement is not high.",
                evidence=(
                    f"Specialist agreement is {expected_agreement}. "
                    "At least one specialist disagrees with the others."
                ),
                required_board_action=(
                    "Board Final must explicitly discuss mixed specialist evidence and "
                    "produce a cautious final assessment."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Hard gate: overall confidence
    # ------------------------------------------------------------------
    if board_confidence < MIN_PASS_OVERALL_CONFIDENCE:
        issues.append(
            make_critic_issue(
                issue_type="confidence",
                severity="blocking",
                message="Critic pass is not allowed because overall confidence is too low.",
                evidence=(
                    f"Board overall_confidence={board_confidence:.2f}, below threshold "
                    f"{MIN_PASS_OVERALL_CONFIDENCE:.2f}."
                ),
                required_board_action=(
                    "Board Final must lower certainty language and clearly state that "
                    "the result is preliminary assessment-support evidence."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Hard gate: specialist confidence
    # ------------------------------------------------------------------
    low_conf_tasks = [
        report.get("task_name", "UnknownTask")
        for report in specialist_reports
        if safe_float(report.get("confidence")) < MIN_PASS_SPECIALIST_CONFIDENCE
    ]

    if low_conf_tasks:
        issues.append(
            make_critic_issue(
                issue_type="confidence",
                severity="blocking",
                message="Critic pass is not allowed because some specialist confidences are too low.",
                evidence=(
                    f"Specialist confidence below {MIN_PASS_SPECIALIST_CONFIDENCE:.2f}: "
                    f"{', '.join(low_conf_tasks)}."
                ),
                required_board_action=(
                    "Board Final must explicitly discuss low-confidence specialist outputs "
                    "and avoid strong conclusions based on them."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Safety: diagnostic overclaiming
    # ------------------------------------------------------------------
    combined_text = " ".join(
        [
            integrated_rationale,
            uncertainty_note,
            recommendation,
        ]
    ).lower()

    unsafe_phrases = [
        "diagnosed with dyslexia",
        "has dyslexia",
        "confirms dyslexia",
        "proves dyslexia",
        "definitely dyslexic",
        "is dyslexic",
        "suffers from dyslexia",
    ]

    if any(phrase in combined_text for phrase in unsafe_phrases):
        issues.append(
            make_critic_issue(
                issue_type="safety",
                severity="blocking",
                message="Board Draft may imply diagnostic overclaiming.",
                evidence="The Board Draft uses language that may imply diagnosis.",
                required_board_action=(
                    "Board Final must use assessment-support language and avoid "
                    "claiming that the system diagnoses dyslexia."
                ),
            )
        )

    # ------------------------------------------------------------------
    # Uncertainty consistency
    # ------------------------------------------------------------------
    if expected_agreement != "high":
        if not contains_any(
            uncertainty_note + " " + integrated_rationale,
            [
                "mixed",
                "disagreement",
                "uneven",
                "partial",
                "inconsistent",
                "limited agreement",
                "cautious",
                "uncertain",
            ],
        ):
            issues.append(
                make_critic_issue(
                    issue_type="calibration",
                    severity="major",
                    message="Board Draft does not clearly explain mixed specialist evidence.",
                    evidence=(
                        f"Expected agreement is {expected_agreement}, but uncertainty "
                        "language is insufficient."
                    ),
                    required_board_action=(
                        "Board Final must explicitly describe disagreement between specialists "
                        "and its effect on the final assessment."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Intervention calibration
    # ------------------------------------------------------------------
    if final_risk_level == "high_risk":
        if not contains_any(
            recommendation,
            [
                "expert",
                "specialist",
                "assessment",
                "screening",
                "review",
            ],
        ):
            issues.append(
                make_critic_issue(
                    issue_type="intervention",
                    severity="major",
                    message="Recommendation does not clearly route the case to expert review.",
                    evidence="Final risk is moderate/high but recommendation lacks expert-review framing.",
                    required_board_action=(
                        "Board Final should frame next steps as expert assessment or review, "
                        "not model-only intervention."
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Version 2 evidence coverage:
    # If specialists contain saccade/global metric evidence, Board should
    # not write a fixation-only synthesis.
    # ------------------------------------------------------------------
    issues.extend(
        build_saccade_metric_coverage_issue(
            specialist_reports=specialist_reports,
            board_report=board_report,
        )
    )

    return issues


def build_fallback_critic_report(
    issues: List[CriticIssue],
) -> CriticReport:
    """
    Deterministic fallback critic report.

    Used only if LLM fallback is explicitly enabled.
    Critic still does not produce final risk or recommendations.
    """
    if issues:
        return CriticReport(
            verdict="revise",
            issues=issues,
            overall_comment=(
                "The Board Draft requires revision based on rule-based Critic checks."
            ),
        )

    return CriticReport(
        verdict="pass",
        issues=[],
        overall_comment=(
            "No blocking critic issues were detected by the fallback checker."
        ),
    )


# =============================================================================
# Prompt construction
# =============================================================================

def build_critic_user_prompt(
    specialist_reports: List[Dict[str, Any]],
    board_report: Dict[str, Any],
    rule_based_issues: List[CriticIssue],
) -> str:
    """
    Build prompt for Critic LLM.

    The Critic receives rule-based issues as hints, but must produce its own
    structured review. It must not produce final risk or final recommendations.

    Version 2:
    Critic sees separated fixation / saccade / global metric evidence.
    """
    compact_specialists: List[Dict[str, Any]] = []

    for report in specialist_reports:
        compact_specialists.append(
            {
                "task_name": report.get("task_name"),
                "predicted_label": report.get("predicted_label"),
                "risk_score": report.get("risk_score"),
                "confidence": report.get("confidence"),

                "difficulty_pattern": report.get("difficulty_pattern"),
                "global_reading_patterns": report.get("global_reading_patterns", [])[:3],

                # Backward-compatible field.
                "top_eye_tracking_evidence": report.get("top_eye_tracking_evidence", [])[:2],

                # Version 2 evidence groups.
                "top_fixation_evidence": report.get("top_fixation_evidence", [])[:2],
                "top_saccade_evidence": report.get("top_saccade_evidence", [])[:2],
                "top_metric_evidence": report.get("top_metric_evidence", [])[:2],

                "top_linguistic_evidence": report.get("top_linguistic_evidence", [])[:2],
                "top_interaction_evidence": report.get("top_interaction_evidence", [])[:2],

                "specific_difficulties": report.get("specific_difficulties", [])[:3],
                "explanation": report.get("explanation"),
            }
        )

    payload = {
        "specialist_reports": compact_specialists,
        "board_draft_report": board_report,
        "rule_based_issue_hints": [
            issue.model_dump()
            for issue in rule_based_issues
        ],
        "version_2_evidence_checks": {
            "specialist_has_saccade_or_metric_evidence": specialist_has_saccade_or_metric_evidence(
                specialist_reports
            ),
            "board_mentions_saccade_or_metric_evidence": board_text_mentions_saccade_or_metric_evidence(
                board_report
            ),
            "critic_should_flag_fixation_only_board_summary": True,
        },
        "critic_role_constraints": {
            "must_not_choose_final_risk_level": True,
            "must_not_write_final_recommendation": True,
            "must_not_write_intervention_suggestions": True,
            "must_only_review_board_draft": True,
        },
        "instructions": [
            "Return only valid JSON matching the CriticReport schema.",
            "Do not include revised_final_risk_level.",
            "Do not include revised_integrated_rationale.",
            "Do not include revised_intervention_suggestions.",
            "Do not make the final assessment decision.",
            "Identify issues that the Final Board Agent must address.",
            "If rule_based_issue_hints contain blocking issues, verdict must be revise.",
            "If specialists contain saccade/global metric evidence but the Board Draft ignores it, verdict should be revise.",
            "Do not allow a fixation-only Board explanation when saccade or global metric evidence is available.",
        ],
    }

    return json.dumps(payload, indent=2, ensure_ascii=False)


# =============================================================================
# LLM invocation
# =============================================================================

def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from LLM response.
    Supports raw JSON or fenced ```json blocks.
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


def normalize_critic_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize LLM Critic output to match the new CriticReport schema.

    Critic must not return revised_* fields.
    If the LLM returns them anyway, they are ignored.
    """
    verdict = payload.get("verdict", "revise")

    if verdict not in {"pass", "revise"}:
        verdict = "revise"

    raw_issues = payload.get("issues", [])

    if raw_issues is None:
        raw_issues = []

    normalized_issues = []

    if isinstance(raw_issues, list):
        for item in raw_issues:
            if isinstance(item, dict):
                issue_type = str(item.get("issue_type", "calibration"))
                if issue_type not in {
                    "calibration",
                    "agreement",
                    "confidence",
                    "evidence_consistency",
                    "safety",
                    "specificity",
                    "conciseness",
                    "intervention",
                }:
                    issue_type = "calibration"

                severity = str(item.get("severity", "major"))
                if severity not in {"minor", "major", "blocking"}:
                    severity = "major"

                normalized_issues.append(
                    {
                        "issue_type": issue_type,
                        "severity": severity,
                        "message": str(item.get("message", "")),
                        "evidence": str(item.get("evidence", "")),
                        "required_board_action": str(
                            item.get(
                                "required_board_action",
                                "Board Final should review and address this issue.",
                            )
                        ),
                    }
                )
            else:
                normalized_issues.append(
                    {
                        "issue_type": "calibration",
                        "severity": "major",
                        "message": str(item),
                        "evidence": "",
                        "required_board_action": (
                            "Board Final should review and address this issue."
                        ),
                    }
                )

    overall_comment = str(payload.get("overall_comment", ""))

    if not overall_comment:
        if verdict == "pass":
            overall_comment = "The Board Draft is acceptable under the Critic criteria."
        else:
            overall_comment = "The Board Draft requires revision before final assessment."

    return {
        "verdict": verdict,
        "issues": normalized_issues,
        "overall_comment": overall_comment,
    }


def call_critic_llm(
    specialist_reports: List[Dict[str, Any]],
    board_report: Dict[str, Any],
    rule_based_issues: List[CriticIssue],
) -> CriticReport:
    """
    Call LLM-based Critic Agent.

    Critic is LLM-based by default.
    If LLM fails, raise unless MACDYS_ALLOW_LLM_FALLBACK=1.
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        msg = (
            "[Critic LLM] langchain_openai import failed. "
            "Install it with: pip install langchain-openai"
        )
        raise RuntimeError(msg) from exc

    try:
        print(f"[Critic LLM] Calling model: {OPENAI_MODEL}")

        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.2,
        )

        messages = [
            ("system", CRITIC_SYSTEM_PROMPT),
            (
                "user",
                build_critic_user_prompt(
                    specialist_reports=specialist_reports,
                    board_report=board_report,
                    rule_based_issues=rule_based_issues,
                ),
            ),
        ]

        response = llm.invoke(messages)
        content = getattr(response, "content", response)

        print("[Critic LLM] Response received.")

        parsed = extract_json_object(str(content))
        parsed = normalize_critic_payload(parsed)

        critic_report = CriticReport.model_validate(parsed)

        print("[Critic LLM] JSON parsed and validated.")

        return critic_report

    except Exception as exc:
        raise RuntimeError(f"[Critic LLM] Failed: {exc}") from exc


# =============================================================================
# Critic node
# =============================================================================

def critic_agent_node(state: GraphState) -> GraphState:
    """
    LLM-based Critic Agent.

    Critic reviews the Board Draft and returns only:
    - verdict
    - structured issues
    - overall_comment

    Critic does NOT produce final risk or final recommendations.
    """
    specialist_reports = get_specialist_reports(state)
    board_report = state.get("board_report", {})

    rule_based_issues = build_rule_based_issues(
        specialist_reports=specialist_reports,
        board_report=board_report,
    )

    try:
        critic_report = call_critic_llm(
            specialist_reports=specialist_reports,
            board_report=board_report,
            rule_based_issues=rule_based_issues,
        )

        print("[Critic] Using LLM-generated critic report.")

    except Exception:
        if not allow_llm_fallback():
            raise

        print("[Critic] Using deterministic fallback critic report.")
        critic_report = build_fallback_critic_report(
            issues=rule_based_issues,
        )

    critic_report = enforce_critic_hard_gates(
        critic_report=critic_report,
        specialist_reports=specialist_reports,
        board_report=board_report,
    )

    needs_revision = critic_report.verdict == "revise"

    return {
        "critic_report": critic_report.model_dump(),
        "needs_revision": needs_revision,
    }


def critic_node(state: GraphState) -> GraphState:
    return critic_agent_node(state)


def allow_llm_fallback() -> bool:
    """
    By default, Critic Agent MUST use LLM.

    Only allow deterministic fallback when explicitly enabled:
    export MACDYS_ALLOW_LLM_FALLBACK=1
    """
    return os.getenv("MACDYS_ALLOW_LLM_FALLBACK", "0") == "1"


def compute_expected_agreement_level(
    specialist_reports: List[Dict[str, Any]],
) -> str:
    """
    Compute agreement from specialist predicted labels.

    With 3 specialists:
    - high: all 3 agree
    - medium: 2 agree, 1 disagrees
    - low: missing/invalid/no majority
    """
    if len(specialist_reports) < 3:
        return "low"

    labels = [
        report.get("predicted_label")
        for report in specialist_reports
    ]

    counts = Counter(labels)

    if len(counts) == 1:
        return "high"

    if counts.most_common(1)[0][1] == 2:
        return "medium"

    return "low"


def enforce_critic_hard_gates(
    critic_report: CriticReport,
    specialist_reports: List[Dict[str, Any]],
    board_report: Dict[str, Any],
) -> CriticReport:
    """
    Enforce non-negotiable Critic pass criteria.

    This function only changes Critic verdict/issues.
    It does NOT decide final risk level.
    """
    hard_gate_issues = build_rule_based_issues(
        specialist_reports=specialist_reports,
        board_report=board_report,
    )

    blocking_issues = [
        issue for issue in hard_gate_issues
        if issue.severity == "blocking"
    ]

    if not blocking_issues:
        return critic_report

    existing = list(critic_report.issues or [])

    merged: List[CriticIssue] = []
    seen = set()

    for issue in existing + blocking_issues:
        key = (
            issue.issue_type,
            issue.severity,
            issue.message,
            issue.evidence,
        )

        if key in seen:
            continue

        seen.add(key)
        merged.append(issue)

    return CriticReport(
        verdict="revise",
        issues=merged,
        overall_comment=(
            "The Board Draft requires revision because one or more hard "
            "critic gates failed. The Final Board Agent must address the "
            "listed issues before producing the final assessment."
        ),
    )


def make_critic_issue(
    issue_type: str,
    severity: str,
    message: str,
    evidence: str,
    required_board_action: str,
) -> CriticIssue:
    """
    Helper for creating structured critic issues.
    """
    return CriticIssue(
        issue_type=issue_type,
        severity=severity,
        message=message,
        evidence=evidence,
        required_board_action=required_board_action,
    )


def evidence_item_feature_name(item: Any) -> str:
    """
    Safely extract feature name from EvidenceItem-like object or dict.
    """
    if hasattr(item, "feature"):
        return str(item.feature)

    if isinstance(item, dict):
        return str(item.get("feature", ""))

    return ""


def specialist_has_saccade_or_metric_evidence(
    specialist_reports: List[Dict[str, Any]],
) -> bool:
    """
    Check whether specialists contain saccade/scanpath/global metric evidence.

    Supports both Version 2 fields and backward-compatible top_eye_tracking_evidence.
    """
    evidence_keywords = [
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
        "metric_",
        "dwell",
        "first_visit",
        "visit",
        "revisit",
        "transit",
        "transition",
        "trial",
        "reading_time",
        "total_reading_time",
        "fixation_rate",
        "saccade_rate",
        "task-level",
        "reading flow",
    ]

    for report in specialist_reports:
        if report.get("top_saccade_evidence"):
            return True

        if report.get("top_metric_evidence"):
            return True

        if report.get("global_reading_patterns"):
            return True

        # Backward-compatible check in case old fields still contain metric/saccade features.
        for field_name in [
            "top_eye_tracking_evidence",
            "top_fixation_evidence",
            "top_interaction_evidence",
        ]:
            for item in report.get(field_name, []) or []:
                feature_name = evidence_item_feature_name(item).lower()

                if any(keyword in feature_name for keyword in evidence_keywords):
                    return True

    return False


def board_text_mentions_saccade_or_metric_evidence(
    board_report: Dict[str, Any],
) -> bool:
    """
    Check whether Board Draft text explicitly mentions saccade/global metric evidence.
    """
    summary = board_report.get("specific_difficulty_summary", []) or []
    suggestions = board_report.get("intervention_suggestions", []) or []

    suggestion_text_parts: List[str] = []

    for item in suggestions:
        if isinstance(item, dict):
            suggestion_text_parts.extend(
                [
                    str(item.get("focus_area", "")),
                    str(item.get("reason", "")),
                    str(item.get("suggested_action", "")),
                ]
            )

    board_text = " ".join(
        [
            str(board_report.get("integrated_rationale", "")),
            " ".join(map(str, summary)),
            str(board_report.get("uncertainty_note", "")),
            str(board_report.get("recommendation", "")),
            " ".join(suggestion_text_parts),
        ]
    ).lower()

    required_terms = [
        "saccade",
        "scanpath",
        "regression",
        "progression",
        "transition",
        "transit",
        "dwell",
        "first-visit",
        "first visit",
        "visit",
        "revisit",
        "reading flow",
        "global metric",
        "reading time",
        "task-level",
        "eye-movement dynamics",
    ]

    return any(term in board_text for term in required_terms)


def build_saccade_metric_coverage_issue(
    specialist_reports: List[Dict[str, Any]],
    board_report: Dict[str, Any],
) -> List[CriticIssue]:
    """
    Create an issue if specialists contain saccade/global metric evidence
    but the Board Draft does not mention it.
    """
    if not specialist_has_saccade_or_metric_evidence(specialist_reports):
        return []

    if board_text_mentions_saccade_or_metric_evidence(board_report):
        return []

    return [
        make_critic_issue(
            issue_type="evidence_consistency",
            severity="major",
            message="Board Draft underuses saccade or global metric evidence.",
            evidence=(
                "Specialist reports include saccade/scanpath evidence, global reading "
                "metric evidence, or global_reading_patterns, but the Board Draft mainly "
                "discusses fixation/AOI difficulty."
            ),
            required_board_action=(
                "Board Final should integrate saccade/global metric evidence into the "
                "final rationale, especially reading flow, dwell time, first-visit, "
                "transit/transition, regression, scanpath, or task-level reading behavior."
            ),
        )
    ]