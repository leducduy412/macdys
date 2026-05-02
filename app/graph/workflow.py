from __future__ import annotations

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from app.agents.board import board_agent_node, board_final_agent_node
from app.agents.critic import critic_agent_node
from app.agents.specialist import (
    meaningful_agent_node,
    pseudotext_agent_node,
    syllables_agent_node,
)
from app.state import GraphState
from app.utils.utils import load_case_from_csv


def initialize_workflow_node(state: GraphState) -> GraphState:
    """
    Initialize shared workflow fields before the graph starts processing.
    """
    return {
        "errors": [],
        "needs_revision": False,
    }


def load_case_node(state: GraphState) -> GraphState:
    """
    Load one subject/case from the input CSV into the graph state.
    """
    case_csv = state["case_csv"]
    subject_id = state["subject_id"]

    case_row = load_case_from_csv(case_csv, subject_id)

    return {
        "case_row": case_row,
    }


# =============================================================================
# Final report helpers
# =============================================================================

def get_specialist_reports(state: GraphState) -> List[Dict[str, Any]]:
    """
    Collect specialist reports in a stable order.
    """
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


def validate_required_workflow_outputs(state: GraphState) -> None:
    """
    Validate that all required upstream reports exist before finalization.
    """
    required_keys = [
        "syllables_report",
        "meaningful_report",
        "pseudotext_report",
        "board_report",
        "critic_report",
    ]

    missing = [key for key in required_keys if key not in state]

    if missing:
        raise ValueError(
            "Workflow cannot be finalized because required outputs are missing: "
            f"{missing}"
        )


def build_final_report(state: GraphState) -> Dict[str, Any]:
    """
    Build final JSON report.

    Important:
    - board_draft_report is the initial Board output before Critic.
    - critic_report is the Critic review only.
    - board_final_report is the final Board output after reading Critic.
    - final_assessment equals board_final_report.
    """
    validate_required_workflow_outputs(state)

    final_board_report = state.get("board_final_report") or state["board_report"]

    final_report = {
        "subject_id": str(state["subject_id"]),
        "specialist_reports": get_specialist_reports(state),

        "board_draft_report": state.get("board_draft_report"),
        "critic_report": state["critic_report"],
        "board_final_report": final_board_report,

        "board_report": final_board_report,
        "final_assessment": final_board_report,
    }

    return final_report


def finalize_workflow_node(state: GraphState) -> GraphState:
    """
    Final node.

    Critic does not patch the final assessment.
    Final assessment is produced by Board Final Agent after reading Critic.
    """
    final_report = build_final_report(state)

    return {
        "final_report": final_report,
    }


# =============================================================================
# Graph construction
# =============================================================================

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Workflow:
        START
          -> initialize_workflow
          -> load_case
          -> [3 specialist agents in parallel]
          -> board_agent          # Board Draft
          -> critic_agent         # Critic reviews Board Draft only
          -> board_final_agent    # Board reads Critic and makes final decision
          -> finalize_workflow
          -> END
    """
    graph = StateGraph(GraphState)

    graph.add_node("initialize_workflow", initialize_workflow_node)
    graph.add_node("load_case", load_case_node)

    graph.add_node("syllables_agent", syllables_agent_node)
    graph.add_node("meaningful_agent", meaningful_agent_node)
    graph.add_node("pseudotext_agent", pseudotext_agent_node)

    graph.add_node("board_agent", board_agent_node)
    graph.add_node("critic_agent", critic_agent_node)
    graph.add_node("board_final_agent", board_final_agent_node)
    graph.add_node("finalize_workflow", finalize_workflow_node)

    graph.add_edge(START, "initialize_workflow")
    graph.add_edge("initialize_workflow", "load_case")

    graph.add_edge("load_case", "syllables_agent")
    graph.add_edge("load_case", "meaningful_agent")
    graph.add_edge("load_case", "pseudotext_agent")

    graph.add_edge(
        ["syllables_agent", "meaningful_agent", "pseudotext_agent"],
        "board_agent",
    )

    graph.add_edge("board_agent", "critic_agent")
    graph.add_edge("critic_agent", "board_final_agent")
    graph.add_edge("board_final_agent", "finalize_workflow")
    graph.add_edge("finalize_workflow", END)

    return graph.compile()


def get_compiled_graph():
    return build_graph()