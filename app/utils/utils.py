from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Type

import pandas as pd
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.config import (
    DEFAULT_OUTPUT_FILENAME_TEMPLATE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OUTPUT_DIR,
)


def get_llm() -> ChatOpenAI:
    """
    Return an OpenAI-compatible chat client.
    Works with OpenRouter or any OpenAI-compatible endpoint.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=LLM_TEMPERATURE,
    )


def _extract_json_block(text: str) -> str:
    """
    Try to extract a JSON object from free-form model output.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # If the whole text is already JSON, return it directly
    if text.startswith("{") and text.endswith("}"):
        return text

    # Fallback: grab the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in model output.")


def invoke_json_model(
    schema_model: Type[BaseModel],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> BaseModel:
    """
    Ask the model to return strict JSON text, then parse and validate it with Pydantic.

    This avoids OpenRouter tool-calling/tool_choice issues and works with more free models.
    """
    llm = get_llm()

    format_instructions = f"""
You must return ONLY a valid JSON object.
Do not include markdown fences.
Do not include explanation before or after the JSON.

The JSON must match this schema exactly:
{schema_model.model_json_schema()}
""".strip()

    messages = [
        ("system", system_prompt),
        ("human", f"{user_prompt}\n\n{format_instructions}"),
    ]

    last_error = None
    for _ in range(max_retries + 1):
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            json_text = _extract_json_block(content)
            payload = json.loads(json_text)
            return schema_model.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_error = e
            # ask model to self-correct
            messages.append(("assistant", content))
            messages.append((
                "human",
                f"The previous output was invalid because: {e}. "
                f"Return ONLY a corrected JSON object that matches the schema exactly."
            ))

    raise ValueError(f"Failed to obtain valid structured JSON output: {last_error}")


def ensure_output_dir(path: str | Path | None = None) -> Path:
    """
    Ensure the output directory exists and return it.
    If path is None, use the default OUTPUT_DIR from config.
    """
    output_path = Path(path) if path is not None else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def build_output_path(subject_id: str, output_dir: str | Path | None = None) -> Path:
    """
    Build the output file path for a subject report.
    """
    out_dir = ensure_output_dir(output_dir)
    filename = DEFAULT_OUTPUT_FILENAME_TEMPLATE.format(subject_id=subject_id)
    return out_dir / filename


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """
    Save a dictionary as a UTF-8 JSON file.
    """
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_case_from_csv(csv_path: str | Path, subject_id: str) -> Dict[str, Any]:
    """
    Load one subject row from a tabular CSV file.

    Expected:
    - a column named 'subject_id'
    - one row per subject
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "subject_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'subject_id' column.")

    row = df[df["subject_id"].astype(str) == str(subject_id)]

    if row.empty:
        raise ValueError(f"Subject '{subject_id}' was not found in {csv_path}.")

    return row.iloc[0].to_dict()


def probability_to_confidence(prob: float) -> float:
    """
    Convert a probability into a simple confidence score.
    """
    return float(min(1.0, abs(prob - 0.5) * 2.0))