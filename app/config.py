import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_API_URL_HERE")
OPENAI_MODEL = os.getenv("LLM_MODEL", "YOUR_MODEL_HERE")
LLM_TEMPERATURE = 0.0

SYLLABLES_MODEL_PATH = MODEL_DIR / "syllables_rf.joblib"
MEANINGFUL_MODEL_PATH = MODEL_DIR / "meaningful_rf.joblib"
PSEUDOTEXT_MODEL_PATH = MODEL_DIR / "pseudotext_rf.joblib"

SYLLABLES_PREFIX = "syll_"
MEANINGFUL_PREFIX = "mean_"
PSEUDOTEXT_PREFIX = "pseudo_"

TASK_SYLLABLES = "Syllables"
TASK_MEANINGFUL = "MeaningfulText"
TASK_PSEUDOTEXT = "PseudoText"

RISK_THRESHOLD = 0.5
TOP_K_EVIDENCE = 3

ALLOW_CRITIC_REVISION = True
DEFAULT_OUTPUT_FILENAME_TEMPLATE = "{subject_id}_assessment_report.json"


def validate_config() -> None:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please set it in your environment or .env file."
        )


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)