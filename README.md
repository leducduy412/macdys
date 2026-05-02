# MacDys: Multi-Agent Dyslexia Risk Assessment Support from Eye-Tracking Evidence

## Overview
**MacDys** is a multi-agent collaboration assessment-support system for analyzing dyslexia-related reading difficulty
patterns from eye-tracking data. The system combines task-specific machine learning models with LLM-based 
reasoning agents to produce an comprehensive assessment report. The system does **not** diagnose dyslexia. 
It provides assessment-support evidence for expert review.

The system is designed around three reading tasks:
1. **Syllables:** Focuses on syllable-level decoding and fluency.
2. **MeaningfulText:** Focuses on connected-text reading, word-level fluency, and reading-flow behavior.
3. **PseudoText:** Focuses on pseudo-word or pseudo-unit decoding where lexical and semantic support is
reduced.

Each task is handled by a **task-specialist agent**. These specialists generate task-level predictions and
evidence. A **Board Agent** then synthesizes all specialist outputs into a final assessment-support report.
A **Critic Agent** reviews the Board Draft and points out issues before the Board produces the final report.

![Overview of the method](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fdrive.google.com/uc?id=1cnrRnfmdzazL1xXUT8C7S60CX8QPUZGk)

## Repository Structure
The MacDys is built to be modular. A typical project structure is:

```
macdys/
├── app/
│   ├── agents/
│   │   ├── specialist.py
│   │   ├── board.py
│   │   └── critic.py
│   ├── graph/
│   │   └── workflow.py
│   ├── config.py
│   ├── explainability.py
│   ├── inference.py
│   ├── main.py
│   ├── schemas.py
│   ├── state.py
│   └── utils/
│       └── utils.py
│
├── scripts/
│   ├── evaluate_models.py
│   └── prepare_features_with_aoi.py
│   └── train_models.py
│   └── split_dataset.py
│
├── requirements.txt
│
└── README.md
```

## Dataset
This project uses **ETDD70: Eye-Tracking Dyslexia Dataset**, an eye-tracking dataset designed for AI-based dyslexia classification. The dataset contains eye-movement recordings from **70 Czech children aged 9–10**, including **35 dyslexic readers** and **35 non-dyslexic readers**. Participants performed three Czech reading tasks: syllable reading, meaningful-text reading, and pseudo-text reading. The dataset provides raw eye-tracking signals, fixation data, saccade data, derived statistical metrics, region-of-interest annotations, task stimuli, class labels, and fixation-image representations. These data are used to analyze reading behavior and build models for distinguishing dyslexic and non-dyslexic readers based on eye-movement patterns.

After downloading the dataset from Zenodo, organize the files as follows:

```
data/
├── fixation/                  # Processed fixation files: *_fixations.csv
├── saccade/                   # Processed saccade files: *_saccades.csv
├── metric/                    # Derived metric files: *_metrics.csv
├── raw/                       # Raw eye-tracking files: *_raw.csv
├── rois/                      # ROI annotation files from rois.zip
├── task_stimuli/              # Stimulus images/files from stimuli.zip
├── dyslexia_class_label.csv   # Participant-level dyslexia labels
└── fixation_images/           # Visual fixation representations from fixation_images.zip
```

## Quickstart
This section describes how to run the complete MacDys pipeline, from raw eye-tracking data preparation to the final multi-agent assessment-support report.
**1. Install dependencies**
```ruby
pip install -r requirements.txt
```
**2. Generate subject-level features**
```ruby
python scripts/prepare_features_with_aoi.py \
  --fixation-dir data/fixation \
  --saccade-dir data/saccade \
  --metrics-dir data/metrics \
  --aoi-dir data/rois \
  --labels-csv data/dyslexia_class_label.csv.csv \
  --output-csv data/processed/features_aoi.csv \
  --aoi-profile-json data/processed/aoi_profiles.json
```
This step will create:
```
data/processed/features_aoi.csv
data/processed/aoi_profiles.json
```
**3.Train the task-specific models**
```ruby
python scripts/train_models.py --csv data/processed_v2/features_aoi.csv
```
After training, the expected model files are:
```
models/final/syllables_rf.joblib     → Syllables task
models/final/meaningful_rf.joblib    → MeaningfulText task
models/final/pseudotext_rf.joblib    → PseudoText task
```
**4.Configure the LLM**
The Board Agent and Critic Agent use an OpenAI-compatible LLM endpoint.
Edit `app/config.py`:
```ruby
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY_HERE")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "YOUR_API_URL_HERE")
OPENAI_MODEL = os.getenv("LLM_MODEL", "YOUR_MODEL_HERE")
```
Example:
```ruby
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
LLM_API_KEY = "your_api_key_here"
LLM_BASE_URL = "https://openrouter.ai/api/v1"
```
For the main pipeline, LLM fallback should be disabled:
```ruby
unset MACDYS_ALLOW_LLM_FALLBACK
```
For debugging only, you may enable fallback mode:
```ruby
export MACDYS_ALLOW_LLM_FALLBACK=1
```
**5. Run full multi-agent inference**
Run the complete MacDys workflow for one subject:
```ruby
python -m app.main \
  --case-csv data/processed/features_aoi.csv \
  --subject-id 1038 \
  --mode holdout \
  --aoi-profile-json data/processed/aoi_profiles.json \
  --syllables-model-path models/final/syllables_rf.joblib \
  --meaningful-model-path models/final/meaningful_rf.joblib \
  --pseudotext-model-path models/final/pseudotext_rf.joblib \
  --pretty-print
````
Replace `1038` with the subject ID you want to evaluate.
The final assessment-support report is saved to:
```ruby
outputs/<subject_id>_assessment_report.json
```
The JSON output contains:
```
{
  "subject_id": "1038",
  "specialist_reports": [],
  "board_draft_report": {},
  "critic_report": {},
  "board_final_report": {},
  "board_report": {},
  "final_assessment": {}
}
```
A successful run should show LLM calls for the Board Agent, Critic Agent, and Board Final Agent:
```
[Board LLM] Calling model: ...
[Board LLM] Response received.
[Board] Using LLM-generated board report.
[Critic LLM] Calling model: ...
[Critic LLM] Response received.
[Critic] Using LLM-generated critic report.
[Board Final LLM] Calling model: ...
[Board Final LLM] Response received.
[Board Final] Using LLM-generated final board report.
```
