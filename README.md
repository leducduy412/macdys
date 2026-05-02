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

# Dataset
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






