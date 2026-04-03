# UG04 — BESSTIE Sequence Classification
**COMM061 Natural Language Processing | University of Surrey | Semester 2, 2025–26**

---

## Group Members

| Name | URN | Task |
|------|-----|------|
| Raaghav Kulshreshtha | 6776090 | Data Analysis & Visualisation |
| Qifan Zheng | 6783820 | Cross-Variety Evaluation |
| Akshyat Dali | 6805993 | Fine-tuning Pre-trained Models |
| Naeem Ahmed | 6779560 | Classical Baseline Models |

---

## Project Overview

This repository contains the code and experiments for our COMM061 group coursework. We build and evaluate sequence classification models for **Sentiment Analysis** and **Sarcasm Detection** across three varieties of English (en-AU, en-IN, en-UK) using the [BESSTIE dataset](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26).

---

## Repository Structure

```
UG04-NLP-COMM061/
│
├── notebooks/
│   ├── main.ipynb                    # Entry point — runs all experiments
│   ├── raaghav_data_analysis.ipynb   # Q1: Data analysis & visualisation
│   ├── naeem_baselines.ipynb         # Q2.1: Classical baseline models
│   ├── qifan_crossvariety.ipynb      # Q2.2: Cross-variety evaluation
│   └── akshyat_finetuning.ipynb      # Q2.3: Fine-tuning & LoRA
│
├── deployment/
│   └── app.py                        # Q5: Gradio/Streamlit deployment app
│
├── report/
│   └── report_UG04.pdf               # Final PDF report (do not ZIP this)
│
├── requirements.txt
└── README.md
```

---

## Dataset

We use the **BESSTIE** benchmark dataset, available on Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("surrey-nlp/BESSTIE-CW-26")
```

- **Varieties:** British English (en-UK), Australian English (en-AU), Indian English (en-IN)
- **Domains:** Google Places reviews, Reddit posts/comments
- **Tasks:** Sentiment (0 = Negative, 1 = Positive), Sarcasm (0 = Not Sarcastic, 1 = Sarcastic)
- **Split sizes:** ~1,200 train / ~300 val / ~300 test per variety

> **Do not commit the dataset to this repo.**

---

## Experiments

| Section | Task | Owner |
|---------|------|-------|
| Q1 | Data analysis & visualisation | Raaghav |
| Q2.1 | TF-IDF + LR/SVM vs RoBERTa baseline | Naeem |
| Q2.2 | Cross-variety evaluation matrix | Qifan |
| Q2.3 | LoRA fine-tuning on 1B–3B LLM | Akshyat |
| Q3 | Evaluation (Macro-F1, precision, recall, confusion matrices) | All |
| Q4 | Sarcasm error analysis & few-shot prompting | All |
| Q5 | Deployment endpoint + efficiency analysis | All |

---

## Setup

```bash
git clone https://github.com/raaghavkk/UG04-NLP-COMM061.git
cd UG04-NLP-COMM061
pip install -r requirements.txt
```

Key libraries: `transformers`, `datasets`, `scikit-learn`, `torch`, `peft`, `pandas`, `matplotlib`, `seaborn`, `wandb`

---

## Development Environment

- **Platform:** Google Colab Pro (A100 / T4 GPU)
- **Python:** 3.10
- **Experiment tracking:** Weights & Biases (W&B)
- **Version control:** This GitHub repository

---

## Submission Checklist

- [ ] `main.ipynb` runs end-to-end without errors
- [ ] All experiment notebooks are included and documented
- [ ] `requirements.txt` is up to date
- [ ] Report PDF is named `report_UG04.pdf` and submitted **outside** the ZIP
- [ ] No trained model weights or dataset files are committed
- [ ] No `/runs` or `/checkpoints` folders included

---

## Submission Deadline

**Wednesday 6th May 2026, 4:00 PM** — SurreyLearn
