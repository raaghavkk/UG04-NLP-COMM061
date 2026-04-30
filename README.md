# UG04 - BESSTIE Sequence Classification
**COMM061 Natural Language Processing | University of Surrey | Semester 2, 2025–26**

---

## Group Members

| Name | URN | Task |
|------|-----|------|
| Raaghav Kulshreshtha | 6776090 | Data Analysis & Visualisation + Cross-Variety Evaluation |
| Qifan Zheng | 6783820 | Evaluation & Metrics |
| Akshyat Dali | 6805993 | Monolingual vs Multilingual (XLM-RoBERTa) |
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
│   ├── main.ipynb                      # Entry point — install dependencies + overview
│   ├── raaghav_data_analysis.ipynb     # Q1: Data analysis & visualisation
│   ├── raaghav_crossvariety.ipynb      # Q2.2: Inner vs Outer Circle evaluation
│   ├── naeem_baselines.ipynb           # Q2.1: Classical baseline models
│   ├── akshyat_finetuning.ipynb        # Q2.3: Monolingual vs Multilingual
│   └── deployment.ipynb               # Q5: Gradio deployment app
│
├── report/
│   └── report_UG04.pdf                 # Final PDF report (submitted outside ZIP)
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
- **Split sizes:** en-AU (1145/95/667), en-IN (1399/117/816), en-UK (1203/101/700) — train/val/test

> **Do not commit the dataset to this repo.**

---

## Experiments

| Section | Task | Owner |
|---------|------|-------|
| Q1 | Data analysis, visualisation, vocabulary analysis & tokenisation | Raaghav |
| Q2.1 | TF-IDF + LR/SVM vs RoBERTa-base (Sentiment task) | Naeem |
| Q2.2 | Inner Circle (en-AU + en-UK) vs Outer Circle (en-IN) + balanced oversampling | Raaghav |
| Q2.3 | Monolingual (RoBERTa) vs Multilingual (XLM-RoBERTa) on Sarcasm task | Akshyat |
| Q3 | Evaluation — Macro-F1, precision, recall, confusion matrices for all experiments | Qifan |
| Q4 | Error analysis using BESSTIE typology + LIME interpretability | All |
| Q5 | Gradio deployment app + inference efficiency analysis | All |

---

## Models

Trained models are hosted on HuggingFace and loaded at runtime via `from_pretrained()`:

- `raaghavkk/roberta-sarcasm-en-AU-seed42`
- `raaghavkk/roberta-sarcasm-en-IN-seed42`
- `raaghavkk/roberta-sarcasm-en-UK-seed42`

> **Do not submit model weights in the ZIP.**

---

## Setup

```bash
git clone https://github.com/raaghavkk/UG04-NLP-COMM061.git
cd UG04-NLP-COMM061
pip install -r requirements.txt
```

Key libraries: `transformers`, `datasets`, `scikit-learn`, `torch`, `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `gradio`, `lime`

---

## Development Environment

- **Platform:** Google Colab (primary), local GPU (RTX 4070) where needed
- **GPU:** A100 / T4 via Colab Pro
- **Python:** 3.12
- **Version control:** This GitHub repository

---

## Submission Checklist

- [ ] `main.ipynb` runs without errors
- [ ] All experiment notebooks are included and documented
- [ ] `requirements.txt` is up to date
- [ ] Report PDF is named `report_UG04.pdf` and submitted **outside** the ZIP
- [ ] No trained model weights or dataset files are committed
- [ ] No `/runs` or `/checkpoints` folders included
- [ ] Declaration of originality included in report

---

## Submission Deadline

**Wednesday 1st May 2026, 4:00 PM** — SurreyLearn
