<div align="center">

# 🛡️ FakeGuard AI
### Real-Time Fake Review Detection & Advanced Multilingual Sentiment Analysis

Final Year Project (FYP) — a two-in-one ML system that detects fake reviews on the
Google Play Store and performs multilingual sentiment analysis on user reviews,
each shipped with its own Streamlit web app.

</div>

---

## 📌 Overview

**FakeGuard AI** bundles two independent but complementary ML pipelines built for
a single FYP:

| Module | Purpose | Key Result |
|---|---|---|
| **Fake Review Detector** | Scrapes live Google Play reviews and flags fake/spam reviews using a rule-based + LightGBM ensemble | ROC-AUC ≈ 0.99 |
| **Sentiment Analysis Engine** | Multilingual (incl. code-mixed/Roman Urdu) sentiment classification using lexicon, rule-based and transformer features combined into one ensemble model | Balanced multi-class sentiment scoring |

Both modules share the same underlying review dataset (scraped from the Google
Play Store) but were trained and deployed as separate systems, each with its own
Streamlit frontend.

---

## 🏗️ Architecture

```
Google Play Store
      │
      ▼
 Scraper (google-play-scraper)
      │
      ▼
 Preprocessing & Feature Engineering
      │
      ├──────────────────────────┐
      ▼                          ▼
Fake Review Pipeline      Sentiment Pipeline
(Rule Scoring + LightGBM)  (Lexicon + Rules + Transformer + Ensemble)
      │                          │
      ▼                          ▼
 fake_review_app (Streamlit)   sentiment_app (Streamlit)
```

See `images/architecture.png` for the full diagram.

---

## 📂 Repository Structure

```
FakeGuard-AI/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                # Raw scraped Google Play reviews
│   ├── processed/          # Cleaned / preprocessed data
│   └── generated/          # Feature-engineered datasets
│
├── notebooks/
│   ├── 01_dataset_creation.ipynb          # Scraping + dataset assembly
│   ├── 02_data_preprocessing.ipynb        # Cleaning, EDA, final dataset
│   ├── 03_fake_review_training.ipynb      # Fake review model (ROC-AUC 0.99)
│   └── 04_sentiment_analysis_training.ipynb  # Multilingual sentiment training
│
├── src/
│   ├── data/
│   │   ├── create_dataset.py       # Scraper (adapted from Phase 1)
│   │   ├── preprocess.py           # Language detection / cleaning (Phase 1)
│   │   └── feature_engineering.py  # Metadata / feature extraction (Phase 3)
│   │
│   ├── fake_review/
│   │   ├── train.py                # LightGBM training script
│   │   ├── predict.py              # Ensemble detector (rule + LightGBM)
│   │   └── model.py                # Rule-based scoring engine
│   │
│   ├── sentiment/
│   │   ├── train.py                # Ensemble sentiment training script
│   │   ├── predict.py              # Prediction pipeline (Phase 8)
│   │   └── model.py                # Text preprocessing (Phase 2)
│   │
│   └── utils.py
│
├── frontend/
│   ├── fake_review_app/       # Standalone Streamlit app — Fake Review Detector
│   │   ├── app.py
│   │   ├── pipeline/          # scraper.py, scorer.py, detector.py, reporter.py
│   │   ├── models/            # local copy of the fake-review model artifacts
│   │   ├── pages/
│   │   └── assets/
│   │
│   └── sentiment_app/         # Standalone Streamlit app — Sentiment Analysis
│       ├── app.py
│       ├── modules/           # phase1..phase9 processing modules
│       ├── utils/
│       ├── models/            # local copy of the sentiment model artifacts
│       ├── pages/
│       └── assets/
│
├── models/
│   ├── fake_review/    # tfidf_vectorizer.pkl, lgbm_model.txt, feature_names.pkl
│   └── sentiment/      # best_model_combine.pkl, tfidf/bow vectorizers, etc.
│
├── outputs/
│   ├── reports/       # Generated detection reports (.txt)
│   ├── figures/       # Charts and plots
│   └── predictions/   # Batch prediction exports
│
├── images/            # architecture.png, fakeguard_demo.png, sentiment_demo.png
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/FakeGuard-AI.git
cd FakeGuard-AI
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Apps

### 1. Fake Review Detector

```bash
cd frontend/fake_review_app
streamlit run app.py
```

Paste a Google Play Store app URL/ID, scrape live reviews, and get each review
scored as genuine or fake with a confidence score, plus a downloadable report.

### 2. Sentiment Analysis App

```bash
cd frontend/sentiment_app
streamlit run app.py
```

Upload reviews or paste text (supports multilingual/code-mixed input) to get
sentiment predictions with rule-based, lexicon, and transformer-driven insight.

---

## 🧠 Models

| Model | Location | Description |
|---|---|---|
| LightGBM (Fake Review) | `models/fake_review/lgbm_model.txt` | Gradient-boosted classifier on TF-IDF + engineered features |
| TF-IDF Vectorizer (Fake Review) | `models/fake_review/tfidf_vectorizer.pkl` | Text vectorizer for similarity/duplication scoring |
| Ensemble Model (Sentiment) | `models/sentiment/best_model_combine.pkl` | Combined classical + transformer-feature ensemble |
| BoW / TF-IDF Vectorizers (Sentiment) | `models/sentiment/*.pkl` | Feature extraction for the sentiment pipeline |

> Large `.pkl` model files are tracked directly in this repo for convenience.
> If GitHub file-size limits become an issue, switch to
> [Git LFS](https://git-lfs.com/) for the `models/` directory.

---

## 📊 Dataset

The dataset was built from scratch by scraping live Google Play Store reviews
(see `notebooks/01_dataset_creation.ipynb`), then cleaned, language-tagged, and
feature-engineered (see `notebooks/02_data_preprocessing.ipynb`) before being
split for the two downstream tasks (fake-review detection and sentiment
analysis).

---

## 🖼️ Demo

- `images/architecture.png` — system architecture diagram.
- `outputs/figures/sample_ensemble_scores.png` — real output of `FakeReviewDetector` run on sample reviews (duplicate/spam reviews correctly flagged `REMOVE`).
- `outputs/figures/sentiment_model_comparison.png` — real model comparison (F1-scores) pulled from `notebooks/02_data_preprocessing.ipynb` training results (XGBoost: 0.994 F1).
- `outputs/predictions/sample_predictions.csv` and `outputs/reports/sample_report.txt` — real end-to-end output from running the fake-review pipeline in this repo.

> App screenshots (`images/fakeguard_demo.png`, `images/sentiment_demo.png`) still need to be added — run the two Streamlit apps and capture the UI.

---

## 🛠️ Tech Stack

`Python` · `Pandas` / `NumPy` · `Scikit-learn` · `LightGBM` · `XGBoost` ·
`Transformers (HuggingFace)` · `VADER Sentiment` · `Streamlit` · `Plotly` /
`Matplotlib` / `Seaborn` · `google-play-scraper`

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

## 🙋 Author

Final Year Project — **FakeGuard AI**
Real-Time Fake Review Detection & Advanced Multilingual Sentiment Analysis
