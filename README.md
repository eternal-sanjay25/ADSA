# ADSA — Autonomous Data Scientist Agent v2

> Upload any dataset → Get automated ML analysis with insights, charts, and a PDF report.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## What It Does

ADSA is a **multi-agent AI pipeline** that automatically:

1. **Plans** — Detects task type (classification/regression/clustering/timeseries)
2. **Profiles** — Analyzes data types, missing values, duplicates, memory usage
3. **Cleans** — Smart imputation (mean/median/KNN), outlier capping, date conversion
4. **Explores** — EDA with correlations, distributions, target analysis
5. **Visualizes** — Dark-themed charts (histograms, boxplots, heatmaps)
6. **Engineers Features** — Encoding, scaling, mutual information selection
7. **Trains Models** — LogisticRegression, RandomForest, XGBoost with cross-validation
8. **Optimizes** — Bayesian hyperparameter tuning with Optuna (20 trials)
9. **Evaluates** — Holdout metrics, confusion matrix, residual plots
10. **Explains** — SHAP feature importance (summary + bar plots)
11. **Generates Insights** — LLM-powered business insights (Gemini Flash)
12. **Reports** — Professional PDF report with all results embedded
13. **Chats** — RAG-based Q&A about your analysis results

## Tech Stack

| Category | Tool |
|----------|------|
| Agent Orchestration | LangGraph |
| LLM (Free) | Google Gemini Flash |
| Data Processing | Pandas |
| ML Models | XGBoost, scikit-learn |
| HPO | Optuna |
| Explainability | SHAP |
| Embeddings | sentence-transformers |
| Vector DB | FAISS |
| Visualization | Matplotlib + Seaborn |
| PDF Reports | ReportLab |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/ADSA.git
cd ADSA
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure API Key (Optional)

```bash
# Get free key at: aistudio.google.com
# Edit .env file:
GOOGLE_API_KEY=your_key_here
```

> Without an API key, ADSA still works — LLM features (insights, chat) fall back to heuristics.

### 3. Generate Sample Data

```bash
python generate_sample_data.py
```

### 4. Run the Server

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker-compose up --build
```

Open **http://localhost:7860**

## Project Structure

```
ADSA/
├── agents/           # 13 specialized AI agents
│   ├── planner.py    # Task detection & planning
│   ├── profiler.py   # Data profiling
│   ├── cleaner.py    # Data cleaning & imputation
│   ├── eda.py        # Exploratory data analysis
│   ├── visualizer.py # Chart generation
│   ├── feature_eng.py# Feature engineering
│   ├── modeler.py    # Model training
│   ├── hpo.py        # Hyperparameter optimization
│   ├── evaluator.py  # Model evaluation
│   ├── explainer.py  # SHAP explainability
│   ├── insight.py    # Business insight generation
│   ├── reporter.py   # PDF report generation
│   └── chat.py       # RAG-based Q&A
├── core/
│   ├── state.py      # Pipeline state definition
│   └── graph.py      # LangGraph pipeline
├── utils/
│   ├── llm_client.py # LLM with fallback chain
│   ├── embedder.py   # FAISS vector store
│   └── audit.py      # Audit logging
├── backend/
│   └── main.py       # FastAPI server
├── frontend/
│   ├── index.html    # Premium dark UI
│   ├── index.css     # Glassmorphism design system
│   └── app.js        # SSE streaming & chat
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## License

MIT — Built by Sanjay ([eternal-sanjay25](https://github.com/eternal-sanjay25))
