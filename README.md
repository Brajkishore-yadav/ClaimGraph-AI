# 🛡️ ClaimGraph AI — Graph-Based Fraud Intelligence Platform

An AI-powered fraud intelligence platform combining machine learning, graph analytics, multimodal evidence and an investigator copilot.

---

## 🌟 Executive Summary & Key Highlights
ClaimGraph AI is an end-to-end insurance fraud detection platform designed to identify organized fraud syndicates (shared devices, repair shops, payment accounts) that traditional isolated tabular ML models miss.

- **Headline Benchmark Result**: **PR-AUC 0.8684** (+0.0236 improvement over tabular baseline).
- **Core Architecture**: Heterogeneous NetworkX Knowledge Graph + XGBoost Classifier + Multimodal PIL Consistency Auditor + LangGraph Agent Copilot.
- **Data Quality Safeguard**: Leakage-free customer-level split (`GroupShuffleSplit`) and complete ground-truth segregation.

---

## 🧪 The Centerpiece: 3-Experiment Ablation Study

| Experiment | Model | Features Included | N Features | Precision | Recall | F1 Score | ROC-AUC | **PR-AUC (Headline)** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** | Logistic Regression | Tabular features | 12 | 0.3200 | 0.9412 | 0.4776 | 0.9645 | 0.7336 |
| **Exp 1: Baseline** | **XGBoost** | Tabular features | 12 | 0.7647 | 0.7647 | 0.7647 | 0.9601 | **0.8448** |
| **Exp 2: + Graph** | Logistic Regression | Tabular + Graph Features | 19 | 0.2857 | 0.9412 | 0.4384 | 0.9654 | 0.7655 |
| **Exp 2: + Graph** | **XGBoost** | Tabular + Graph Features | 19 | 0.7778 | 0.8235 | 0.8000 | 0.9467 | **0.8673** |
| **Exp 3: + Multimodal** | Logistic Regression | Tabular + Graph + Image Score | 21 | 0.2857 | 0.9412 | 0.4384 | 0.9654 | 0.7666 |
| **Exp 3: + Multimodal** | **XGBoost** | Tabular + Graph + Image Score | 21 | 0.7778 | 0.8235 | 0.8000 | 0.9496 | **0.8684** |

> **95% Bootstrap Confidence Interval for Top Model PR-AUC**: `[0.6979, 0.9879]` (1,000 iterations).

---

## 🚀 Quickstart & Pipeline Commands

```bash
# 1. Clone & install dependencies
git clone https://github.com/Brajkishore-yadav/ClaimGraph-AI.git
cd ClaimGraph-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run Synthetic Data Generation
python scripts/generate_data.py --customers 3000 --policies 5000 --claims 7000 --seed 42

# 3. Execute End-to-End Pipeline
python scripts/run_pipeline.py

# 4. Launch Streamlit Analytics Dashboard
streamlit run frontend/app.py

# 5. Launch FastAPI Backend
uvicorn src.api.main:app --reload
```

---

## 🐋 Docker & Local Compose Deployment

```bash
# Spin up complete system (FastAPI, Streamlit, PostgreSQL, Neo4j)
docker-compose up --build
```

---

## 📂 Repository Structure

```
ClaimGraph-AI/
├── .github/workflows/ci.yml    # CI automated testing pipeline
├── data/                       # Generated & processed datasets
├── docs/                       # Comprehensive documentation & PowerPoint deck
│   ├── FRAUD_INJECTION.md
│   ├── RISK_SCORING.md
│   ├── INTERVIEW_PREP.md
│   └── ClaimGraph_AI_Presentation.pptx
├── frontend/
│   └── app.py                  # 5-page Streamlit Dashboard
├── models/                     # Saved trained models & scalers
├── scripts/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── run_pipeline.py         # Master pipeline script
│   └── generate_pptx.py        # PowerPoint deck generator
├── src/
│   ├── agent/                  # LangGraph Copilot agent
│   ├── api/                    # FastAPI REST API endpoints
│   ├── data_engineering/       # Validation, schema & features
│   ├── graph/                  # NetworkX & Louvain community detection
│   ├── ml/                     # Model training & ablation experiments
│   ├── multimodal/             # Image-text consistency scoring
│   └── risk/                   # Risk scoring engine & SHAP explanations
├── tests/                      # Pytest unit tests
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

## 📚 Technical Preparation Guide
For detailed technical Q&A defense on graph algorithms, PR-AUC vs ROC-AUC, and class imbalance handling, refer to [`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md).
