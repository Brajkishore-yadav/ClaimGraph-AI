"""
ClaimGraph AI — FastAPI Backend Service
=======================================
REST API providing endpoints for:
- Claim Risk Scoring & SHAP Explanations
- Graph Neighborhood & Fraud Ring Retrieval
- Multimodal Consistency Scoring
- 3-Experiment Ablation Table
- Conversational Investigation Copilot Chatbot
- System Health & Monitoring Metrics
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.graph import ClaimInvestigationAgent
from src.risk.scoring import RiskScoringEngine
from src.graph.algorithms import ClaimGraphAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claimgraph_api")

app = FastAPI(
    title="ClaimGraph AI API",
    description="Portfolio-grade Fraud Intelligence Platform for Assurant 2027 Data Science & Analytics",
    version="1.0.0"
)

# Enable CORS for local & cloud Streamlit frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Global Singletons
agent = ClaimInvestigationAgent(data_dir=str(DATA_DIR), models_dir=str(MODEL_DIR))

# Pydantic Schemas
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    response: str
    sources: List[str]
    claim_id: Optional[str] = None
    ring_id: Optional[str] = None

class ScoreClaimRequest(BaseModel):
    claim_id: str
    claim_amount: float
    customer_id: str
    policy_id: str
    device_id: str
    claim_type: str = "accidental_damage"
    description: str = "Screen cracked after accidental drop"

# Endpoints

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "ClaimGraph AI",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": (MODEL_DIR / "exp3_plus_multimodal_xgboost.joblib").exists(),
        "data_processed": (DATA_DIR / "scored_claims.csv").exists()
    }

@app.get("/demo-claim")
def get_demo_claim():
    """Return pre-computed evaluation & details for demo claim CLM-DEMO-001"""
    scored_file = DATA_DIR / "scored_claims.csv"
    if scored_file.exists():
        df = pd.read_csv(scored_file)
        match = df[df["claim_id"] == "CLM-DEMO-001"]
        if not match.empty:
            return match.iloc[0].to_dict()

    return {
        "claim_id": "CLM-DEMO-001",
        "customer_id": "CUS-DEMO-99",
        "risk_level": "HIGH",
        "risk_score": 0.999,
        "recommendation": "ESCALATE_FOR_INVESTIGATION",
        "ml_probability": 0.999,
        "anomaly_score": 0.85,
        "graph_risk_indicator": True,
        "multimodal_inconsistency": True,
        "top_risk_factors": [
            "⚠️ Part of flagged community (graph signal)",
            "⚠️ Image-text inconsistency detected",
            "n_shared_devices: ↑ fraud (0.412)",
            "amount_vs_avg: ↑ fraud (0.354)"
        ]
    }

@app.get("/claims/{claim_id}")
def get_claim_details(claim_id: str):
    """Retrieve details and risk score for a specific claim"""
    scored_file = DATA_DIR / "scored_claims.csv"
    if scored_file.exists():
        df = pd.read_csv(scored_file)
        match = df[df["claim_id"] == claim_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

@app.get("/graph/rings")
def get_fraud_rings():
    """Retrieve all detected fraud rings from knowledge graph analysis"""
    rings_file = DATA_DIR / "detected_fraud_rings.csv"
    if rings_file.exists():
        df = pd.read_csv(rings_file)
        return {
            "total_rings": int(df["detected_ring_id"].nunique()),
            "total_flagged_claims": len(df),
            "rings": df.to_dict(orient="records")
        }
    return {"total_rings": 0, "rings": []}

@app.get("/experiments/ablation")
def get_ablation_results():
    """Return the centerpiece 3-experiment ablation study table"""
    ablation_file = DATA_DIR / "experiments" / "ablation_table.csv"
    if ablation_file.exists():
        df = pd.read_csv(ablation_file)
        return {
            "experiments": df.to_dict(orient="records")
        }
    return {"experiments": []}

@app.get("/validation/report")
def get_validation_report():
    """Return data validation report"""
    val_file = DATA_DIR / "validation_report.json"
    if val_file.exists():
        with open(val_file) as f:
            return json.load(f)
    return {"status": "No validation report found."}

@app.post("/chat", response_model=ChatResponse)
def copilot_chat(req: ChatRequest):
    """Conversational Copilot powered by LangGraph"""
    result = agent.run(req.message)
    return ChatResponse(**result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
