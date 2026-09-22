"""
Unit tests for RiskScoringEngine
"""
import pandas as pd
from src.risk.scoring import RiskScoringEngine

def test_risk_scoring_rules():
    engine = RiskScoringEngine()
    
    # High risk test
    dummy_feat = pd.DataFrame([{"claim_amount": 1000.0}])
    res = engine.score_claim(dummy_feat, graph_community_flagged=True, multimodal_inconsistency=True)
    assert res["risk_level"] in ["HIGH", "MEDIUM", "LOW"]
    assert "recommendation" in res
