"""
Unit tests for ClaimGraphAnalyzer
"""
import pandas as pd
from src.graph.algorithms import ClaimGraphAnalyzer

def test_graph_construction():
    claims = pd.DataFrame({
        "claim_id": ["CLM-001", "CLM-002"],
        "customer_id": ["CUS-001", "CUS-002"],
        "policy_id": ["POL-001", "POL-002"],
        "device_id": ["DEV-SHARED", "DEV-SHARED"],
        "repair_shop_id": ["SHP-001", "SHP-001"],
        "address_id": ["ADDR-01", "ADDR-02"],
        "payment_account_id": ["PAY-01", "PAY-02"]
    })
    customers = pd.DataFrame({"customer_id": ["CUS-001", "CUS-002"], "address_id": ["ADDR-01", "ADDR-02"], "payment_account_id": ["PAY-01", "PAY-02"]})
    devices = pd.DataFrame({"device_id": ["DEV-SHARED"]})
    shops = pd.DataFrame({"repair_shop_id": ["SHP-001"]})
    addresses = pd.DataFrame({"address_id": ["ADDR-01", "ADDR-02"]})
    payments = pd.DataFrame({"payment_account_id": ["PAY-01", "PAY-02"]})

    analyzer = ClaimGraphAnalyzer()
    graph = analyzer.build_graph(claims, customers, devices, shops, addresses, payments)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    feats = analyzer.extract_claim_features(claims)
    assert len(feats) == 2
    assert "n_shared_devices" in feats.columns
    assert feats.loc[feats["claim_id"] == "CLM-001", "n_shared_devices"].values[0] == 1
