"""
Unit tests for DataValidator
"""
import pandas as pd
from src.data_engineering.validation import DataValidator

def test_data_validator_integrity():
    claims = pd.DataFrame({"claim_id": ["CLM-001"], "customer_id": ["CUS-001"], "policy_id": ["POL-001"], "device_id": ["DEV-001"], "repair_shop_id": [None], "claim_amount": [500.0]})
    customers = pd.DataFrame({"customer_id": ["CUS-001"], "first_name": ["John"], "last_name": ["Doe"]})
    policies = pd.DataFrame({"policy_id": ["POL-001"], "customer_id": ["CUS-001"]})
    devices = pd.DataFrame({"device_id": ["DEV-001"]})
    shops = pd.DataFrame({"repair_shop_id": []})
    addresses = pd.DataFrame({"address_id": ["ADDR-01"], "street": ["123 Main St"], "city": ["New York"], "state": ["NY"], "zip_code": ["10001"]})
    payments = pd.DataFrame({"payment_account_id": ["PAY-01"]})

    validator = DataValidator()
    report = validator.validate_all(claims, customers, policies, devices, addresses, shops, payments)
    assert report["errors"] == []
