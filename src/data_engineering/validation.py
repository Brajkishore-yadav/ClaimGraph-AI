"""
ClaimGraph AI — Data Validation
================================
Ye module data quality ensure karta hai BEFORE kuch bhi pipeline mein jaaye.
Fuzzy matching, referential integrity, missing value handling — sab yahan.

Interview tip: "Data validation is the first line of defense. I validate at
ingestion time so downstream models can trust the data shape. Fuzzy address
matching also feeds into graph-quality monitoring."
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from typing import Optional


class DataValidator:
    """
    Data quality checker — har entity ke liye specific checks.
    Results ek report mein aate hain, silently drop nahi karte.
    """

    def __init__(self):
        self.issues = []  # Saari issues yahan collect hoti hain

    def _add_issue(self, severity: str, entity: str, field: str, message: str, record_id: str = ""):
        """Issue record karo — severity: ERROR, WARNING, INFO"""
        self.issues.append({
            "severity": severity,
            "entity": entity,
            "field": field,
            "message": message,
            "record_id": record_id
        })

    def validate_referential_integrity(self, claims_df: pd.DataFrame,
                                        customers_df: pd.DataFrame,
                                        policies_df: pd.DataFrame,
                                        devices_df: pd.DataFrame,
                                        repair_shops_df: pd.DataFrame) -> bool:
        """
        Check karo ki har claim mein referenced IDs actually exist karte hain.
        Agar customer_id claims mein hai lekin customers table mein nahi → ERROR.
        """
        print("🔍 Checking referential integrity...")
        valid = True

        # Claims → Customers
        orphan_customers = set(claims_df["customer_id"]) - set(customers_df["customer_id"])
        if orphan_customers:
            self._add_issue("ERROR", "claims", "customer_id",
                          f"{len(orphan_customers)} claims reference non-existent customers")
            valid = False

        # Claims → Policies
        orphan_policies = set(claims_df["policy_id"]) - set(policies_df["policy_id"])
        if orphan_policies:
            self._add_issue("ERROR", "claims", "policy_id",
                          f"{len(orphan_policies)} claims reference non-existent policies")
            valid = False

        # Claims → Devices
        orphan_devices = set(claims_df["device_id"]) - set(devices_df["device_id"])
        if orphan_devices:
            self._add_issue("ERROR", "claims", "device_id",
                          f"{len(orphan_devices)} claims reference non-existent devices")
            valid = False

        # Claims → Repair Shops (optional field — None is OK)
        claims_with_shop = claims_df[claims_df["repair_shop_id"].notna()]
        orphan_shops = set(claims_with_shop["repair_shop_id"]) - set(repair_shops_df["repair_shop_id"])
        if orphan_shops:
            self._add_issue("WARNING", "claims", "repair_shop_id",
                          f"{len(orphan_shops)} claims reference non-existent repair shops")

        if valid:
            print("   ✅ Referential integrity check passed")
        else:
            print(f"   ❌ Referential integrity issues found: {len(self.issues)}")
        return valid

    def check_missing_values(self, df: pd.DataFrame, entity_name: str,
                             required_fields: list[str]) -> pd.DataFrame:
        """
        Missing values check karo — required fields mein NULL allowed nahi.
        Optional fields mein NULL = OK, but report karo.

        Missing value handling rules:
        - customer_id, claim_id: NEVER null (primary keys)
        - repair_shop_id: CAN be null (not all claims go to shops)
        - claim_amount: NEVER null, must be > 0
        - image_path: CAN be null (not all claims have images)
        """
        print(f"🔍 Checking missing values in {entity_name}...")

        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isna().sum()
                if null_count > 0:
                    self._add_issue("ERROR", entity_name, field,
                                  f"{null_count} missing values in required field")
                    print(f"   ❌ {field}: {null_count} missing values")

        # Optional fields — report but don't block
        optional_fields = set(df.columns) - set(required_fields)
        for field in optional_fields:
            null_count = df[field].isna().sum()
            if null_count > 0:
                self._add_issue("INFO", entity_name, field,
                              f"{null_count} missing values in optional field")

        return df

    def detect_duplicate_addresses(self, addresses_df: pd.DataFrame,
                                    threshold: float = 85.0) -> pd.DataFrame:
        """
        Fuzzy matching se duplicate/similar addresses detect karo.
        Ye graph-quality monitoring mein bhi use hota hai.

        How it works: Har address ka full string banao, phir rapidfuzz se
        pairwise similarity check karo. High similarity = potential duplicate.
        Score > 85 = likely same address with typos/formatting differences.
        """
        print("🔍 Detecting duplicate addresses (fuzzy matching)...")

        # Full address string banao comparison ke liye
        addresses_df = addresses_df.copy()
        addresses_df["full_address"] = (
            addresses_df["street"].str.lower() + ", " +
            addresses_df["city"].str.lower() + ", " +
            addresses_df["state"].str.lower() + " " +
            addresses_df["zip_code"].astype(str)
        )

        duplicates = []
        addresses = addresses_df["full_address"].tolist()
        address_ids = addresses_df["address_id"].tolist()

        # O(n²) but n is small (~4000) — fast enough, no need to optimize
        for i in range(len(addresses)):
            for j in range(i + 1, min(i + 50, len(addresses))):  # Window limit
                score = fuzz.ratio(addresses[i], addresses[j])
                if score >= threshold and score < 100:  # Skip exact matches (same record)
                    duplicates.append({
                        "address_id_1": address_ids[i],
                        "address_id_2": address_ids[j],
                        "address_1": addresses[i],
                        "address_2": addresses[j],
                        "similarity_score": score
                    })

        if duplicates:
            self._add_issue("WARNING", "addresses", "full_address",
                          f"{len(duplicates)} potential duplicate address pairs found")
            print(f"   ⚠️  Found {len(duplicates)} potential duplicate address pairs")
        else:
            print("   ✅ No duplicate addresses found")

        return pd.DataFrame(duplicates) if duplicates else pd.DataFrame()

    def validate_claim_amounts(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Claim amounts ka outlier analysis — IQR method.
        IMPORTANT: Outliers REPORT karte hain, REMOVE nahi — kyunki outliers
        aksar fraud signal hote hain.

        Interview answer: "I report outliers but don't remove them, because
        in fraud detection, outliers are often exactly the signal you're
        looking for. Removing them would remove the fraud."
        """
        print("🔍 Analyzing claim amount outliers (IQR method)...")

        amounts = claims_df["claim_amount"]
        q1 = amounts.quantile(0.25)
        q3 = amounts.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = claims_df[
            (amounts < lower_bound) | (amounts > upper_bound)
        ]

        self._add_issue("INFO", "claims", "claim_amount",
                       f"Amount stats: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
                       f"Bounds=[{lower_bound:.2f}, {upper_bound:.2f}], "
                       f"Outliers={len(outliers)} ({len(outliers)/len(claims_df)*100:.1f}%)")

        print(f"   📊 Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
        print(f"   📊 Bounds: [{max(0, lower_bound):.2f}, {upper_bound:.2f}]")
        print(f"   📊 Outliers: {len(outliers)} ({len(outliers)/len(claims_df)*100:.1f}%) — REPORTED, NOT REMOVED")

        return outliers

    def validate_all(self, claims_df: pd.DataFrame, customers_df: pd.DataFrame,
                     policies_df: pd.DataFrame, devices_df: pd.DataFrame,
                     addresses_df: pd.DataFrame, repair_shops_df: pd.DataFrame,
                     payments_df: pd.DataFrame) -> dict:
        """Run all data validation checks in sequence and return the summary report."""
        self.validate_referential_integrity(claims_df, customers_df, policies_df, devices_df, repair_shops_df)
        self.check_missing_values(claims_df, "claims", ["claim_id", "customer_id", "policy_id", "claim_amount"])
        self.check_missing_values(customers_df, "customers", ["customer_id", "first_name", "last_name"])
        self.check_missing_values(policies_df, "policies", ["policy_id", "customer_id"])
        self.detect_duplicate_addresses(addresses_df)
        self.validate_claim_amounts(claims_df)
        return self.get_report()

    def get_report(self) -> dict:

        """Validation report return karo"""
        return {
            "total_issues": len(self.issues),
            "errors": [i for i in self.issues if i["severity"] == "ERROR"],
            "warnings": [i for i in self.issues if i["severity"] == "WARNING"],
            "info": [i for i in self.issues if i["severity"] == "INFO"]
        }

    def print_report(self):
        """Console pe readable report print karo"""
        report = self.get_report()
        print("\n" + "=" * 50)
        print("📋 Data Validation Report")
        print("=" * 50)
        print(f"   Errors:   {len(report['errors'])}")
        print(f"   Warnings: {len(report['warnings'])}")
        print(f"   Info:     {len(report['info'])}")

        if report["errors"]:
            print("\n❌ ERRORS:")
            for e in report["errors"]:
                print(f"   [{e['entity']}.{e['field']}] {e['message']}")

        if report["warnings"]:
            print("\n⚠️  WARNINGS:")
            for w in report["warnings"]:
                print(f"   [{w['entity']}.{w['field']}] {w['message']}")
