"""
ClaimGraph AI — Feature Engineering
=====================================
Ye module raw data ko ML-ready features mein convert karta hai.
Graph features alag module mein hain (src/graph/algorithms.py).

Key features (interview ke liye):
1. Claim amount vs customer ka historical average — anomaly detect karta hai
2. Claim frequency — ek customer kitni baar claim karta hai
3. Time since policy start — naye policies pe jaldi claim = suspicious
4. Categorical encoding — claim type, device type ko numbers mein convert
5. Numerical normalization — saare features same scale pe laao

IMPORTANT: Customer-level split — same customer ke claims alag splits mein nahi jaate.
"I split by customer, not by row, so the same customer's claims don't leak
across train/val/test. This prevents the model from memorizing customer-specific
patterns during training and then being tested on the same customer's other claims."
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple
import json
import os


class FeatureEngineer:
    """
    Feature engineering pipeline — raw CSV se ML-ready features tak.
    Saare transformations reproducible hain (fitted transformers save hote hain).
    """

    def __init__(self):
        self.label_encoders = {}  # Categorical → numeric mapping
        self.scaler = StandardScaler()  # Numerical normalization
        self.feature_columns = []  # Final feature list

    @property
    def feature_cols(self) -> list:
        return self.feature_columns

    def create_full_dataset(self, claims_df: pd.DataFrame,
                            customers_df: pd.DataFrame,
                            policies_df: pd.DataFrame,
                            devices_df: pd.DataFrame = None,
                            graph_features_df: pd.DataFrame = None,
                            multimodal_features_df: pd.DataFrame = None) -> pd.DataFrame:
        """Helper to chain tabular, graph, and multimodal feature engineering."""
        if devices_df is None:
            # Create a mock devices_df if not provided
            devices_df = pd.DataFrame({"device_id": claims_df["device_id"].unique() if "device_id" in claims_df.columns else []})
            devices_df["device_type"] = "smartphone"
            devices_df["purchase_date"] = "2023-01-01"
            devices_df["purchase_price"] = 500.0

        df = self.build_tabular_features(claims_df, customers_df, policies_df, devices_df)
        if graph_features_df is not None:
            df = self.add_graph_features(df, graph_features_df)
        if multimodal_features_df is not None:
            df = self.add_multimodal_features(df, multimodal_features_df)
        return df

    def build_tabular_features(self, claims_df: pd.DataFrame,

                                customers_df: pd.DataFrame,
                                policies_df: pd.DataFrame,
                                devices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tabular features banao — ye Experiment 1 (baseline) ka feature set hai.

        Features:
        1. claim_amount — raw amount
        2. amount_vs_avg — claim amount / customer ka average (anomaly signal)
        3. claim_frequency — customer ne total kitne claims kiye
        4. days_since_policy_start — policy start se claim tak ke din
        5. claim_type_encoded — categorical → numeric
        6. device_type_encoded — categorical → numeric
        7. policy_type_encoded — categorical → numeric
        8. device_age_days — device purchase se claim tak ke din
        9. premium_amount — policy premium
        10. coverage_utilization — claim_amount / coverage_limit
        """
        print("🔧 Building tabular features...")

        df = claims_df.copy()

        # === Amount-based features ===

        # Customer ka historical average claim amount
        customer_avg = df.groupby("customer_id")["claim_amount"].transform("mean")
        df["amount_vs_avg"] = df["claim_amount"] / customer_avg.replace(0, 1)

        # Customer claim frequency (kitne claims kiye)
        df["claim_frequency"] = df.groupby("customer_id")["claim_id"].transform("count")

        # === Time-based features ===

        # Policy info merge karo
        policy_info = policies_df[["policy_id", "start_date", "end_date",
                                    "premium_amount", "coverage_limit", "policy_type"]].copy()
        policy_info["start_date"] = pd.to_datetime(policy_info["start_date"])
        policy_info["end_date"] = pd.to_datetime(policy_info["end_date"])

        df = df.merge(policy_info, on="policy_id", how="left", suffixes=("", "_policy"))

        # Days since policy start — jaldi claim = suspicious
        df["claim_date"] = pd.to_datetime(df["claim_date"])
        df["days_since_policy_start"] = (df["claim_date"] - df["start_date"]).dt.days
        df["days_since_policy_start"] = df["days_since_policy_start"].fillna(0).clip(lower=0)

        # Coverage utilization — kitna coverage use ho raha hai
        df["coverage_utilization"] = df["claim_amount"] / df["coverage_limit"].replace(0, 1)

        # === Device features ===
        device_info = devices_df[["device_id", "device_type", "purchase_date", "purchase_price"]].copy()
        device_info["purchase_date"] = pd.to_datetime(device_info["purchase_date"])
        df = df.merge(device_info, on="device_id", how="left", suffixes=("", "_device"))

        # Device age at claim time
        df["device_age_days"] = (df["claim_date"] - df["purchase_date"]).dt.days
        df["device_age_days"] = df["device_age_days"].fillna(365).clip(lower=0)

        # Claim amount vs device price ratio
        df["amount_vs_device_price"] = df["claim_amount"] / df["purchase_price"].replace(0, 1)

        # === Categorical encoding ===
        categorical_cols = ["claim_type", "device_type", "policy_type"]
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Has repair shop (binary feature)
        df["has_repair_shop"] = df["repair_shop_id"].notna().astype(int)

        # === Final feature columns ===
        self.feature_columns = [
            "claim_amount", "amount_vs_avg", "claim_frequency",
            "days_since_policy_start", "coverage_utilization",
            "device_age_days", "amount_vs_device_price", "premium_amount",
            "claim_type_encoded", "device_type_encoded", "policy_type_encoded",
            "has_repair_shop"
        ]

        # Missing values handle karo — NaN ko median se fill karo, drop nahi
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        print(f"   ✅ Built {len(self.feature_columns)} tabular features")
        return df

    def add_graph_features(self, features_df: pd.DataFrame,
                           graph_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Graph features merge karo — ye Experiment 2 ka addition hai.
        Graph features src/graph/algorithms.py se aate hain.

        Added features:
        - node_degree: kitne entities se connected hai
        - community_id: Louvain community membership
        - community_size: community mein kitne nodes hain
        - pagerank_score: PageRank centrality
        - n_shared_devices: kitne claims same device share karte hain
        - n_shared_addresses: kitne claims same address share karte hain
        - n_shared_shops: kitne claims same repair shop share karte hain
        - n_shared_payments: kitne claims same payment account share karte hain
        """
        print("🔧 Adding graph features...")

        # Merge on claim_id
        df = features_df.merge(graph_features_df, on="claim_id", how="left")

        # Graph feature columns
        graph_cols = [
            "node_degree", "community_size", "pagerank_score",
            "n_shared_devices", "n_shared_addresses",
            "n_shared_shops", "n_shared_payments"
        ]

        # Missing graph features = 0 (isolated nodes)
        for col in graph_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        self.feature_columns.extend([c for c in graph_cols if c in df.columns])
        print(f"   ✅ Added {len(graph_cols)} graph features (total: {len(self.feature_columns)})")
        return df

    def add_multimodal_features(self, features_df: pd.DataFrame,
                                multimodal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Multimodal consistency score merge karo — ye Experiment 3 ka addition.
        Multimodal score src/multimodal/consistency.py se aata hai.
        """
        print("🔧 Adding multimodal features...")

        df = features_df.merge(multimodal_df[["claim_id", "consistency_score"]],
                              on="claim_id", how="left")

        # Jo claims ke paas image nahi hai, unka score neutral = 0.5
        df["consistency_score"] = df["consistency_score"].fillna(0.5)
        # Inconsistency flag — low consistency = suspicious
        df["multimodal_inconsistency"] = (df["consistency_score"] < 0.3).astype(int)

        self.feature_columns.extend(["consistency_score", "multimodal_inconsistency"])
        print(f"   ✅ Added multimodal features (total: {len(self.feature_columns)})")
        return df

    def normalize_features(self, df: pd.DataFrame,
                          fit: bool = True) -> pd.DataFrame:
        """
        Numerical features ko StandardScaler se normalize karo.
        fit=True training pe, fit=False test/inference pe.
        """
        numeric_cols = [c for c in self.feature_columns
                       if c in df.columns and df[c].dtype in ["float64", "int64", "float32"]]

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def customer_level_split(self, df: pd.DataFrame,
                             test_size: float = 0.15,
                             val_size: float = 0.15,
                             seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Customer-level split — CRITICAL for preventing data leakage.

        Kyu: Agar same customer ke claims train aur test dono mein hain,
        model customer-specific patterns memorize kar leta hai (e.g., "CUS-001234
        always files high claims") instead of generalizable fraud patterns.

        How: GroupShuffleSplit use karte hain jahan group = customer_id.
        Ek customer ke SAARE claims same split mein jaate hain.

        Interview answer: "I split by customer, not by row. If I split by row,
        the model could see customer CUS-001's first 3 claims in training and
        then be tested on their 4th claim — that's label leakage through
        customer identity."
        """
        print("🔧 Performing customer-level train/val/test split...")

        # First split: train+val vs test
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=df["customer_id"]))

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        # Second split: train vs val
        adjusted_val_size = val_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["customer_id"]))

        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

        # Verify no customer overlap
        train_customers = set(train_df["customer_id"])
        val_customers = set(val_df["customer_id"])
        test_customers = set(test_df["customer_id"])

        assert len(train_customers & val_customers) == 0, "Customer leak: train ∩ val"
        assert len(train_customers & test_customers) == 0, "Customer leak: train ∩ test"
        assert len(val_customers & test_customers) == 0, "Customer leak: val ∩ test"

        print(f"   ✅ Train: {len(train_df)} claims ({len(train_customers)} customers)")
        print(f"   ✅ Val:   {len(val_df)} claims ({len(val_customers)} customers)")
        print(f"   ✅ Test:  {len(test_df)} claims ({len(test_customers)} customers)")
        print(f"   ✅ No customer overlap between splits — leakage-free ✓")

        # Fraud distribution check karo
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            fraud_rate = split_df["is_fraud"].mean() * 100
            print(f"   📊 {split_name} fraud rate: {fraud_rate:.2f}%")

        return train_df, val_df, test_df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Feature matrix (X) aur labels (y) return karo"""
        available_features = [c for c in self.feature_columns if c in df.columns]
        X = df[available_features].copy()
        y = df["is_fraud"].astype(int)
        return X, y

    def save_feature_info(self, output_dir: str):
        """Feature engineering metadata save karo — reproducibility ke liye"""
        os.makedirs(output_dir, exist_ok=True)
        info = {
            "feature_columns": self.feature_columns,
            "label_encoders": {k: list(v.classes_) for k, v in self.label_encoders.items()},
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None else []
        }
        with open(os.path.join(output_dir, "feature_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"   ✅ Feature info saved to {output_dir}/feature_info.json")
