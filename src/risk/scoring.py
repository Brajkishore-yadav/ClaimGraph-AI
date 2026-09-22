"""
ClaimGraph AI — Risk Scoring Engine
======================================
3 signals combine karke ek final risk score banao.

Signals:
1. ML fraud probability (XGBoost) — primary signal
2. Graph/community risk indicator — booster
3. Multimodal inconsistency score — booster

Plus: Isolation Forest anomaly score — unsupervised check

Rule-based escalation (NOT opaque linear blend):
- ML prob > 0.7 → HIGH RISK
- ML prob 0.4-0.7 AND (flagged community OR multimodal inconsistency) → HIGH RISK
- ML prob 0.4-0.7 alone → MEDIUM RISK
- ML prob < 0.4 AND anomaly detected → MEDIUM RISK
- ML prob < 0.4 → LOW RISK

IMPORTANT: System sirf RECOMMEND karta hai — APPROVE, REVIEW, ESCALATE.
KABHI autonomous approve/deny decision nahi.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional

# SHAP import — conditional to avoid import errors
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class RiskScoringEngine:
    """
    Risk scoring engine — explainable, rule-based, auditable.

    Interview tip: "Instead of a black-box weighted average, I use the ML
    model's calibrated probability as the primary score, with graph-community
    and multimodal flags as boosters. This is easier to explain to an
    investigator — they can see exactly why a claim was escalated."
    """

    def __init__(self, model_path: str = None):
        self.ml_model = None
        self.isolation_forest = None
        self.shap_explainer = None

        if model_path and os.path.exists(model_path):
            self.ml_model = joblib.load(model_path)
            print(f"   ✅ ML model loaded from {model_path}")

    def fit_isolation_forest(self, X_train: pd.DataFrame, seed: int = 42):
        """
        Isolation Forest fit karo — unsupervised anomaly detection.

        Ye supervised model ka complement hai:
        - Supervised model known fraud patterns detect karta hai
        - Isolation Forest UNKNOWN anomalies detect karta hai
          jo training data mein nahi the

        Interview answer: "Isolation Forest randomly splits features and measures
        how quickly a data point gets isolated. Anomalies get isolated quickly
        because they're different from the majority. I use it as a safety net
        for fraud patterns the supervised model wasn't trained on."
        """
        print("🌲 Fitting Isolation Forest for anomaly detection...")
        X_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # ~5% anomalies expected
            random_state=seed
        )
        self.isolation_forest.fit(X_clean)
        print("   ✅ Isolation Forest fitted")

    def setup_shap_explainer(self, X_background: pd.DataFrame):
        """
        SHAP explainer setup karo — TreeExplainer for XGBoost.

        SHAP batata hai ki har feature ne prediction ko kitna push kiya
        (positive = toward fraud, negative = toward legitimate).

        Interview answer: "SHAP shows how much each feature pushed the
        prediction up or down for THIS SPECIFIC claim. It's based on
        game theory — each feature gets a fair credit for its contribution."
        """
        if not SHAP_AVAILABLE:
            print("   ⚠️ SHAP not available, skipping explainer setup")
            return

        if self.ml_model is None:
            print("   ⚠️ No ML model loaded, skipping SHAP setup")
            return

        print("🔍 Setting up SHAP TreeExplainer...")
        X_bg = X_background.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Use a sample for background (faster)
        sample_size = min(100, len(X_bg))
        X_sample = X_bg.sample(n=sample_size, random_state=42)
        self.shap_explainer = shap.TreeExplainer(self.ml_model, X_sample)
        print("   ✅ SHAP explainer ready")

    def score_claim(self, features: pd.DataFrame,
                    graph_community_flagged: bool = False,
                    multimodal_inconsistency: bool = False,
                    feature_names: list = None) -> Dict:
        """
        Single claim ko score karo — all signals combine karke.

        Returns:
        {
            risk_level: "HIGH" / "MEDIUM" / "LOW",
            risk_score: 0.0-1.0,
            recommendation: "APPROVE" / "REVIEW" / "ESCALATE_FOR_INVESTIGATION",
            ml_probability: float,
            anomaly_score: float,
            graph_risk: bool,
            multimodal_risk: bool,
            top_factors: [...],
            shap_values: {...}
        }
        """
        features_clean = features.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 1. ML probability
        if self.ml_model is not None:
            ml_prob = float(self.ml_model.predict_proba(features_clean)[:, 1][0])
        else:
            ml_prob = 0.5  # Default if no model

        # 2. Isolation Forest anomaly score
        if self.isolation_forest is not None:
            anomaly_raw = self.isolation_forest.decision_function(features_clean)[0]
            # Convert to 0-1 (lower = more anomalous)
            anomaly_score = max(0, min(1, 0.5 - anomaly_raw))
        else:
            anomaly_score = 0.0

        # 3. SHAP explanation
        shap_dict = {}
        top_factors = []
        if self.shap_explainer is not None and feature_names:
            try:
                shap_values = self.shap_explainer.shap_values(features_clean)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]  # Class 1 (fraud)
                else:
                    shap_vals = shap_values[0]

                # Top contributing features
                for fname, sval in sorted(zip(feature_names, shap_vals),
                                         key=lambda x: abs(x[1]), reverse=True)[:5]:
                    shap_dict[fname] = round(float(sval), 4)
                    direction = "↑ fraud" if sval > 0 else "↓ legitimate"
                    top_factors.append(f"{fname}: {direction} ({abs(sval):.3f})")
            except Exception:
                pass

        # === RISK SCORING RULES (documented in docs/RISK_SCORING.md) ===

        if ml_prob > 0.7:
            risk_level = "HIGH"
            risk_score = ml_prob
            recommendation = "ESCALATE_FOR_INVESTIGATION"
        elif ml_prob > 0.4:
            if graph_community_flagged or multimodal_inconsistency:
                # Borderline ML + graph/multimodal signal = escalate
                risk_level = "HIGH"
                risk_score = min(ml_prob + 0.15, 0.95)
                recommendation = "ESCALATE_FOR_INVESTIGATION"
                if graph_community_flagged:
                    top_factors.insert(0, "⚠️ Part of flagged community (graph signal)")
                if multimodal_inconsistency:
                    top_factors.insert(0, "⚠️ Image-text inconsistency detected")
            else:
                risk_level = "MEDIUM"
                risk_score = ml_prob
                recommendation = "REVIEW"
        elif anomaly_score > 0.6:
            risk_level = "MEDIUM"
            risk_score = max(ml_prob, anomaly_score * 0.7)
            recommendation = "REVIEW"
            top_factors.insert(0, "⚠️ Isolation Forest anomaly detected")
        else:
            risk_level = "LOW"
            risk_score = ml_prob
            recommendation = "APPROVE"

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 4),
            "recommendation": recommendation,
            "ml_probability": round(ml_prob, 4),
            "anomaly_score": round(anomaly_score, 4),
            "graph_risk_indicator": graph_community_flagged,
            "multimodal_inconsistency": multimodal_inconsistency,
            "top_risk_factors": top_factors[:5],
            "shap_values": shap_dict
        }

    def score_all_claims(self, features_df: pd.DataFrame,
                         graph_features_df: pd.DataFrame = None,
                         feature_columns: list = None) -> pd.DataFrame:
        """
        Saare claims ko score karo — batch processing.
        """
        print("🎯 Scoring all claims...")

        results = []
        for idx, row in features_df.iterrows():
            claim_id = row.get("claim_id", f"CLM-{idx}")

            # Get feature vector
            if feature_columns:
                available_cols = [c for c in feature_columns if c in features_df.columns]
                feat = pd.DataFrame([row[available_cols]])
            else:
                feat = pd.DataFrame([row.drop(["claim_id", "is_fraud"], errors="ignore")])

            # Graph flags
            graph_flagged = False
            if graph_features_df is not None and claim_id in graph_features_df["claim_id"].values:
                gf = graph_features_df[graph_features_df["claim_id"] == claim_id].iloc[0]
                graph_flagged = gf.get("community_size", 0) >= 4 and gf.get("n_shared_devices", 0) >= 2

            # Multimodal flag
            mm_inconsistency = row.get("multimodal_inconsistency", 0) == 1

            result = self.score_claim(
                feat, graph_flagged, mm_inconsistency,
                feature_names=list(feat.columns) if feature_columns else None
            )
            result["claim_id"] = claim_id
            results.append(result)

        results_df = pd.DataFrame(results)
        print(f"   ✅ Scored {len(results_df)} claims")
        print(f"   📊 HIGH: {(results_df['risk_level']=='HIGH').sum()}, "
              f"MEDIUM: {(results_df['risk_level']=='MEDIUM').sum()}, "
              f"LOW: {(results_df['risk_level']=='LOW').sum()}")

        return results_df

    def save_engine(self, output_dir: str):
        """Engine state save karo"""
        os.makedirs(output_dir, exist_ok=True)
        if self.isolation_forest:
            joblib.dump(self.isolation_forest,
                       os.path.join(output_dir, "isolation_forest.joblib"))
        print(f"   ✅ Risk engine saved to {output_dir}/")
