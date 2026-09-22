"""
ClaimGraph AI — ML Training Pipeline
=======================================
Ye module ML models train karta hai — 3 experiments × 2 models.

THE CENTERPIECE EXPERIMENT (Section 10):
Experiment 1: Tabular baseline features only
Experiment 2: + Graph features (degree, community, PageRank, shared entities)
Experiment 3: + Multimodal consistency score

Models: LogisticRegression (interpretable baseline) + XGBoost (strong model)
Imbalance handling: class_weight='balanced' (not SMOTE)

Interview answer for class weights vs SMOTE:
"I used class weights instead of SMOTE because SMOTE creates synthetic minority
samples, which on graph-connected data could create misleading synthetic nodes.
Class weights simply tell the model to pay more attention to the minority class
during training — simpler and more honest."

Interview answer for PR-AUC as headline metric:
"ROC-AUC can be misleadingly optimistic under severe class imbalance because
it includes true negatives, which are easy to get right when 95% of data is
negative. PR-AUC focuses on precision and recall of the POSITIVE (fraud) class,
which is what we actually care about — how well do we find fraud without
too many false alarms?"
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception as e:
    print(f"⚠️ Warning: XGBoost could not be loaded ({e}). Falling back to HistGradientBoostingClassifier.")
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    HAS_XGBOOST = False



class FraudModelTrainer:
    """
    ML fraud model trainer — 3 experiments, 2 models each.
    Saare results honestly report hote hain — no cherry picking.
    """

    def __init__(self, experiments_dir: str = "experiments/runs",
                 models_dir: str = "models"):
        self.experiments_dir = experiments_dir
        self.models_dir = models_dir
        os.makedirs(experiments_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        self.results = {}  # Store all experiment results

    def run_ablation_study(self, full_dataset_df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
        """Run full 3-experiment ablation study on customer-level split dataset."""
        from src.data_engineering.features import FeatureEngineer
        fe = FeatureEngineer()
        train_df, val_df, test_df = fe.customer_level_split(full_dataset_df)

        tabular_cols = [
            "claim_amount", "amount_vs_avg", "claim_frequency",
            "days_since_policy_start", "coverage_utilization",
            "device_age_days", "amount_vs_device_price", "premium_amount",
            "claim_type_encoded", "device_type_encoded", "policy_type_encoded",
            "has_repair_shop"
        ]
        graph_cols = [
            "node_degree", "community_size", "pagerank_score",
            "n_shared_devices", "n_shared_addresses",
            "n_shared_shops", "n_shared_payments"
        ]
        multimodal_cols = ["consistency_score", "multimodal_inconsistency"]

        table = self.run_all_experiments(
            train_df, val_df, test_df,
            tabular_features=tabular_cols,
            graph_features=graph_cols,
            multimodal_features=multimodal_cols
        )
        self.compute_bootstrap_ci(test_df[tabular_cols + graph_cols + multimodal_cols], test_df["is_fraud"])
        return table

    def get_ablation_summary(self) -> dict:
        return self.results


    def _get_model(self, model_name: str, seed: int = 42):
        """
        Model create karo — sirf LogReg aur XGBoost, baaki kuch nahi.

        LogReg = interpretable baseline (coefficients directly tell you feature importance)
        XGBoost = strong tabular model (handles non-linear interactions, feature importance via gain)
        """
        if model_name == "logistic_regression":
            return LogisticRegression(
                class_weight="balanced",  # Imbalance handle karo
                max_iter=1000,
                random_state=seed,
                solver="lbfgs"
            )
        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,  # Fraud class ko zyada weight do
                random_state=seed,
                eval_metric="aucpr",  # PR-AUC as eval metric
                use_label_encoder=False,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_and_evaluate(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           experiment_name: str, feature_set: str,
                           model_name: str) -> Dict:
        """
        Ek model train karo aur evaluate karo.
        Results JSON manifest mein save hote hain.
        """
        print(f"\n{'='*50}")
        print(f"🏋️ Training: {model_name} | {experiment_name}")
        print(f"   Features: {feature_set} ({X_train.shape[1]} cols)")
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Fraud rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}")
        print(f"{'='*50}")

        # Handle inf/nan values
        for df in [X_train, X_val, X_test]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

        # Train
        start_time = time.time()
        model = self._get_model(model_name)
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics calculate karo
        metrics = {
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_time_seconds": round(train_time, 2)
        }

        # Print results
        print(f"\n📊 Results ({model_name}):")
        print(f"   Precision: {metrics['precision']}")
        print(f"   Recall:    {metrics['recall']}")
        print(f"   F1:        {metrics['f1']}")
        print(f"   ROC-AUC:   {metrics['roc_auc']}")
        print(f"   PR-AUC:    {metrics['pr_auc']} ← headline metric")
        print(f"   Confusion Matrix: {metrics['confusion_matrix']}")

        # Save model
        model_path = os.path.join(self.models_dir,
                                  f"{experiment_name}_{model_name}.joblib")
        joblib.dump(model, model_path)

        # Save experiment manifest
        manifest = {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "feature_set": feature_set,
            "n_features": X_train.shape[1],
            "feature_columns": list(X_train.columns),
            "metrics": metrics,
            "hyperparameters": model.get_params(),
            "data_stats": {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_fraud_rate": round(float(y_train.mean()), 4),
                "test_fraud_rate": round(float(y_test.mean()), 4)
            },
            "model_path": model_path,
            "timestamp": str(datetime.now()),
        }

        manifest_path = os.path.join(
            self.experiments_dir,
            f"{experiment_name}_{model_name}.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        # Store in results dict
        key = f"{experiment_name}_{model_name}"
        self.results[key] = manifest

        return manifest

    def run_all_experiments(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           tabular_features: list,
                           graph_features: list,
                           multimodal_features: list) -> pd.DataFrame:
        """
        MAIN METHOD: Saare 3 experiments × 2 models run karo.

        Returns: ablation table DataFrame — THE centerpiece artifact.
        """
        print("\n" + "=" * 70)
        print("🧪 RUNNING ALL EXPERIMENTS — The Centerpiece Comparison")
        print("=" * 70)

        experiments = [
            ("exp1_tabular_baseline", "Tabular baseline", tabular_features),
            ("exp2_plus_graph", "+ Graph features", tabular_features + graph_features),
            ("exp3_plus_multimodal", "+ Multimodal signal",
             tabular_features + graph_features + multimodal_features),
        ]

        models = ["logistic_regression", "xgboost"]

        for exp_name, feature_set_desc, feature_cols in experiments:
            available_cols = [c for c in feature_cols
                            if c in train_df.columns]

            X_train = train_df[available_cols].copy()
            y_train = train_df["is_fraud"].astype(int)
            X_val = val_df[available_cols].copy()
            y_val = val_df["is_fraud"].astype(int)
            X_test = test_df[available_cols].copy()
            y_test = test_df["is_fraud"].astype(int)

            for model_name in models:
                self.train_and_evaluate(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    experiment_name=exp_name,
                    feature_set=feature_set_desc,
                    model_name=model_name
                )

        # Build ablation table
        return self.build_ablation_table()

    def build_ablation_table(self) -> pd.DataFrame:
        """
        The 3-experiment comparison table — MOST IMPORTANT visual.
        Ye table README, dashboard, aur PowerPoint mein jaata hai.
        """
        rows = []
        for key, result in self.results.items():
            rows.append({
                "Experiment": result["experiment_name"],
                "Model": result["model_name"],
                "Features": result["feature_set"],
                "N_Features": result["n_features"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1": result["metrics"]["f1"],
                "ROC_AUC": result["metrics"]["roc_auc"],
                "PR_AUC": result["metrics"]["pr_auc"],
            })

        table = pd.DataFrame(rows)
        table = table.sort_values(["Experiment", "Model"])

        print("\n" + "=" * 70)
        print("📊 ABLATION TABLE — The Centerpiece")
        print("=" * 70)
        print(table.to_string(index=False))
        print("=" * 70)

        # Save table
        table_path = os.path.join(self.experiments_dir, "ablation_table.csv")
        table.to_csv(table_path, index=False)
        print(f"   ✅ Saved to {table_path}")

        return table

    def compute_bootstrap_ci(self, X_test: pd.DataFrame, y_test: pd.Series,
                             model_name: str = "xgboost",
                             experiment: str = "exp3_plus_multimodal",
                             n_bootstrap: int = 1000,
                             ci_level: float = 0.95,
                             seed: int = 42) -> Dict:
        """
        Bootstrap confidence interval for PR-AUC.

        How it works: Test set ko N baar resample karo (with replacement),
        har baar PR-AUC calculate karo, phir 2.5th aur 97.5th percentile lo.
        Ye batata hai ki hamari PR-AUC estimate kitni stable hai.

        Interview answer: "I resampled the test set 1000 times with replacement,
        computed PR-AUC each time, and reported the 95% confidence interval.
        This shows that my PR-AUC estimate isn't just one lucky test set."
        """
        print(f"\n📊 Computing bootstrap CI for PR-AUC ({n_bootstrap} iterations)...")

        # Load the model
        model_path = os.path.join(self.models_dir, f"{experiment}_{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"   ⚠️ Model not found at {model_path}, skipping bootstrap")
            return {}

        model = joblib.load(model_path)

        rng = np.random.RandomState(seed)
        pr_aucs = []

        X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = rng.choice(len(X_test_clean), size=len(X_test_clean), replace=True)
            X_boot = X_test_clean.iloc[indices]
            y_boot = y_test.iloc[indices]

            # Skip if only one class in bootstrap sample
            if len(y_boot.unique()) < 2:
                continue

            y_proba = model.predict_proba(X_boot)[:, 1]
            pr_auc = average_precision_score(y_boot, y_proba)
            pr_aucs.append(pr_auc)

        alpha = (1 - ci_level) / 2
        lower = np.percentile(pr_aucs, alpha * 100)
        upper = np.percentile(pr_aucs, (1 - alpha) * 100)
        mean_pr_auc = np.mean(pr_aucs)

        result = {
            "metric": "PR-AUC",
            "mean": round(mean_pr_auc, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "ci_level": ci_level,
            "n_bootstrap": n_bootstrap,
            "model": model_name,
            "experiment": experiment
        }

        print(f"   ✅ PR-AUC: {mean_pr_auc:.4f} "
              f"({ci_level*100:.0f}% CI: [{lower:.4f}, {upper:.4f}])")

        # Save
        ci_path = os.path.join(self.experiments_dir, "bootstrap_ci.json")
        with open(ci_path, "w") as f:
            json.dump(result, f, indent=2)

        return result
