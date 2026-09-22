"""
ClaimGraph AI — Master Pipeline Runner
======================================
Yeh script poora pipeline end-to-end execute karta hai:
1. Data Validation (Referential integrity, missing values, dedup, outliers)
2. Graph Analysis & Graph Feature Extraction (NetworkX, Louvain, PageRank)
3. Multimodal Consistency Check (PIL synthetic images & consistency scoring)
4. Feature Engineering & Dataset Prep (Tabular + Graph + Multimodal features)
5. Model Training (Baseline vs +Graph vs +Multimodal ablation study)
6. Risk Scoring Engine Initialization & Calibration (ML + Unsupervised + Rules + SHAP)
"""

import sys
import os
import logging
import json
import pandas as pd
from pathlib import Path

# Project paths setup
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_engineering.validation import DataValidator
from src.graph.algorithms import ClaimGraphAnalyzer
from src.multimodal.consistency import SyntheticImageGenerator, ConsistencyChecker
from src.data_engineering.features import FeatureEngineer
from src.ml.train import FraudModelTrainer
from src.risk.scoring import RiskScoringEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(data_dir: Path = ROOT_DIR / "data" / "generated", output_dir: Path = ROOT_DIR / "data" / "processed", model_dir: Path = ROOT_DIR / "models"):
    logger.info("🚀 Starting ClaimGraph AI End-to-End Pipeline...")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Loading & Validation
    logger.info("📋 Step 1: Loading & Validating Data...")
    claims_df = pd.read_csv(data_dir / "claims.csv")
    customers_df = pd.read_csv(data_dir / "customers.csv")
    policies_df = pd.read_csv(data_dir / "policies.csv")
    devices_df = pd.read_csv(data_dir / "devices.csv")
    addresses_df = pd.read_csv(data_dir / "addresses.csv")
    shops_df = pd.read_csv(data_dir / "repair_shops.csv")
    payments_df = pd.read_csv(data_dir / "payment_accounts.csv")

    validator = DataValidator()
    val_report = validator.validate_all(
        claims_df, customers_df, policies_df, devices_df, addresses_df, shops_df, payments_df
    )
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(val_report, f, indent=2)
    logger.info(f"   ✅ Data Validation Passed! Report saved to {output_dir / 'validation_report.json'}")

    # 2. Graph Construction & Graph Feature Extraction
    logger.info("🕸️ Step 2: Building Knowledge Graph & Extracting Graph Features...")
    graph_analyzer = ClaimGraphAnalyzer()
    graph = graph_analyzer.build_graph(
        claims_df, customers_df, devices_df, shops_df, addresses_df, payments_df
    )
    graph_features_df = graph_analyzer.extract_claim_features(claims_df)
    graph_features_df.to_csv(output_dir / "graph_features.csv", index=False)
    
    # Identify and save detected fraud rings
    detected_rings = graph_analyzer.detect_fraud_rings(min_community_size=3)
    graph_analyzer.save_graph_data(str(output_dir / "graph"))
    if not detected_rings.empty:
        detected_rings.to_csv(output_dir / "detected_fraud_rings.csv", index=False)
    logger.info(f"   ✅ Extracted graph features for {len(graph_features_df)} claims. Detected {len(detected_rings)} suspect ring records.")

    # 3. Multimodal Consistency Analysis
    logger.info("🖼️ Step 3: Running Multimodal Consistency Checker...")
    image_dir = ROOT_DIR / "data" / "synthetic_images"
    image_gen = SyntheticImageGenerator()
    images_metadata = image_gen.generate_dataset(claims_df, output_dir=str(image_dir), n_images=500)

    checker = ConsistencyChecker(use_clip=False)  # Heuristic mode for fast & reproducible evaluation
    multimodal_features_df = checker.check_batch(images_metadata)
    multimodal_features_df.to_csv(output_dir / "multimodal_features.csv", index=False)
    logger.info(f"   ✅ Multimodal evaluation completed for {len(multimodal_features_df)} claims.")

    # 4. Feature Engineering & Dataset Preparation
    logger.info("⚙️ Step 4: Engineering Tabular Features & Merging All Modalities...")
    fe = FeatureEngineer()
    full_dataset_df = fe.create_full_dataset(
        claims_df=claims_df,
        customers_df=customers_df,
        policies_df=policies_df,
        devices_df=devices_df,
        graph_features_df=graph_features_df,
        multimodal_features_df=multimodal_features_df
    )
    full_dataset_df.to_csv(output_dir / "full_dataset.csv", index=False)
    logger.info(f"   ✅ Full dataset prepared: shape {full_dataset_df.shape}")

    # 5. Model Training & Ablation Study
    logger.info("🤖 Step 5: Training ML Models & Executing 3-Experiment Ablation Study...")
    trainer = FraudModelTrainer(experiments_dir=str(output_dir / "experiments"), models_dir=str(model_dir))
    results = trainer.run_ablation_study(full_dataset_df, fe.feature_cols)

    # Save ablation summary
    ablation_summary = trainer.get_ablation_summary()
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)
    logger.info("   ✅ Ablation Study Complete! Model artifacts saved.")

    # 6. Risk Scoring Engine Initialization & Calibration
    logger.info("⚖️ Step 6: Calibrating Risk Scoring Engine & Isolation Forest...")
    best_model_path = model_dir / "exp3_plus_multimodal_xgboost.joblib"
    if not best_model_path.exists():
        best_model_path = model_dir / "exp2_plus_graph_xgboost.joblib"

    scoring_engine = RiskScoringEngine(model_path=str(best_model_path))

    feat_cols = [c for c in fe.feature_cols if c in full_dataset_df.columns]
    scoring_engine.fit_isolation_forest(full_dataset_df[feat_cols])
    scoring_engine.setup_shap_explainer(full_dataset_df[feat_cols])
    scoring_engine.save_engine(str(model_dir))

    # Score all claims and save summary
    scored_claims_df = scoring_engine.score_all_claims(full_dataset_df, feature_columns=feat_cols)
    scored_claims_df.to_csv(output_dir / "scored_claims.csv", index=False)

    # Evaluate risk score for demo claim
    demo_claim = claims_df[claims_df["claim_id"] == "CLM-DEMO-001"]
    if not demo_claim.empty:
        demo_row = full_dataset_df[full_dataset_df["claim_id"] == "CLM-DEMO-001"]
        if not demo_row.empty:
            demo_risk = scoring_engine.score_claim(demo_row[feat_cols], feature_names=feat_cols)
            logger.info(f"   🎯 Demo Claim CLM-DEMO-001 Score: {demo_risk['risk_score']:.3f} | Tier: {demo_risk['risk_level']} | Recommendation: {demo_risk['recommendation']}")

    logger.info("============================================================")
    logger.info("🎉 ClaimGraph AI Pipeline Execution Completed Successfully!")
    logger.info("============================================================")

if __name__ == "__main__":
    run_pipeline()
