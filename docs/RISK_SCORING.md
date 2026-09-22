# ClaimGraph AI — Risk Scoring Engine & Escalation Rules

## Architecture
The Risk Scoring Engine combines three complementary signals to assign every insurance claim an auditable risk score and action recommendation.

## Signal Composition
1. **Supervised ML Model (XGBoost)**: Provides primary calibrated probability ($P_{\text{fraud}}$).
2. **Unsupervised Anomaly Score (Isolation Forest)**: Identifies out-of-distribution claims not captured by supervised training.
3. **Graph Community & Multimodal Signals**: Acts as high-precision risk boosters.

## Decision & Escalation Matrix
- **HIGH RISK** ($P_{\text{fraud}} > 0.70$ OR $P_{\text{fraud}} > 0.40$ + Flagged Graph/Multimodal) $\rightarrow$ **ESCALATE_FOR_INVESTIGATION**
- **MEDIUM RISK** ($P_{\text{fraud}} \in [0.40, 0.70]$ OR Isolation Forest score $> 0.60$) $\rightarrow$ **REVIEW**
- **LOW RISK** ($P_{\text{fraud}} < 0.40$) $\rightarrow$ **APPROVE**

> [!IMPORTANT]
> The system strictly output recommendations (`APPROVE`, `REVIEW`, `ESCALATE_FOR_INVESTIGATION`) and never performs autonomous denial.
