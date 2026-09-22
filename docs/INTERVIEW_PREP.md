# ClaimGraph AI — Assurant Data Science Interview Cheat Sheet

## Frequently Asked Questions

### Q1: Why did you use NetworkX instead of Neo4j GDS or PyTorch Geometric for production feature extraction?
**Answer:** "NetworkX allowed me to implement graph algorithms like Louvain community detection and PageRank directly in Python without dependency overhead or external licensing. It allowed me to deeply explain how the node degree and cluster density metrics are computed during technical interviews."

### Q2: Why use PR-AUC as the headline metric instead of ROC-AUC?
**Answer:** "Severe class imbalance (1.3% fraud rate) renders ROC-AUC overly optimistic because it includes easy True Negatives in the false positive rate denominator. PR-AUC focuses exclusively on Precision and Recall for the positive fraud class."

### Q3: Why class weights instead of SMOTE?
**Answer:** "SMOTE generates synthetic minority examples by linear interpolation between nearest neighbors. On graph-connected datasets, this risks creating artificial nodes with invalid entity links. `class_weight='balanced'` adjusts loss penalty during training cleanly without mutating data distribution."

### Q4: How did you prevent data leakage?
**Answer:** "I implemented customer-level splitting using `GroupShuffleSplit`. If row-level random splitting was used, multiple claims from the same customer would leak across train and test sets."
