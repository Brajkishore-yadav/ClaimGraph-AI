# ClaimGraph AI — Synthetic Fraud Injection Specification

## Overview
This document outlines the exact methodology used to generate synthetic claims and inject realistic fraud ring topology for Assurant 2027 Data Science internship demonstration.

## Fraud Ring Typologies
1. **Shared Device Ring**: Multiple distinct customer accounts filing claims from the exact same IMEI/Device ID within a 14-day window.
2. **Shared Repair Shop Syndicate**: Suspicious repair shop (`SHP-0042`) inflating repair costs and appearing across claims from unrelated policyholders.
3. **Shared Payment Account Ring**: Payouts requested to the same bank account or virtual card across different identity details.
4. **Mixed Entity Network**: Complex multi-hop collusion combining shared address and repair shop.

## Safeguard Against Data Leakage
- `fraud_ring_ground_truth.csv` is maintained in isolation during data generation.
- The ground truth column `is_fraud` is **NEVER** exposed as a feature input.
- Customer-level splitting (`GroupShuffleSplit`) ensures claims from the same customer never span across train, validation, and test splits.
