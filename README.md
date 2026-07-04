# Hard Drive Failure Predictor

Predictive maintenance system built on real telemetry data from Backblaze's data centers.
The goal is to predict whether a hard drive will fail within the next 30 days using daily 
SMART diagnostic attributes.

## Problem Statement

Hard drive failures in data centers cause data loss and costly downtime. Backblaze publishes
daily SMART telemetry for every drive in their fleet. Using this data, we can build a model
that flags at-risk drives before they fail — giving engineers time to act.

## Dataset

- **Source:** Backblaze Hard Drive Test Data (2016), via Kaggle
- **Size:** 3.1 million rows, 95 raw columns
- **Failures:** 215 out of 3.1 million drive-days (0.006%)
- **Features:** Daily SMART diagnostic attributes per drive

## Data Preprocessing

### Column Reduction
Dropped 63 columns:
- 36 columns with >90% null values — entirely unreported attributes
- 27 normalized SMART columns — Backblaze research confirms raw values
  are more predictive than manufacturer-normalized equivalents

Final shape after dropping: 3.1M rows × 32 columns

### Null Handling
Remaining nulls were filled with 0 rather than median or mean.

SMART attributes are counters and flags that increment only when a fault occurs.
A null value indicates the attribute was never triggered by that drive — 
functionally identical to 0. Filling with median would incorrectly imply 
the drive had an average fault count.

## Project Phases
- [x] Phase 1 — EDA & Data Cleaning
- [ ] Phase 2 — Target Label Engineering (30-day failure window)
- [ ] Phase 3 — Feature Engineering
- [ ] Phase 4 — Handling Class Imbalance
- [ ] Phase 5 — Model Training & Comparison
- [ ] Phase 6 — SHAP Explainability
- [ ] Phase 7 — Streamlit Dashboard
