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

### dtype Correction
12 SMART columns were incorrectly read as object type due to mixed string/float 
values in the raw CSV. Converted to float64 using pd.to_numeric with errors='coerce', 
then filled resulting nulls with 0.

### Duplicate Removal
120,497 duplicate drive-date combinations were found and removed, keeping the first 
occurrence. These were true duplicates with identical SMART values — a data collection 
artifact in the raw dataset.

### Corrupted Column
`capacity_bytes` was dropped entirely. All values were corrupted near-zero floats 
(e.g. 1.48e-311) rather than real capacity values (expected ~1e12 for TB-range drives).

## Evaluation Strategy

Accuracy is not a suitable metric for this dataset. A model predicting 
'no failure' for every row achieves 99.91% accuracy while missing every 
failure entirely.

Instead we optimize for **Recall** — minimizing missed failures — because:
- Missing a failure = data loss, downtime, costly emergency replacement
- False alarm = unnecessary but cheap drive inspection

Primary metric: **Precision-Recall AUC**
Secondary metric: **Recall at fixed Precision threshold**

## Feature Engineering

Three categories of features were engineered from raw SMART attributes:

### Drive Age
Days since the drive's first appearance in the dataset, calculated per drive 
using groupby + transform. Captures the bathtub curve effect — older drives 
are more failure-prone.

### Delta Features
Day-over-day change in Backblaze's top 5 failure-predictive SMART attributes:
smart_5, smart_187, smart_188, smart_197, smart_198. A sudden spike in 
reallocated sectors is more alarming than a stable high value.
First row per drive filled with 0 (no previous reading available).

### Rolling Averages (7-day)
7-day rolling mean of the same 5 SMART attributes. Smooths daily noise and 
captures sustained trends rather than transient spikes. min_periods=1 ensures 
drives with fewer than 7 days of history are still included.

## Project Phases
- [x] Phase 1 — EDA & Data Cleaning
- [x] Phase 2 — Target Label Engineering (30-day failure window)
- [ ] Phase 3 — Feature Engineering
- [ ] Phase 4 — Handling Class Imbalance
- [ ] Phase 5 — Model Training & Comparison
- [ ] Phase 6 — SHAP Explainability
- [ ] Phase 7 — Streamlit Dashboard
