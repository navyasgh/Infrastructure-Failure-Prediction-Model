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

## Handling Class Imbalance

Class weighting was chosen over SMOTE and undersampling because:
- Undersampling would discard 99.9% of majority class data
- SMOTE generates synthetic rows that don't represent realistic 
  drive readings in a time-series context
- Class weighting penalizes misclassification of failures during 
  training without modifying the dataset

sklearn's class_weight='balanced' assigns weight ~554x higher to 
failure rows, reflecting the 1:554 class ratio in the dataset.

## Corrupted SMART Columns (dtype Reinterpretation Bug)

During evaluation, three model families (Logistic Regression, Random Forest, XGBoost)
all failed to rank true failures above healthy drives — a symptom of missing feature
signal rather than a modeling or threshold issue. Investigating the raw SMART columns
surfaced a second, more subtle instance of the corruption pattern already seen in
`capacity_bytes`.

**Symptom:** `smart_7`, `smart_188`, `smart_240`, `smart_241`, and `smart_242` showed
`std == 0.0` alongside mean values on the order of `1e-314` to `1e-316` — subnormal
floats at the edge of float64 precision, functionally indistinguishable from `0.0` to
any model or statistical test.

**Root cause:** These values are the result of integer bytes being bit-reinterpreted
as float64 (a `.view()`-style cast) rather than properly converted (`.astype()`),
upstream in the source file — before this dataset was ever loaded. Confirmed via
reversal: `column.values.view(np.int64)` on affected rows recovers plausible integer
SMART readings (e.g. `4.940656e-324` → `1`, matching the smallest possible subnormal
double reinterpreted from the integer `1`).

**Why this broke the models:** Unlike ordinary outlier-heavy SMART attributes (e.g.
`smart_197`, `smart_198`, which are legitimately zero-inflated with rare large spikes),
these columns collapsed *every* value — healthy and failed drives alike — into the same
near-zero neighborhood. This erased real variance rather than representing it, making
the affected columns statistically inert regardless of algorithm.

**Fix:**
- Detected affected columns programmatically via the signature `std == 0.0` and
  `0 < |mean| < np.finfo(np.float64).tiny`.
- Recovered true integer values via NaN-safe bit reinterpretation: non-null rows are
  passed through `.view(np.int64)`; true nulls (sensor never reported) are left as
  missing rather than zero-filled, to avoid re-introducing the same signal-flattening
  problem via a different mechanism.
- Rebuilt `smart_188_raw_delta` and `smart_188_raw_roll_avg`, since these engineered
  features were originally computed on corrupted values and inherited `std == 0.0`
  themselves.
- `smart_7`, `smart_240`, `smart_241`, `smart_242` were not included in delta/rolling
  feature engineering (outside Backblaze's top-5 predictive attributes used for that
  step), so recovery is limited to the raw columns for those four.

**Status:** Corruption confirmed and reversible. Models are being re-trained on
recovered features; results comparison against the pre-fix baseline (PR-AUC 0.0042,
0% recall on XGBoost) to follow.

## Project Phases
- [x] Phase 1 — EDA & Data Cleaning
- [x] Phase 2 — Target Label Engineering (30-day failure window)
- [x] Phase 3 — Feature Engineering
- [x] Phase 4 — Handling Class Imbalance
- [ ] Phase 5 — Model Training & Comparison
- [ ] Phase 6 — SHAP Explainability
- [ ] Phase 7 — Streamlit Dashboard
