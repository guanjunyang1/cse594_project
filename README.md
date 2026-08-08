# CSE 594 Project: Two-Stage Low-Cost ADHD Screening

This repository contains the implementation and experimental code for the project **“A Two-Stage Low-Cost ADHD Screening Framework Using Passive Mobile Interaction and Camera-Based Eye Tracking.”**

The project evaluates a cost-aware cascaded screening framework in which:

- **Stage 1** uses low-cost mobile-interaction and behavioral features.
- **Stage 2** uses eye-tracking features only for samples whose Stage 1 prediction is uncertain.
- Logistic Regression (LR), Random Forest (RF), and XGBoost are evaluated.
- All current results are based on a **synthetic proof-of-concept dataset** and should not be interpreted as clinical validation.

---

## Main Python Files

The final implementation is organized around four main Python scripts.

### 1. `generate_synthetic_dataset.py`

Generates the synthetic multimodal dataset used by the project.

Main characteristics:

- 200 simulated participants:
  - 100 ADHD
  - 100 TDC
- 15 Stage 1 features
- 13 Stage 2 features
- 28 numerical features total
- Stage 1 reference-based features are derived from published group statistics.
- Five Stage 1 touchscreen/session variables are literature-informed proxy features.
- Stage 2 eye-tracking variables are based on published ADHD/control group statistics.
- The generated dataset is saved as:

```text
synthetic_stage1_stage2_dataset.csv
```

This file corresponds mainly to the **Synthetic Data** section of the paper.

---

### 2. `uniform_three_model_cascade_test.py`

Runs the main baseline comparison on the **original synthetic dataset**.

Models:

- Logistic Regression
- Random Forest
- XGBoost

Configurations:

- Stage 1 only
- Stage 1 + Stage 2
- Cascade

Experimental setup:

- 30 repeated stratified 70/30 train-test splits
- The same split indices are reused across all three classifiers.
- Median imputation is included in the preprocessing pipeline.
- Logistic Regression uses standardization.
- Random Forest and XGBoost are evaluated without feature standardization.
- Default cascade uncertainty interval:

```text
0.30 – 0.70
```

Main output files:

```text
uniform_three_model_results/
├── uniform_model_comparison_runs.csv
└── uniform_model_comparison_summary.csv
```

Representative baseline AUC results reported in the paper:

| Model | Stage 1 Only | Stage 1 + Stage 2 |
|---|---:|---:|
| Logistic Regression | 0.969 | 0.964 |
| Random Forest | 0.964 | 0.975 |
| XGBoost | 0.935 | 0.942 |

Representative cascade results using the 0.30–0.70 interval:

| Model | Cascade Accuracy | Full-Model Accuracy | Stage 2 Usage |
|---|---:|---:|---:|
| Logistic Regression | 0.924 ± 0.029 | 0.918 ± 0.034 | 15.2 ± 4.9% |
| Random Forest | 0.921 ± 0.039 | 0.921 ± 0.037 | 40.3 ± 4.9% |
| XGBoost | 0.871 ± 0.036 | 0.872 ± 0.041 | 17.5 ± 4.6% |

This script corresponds mainly to the paper sections on **Linear and Nonlinear Model Comparison** and **Cost–Performance Trade-Off**.

---

### 3. `stage1_signal_scenarios_test_fixed.py`

Runs the Stage 1 signal-strength sensitivity experiments.

Three controlled Stage 1 conditions are evaluated:

```text
Weak   : 0.30
Medium : 0.60
Strong : 1.00
```

For each Stage 1 feature, the class-mean separation is scaled according to the selected signal strength. The current implementation also:

- uses a shared, label-independent residual distribution,
- reduces class-specific residual-shape information,
- leaves Stage 2 features unchanged across scenarios,
- shifts originally nonnegative transformed variables upward when necessary instead of clipping them to zero.

Models:

- Logistic Regression
- Random Forest
- XGBoost

Configurations:

- Stage 1 only
- Stage 1 + Stage 2
- Cascade

Default uncertainty thresholds:

```text
0.20 – 0.80
0.30 – 0.70
0.40 – 0.60
```

Default experimental settings:

```text
n_repeats = 30
test_size = 0.30
random_seed = 42
```

Representative Stage 1-only AUC results:

| Signal | LR | RF | XGBoost |
|---|---:|---:|---:|
| Weak | 0.652 | 0.934 | 0.931 |
| Medium | 0.833 | 0.993 | 0.982 |
| Strong | 0.949 | 0.998 | 0.993 |

Representative Stage 1 + Stage 2 AUC results:

| Signal | LR | RF | XGBoost |
|---|---:|---:|---:|
| Weak | 0.820 | 0.950 | 0.931 |
| Medium | 0.883 | 0.992 | 0.977 |
| Strong | 0.951 | 0.998 | 0.990 |

Representative Stage 2 usage for the cascaded models:

| Signal | LR | RF | XGBoost |
|---|---:|---:|---:|
| Weak | 59.1% | 25.9% | 14.6% |
| Medium | 31.9% | 20.9% | 10.9% |
| Strong | 14.6% | 14.9% | 7.0% |

The main trend is that Stage 2 becomes most useful when Stage 1 is weak, while stronger Stage 1 signals reduce the number of samples that need additional eye-tracking assessment.

Typical outputs include:

- per-run metrics,
- summary metrics,
- threshold-sensitivity results,
- test-set predictions,
- ROC plots,
- Stage 1 AUC vs. signal-strength plots,
- Stage 2 usage plots,
- F1 vs. Stage 2 usage plots,
- scenario datasets,
- experiment metadata.

This script corresponds mainly to the paper sections on **Signal Strength Analysis for Stage 1** and **Threshold Sensitivity Analysis**.

---

### 4. `feature_explainability_30splits.py`

Performs feature-importance analysis over 30 repeated stratified train-test splits.

Methods:

- **Logistic Regression:** standardized coefficients
- **Random Forest:** held-out permutation importance
- **XGBoost:** held-out permutation importance

The analysis is used to determine which Stage 1 and Stage 2 features contribute most strongly to each classifier.

Main figure outputs include:

```text
lr_top10.png
rf_top10.png
xgb_top10.png
```

The paper reports that Stage 1 behavioral features dominate many of the highest-ranked positions, while several Stage 2 fixation- and saccade-related variables also contribute.

These importance values describe model dependence on the synthetic feature distributions and should **not** be interpreted as clinical or causal biomarker rankings.

This script corresponds to the paper section on **Feature Importance and Model Explainability**.

---

## Recommended Execution Order

Run the scripts from the project directory in the following order.

### Step 1: Generate the synthetic dataset

```bash
python generate_synthetic_dataset.py
```

Expected main output:

```text
synthetic_stage1_stage2_dataset.csv
```

### Step 2: Run the original-dataset model comparison

```bash
python uniform_three_model_cascade_test.py
```

### Step 3: Run Stage 1 signal-strength and threshold experiments

```bash
python stage1_signal_scenarios_test_fixed.py \
    --input synthetic_stage1_stage2_dataset.csv
```

### Step 4: Run feature-importance analysis

```bash
python feature_explainability_30splits.py
```

---

## Main Software Dependencies

The project uses Python 3 and the following packages:

```text
numpy
pandas
scikit-learn
xgboost
matplotlib
```

Install the main dependencies with:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

---

## Suggested Project Structure

```text
cse594_project/
├── generate_synthetic_dataset.py
├── uniform_three_model_cascade_test.py
├── stage1_signal_scenarios_test_fixed.py
├── feature_explainability_30splits.py
├── synthetic_stage1_stage2_dataset.csv
├── README.md
│
├── uniform_three_model_results/
├── signal_scenario_results/
├── feature_explainability_results/
│
└── archive/
    └── older experimental scripts
```

Older baseline or intermediate scripts can be moved to `archive/` if they are no longer used to generate the final paper results.

---

## Interpretation of the Results

The experiments support three main proof-of-concept observations:

1. **Stage 2 utility depends on Stage 1 signal strength and classifier choice.**  
   The clearest gain from Stage 2 occurs for Logistic Regression under the weak Stage 1 condition.

2. **The cascade can reduce Stage 2 measurement burden.**  
   When Stage 1 predictions are sufficiently confident, eye-tracking assessment can be skipped for a substantial subset of samples.

3. **Threshold selection is classifier-dependent.**  
   The same uncertainty interval does not produce the same Stage 2 usage or performance trade-off across LR, RF, and XGBoost.

---

## Important Limitation

All experiments use synthetic data generated from published group-level statistics and literature-informed assumptions.

The current results therefore demonstrate:

- the behavior of the proposed cascade,
- the feasibility of selective Stage 2 acquisition,
- model and threshold trade-offs under controlled conditions.

They do **not** establish clinical effectiveness or the real-world incremental value of eye tracking for ADHD screening.

Future validation requires same-participant multimodal data containing:

- mobile-interaction measurements,
- eye-tracking measurements,
- clinician-confirmed ADHD assessments.

---

## Project Purpose

This implementation is intended as a proof-of-concept for a **screening and referral framework**, not as a standalone ADHD diagnostic system.
