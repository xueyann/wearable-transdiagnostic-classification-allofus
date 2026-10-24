# Reviewer sensitivity analyses: seven-day past-diagnosis primary cohort

## Scope and primary-analysis definition

This is the new reviewer-facing sensitivity-analysis framework. The proposed primary analysis is **seven consecutive eligible Fitbit days**, restricted to **past-diagnosis cases**. For each disease area, a case qualifies if at least one mapped diagnosis was recorded on or before the selected 7-day streak start and within the disease-specific pre-index lookback window. Recency is calculated from the most recent qualifying **pre-index** diagnosis; diagnoses recorded after the Fitbit index do not affect case eligibility. The **latest eligible** 7-day streak is selected per case to maximize the usable past-diagnosis sample. Controls have no recorded target-condition diagnosis or ICD exclusion at any time in their available EHR history and use their latest eligible 7-day streak. This deliberately strict comparator definition reduces documented-disease contamination but does not establish absence of undiagnosed disease and may reflect differences in available observation time.

This is a disease-versus-control association analysis of recorded, potentially treated disease. It is not an incident-disease prediction analysis or a causal medication-effect analysis.

Key implementation/results:

- Cohort and streak-selection logic: `src/past_diagnosis_streak_selection.py`, `src/streak_timing_sensitivity_models.py`.
- Seven-day past model performance: `results/streak_timing_sensitivity/`.
- Seven-day past coefficient audit: `src/past_7day_logreg_coefficients.py`, `results/past_7day_logreg_coefficients/`.
- The deterministic participant split uses seed 42; coefficients are unregularized `statsmodels.Logit` estimates fitted in the 70% training partition.

## 1. Multimorbidity

### Required descriptive analysis

Because past-streak selection is disease-specific, there is not one common seven-day analytic cohort. Report (1) a participant-level, descriptive pairwise overlap matrix based on qualifying pre-index diagnosis histories, clearly labelled as descriptive; and (2) within each disease-specific analytic cohort, the number of additional target conditions qualifying at that analysis's selected Fitbit index. Do not copy the prior 21-day mixed-timing matrix.

The prior 21-day mixed-timing cohort had substantial multimorbidity: 3,108/6,423 cases (48.4%) had two or more target conditions. Its most common overlaps were MDD–anxiety, CVD–OSA, OSA–anxiety, OSA–MDD, and diabetes–OSA. This motivates the sensitivity but is not the final seven-day-past matrix.

- Prior matrix (context only): `results/reviewer_sensitivity/condition_overlap_matrix.csv`.
- Required seven-day-past matrix: generate from the condition-specific case selections in `results/streak_timing_sensitivity/selected_streaks.csv` and document non-exclusive disease membership.

### Interpretation

The primary estimates should not be described as disease-specific signatures independent of multimorbidity. Single-condition results below isolate this issue.

## 2. Single-condition analysis

**Definition:** for each disease-specific analysis, controls plus cases with the target phenotype but no other target-condition phenotype qualifying at the same selected Fitbit index under the relevant pre-index rules. Medication exposure is not excluded.

**Status:** requires rerun under the new primary definition. The existing results in `results/reviewer_sensitivity/single_condition_performance.csv` are 21-day, mixed-timing results and must be retained only as historical context, not reported as seven-day-past findings.

**Implementation:** adapt `src/reviewer_sensitivity_analyses.py` to begin with each disease-specific seven-day past-diagnosis frame, then restrict cases to target-condition count = 1 at that outcome's selected Fitbit index. Fit the same seeded logistic and 5-fold tuned XGBoost models; report cases/controls, AUROC/BSS with 95% bootstrap CIs, and matched-test paired DeLong comparisons with the seven-day past primary model.

## 3. Stricter-comparator analysis

**Definition:** all seven-day past-diagnosis cases versus controls with no record at any time of a listed sleep- or heart-rate-altering medication. This is a deliberately stricter comparator definition, not a medication-effect analysis and not a test of whether observed differences are caused by medications.

**Status:** requires rerun. The existing `results/reviewer_sensitivity/clean_control_performance.csv` is not compatible with the new timing/streak definition.

**Implementation:** apply the original medication list in `data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv` to the seven-day past control set; retain all eligible cases. Clearly report that this control definition uses any available medication history, including post-index records. Use the same models, split, CIs, and matched-test comparison procedure as above.

## 4. Medication-unexposed sensitivity analysis

**Definition:** past-diagnosis cases and controls without recorded active exposure to prespecified medication classes during the selected seven-day Fitbit period. A participant is eligible if no recorded exposure interval overlaps `[streak_start, streak_end]`. The same pre-index medication data source, drug list, and exposure-overlap rule are applied to cases and controls. Missing medication records are treated as no **recorded** exposure, not evidence of medication non-use.

**Status:** requires rerun. The old result (`results/reviewer_sensitivity/premedication_cases_clean_control_performance.csv`) used a 21-day first-exposure rule and is not the proposed definition.

**Medication definition:** Where medication start and end dates are available, active exposure is determined from the recorded exposure interval. Where an end date is unavailable, infer duration from days supplied using a prespecified rule; if only medication-list or prescription-order records are available, describe the definition as no *recorded relevant medication* rather than medication-free. Prespecify condition-relevant medication classes and provide the full code list in the supplement.

**Interpretation:** this is a sensitivity to documented active medication exposure, not a treatment-effect analysis. It evaluates whether associations persist among participants without recorded active relevant pharmacotherapy; it does not isolate a causal disease effect because medication restriction may select participants with different disease severity, access to care, or health-care use. Unrecorded medications and treatment indication remain limitations.

## 5. Lookback sensitivity (completed)

Seven-day past-diagnosis models have been run for:

1. **No lookback:** any lifetime first diagnosis before the selected streak; no upper recency restriction.
2. **Current rules:** CVD/diabetes 10 years, OSA/ADHD 3 years, MDD/anxiety 2 years, bipolar 5 years, calculated from the most recent qualifying pre-index diagnosis.
3. **Halved rules:** CVD/diabetes 5 years, OSA/ADHD 1.5 years, MDD/anxiety 1 year, bipolar 2.5 years.

Results: `results/past_lookback_sensitivity_7day/past_lookback_sensitivity_results.md`, `performance.csv`, and `sample_sizes.csv`. The selection/model code is `src/past_lookback_sensitivity.py`.

## 6. Seven-day versus 14- and 21-day streak definitions

The seven-day past-diagnosis model is the primary definition because it materially increases usable sample size while retaining broadly consistent AUROC and principal directions of association across 7, 14, and 21 consecutive days. This decision should be justified primarily by retention and interpretability, not by selecting the highest observed AUROC.

Performance comparisons:

- `results/streak_timing_sensitivity/streak_timing_sensitivity_results.md`
- `results/streak_timing_sensitivity/performance.csv`
- `results/streak_timing_sensitivity/sample_sizes.csv`

Coefficient comparisons (same seed-42 training-set estimation):

- 7 days: `results/past_7day_logreg_coefficients/`
- 14 days: `results/past_14day_logreg_coefficients/`
- 21 days: `results/past_21day_logreg_coefficients/`

A formal comparison of seven versus 14/21-day performance should use paired DeLong tests on a common held-out participant subset; non-overlapping bootstrap intervals alone are not a test. The sensitivity report should state whether coefficient directions and FDR-significant feature categories are concordant, while avoiding direct comparisons of raw coefficient magnitude across changing cohorts.

## 7. Stricter diagnosis confirmation

**Definition:** a case must have at least two mapped diagnosis records for the same disease area on **two distinct calendar dates** on or before the selected seven-day Fitbit streak start, with at least one qualifying pre-index diagnosis satisfying the selected lookback rule. Repeated records on the same date do not satisfy this criterion.

**Purpose:** this sensitivity addresses potential phenotype misclassification from a single ICD-coded diagnosis. It is a stricter EHR-record confirmation rule, not evidence of clinical gold-standard diagnosis.

**Status:** requires rerun for the seven-day past-diagnosis cohort. Build the qualifying diagnosis set from `condition_df_parent_phecode_removed.parquet`, count distinct `condition_start_date` values by participant and disease area, then apply the same selected-streak, medication, complete-case, model, and reporting procedures as the primary analysis. Report resulting case attrition and performance separately by disease area.

## 8. Past versus future diagnosis performance

**Primary timing analysis:** past diagnosis only, as defined above.

**Exploratory appendix analysis:** future diagnosis only. A future case has its first mapped diagnosis after the end of its **first** qualifying seven-day streak; diagnoses during the streak are excluded. The first-streak rule maximizes the future-case sample but does not establish near-term prospective prediction, because first recorded diagnosis may be remote from clinical onset or reflect later documentation.

Existing seven-day past/future results are available in:

- `results/streak_timing_sensitivity/streak_timing_sensitivity_results.md`
- `results/streak_timing_sensitivity/performance.csv`
- `results/streak_timing_sensitivity/sample_sizes.csv`
- Analysis code: `src/streak_timing_sensitivity_models.py`

Future results should be placed in the appendix/supplement only, labelled exploratory, and should not be used to claim incident-disease prediction or alter the main manuscript’s past-diagnosis primary estimates. They are not an incident-disease prediction analysis because controls were not required to have comparable post-index observation and first recorded diagnosis may lag disease onset or reflect later documentation.

## 9. Missingness and complete-case assessment

**Required reporting:** Report cohort flow and variable-level missingness before and after applying wearable-data eligibility and complete-case requirements. Compare the characteristics of participants retained versus excluded where feasible.

**Interpretation:** Do not impute unavailable Fitbit-derived features, because absent eligible wearable data represent structural non-observation rather than a conventional partially missing covariate. For ordinary non-wearable covariates, report the extent and pattern of missingness and consider a multiple-imputation sensitivity analysis only if missingness is material. The complete-case decision should distinguish selection caused by wearable-data availability from the incremental loss caused by missing analytic covariates.

## Reporting requirements

For every re-estimated analysis, report per disease area: case/control N, AUROC and Brier skill score with 95% percentile-bootstrap CIs, model estimator/fallback status, and the exact diagnosis, medication, streak, and timing rules. Apply BH FDR correction separately to the pre-specified family of matched DeLong comparisons. Retain the full coefficient tables in supplement only.

No manuscript files were modified.
