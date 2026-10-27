# Reviewer sensitivity analyses: seven-day past-diagnosis primary cohort

## Scope and primary-analysis definition

This is the new reviewer-facing sensitivity-analysis framework. The proposed primary analysis is **seven consecutive eligible Fitbit days**, restricted to **past-diagnosis cases**. For each disease area, a case qualifies if at least one mapped diagnosis was recorded on or before the selected 7-day streak start and within the disease-specific pre-index lookback window. Recency is calculated from the most recent qualifying **pre-index** diagnosis; diagnoses recorded after the Fitbit index do not affect case eligibility. The **latest eligible** 7-day streak is selected per case to maximize the usable past-diagnosis sample. Controls have no recorded target-condition diagnosis or ICD exclusion at any time in their available EHR history and use their latest eligible 7-day streak. This deliberately strict comparator definition reduces documented-disease contamination but does not establish absence of undiagnosed disease and may reflect differences in available observation time.

This is a disease-versus-control association analysis of recorded, potentially treated disease. It is not an incident-disease prediction analysis or a causal medication-effect analysis.

Key implementation/results:

- Cohort and streak-selection logic: `src/seven_day_past_preindex.py`.
- Seven-day past model performance and reviewer sensitivities: `src/seven_day_past_preindex_sensitivities.py`, `results/seven_day_past_preindex_sensitivities/`.
- Corrected 7/14/21-day duration and coefficient comparison: `src/preindex_streak_duration_sensitivity.py`, `src/render_preindex_streak_duration_report.py`, and `results/preindex_streak_duration_sensitivity/`.
- The deterministic participant split uses seed 42; coefficients are unregularized `statsmodels.Logit` estimates fitted in the 70% training partition.

## Results summary

Each cell is **AUROC (95% percentile-bootstrap CI); cases/controls**. All entries use seven consecutive eligible Fitbit days, the corrected past-diagnosis pre-index rule, and the disease-specific current lookback window. Medication-unexposed results use the revised ±30-day medication-free window and a one-day fallback when an exposure end date is absent. All other results remain in `results/seven_day_past_preindex_sensitivities/`.

### Logistic regression

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.835 (0.819–0.851); 3,536/4,283 | 0.813 (0.791–0.833); 1,486/4,283 | 0.843 (0.823–0.861); 3,536/2,536 | 0.836 (0.819–0.854); 2,399/4,061 | 0.864 (0.850–0.880); 2,534/4,283 |
| Diabetes | 0.830 (0.811–0.846); 3,050/4,283 | 0.769 (0.743–0.795); 1,177/4,283 | 0.835 (0.817–0.855); 3,050/2,536 | 0.827 (0.809–0.845); 2,106/4,061 | 0.829 (0.810–0.847); 2,154/4,283 |
| OSA | 0.826 (0.810–0.844); 3,346/4,283 | 0.770 (0.744–0.797); 971/4,283 | 0.832 (0.813–0.851); 3,346/2,536 | 0.826 (0.805–0.845); 2,033/4,061 | 0.854 (0.838–0.869); 2,809/4,283 |
| MDD | 0.748 (0.727–0.768); 3,014/4,283 | 0.688 (0.647–0.728); 617/4,283 | 0.759 (0.734–0.782); 3,014/2,536 | 0.699 (0.672–0.726); 1,479/4,061 | 0.757 (0.735–0.777); 2,414/4,283 |
| Anxiety | 0.711 (0.691–0.731); 3,578/4,283 | 0.652 (0.618–0.681); 1,133/4,283 | 0.715 (0.692–0.739); 3,578/2,536 | 0.673 (0.647–0.699); 2,025/4,061 | 0.740 (0.719–0.761); 2,658/4,283 |
| Bipolar | 0.760 (0.731–0.791); 1,028/4,283 | 0.702 (0.641–0.758); 254/4,283 | 0.772 (0.742–0.802); 1,028/2,536 | 0.747 (0.706–0.784); 510/4,061 | 0.809 (0.782–0.835); 777/4,283 |
| ADHD | 0.719 (0.681–0.758); 783/4,283 | 0.641 (0.568–0.713); 187/4,283 | 0.745 (0.707–0.780); 783/2,536 | 0.706 (0.648–0.762); 360/4,061 | 0.764 (0.726–0.800); 592/4,283 |

### XGBoost

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.835 (0.819–0.850); 3,536/4,283 | 0.812 (0.791–0.833); 1,486/4,283 | 0.841 (0.820–0.860); 3,536/2,536 | 0.834 (0.817–0.852); 2,399/4,061 | 0.858 (0.844–0.874); 2,534/4,283 |
| Diabetes | 0.829 (0.812–0.846); 3,050/4,283 | 0.778 (0.754–0.802); 1,177/4,283 | 0.828 (0.809–0.849); 3,050/2,536 | 0.825 (0.806–0.842); 2,106/4,061 | 0.832 (0.814–0.850); 2,154/4,283 |
| OSA | 0.826 (0.810–0.843); 3,346/4,283 | 0.773 (0.748–0.800); 971/4,283 | 0.830 (0.810–0.849); 3,346/2,536 | 0.824 (0.803–0.843); 2,033/4,061 | 0.850 (0.833–0.866); 2,809/4,283 |
| MDD | 0.739 (0.718–0.759); 3,014/4,283 | 0.622 (0.582–0.667); 617/4,283 | 0.751 (0.728–0.774); 3,014/2,536 | 0.690 (0.662–0.718); 1,479/4,061 | 0.758 (0.736–0.777); 2,414/4,283 |
| Anxiety | 0.704 (0.683–0.724); 3,578/4,283 | 0.629 (0.595–0.658); 1,133/4,283 | 0.712 (0.689–0.735); 3,578/2,536 | 0.658 (0.631–0.684); 2,025/4,061 | 0.733 (0.712–0.754); 2,658/4,283 |
| Bipolar | 0.749 (0.720–0.780); 1,028/4,283 | 0.649 (0.577–0.711); 254/4,283 | 0.781 (0.753–0.811); 1,028/2,536 | 0.715 (0.672–0.757); 510/4,061 | 0.804 (0.774–0.831); 777/4,283 |
| ADHD | 0.715 (0.681–0.753); 783/4,283 | 0.606 (0.531–0.683); 187/4,283 | 0.734 (0.697–0.773); 783/2,536 | 0.691 (0.636–0.749); 360/4,061 | 0.744 (0.706–0.783); 592/4,283 |

The section-specific paired-comparison tables below report raw two-sided DeLong P-values; no multiplicity correction is applied to those requested comparisons.

**Main interpretation:** Primary AUROCs were highest for CVD, diabetes, and OSA (approximately 0.83 for both estimators), intermediate for bipolar disorder and MDD, and lower for anxiety and ADHD. The stricter-comparator analysis was similar to or modestly higher than the primary analysis. Restricting to a single condition reduced both case counts and observed AUROC, especially for mental-health outcomes. Under the buffered medication-unexposed definition, CVD, diabetes, and OSA performance was similar to the primary model; MDD and anxiety were lower in both estimators on raw paired tests. This difference is descriptive of the selected medication-unexposed cohort, not a medication effect. Requiring two diagnosis dates generally increased observed AUROC, which is consistent with a more stringently documented, potentially more homogeneous case definition and should not be interpreted as an improvement attributable to the model.

## 1. Multimorbidity

### Required descriptive analysis

Because past-streak selection is disease-specific, there is not one common seven-day analytic cohort. Report (1) a participant-level, descriptive pairwise overlap matrix based on qualifying pre-index diagnosis histories, clearly labelled as descriptive; and (2) within each disease-specific analytic cohort, the number of additional target conditions qualifying at that analysis's selected Fitbit index. Do not copy the prior 21-day mixed-timing matrix.

The prior 21-day mixed-timing cohort had substantial multimorbidity: 3,108/6,423 cases (48.4%) had two or more target conditions. Its most common overlaps were MDD–anxiety, CVD–OSA, OSA–anxiety, OSA–MDD, and diabetes–OSA. This motivates the sensitivity but is not the final seven-day-past matrix.

- Prior matrix (context only): `results/reviewer_sensitivity/condition_overlap_matrix.csv`.
- Completed seven-day-past descriptive outputs: `results/seven_day_past_preindex_sensitivities/descriptive_overlap_matrix.csv` (wide co-occurrence matrix), `descriptive_overlap_long.csv`, and `within_cohort_additional_conditions.csv`. These are non-exclusive, condition-specific selections.

### Interpretation

The primary estimates should not be described as disease-specific signatures independent of multimorbidity. Single-condition results below isolate this issue.

### Results summary table

| Primary-case cohort | ADHD | Anxiety | Bipolar | CVD | Diabetes | MDD | OSA |
|---|---|---|---|---|---|---|---|
| ADHD | 783 | 410 | 147 | 118 | 110 | 384 | 205 |
| Anxiety | 410 | 3578 | 447 | 723 | 639 | 1615 | 976 |
| Bipolar | 147 | 447 | 1028 | 240 | 255 | 466 | 327 |
| CVD | 118 | 723 | 240 | 3536 | 1020 | 713 | 1177 |
| Diabetes | 110 | 639 | 255 | 1020 | 3050 | 665 | 1059 |
| MDD | 384 | 1615 | 466 | 713 | 665 | 3014 | 1014 |
| OSA | 205 | 976 | 327 | 1177 | 1059 | 1014 | 3346 |

## 2. Single-condition analysis

**Definition:** for each disease-specific analysis, controls plus cases with the target phenotype but no other target-condition phenotype qualifying at the same selected Fitbit index under the relevant pre-index rules. Medication exposure is not excluded.

**Status:** completed under the corrected seven-day pre-index definition; results are in `results/seven_day_past_preindex_sensitivities/` (`performance.csv`, `sample_sizes.csv`, and `matched_test_delong.csv`). The historical 21-day mixed-timing result remains context only.

**Implementation:** completed in `src/seven_day_past_preindex_sensitivities.py`, using shared cohort functions in `src/seven_day_past_preindex.py`.

### Results summary table

| Disease area | Cases / controls | LogReg AUROC (95% CI) | XGBoost AUROC (95% CI) |
|---|---|---|---|
| CVD | 1,486 / 4,283 | 0.813 (0.791–0.833) | 0.812 (0.791–0.833) |
| Diabetes | 1,177 / 4,283 | 0.769 (0.743–0.795) | 0.778 (0.754–0.802) |
| OSA | 971 / 4,283 | 0.770 (0.744–0.797) | 0.773 (0.748–0.800) |
| MDD | 617 / 4,283 | 0.688 (0.647–0.728) | 0.622 (0.582–0.667) |
| Anxiety | 1,133 / 4,283 | 0.652 (0.618–0.681) | 0.629 (0.595–0.658) |
| Bipolar | 254 / 4,283 | 0.702 (0.641–0.758) | 0.649 (0.577–0.711) |
| ADHD | 187 / 4,283 | 0.641 (0.568–0.713) | 0.606 (0.531–0.683) |

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on the shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| single condition | CVD | 1,736 | 0.803 | 0.813 | +0.010 (+0.003 to +0.017) | **0.004** |
| single condition | Diabetes | 1,625 | 0.778 | 0.769 | -0.009 (-0.018 to +0.000) | 0.059 |
| single condition | OSA | 1,572 | 0.780 | 0.770 | -0.009 (-0.019 to +0.001) | 0.068 |
| single condition | MDD | 1,456 | 0.689 | 0.688 | -0.001 (-0.024 to +0.022) | 0.919 |
| single condition | Anxiety | 1,620 | 0.642 | 0.652 | +0.010 (-0.011 to +0.031) | 0.367 |
| single condition | Bipolar | 1,372 | 0.727 | 0.702 | -0.025 (-0.070 to +0.019) | 0.259 |
| single condition | ADHD | 1,352 | 0.633 | 0.641 | +0.007 (-0.065 to +0.079) | 0.842 |

### Paired AUROC comparison: XGBoost

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| single condition | CVD | 1,736 | 0.800 | 0.812 | +0.013 (+0.003 to +0.022) | **0.009** |
| single condition | Diabetes | 1,625 | 0.786 | 0.778 | -0.008 (-0.019 to +0.004) | 0.188 |
| single condition | OSA | 1,572 | 0.784 | 0.773 | -0.010 (-0.024 to +0.003) | 0.143 |
| single condition | MDD | 1,456 | 0.663 | 0.622 | -0.040 (-0.071 to -0.010) | **0.009** |
| single condition | Anxiety | 1,620 | 0.632 | 0.629 | -0.003 (-0.030 to +0.024) | 0.831 |
| single condition | Bipolar | 1,372 | 0.714 | 0.649 | -0.065 (-0.121 to -0.008) | **0.026** |
| single condition | ADHD | 1,352 | 0.645 | 0.606 | -0.039 (-0.125 to +0.047) | 0.375 |

## 3. Stricter-comparator analysis

**Definition:** all seven-day past-diagnosis cases versus controls with no record at any time of a listed sleep- or heart-rate-altering medication. This is a deliberately stricter comparator definition, not a medication-effect analysis and not a test of whether observed differences are caused by medications.

**Status:** completed as `stricter_comparator` under the corrected seven-day pre-index definition. Results: `results/seven_day_past_preindex_sensitivities/performance.csv`, `sample_sizes.csv`, and `matched_test_delong.csv`.

**Implementation:** completed in `src/seven_day_past_preindex_sensitivities.py`; the medication source is `data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv`. This comparator uses any recorded medication history, including post-index records.

### Results summary table

| Disease area | Cases / controls | LogReg AUROC (95% CI) | XGBoost AUROC (95% CI) |
|---|---|---|---|
| CVD | 3,536 / 2,536 | 0.843 (0.823–0.861) | 0.841 (0.820–0.860) |
| Diabetes | 3,050 / 2,536 | 0.835 (0.817–0.855) | 0.828 (0.809–0.849) |
| OSA | 3,346 / 2,536 | 0.832 (0.813–0.851) | 0.830 (0.810–0.849) |
| MDD | 3,014 / 2,536 | 0.759 (0.734–0.782) | 0.751 (0.728–0.774) |
| Anxiety | 3,578 / 2,536 | 0.715 (0.692–0.739) | 0.712 (0.689–0.735) |
| Bipolar | 1,028 / 2,536 | 0.772 (0.742–0.802) | 0.781 (0.753–0.811) |
| ADHD | 783 / 2,536 | 0.745 (0.707–0.780) | 0.734 (0.697–0.773) |

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on the shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| stricter comparator | CVD | 1,841 | 0.842 | 0.843 | +0.002 (-0.003 to +0.006) | 0.465 |
| stricter comparator | Diabetes | 1,670 | 0.833 | 0.835 | +0.003 (-0.003 to +0.008) | 0.338 |
| stricter comparator | OSA | 1,761 | 0.828 | 0.832 | +0.004 (-0.001 to +0.008) | 0.156 |
| stricter comparator | MDD | 1,691 | 0.753 | 0.759 | +0.006 (-0.001 to +0.014) | 0.090 |
| stricter comparator | Anxiety | 1,845 | 0.707 | 0.715 | +0.008 (-0.001 to +0.016) | 0.078 |
| stricter comparator | Bipolar | 1,047 | 0.766 | 0.772 | +0.007 (-0.004 to +0.017) | 0.225 |
| stricter comparator | ADHD | 990 | 0.725 | 0.745 | +0.020 (+0.005 to +0.035) | **0.010** |

### Paired AUROC comparison: XGBoost

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| stricter comparator | CVD | 1,841 | 0.844 | 0.841 | -0.003 (-0.010 to +0.004) | 0.371 |
| stricter comparator | Diabetes | 1,670 | 0.838 | 0.828 | -0.009 (-0.017 to -0.002) | **0.015** |
| stricter comparator | OSA | 1,761 | 0.833 | 0.830 | -0.003 (-0.012 to +0.005) | 0.422 |
| stricter comparator | MDD | 1,691 | 0.743 | 0.751 | +0.008 (-0.002 to +0.019) | 0.114 |
| stricter comparator | Anxiety | 1,845 | 0.702 | 0.712 | +0.010 (-0.003 to +0.022) | 0.130 |
| stricter comparator | Bipolar | 1,047 | 0.762 | 0.781 | +0.019 (+0.002 to +0.036) | **0.030** |
| stricter comparator | ADHD | 990 | 0.722 | 0.734 | +0.012 (-0.012 to +0.035) | 0.325 |

## 4. Medication-unexposed sensitivity analysis

**Definition:** past-diagnosis cases and controls without recorded exposure to prespecified medication classes during the Fitbit interval expanded by 30 calendar days on each side: `[streak_start − 30 days, streak_end + 30 days]`. A recorded exposure end date is used when available. If it is absent, the record is treated as a one-day exposure (`end date = start date`). Any resolved medication interval overlapping the expanded Fitbit window excludes the participant. The same rule and drug list are applied to cases and controls.

**Implementation:** `src/seven_day_past_preindex_sensitivities.py` defines the medication-overlap rule. `src/medication_unexposed_30day_window.py` reruns this sensitivity only, writing results to `results/medication_unexposed_30day_window/`. The medication source is `data/02_intermediate/study_cohorts_20251222/hr_and_sleep_drugs.csv`; no days-supply field is used.

**Interpretation:** this sensitivity evaluates robustness among participants without *recorded* relevant medication exposure in the buffered interval; it is not a treatment-effect analysis. The restriction preferentially removes many psychiatric cases and may select for disease severity, care access, and health-care use. Consequently, lower AUROCs cannot be attributed causally to medication exposure.

### Results summary table

Each cell is **AUROC (95% percentile-bootstrap CI); cases/controls**.

| Disease area | Primary LogReg | Medication-unexposed LogReg | Primary XGBoost | Medication-unexposed XGBoost |
|---|---|---|---|---|
| CVD | 0.835 (0.819–0.851); 3,536/4,283 | 0.836 (0.819–0.854); 2,399/4,061 | 0.835 (0.819–0.850); 3,536/4,283 | 0.834 (0.817–0.852); 2,399/4,061 |
| Diabetes | 0.830 (0.811–0.846); 3,050/4,283 | 0.827 (0.809–0.845); 2,106/4,061 | 0.829 (0.812–0.846); 3,050/4,283 | 0.825 (0.806–0.842); 2,106/4,061 |
| OSA | 0.826 (0.810–0.844); 3,346/4,283 | 0.826 (0.805–0.845); 2,033/4,061 | 0.826 (0.810–0.843); 3,346/4,283 | 0.824 (0.803–0.843); 2,033/4,061 |
| MDD | 0.748 (0.727–0.768); 3,014/4,283 | 0.699 (0.672–0.726); 1,479/4,061 | 0.739 (0.718–0.759); 3,014/4,283 | 0.690 (0.662–0.718); 1,479/4,061 |
| Anxiety | 0.711 (0.691–0.731); 3,578/4,283 | 0.673 (0.647–0.699); 2,025/4,061 | 0.704 (0.683–0.724); 3,578/4,283 | 0.658 (0.631–0.684); 2,025/4,061 |
| Bipolar | 0.760 (0.731–0.791); 1,028/4,283 | 0.747 (0.706–0.784); 510/4,061 | 0.749 (0.720–0.780); 1,028/4,283 | 0.715 (0.672–0.757); 510/4,061 |
| ADHD | 0.719 (0.681–0.758); 783/4,283 | 0.706 (0.648–0.762); 360/4,061 | 0.715 (0.681–0.753); 783/4,283 | 0.691 (0.636–0.749); 360/4,061 |

The buffered rule retained 4,061 controls for every disease area. It retained 2,399 CVD, 2,106 diabetes, 2,033 OSA, 1,479 MDD, 2,025 anxiety, 510 bipolar, and 360 ADHD cases.

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on their shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,943 | 0.836 | 0.836 | -0.000 (-0.002 to +0.002) | 0.993 |
| Diabetes | 1,842 | 0.829 | 0.827 | -0.002 (-0.004 to +0.001) | 0.169 |
| OSA | 1,808 | 0.823 | 0.826 | +0.003 (-0.000 to +0.006) | 0.070 |
| MDD | 1,659 | 0.714 | 0.699 | -0.015 (-0.022 to -0.008) | **<0.001** |
| Anxiety | 1,817 | 0.681 | 0.673 | -0.008 (-0.014 to -0.001) | **0.018** |
| Bipolar | 1,392 | 0.758 | 0.747 | -0.011 (-0.024 to +0.001) | 0.077 |
| ADHD | 1,329 | 0.698 | 0.706 | +0.008 (-0.011 to +0.026) | 0.404 |

### Paired AUROC comparison: XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,943 | 0.832 | 0.834 | +0.002 (-0.003 to +0.008) | 0.400 |
| Diabetes | 1,842 | 0.825 | 0.825 | -0.001 (-0.007 to +0.005) | 0.784 |
| OSA | 1,808 | 0.824 | 0.824 | -0.000 (-0.007 to +0.006) | 0.899 |
| MDD | 1,659 | 0.705 | 0.690 | -0.015 (-0.028 to -0.003) | **0.014** |
| Anxiety | 1,817 | 0.670 | 0.658 | -0.012 (-0.024 to -0.000) | **0.045** |
| Bipolar | 1,392 | 0.741 | 0.715 | -0.026 (-0.056 to +0.003) | 0.076 |
| ADHD | 1,329 | 0.703 | 0.691 | -0.012 (-0.053 to +0.029) | 0.562 |

## 5. Lookback sensitivity (completed)

Seven-day past-diagnosis models have been run for:

1. **No lookback:** any lifetime first diagnosis before the selected streak; no upper recency restriction.
2. **Current rules:** CVD/diabetes 10 years, OSA/ADHD 3 years, MDD/anxiety 2 years, bipolar 5 years, calculated from the most recent qualifying pre-index diagnosis.
3. **Halved rules:** CVD/diabetes 5 years, OSA/ADHD 1.5 years, MDD/anxiety 1 year, bipolar 2.5 years.

Completed corrected pre-index results: `results/past_lookback_sensitivity_7day_preindex/past_lookback_sensitivity_results.md`, `performance.csv`, `sample_sizes.csv`, and raw paired-test output `matched_test_delong.csv`. The selection/model code is `src/past_lookback_sensitivity.py`; paired-test code is `src/past_lookback_paired_tests.py`. The earlier non-pre-index folder is superseded.

### Results summary table

Each cell is **LogReg AUROC (95% CI) / XGBoost AUROC (95% CI); cases/controls**.

| Disease area | No lookback | Current rules | Halved rules |
|---|---|---|---|
| CVD | 0.840 (0.825–0.854) / 0.833 (0.816–0.847); 3,681/4,283 | 0.835 (0.819–0.851) / 0.835 (0.819–0.850); 3,536/4,283 | 0.836 (0.819–0.852) / 0.836 (0.819–0.852); 3,229/4,283 |
| Diabetes | 0.828 (0.810–0.845) / 0.829 (0.812–0.844); 3,225/4,283 | 0.830 (0.811–0.846) / 0.829 (0.812–0.846); 3,050/4,283 | 0.825 (0.807–0.842) / 0.821 (0.803–0.839); 2,705/4,283 |
| OSA | 0.827 (0.810–0.843) / 0.831 (0.816–0.847); 4,107/4,283 | 0.826 (0.810–0.844) / 0.826 (0.810–0.843); 3,346/4,283 | 0.839 (0.823–0.855) / 0.836 (0.820–0.853); 2,956/4,283 |
| MDD | 0.738 (0.720–0.756) / 0.742 (0.724–0.760); 4,583/4,283 | 0.748 (0.727–0.768) / 0.739 (0.718–0.759); 3,014/4,283 | 0.756 (0.734–0.775) / 0.757 (0.735–0.778); 2,498/4,283 |
| Anxiety | 0.694 (0.675–0.712) / 0.681 (0.662–0.699); 5,654/4,283 | 0.711 (0.691–0.731) / 0.704 (0.683–0.724); 3,578/4,283 | 0.740 (0.718–0.761) / 0.735 (0.715–0.756); 2,955/4,283 |
| Bipolar | 0.763 (0.738–0.787) / 0.758 (0.729–0.781); 1,458/4,283 | 0.760 (0.731–0.791) / 0.749 (0.720–0.780); 1,028/4,283 | 0.768 (0.735–0.801) / 0.758 (0.727–0.790); 793/4,283 |
| ADHD | 0.714 (0.681–0.747) / 0.706 (0.672–0.738); 1,092/4,283 | 0.719 (0.681–0.758) / 0.715 (0.681–0.753); 783/4,283 | 0.702 (0.662–0.743) / 0.670 (0.626–0.710); 662/4,283 |

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on the shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Lookback policy | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| no lookback | CVD | 741 | 0.846 | 0.847 | +0.000 (-0.005 to +0.005) | 0.955 |
| no lookback | Diabetes | 651 | 0.840 | 0.837 | -0.002 (-0.009 to +0.004) | 0.479 |
| no lookback | OSA | 728 | 0.800 | 0.795 | -0.006 (-0.011 to +0.000) | 0.067 |
| no lookback | MDD | 636 | 0.770 | 0.755 | -0.015 (-0.032 to +0.002) | 0.087 |
| no lookback | Anxiety | 715 | 0.705 | 0.700 | -0.005 (-0.021 to +0.011) | 0.539 |
| no lookback | Bipolar | 482 | 0.754 | 0.768 | +0.014 (-0.013 to +0.042) | 0.298 |
| no lookback | ADHD | 463 | 0.709 | 0.716 | +0.008 (-0.019 to +0.034) | 0.575 |
| halved rules | CVD | 680 | 0.825 | 0.826 | +0.001 (-0.006 to +0.007) | 0.795 |
| halved rules | Diabetes | 649 | 0.838 | 0.837 | -0.002 (-0.008 to +0.005) | 0.603 |
| halved rules | OSA | 662 | 0.848 | 0.844 | -0.004 (-0.010 to +0.001) | 0.133 |
| halved rules | MDD | 591 | 0.762 | 0.771 | +0.009 (-0.004 to +0.023) | 0.187 |
| halved rules | Anxiety | 643 | 0.736 | 0.733 | -0.003 (-0.014 to +0.008) | 0.607 |
| halved rules | Bipolar | 461 | 0.787 | 0.779 | -0.008 (-0.028 to +0.013) | 0.460 |
| halved rules | ADHD | 452 | 0.635 | 0.640 | +0.005 (-0.018 to +0.027) | 0.681 |

### Paired AUROC comparison: XGBoost

| Lookback policy | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| no lookback | CVD | 741 | 0.840 | 0.836 | -0.004 (-0.013 to +0.006) | 0.424 |
| no lookback | Diabetes | 651 | 0.837 | 0.834 | -0.002 (-0.013 to +0.008) | 0.645 |
| no lookback | OSA | 728 | 0.806 | 0.802 | -0.004 (-0.015 to +0.006) | 0.445 |
| no lookback | MDD | 636 | 0.754 | 0.752 | -0.003 (-0.023 to +0.018) | 0.795 |
| no lookback | Anxiety | 715 | 0.699 | 0.691 | -0.008 (-0.031 to +0.014) | 0.485 |
| no lookback | Bipolar | 482 | 0.736 | 0.747 | +0.011 (-0.021 to +0.043) | 0.484 |
| no lookback | ADHD | 463 | 0.693 | 0.726 | +0.033 (-0.010 to +0.076) | 0.131 |
| halved rules | CVD | 680 | 0.820 | 0.828 | +0.009 (-0.002 to +0.019) | 0.104 |
| halved rules | Diabetes | 649 | 0.834 | 0.828 | -0.007 (-0.019 to +0.005) | 0.270 |
| halved rules | OSA | 662 | 0.840 | 0.836 | -0.004 (-0.015 to +0.007) | 0.482 |
| halved rules | MDD | 591 | 0.752 | 0.762 | +0.010 (-0.010 to +0.030) | 0.310 |
| halved rules | Anxiety | 643 | 0.735 | 0.741 | +0.006 (-0.016 to +0.028) | 0.588 |
| halved rules | Bipolar | 461 | 0.763 | 0.778 | +0.015 (-0.022 to +0.051) | 0.432 |
| halved rules | ADHD | 452 | 0.673 | 0.620 | -0.053 (-0.103 to -0.004) | **0.035** |

## 6. Seven-day versus 14- and 21-day streak definitions

The seven-day past-diagnosis model is the primary definition because it materially increases usable sample size while retaining broadly consistent AUROC and principal directions of association across 7, 14, and 21 consecutive days. This decision should be justified primarily by retention and interpretability, not by selecting the highest observed AUROC.

**Status:** completed under the corrected pre-index past-diagnosis definition. Results are in `results/preindex_streak_duration_sensitivity/preindex_streak_duration_sensitivity_results.md` with machine-readable `performance.csv`, `sample_sizes.csv`, `matched_test_delong.csv`, `logreg_coefficients.csv`, and `coefficient_direction_concordance.csv`. Code: `src/preindex_streak_duration_sensitivity.py`; report renderer: `src/render_preindex_streak_duration_report.py`.

The 7-day model retains substantially more analytic participants (e.g., CVD 3,536 cases / 4,283 controls, versus 2,695 / 3,445 at 14 days and 2,075 / 2,813 at 21 days). Raw two-sided paired DeLong P-values for 14- or 21-day versus seven-day models are shown below without multiplicity correction. All coefficients that were FDR-significant at every duration had the same direction across 7, 14, and 21 days. Raw coefficient signs/magnitudes are not directly comparable across the changing cohorts.

### Results summary table

Each cell is **LogReg AUROC (95% CI) / XGBoost AUROC (95% CI); cases/controls**.

| Disease area | 7 days | 14 days | 21 days |
|---|---|---|---|
| CVD | 0.835 (0.819–0.851) / 0.835 (0.819–0.850); 3,536/4,283 | 0.845 (0.827–0.863) / 0.842 (0.824–0.859); 2,695/3,445 | 0.857 (0.839–0.875) / 0.857 (0.838–0.876); 2,075/2,813 |
| Diabetes | 0.830 (0.811–0.846) / 0.829 (0.812–0.846); 3,050/4,283 | 0.822 (0.802–0.841) / 0.825 (0.805–0.845); 2,153/3,445 | 0.821 (0.795–0.842) / 0.826 (0.802–0.847); 1,608/2,813 |
| OSA | 0.826 (0.810–0.844) / 0.826 (0.810–0.843); 3,346/4,283 | 0.830 (0.810–0.850) / 0.829 (0.809–0.849); 2,504/3,445 | 0.829 (0.806–0.850) / 0.818 (0.794–0.839); 1,888/2,813 |
| MDD | 0.748 (0.727–0.768) / 0.739 (0.718–0.759); 3,014/4,283 | 0.737 (0.713–0.759) / 0.738 (0.713–0.762); 2,184/3,445 | 0.748 (0.721–0.774) / 0.734 (0.704–0.760); 1,632/2,813 |
| Anxiety | 0.711 (0.691–0.731) / 0.704 (0.683–0.724); 3,578/4,283 | 0.722 (0.697–0.745) / 0.719 (0.695–0.742); 2,638/3,445 | 0.707 (0.678–0.735) / 0.696 (0.669–0.725); 1,964/2,813 |
| Bipolar | 0.760 (0.731–0.791) / 0.749 (0.720–0.780); 1,028/4,283 | 0.804 (0.771–0.837) / 0.779 (0.746–0.809); 734/3,445 | 0.784 (0.748–0.821) / 0.761 (0.719–0.801); 516/2,813 |
| ADHD | 0.719 (0.681–0.758) / 0.715 (0.681–0.753); 783/4,283 | 0.696 (0.647–0.741) / 0.658 (0.607–0.703); 533/3,445 | 0.709 (0.656–0.761) / 0.649 (0.589–0.703); 385/2,813 |

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on the shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Streak comparison | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| 14 vs 7 days | CVD | 553 | 0.835 | 0.847 | +0.012 (+0.001 to +0.023) | **0.030** |
| 14 vs 7 days | Diabetes | 500 | 0.847 | 0.844 | -0.004 (-0.016 to +0.008) | 0.541 |
| 14 vs 7 days | OSA | 549 | 0.829 | 0.823 | -0.007 (-0.016 to +0.003) | 0.180 |
| 14 vs 7 days | MDD | 485 | 0.727 | 0.741 | +0.014 (-0.010 to +0.037) | 0.250 |
| 14 vs 7 days | Anxiety | 562 | 0.701 | 0.694 | -0.007 (-0.028 to +0.014) | 0.506 |
| 14 vs 7 days | Bipolar | 390 | 0.794 | 0.808 | +0.014 (-0.009 to +0.037) | 0.242 |
| 14 vs 7 days | ADHD | 365 | 0.653 | 0.715 | +0.061 (+0.017 to +0.105) | **0.006** |
| 21 vs 7 days | CVD | 441 | 0.851 | 0.841 | -0.009 (-0.025 to +0.007) | 0.254 |
| 21 vs 7 days | Diabetes | 394 | 0.846 | 0.846 | -0.000 (-0.014 to +0.014) | 0.978 |
| 21 vs 7 days | OSA | 420 | 0.821 | 0.813 | -0.008 (-0.021 to +0.005) | 0.235 |
| 21 vs 7 days | MDD | 418 | 0.734 | 0.736 | +0.002 (-0.023 to +0.027) | 0.869 |
| 21 vs 7 days | Anxiety | 415 | 0.695 | 0.704 | +0.009 (-0.018 to +0.036) | 0.501 |
| 21 vs 7 days | Bipolar | 311 | 0.762 | 0.757 | -0.005 (-0.050 to +0.040) | 0.831 |
| 21 vs 7 days | ADHD | 311 | 0.708 | 0.653 | -0.055 (-0.117 to +0.008) | 0.087 |

### Paired AUROC comparison: XGBoost

| Streak comparison | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| 14 vs 7 days | CVD | 553 | 0.830 | 0.836 | +0.006 (-0.007 to +0.019) | 0.340 |
| 14 vs 7 days | Diabetes | 500 | 0.826 | 0.841 | +0.015 (-0.001 to +0.031) | 0.063 |
| 14 vs 7 days | OSA | 549 | 0.830 | 0.826 | -0.003 (-0.020 to +0.013) | 0.682 |
| 14 vs 7 days | MDD | 485 | 0.722 | 0.738 | +0.016 (-0.011 to +0.044) | 0.250 |
| 14 vs 7 days | Anxiety | 562 | 0.697 | 0.690 | -0.008 (-0.035 to +0.020) | 0.586 |
| 14 vs 7 days | Bipolar | 390 | 0.764 | 0.772 | +0.008 (-0.040 to +0.056) | 0.735 |
| 14 vs 7 days | ADHD | 365 | 0.654 | 0.649 | -0.005 (-0.079 to +0.069) | 0.895 |
| 21 vs 7 days | CVD | 441 | 0.849 | 0.849 | +0.000 (-0.016 to +0.017) | 0.976 |
| 21 vs 7 days | Diabetes | 394 | 0.834 | 0.841 | +0.007 (-0.015 to +0.029) | 0.532 |
| 21 vs 7 days | OSA | 420 | 0.820 | 0.794 | -0.026 (-0.043 to -0.009) | **0.003** |
| 21 vs 7 days | MDD | 418 | 0.716 | 0.719 | +0.003 (-0.032 to +0.037) | 0.873 |
| 21 vs 7 days | Anxiety | 415 | 0.668 | 0.691 | +0.024 (-0.013 to +0.060) | 0.209 |
| 21 vs 7 days | Bipolar | 311 | 0.725 | 0.733 | +0.008 (-0.042 to +0.058) | 0.757 |
| 21 vs 7 days | ADHD | 311 | 0.681 | 0.603 | -0.078 (-0.167 to +0.011) | 0.085 |

## 7. Stricter diagnosis confirmation

**Definition:** a case must have at least two mapped diagnosis records for the same disease area on **two distinct calendar dates** on or before the selected seven-day Fitbit streak start, with at least one qualifying pre-index diagnosis satisfying the selected lookback rule. Repeated records on the same date do not satisfy this criterion.

**Purpose:** this sensitivity addresses potential phenotype misclassification from a single ICD-coded diagnosis. It is a stricter EHR-record confirmation rule, not evidence of clinical gold-standard diagnosis.

**Status:** completed as `two_date_confirmation` under the corrected seven-day pre-index definition. Results: `results/seven_day_past_preindex_sensitivities/performance.csv`, `sample_sizes.csv`, and `matched_test_delong.csv`.

### Results summary table

| Disease area | Cases / controls | LogReg AUROC (95% CI) | XGBoost AUROC (95% CI) |
|---|---|---|---|
| CVD | 2,534 / 4,283 | 0.864 (0.850–0.880) | 0.858 (0.844–0.874) |
| Diabetes | 2,154 / 4,283 | 0.829 (0.810–0.847) | 0.832 (0.814–0.850) |
| OSA | 2,809 / 4,283 | 0.854 (0.838–0.869) | 0.850 (0.833–0.866) |
| MDD | 2,414 / 4,283 | 0.757 (0.735–0.777) | 0.758 (0.736–0.777) |
| Anxiety | 2,658 / 4,283 | 0.740 (0.719–0.761) | 0.733 (0.712–0.754) |
| Bipolar | 777 / 4,283 | 0.809 (0.782–0.835) | 0.804 (0.774–0.831) |
| ADHD | 592 / 4,283 | 0.764 (0.726–0.800) | 0.744 (0.706–0.783) |

Paired, two-sided DeLong tests compare independently refit primary and sensitivity models on the shared held-out participants. **Bold P-values denote raw P < 0.05; no multiple-testing correction was applied.**

### Paired AUROC comparison: Logistic regression

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| two date confirmation | CVD | 617 | 0.868 | 0.874 | +0.006 (-0.001 to +0.014) | 0.078 |
| two date confirmation | Diabetes | 581 | 0.812 | 0.821 | +0.008 (-0.000 to +0.017) | 0.057 |
| two date confirmation | OSA | 619 | 0.847 | 0.848 | +0.001 (-0.005 to +0.007) | 0.749 |
| two date confirmation | MDD | 622 | 0.748 | 0.745 | -0.003 (-0.013 to +0.007) | 0.510 |
| two date confirmation | Anxiety | 629 | 0.751 | 0.760 | +0.010 (-0.002 to +0.021) | 0.098 |
| two date confirmation | Bipolar | 457 | 0.796 | 0.792 | -0.004 (-0.018 to +0.009) | 0.524 |
| two date confirmation | ADHD | 465 | 0.735 | 0.752 | +0.016 (-0.014 to +0.046) | 0.284 |

### Paired AUROC comparison: XGBoost

| Sensitivity | Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (sensitivity − primary, 95% CI) | Raw paired P |
|---|---|---|---|---|---|---|
| two date confirmation | CVD | 617 | 0.862 | 0.866 | +0.004 (-0.007 to +0.016) | 0.448 |
| two date confirmation | Diabetes | 581 | 0.808 | 0.827 | +0.019 (+0.006 to +0.032) | **0.005** |
| two date confirmation | OSA | 619 | 0.837 | 0.837 | -0.000 (-0.011 to +0.011) | 0.958 |
| two date confirmation | MDD | 622 | 0.722 | 0.737 | +0.015 (-0.004 to +0.034) | 0.121 |
| two date confirmation | Anxiety | 629 | 0.740 | 0.756 | +0.016 (-0.003 to +0.036) | 0.105 |
| two date confirmation | Bipolar | 457 | 0.796 | 0.785 | -0.011 (-0.045 to +0.024) | 0.549 |
| two date confirmation | ADHD | 465 | 0.697 | 0.708 | +0.012 (-0.035 to +0.058) | 0.628 |

## 8. Past versus future diagnosis performance

**Primary timing analysis:** past diagnosis only, as defined above.

**Exploratory appendix analysis:** future diagnosis only. A future case has its first mapped diagnosis after the end of its **first** qualifying seven-day streak; diagnoses during the streak are excluded. The first-streak rule maximizes the future-case sample but does not establish near-term prospective prediction, because first recorded diagnosis may be remote from clinical onset or reflect later documentation.

Existing seven-day past/future results are available in:

- `results/streak_timing_sensitivity/streak_timing_sensitivity_results.md`
- `results/streak_timing_sensitivity/performance.csv`
- `results/streak_timing_sensitivity/sample_sizes.csv`
- Analysis code: `src/streak_timing_sensitivity_models.py`

Future results should be placed in the appendix/supplement only, labelled exploratory, and should not be used to claim incident-disease prediction or alter the main manuscript’s past-diagnosis primary estimates. They are not an incident-disease prediction analysis because controls were not required to have comparable post-index observation and first recorded diagnosis may lag disease onset or reflect later documentation.

### Results summary table

| Disease area | Cases / controls | LogReg AUROC (95% CI) | XGBoost AUROC (95% CI) |
|---|---|---|---|
| CVD | 756 / 4,296 | 0.829 (0.799–0.858) | 0.833 (0.806–0.859) |
| Diabetes | 586 / 4,296 | 0.842 (0.811–0.870) | 0.850 (0.818–0.880) |
| OSA | 819 / 4,296 | 0.807 (0.777–0.837) | 0.819 (0.791–0.847) |
| MDD | 750 / 4,296 | 0.799 (0.768–0.825) | 0.802 (0.770–0.831) |
| Anxiety | 1,272 / 4,296 | 0.757 (0.728–0.784) | 0.766 (0.740–0.791) |
| Bipolar | 167 / 4,296 | 0.786 (0.709–0.854) | 0.788 (0.709–0.857) |
| ADHD | 242 / 4,296 | 0.833 (0.770–0.888) | 0.836 (0.786–0.884) |

## 9. Missingness and complete-case assessment

**Status:** completed. Cohort flow by disease is in `results/seven_day_past_preindex_missingness/cohort_flow_complete_case.csv`; demographic and survey variable missingness are in `person_variable_missingness.csv` and `survey_variable_missingness.csv`; retained-versus-excluded descriptive characteristics are in `retention_characteristics.csv` in the same folder. Code: `src/seven_day_preindex_missingness.py`.

**Required reporting:** Report cohort flow and variable-level missingness before and after applying wearable-data eligibility and complete-case requirements. Compare the characteristics of participants retained versus excluded where feasible.

**Interpretation:** Do not impute unavailable Fitbit-derived features, because absent eligible wearable data represent structural non-observation rather than a conventional partially missing covariate. For ordinary non-wearable covariates, report the extent and pattern of missingness and consider a multiple-imputation sensitivity analysis only if missingness is material. The complete-case decision should distinguish selection caused by wearable-data availability from the incremental loss caused by missing analytic covariates.

### Results summary table

| Disease area | Cases selected | Controls selected | Cases after static | Controls after static | Complete-case cases | Complete-case controls | Complete-case total |
|---|---|---|---|---|---|---|---|
| CVD | 3712 | 4671 | 3555 | 4316 | 3536 | 4283 | 7819 |
| Diabetes | 3247 | 4671 | 3068 | 4316 | 3050 | 4283 | 7333 |
| OSA | 3479 | 4671 | 3365 | 4316 | 3346 | 4283 | 7629 |
| MDD | 3140 | 4671 | 3027 | 4316 | 3014 | 4283 | 7297 |
| Anxiety | 3739 | 4671 | 3593 | 4316 | 3578 | 4283 | 7861 |
| Bipolar | 1086 | 4671 | 1037 | 4316 | 1028 | 4283 | 5311 |
| ADHD | 829 | 4671 | 786 | 4316 | 783 | 4283 | 5066 |

## Reporting requirements

The full completed suite is summarized in `results/seven_day_past_preindex_sensitivities/seven_day_past_preindex_sensitivity_results.md`; code is `src/seven_day_past_preindex_sensitivities.py` and shared cohort construction is `src/seven_day_past_preindex.py`.

For every re-estimated analysis, report per disease area: case/control N, AUROC and Brier skill score with 95% percentile-bootstrap CIs, model estimator/fallback status, and the exact diagnosis, medication, streak, and timing rules. Report raw paired two-sided DeLong P-values for the requested comparisons, without multiplicity correction. Retain the full coefficient tables in supplement only.

No manuscript files were modified.
