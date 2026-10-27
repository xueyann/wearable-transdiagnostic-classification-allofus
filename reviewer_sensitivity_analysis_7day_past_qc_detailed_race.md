# QC-filtered seven-day past-diagnosis sensitivity analysis: detailed-race rerun

## Scope and primary definition

This complete rerun uses the QC-filtered Fitbit summary cohort: each eligible day has at least 600 HR-zone minutes and 100 steps before continuous-streak selection. HR-zone minutes are a provisional summary-level proxy for observed heart-rate coverage, not validated minute-level coverage.

The primary model uses seven consecutive QC-eligible days and the latest eligible streak. Cases have a qualifying diagnosis on or before streak start and meet the disease-specific current lookback rule. Controls are lifetime-free of all target diagnoses and relevant ICD exclusions. The deterministic seed-42 participant split is used throughout.

Race is coded with White as the reference and four indicators: Black or African American, Asian, more than one population, and other/small/unknown. The final group combines all remaining recorded values, including sparse categories and no-response values. This is the primary covariate coding in every analysis below; no binary-race versus detailed-race comparison is reported.

**Results/code:** duration: `src/preindex_streak_duration_qc_detailed_race.py`, `results/preindex_streak_duration_qc_detailed_race/`; core: `src/seven_day_past_preindex_qc_detailed_race_sensitivities.py`, `results/seven_day_past_preindex_qc_detailed_race_sensitivities/`; lookback: `src/past_lookback_qc_detailed_race_sensitivity.py`, `src/past_lookback_qc_detailed_race_paired_tests.py`; future timing: `src/qc_future_diagnosis_detailed_race_exploratory.py`; missingness: `src/seven_day_preindex_qc_detailed_race_missingness.py`; demographic control: `src/qc_detailed_race_demographic_control.py`.

Paired comparisons use two-sided DeLong tests for correlated AUROCs on shared held-out participants. Bold P-values are raw P < 0.05; no multiplicity correction was applied.

## 1. Primary QC model and duration sensitivity

### Logistic regression

| Disease area | 7 days (primary) | 14 days | 21 days |
|---|---|---|---|
| CVD | 0.850 (0.834–0.864); 3,528/4,274 | 0.832 (0.813–0.850); 2,675/3,433 | 0.842 (0.821–0.863); 2,049/2,791 |
| Diabetes | 0.832 (0.815–0.846); 3,040/4,274 | 0.824 (0.803–0.843); 2,136/3,433 | 0.826 (0.802–0.849); 1,595/2,791 |
| OSA | 0.832 (0.816–0.848); 3,335/4,274 | 0.842 (0.824–0.861); 2,485/3,433 | 0.823 (0.801–0.843); 1,859/2,791 |
| MDD | 0.746 (0.725–0.765); 3,003/4,274 | 0.729 (0.704–0.752); 2,170/3,433 | 0.755 (0.728–0.781); 1,610/2,791 |
| Anxiety | 0.705 (0.683–0.725); 3,561/4,274 | 0.726 (0.702–0.747); 2,618/3,433 | 0.680 (0.651–0.708); 1,934/2,791 |
| Bipolar | 0.774 (0.746–0.803); 1,027/4,274 | 0.775 (0.744–0.805); 728/3,433 | 0.781 (0.737–0.819); 509/2,791 |
| ADHD | 0.722 (0.688–0.757); 778/4,274 | 0.735 (0.691–0.771); 532/3,433 | 0.741 (0.691–0.793); 381/2,791 |

### XGBoost

| Disease area | 7 days (primary) | 14 days | 21 days |
|---|---|---|---|
| CVD | 0.837 (0.821–0.852); 3,528/4,274 | 0.829 (0.810–0.847); 2,675/3,433 | 0.832 (0.811–0.852); 2,049/2,791 |
| Diabetes | 0.836 (0.820–0.853); 3,040/4,274 | 0.823 (0.802–0.842); 2,136/3,433 | 0.825 (0.803–0.847); 1,595/2,791 |
| OSA | 0.828 (0.812–0.845); 3,335/4,274 | 0.830 (0.811–0.849); 2,485/3,433 | 0.817 (0.794–0.837); 1,859/2,791 |
| MDD | 0.743 (0.721–0.764); 3,003/4,274 | 0.726 (0.702–0.751); 2,170/3,433 | 0.750 (0.723–0.778); 1,610/2,791 |
| Anxiety | 0.713 (0.693–0.733); 3,561/4,274 | 0.715 (0.693–0.738); 2,618/3,433 | 0.660 (0.631–0.690); 1,934/2,791 |
| Bipolar | 0.783 (0.756–0.811); 1,027/4,274 | 0.764 (0.730–0.798); 728/3,433 | 0.770 (0.732–0.809); 509/2,791 |
| ADHD | 0.722 (0.689–0.756); 778/4,274 | 0.725 (0.682–0.762); 532/3,433 | 0.678 (0.627–0.731); 381/2,791 |

### Paired duration comparisons

| Comparison | Disease area | Model | Matched test N | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| 14 vs 7 days | CVD | logistic | 568 | -0.001 (-0.010 to +0.008) | 0.790 |
| 14 vs 7 days | CVD | xgboost | 568 | -0.010 (-0.025 to +0.005) | 0.176 |
| 14 vs 7 days | Diabetes | logistic | 521 | +0.001 (-0.012 to +0.015) | 0.834 |
| 14 vs 7 days | Diabetes | xgboost | 521 | -0.004 (-0.020 to +0.012) | 0.631 |
| 14 vs 7 days | OSA | logistic | 560 | +0.010 (+0.001 to +0.019) | **0.036** |
| 14 vs 7 days | OSA | xgboost | 560 | -0.002 (-0.017 to +0.013) | 0.814 |
| 14 vs 7 days | MDD | logistic | 514 | -0.014 (-0.036 to +0.007) | 0.194 |
| 14 vs 7 days | MDD | xgboost | 514 | -0.004 (-0.034 to +0.025) | 0.774 |
| 14 vs 7 days | Anxiety | logistic | 547 | -0.005 (-0.026 to +0.017) | 0.672 |
| 14 vs 7 days | Anxiety | xgboost | 547 | -0.022 (-0.051 to +0.007) | 0.135 |
| 14 vs 7 days | Bipolar | logistic | 370 | +0.007 (-0.024 to +0.038) | 0.656 |
| 14 vs 7 days | Bipolar | xgboost | 370 | +0.009 (-0.039 to +0.057) | 0.711 |
| 14 vs 7 days | ADHD | logistic | 346 | +0.026 (-0.015 to +0.068) | 0.216 |
| 14 vs 7 days | ADHD | xgboost | 346 | -0.002 (-0.065 to +0.061) | 0.954 |
| 21 vs 7 days | CVD | logistic | 434 | -0.012 (-0.024 to +0.000) | 0.059 |
| 21 vs 7 days | CVD | xgboost | 434 | +0.004 (-0.016 to +0.024) | 0.683 |
| 21 vs 7 days | Diabetes | logistic | 398 | -0.014 (-0.033 to +0.005) | 0.144 |
| 21 vs 7 days | Diabetes | xgboost | 398 | -0.007 (-0.028 to +0.015) | 0.533 |
| 21 vs 7 days | OSA | logistic | 426 | +0.009 (-0.003 to +0.021) | 0.133 |
| 21 vs 7 days | OSA | xgboost | 426 | +0.002 (-0.016 to +0.021) | 0.816 |
| 21 vs 7 days | MDD | logistic | 385 | +0.005 (-0.023 to +0.033) | 0.744 |
| 21 vs 7 days | MDD | xgboost | 385 | -0.021 (-0.055 to +0.013) | 0.221 |
| 21 vs 7 days | Anxiety | logistic | 396 | -0.001 (-0.025 to +0.023) | 0.939 |
| 21 vs 7 days | Anxiety | xgboost | 396 | -0.042 (-0.076 to -0.007) | **0.019** |
| 21 vs 7 days | Bipolar | logistic | 304 | +0.065 (+0.020 to +0.109) | **0.004** |
| 21 vs 7 days | Bipolar | xgboost | 304 | +0.019 (-0.049 to +0.087) | 0.589 |
| 21 vs 7 days | ADHD | logistic | 262 | -0.026 (-0.086 to +0.034) | 0.397 |
| 21 vs 7 days | ADHD | xgboost | 262 | -0.089 (-0.155 to -0.022) | **0.009** |

## 2. Multimorbidity

The non-exclusive overlap matrix is descriptive; condition-specific case cohorts are not a single common cohort.

| Primary-case cohort | ADHD | Anxiety | Bipolar | CVD | Diabetes | MDD | OSA |
|---|---|---|---|---|---|---|---|
| ADHD | 778 | 408 | 147 | 117 | 110 | 382 | 204 |
| Anxiety | 408 | 3561 | 446 | 718 | 634 | 1608 | 968 |
| Bipolar | 147 | 446 | 1027 | 239 | 253 | 464 | 325 |
| CVD | 117 | 718 | 239 | 3528 | 1015 | 709 | 1174 |
| Diabetes | 110 | 634 | 253 | 1015 | 3040 | 659 | 1055 |
| MDD | 382 | 1608 | 464 | 709 | 659 | 3003 | 1007 |
| OSA | 204 | 968 | 325 | 1174 | 1055 | 1007 | 3335 |

## 3. Core robustness analyses

### Logistic regression

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.850 (0.834–0.864); 3,528/4,274 | 0.839 (0.819–0.859); 1,483/4,274 | 0.857 (0.837–0.875); 3,528/2,526 | 0.842 (0.825–0.859); 2,391/4,052 | 0.864 (0.849–0.880); 2,526/4,274 |
| Diabetes | 0.832 (0.815–0.846); 3,040/4,274 | 0.770 (0.745–0.794); 1,177/4,274 | 0.838 (0.819–0.857); 3,040/2,526 | 0.821 (0.799–0.841); 2,098/4,052 | 0.846 (0.829–0.863); 2,145/4,274 |
| OSA | 0.832 (0.816–0.848); 3,335/4,274 | 0.767 (0.739–0.793); 968/4,274 | 0.834 (0.817–0.854); 3,335/2,526 | 0.824 (0.804–0.843); 2,028/4,052 | 0.839 (0.825–0.856); 2,797/4,274 |
| MDD | 0.746 (0.725–0.765); 3,003/4,274 | 0.659 (0.616–0.701); 619/4,274 | 0.765 (0.745–0.787); 3,003/2,526 | 0.714 (0.689–0.741); 1,475/4,052 | 0.759 (0.739–0.781); 2,411/4,274 |
| Anxiety | 0.705 (0.683–0.725); 3,561/4,274 | 0.654 (0.620–0.688); 1,129/4,274 | 0.712 (0.688–0.736); 3,561/2,526 | 0.674 (0.647–0.699); 2,018/4,052 | 0.731 (0.710–0.753); 2,654/4,274 |
| Bipolar | 0.774 (0.746–0.803); 1,027/4,274 | 0.692 (0.618–0.758); 255/4,274 | 0.781 (0.749–0.813); 1,027/2,526 | 0.780 (0.741–0.820); 511/4,052 | 0.767 (0.733–0.798); 777/4,274 |
| ADHD | 0.722 (0.688–0.757); 778/4,274 | 0.685 (0.609–0.759); 185/4,274 | 0.742 (0.705–0.778); 778/2,526 | 0.670 (0.623–0.716); 364/4,052 | 0.721 (0.680–0.761); 587/4,274 |

### XGBoost

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.837 (0.821–0.852); 3,528/4,274 | 0.835 (0.814–0.857); 1,483/4,274 | 0.847 (0.827–0.865); 3,528/2,526 | 0.843 (0.827–0.861); 2,391/4,052 | 0.858 (0.842–0.874); 2,526/4,274 |
| Diabetes | 0.836 (0.820–0.853); 3,040/4,274 | 0.786 (0.761–0.811); 1,177/4,274 | 0.846 (0.827–0.864); 3,040/2,526 | 0.824 (0.804–0.844); 2,098/4,052 | 0.837 (0.819–0.855); 2,145/4,274 |
| OSA | 0.828 (0.812–0.845); 3,335/4,274 | 0.767 (0.740–0.793); 968/4,274 | 0.829 (0.810–0.848); 3,335/2,526 | 0.815 (0.795–0.834); 2,028/4,052 | 0.834 (0.819–0.850); 2,797/4,274 |
| MDD | 0.743 (0.721–0.764); 3,003/4,274 | 0.636 (0.593–0.678); 619/4,274 | 0.767 (0.744–0.789); 3,003/2,526 | 0.707 (0.679–0.736); 1,475/4,052 | 0.762 (0.741–0.783); 2,411/4,274 |
| Anxiety | 0.713 (0.693–0.733); 3,561/4,274 | 0.650 (0.616–0.683); 1,129/4,274 | 0.719 (0.695–0.741); 3,561/2,526 | 0.657 (0.631–0.682); 2,018/4,052 | 0.730 (0.710–0.751); 2,654/4,274 |
| Bipolar | 0.783 (0.756–0.811); 1,027/4,274 | 0.672 (0.606–0.733); 255/4,274 | 0.794 (0.762–0.824); 1,027/2,526 | 0.754 (0.712–0.791); 511/4,052 | 0.776 (0.745–0.806); 777/4,274 |
| ADHD | 0.722 (0.689–0.756); 778/4,274 | 0.593 (0.525–0.666); 185/4,274 | 0.701 (0.665–0.738); 778/2,526 | 0.662 (0.609–0.715); 364/4,052 | 0.720 (0.679–0.761); 587/4,274 |

### Single-condition paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,709 | 0.834 | 0.839 | +0.005 (-0.003 to +0.013) | 0.189 |
| Diabetes | 1,635 | 0.780 | 0.770 | -0.011 (-0.019 to -0.002) | **0.020** |
| OSA | 1,572 | 0.769 | 0.767 | -0.002 (-0.013 to +0.009) | 0.738 |
| MDD | 1,447 | 0.652 | 0.659 | +0.007 (-0.014 to +0.028) | 0.505 |
| Anxiety | 1,576 | 0.639 | 0.654 | +0.015 (-0.005 to +0.035) | 0.146 |
| Bipolar | 1,365 | 0.723 | 0.692 | -0.030 (-0.070 to +0.009) | 0.130 |
| ADHD | 1,348 | 0.671 | 0.685 | +0.014 (-0.054 to +0.083) | 0.683 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,709 | 0.815 | 0.835 | +0.020 (+0.009 to +0.031) | **0.000** |
| Diabetes | 1,635 | 0.791 | 0.786 | -0.005 (-0.016 to +0.006) | 0.385 |
| OSA | 1,572 | 0.765 | 0.767 | +0.002 (-0.011 to +0.015) | 0.768 |
| MDD | 1,447 | 0.650 | 0.636 | -0.014 (-0.044 to +0.017) | 0.386 |
| Anxiety | 1,576 | 0.652 | 0.650 | -0.002 (-0.029 to +0.025) | 0.887 |
| Bipolar | 1,365 | 0.732 | 0.672 | -0.060 (-0.124 to +0.004) | 0.066 |
| ADHD | 1,348 | 0.682 | 0.593 | -0.089 (-0.153 to -0.026) | **0.006** |

### Stricter-comparator paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,812 | 0.851 | 0.857 | +0.006 (+0.002 to +0.010) | **0.005** |
| Diabetes | 1,669 | 0.833 | 0.838 | +0.005 (-0.001 to +0.010) | 0.090 |
| OSA | 1,775 | 0.831 | 0.834 | +0.003 (-0.002 to +0.008) | 0.201 |
| MDD | 1,665 | 0.747 | 0.765 | +0.018 (+0.010 to +0.026) | **0.000** |
| Anxiety | 1,838 | 0.701 | 0.712 | +0.011 (+0.002 to +0.019) | **0.014** |
| Bipolar | 1,051 | 0.777 | 0.781 | +0.004 (-0.006 to +0.014) | 0.429 |
| ADHD | 1,000 | 0.724 | 0.742 | +0.018 (+0.003 to +0.033) | **0.016** |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,812 | 0.840 | 0.847 | +0.007 (+0.000 to +0.014) | **0.049** |
| Diabetes | 1,669 | 0.841 | 0.846 | +0.004 (-0.003 to +0.012) | 0.280 |
| OSA | 1,775 | 0.834 | 0.829 | -0.005 (-0.014 to +0.004) | 0.241 |
| MDD | 1,665 | 0.744 | 0.767 | +0.023 (+0.011 to +0.035) | **0.000** |
| Anxiety | 1,838 | 0.714 | 0.719 | +0.005 (-0.008 to +0.018) | 0.434 |
| Bipolar | 1,051 | 0.791 | 0.794 | +0.003 (-0.013 to +0.018) | 0.739 |
| ADHD | 1,000 | 0.728 | 0.701 | -0.027 (-0.055 to +0.002) | 0.070 |

### Medication-unexposed paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,904 | 0.844 | 0.842 | -0.002 (-0.004 to +0.001) | 0.252 |
| Diabetes | 1,853 | 0.824 | 0.821 | -0.003 (-0.006 to -0.000) | **0.021** |
| OSA | 1,836 | 0.820 | 0.824 | +0.004 (+0.001 to +0.007) | **0.023** |
| MDD | 1,660 | 0.718 | 0.714 | -0.004 (-0.011 to +0.003) | 0.280 |
| Anxiety | 1,788 | 0.678 | 0.674 | -0.004 (-0.010 to +0.001) | 0.138 |
| Bipolar | 1,381 | 0.786 | 0.780 | -0.007 (-0.020 to +0.007) | 0.334 |
| ADHD | 1,336 | 0.678 | 0.670 | -0.008 (-0.031 to +0.015) | 0.489 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1,904 | 0.832 | 0.843 | +0.012 (+0.005 to +0.018) | **0.000** |
| Diabetes | 1,853 | 0.826 | 0.824 | -0.002 (-0.008 to +0.004) | 0.536 |
| OSA | 1,836 | 0.818 | 0.815 | -0.003 (-0.009 to +0.004) | 0.390 |
| MDD | 1,660 | 0.721 | 0.707 | -0.013 (-0.026 to -0.001) | **0.033** |
| Anxiety | 1,788 | 0.689 | 0.657 | -0.032 (-0.045 to -0.019) | **0.000** |
| Bipolar | 1,381 | 0.780 | 0.754 | -0.025 (-0.048 to -0.003) | **0.028** |
| ADHD | 1,336 | 0.692 | 0.662 | -0.030 (-0.074 to +0.014) | 0.179 |

### Two-date-confirmation paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 611 | 0.856 | 0.861 | +0.005 (-0.003 to +0.014) | 0.217 |
| Diabetes | 556 | 0.843 | 0.854 | +0.011 (+0.001 to +0.020) | **0.030** |
| OSA | 643 | 0.849 | 0.853 | +0.004 (-0.002 to +0.010) | 0.201 |
| MDD | 613 | 0.755 | 0.753 | -0.002 (-0.011 to +0.007) | 0.624 |
| Anxiety | 654 | 0.707 | 0.707 | -0.000 (-0.010 to +0.009) | 0.974 |
| Bipolar | 438 | 0.792 | 0.779 | -0.013 (-0.044 to +0.018) | 0.408 |
| ADHD | 442 | 0.743 | 0.768 | +0.025 (-0.003 to +0.052) | 0.085 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 611 | 0.842 | 0.855 | +0.013 (+0.001 to +0.025) | **0.035** |
| Diabetes | 556 | 0.847 | 0.846 | -0.001 (-0.016 to +0.013) | 0.842 |
| OSA | 643 | 0.838 | 0.840 | +0.002 (-0.010 to +0.013) | 0.775 |
| MDD | 613 | 0.750 | 0.763 | +0.013 (-0.003 to +0.029) | 0.113 |
| Anxiety | 654 | 0.733 | 0.721 | -0.013 (-0.030 to +0.005) | 0.163 |
| Bipolar | 438 | 0.774 | 0.776 | +0.001 (-0.024 to +0.026) | 0.934 |
| ADHD | 442 | 0.731 | 0.750 | +0.019 (-0.037 to +0.075) | 0.511 |

**Definitions.** The stricter comparator excludes controls with any listed relevant medication record. The medication-unexposed analysis excludes cases and controls with a listed medication overlapping `[streak start − 30 days, streak end + 30 days]`; missing end dates are treated as one-day records. Two-date confirmation requires diagnoses on two distinct dates.

## 4. Lookback sensitivity

Cases use the latest eligible seven-day streak under each policy. `no_lookback` accepts any diagnosis before index; `current_rules` is primary; `halved_rules` halves each disease-specific upper limit.

### Logistic regression

| Disease area | Current (primary) | No lookback | Halved lookback |
|---|---|---|---|
| CVD | 0.850 (0.834–0.864); 3,528/4,274 | 0.840 (0.825–0.856); 3,673/4,274 | 0.844 (0.827–0.860); 3,221/4,274 |
| Diabetes | 0.832 (0.815–0.846); 3,040/4,274 | 0.828 (0.811–0.845); 3,214/4,274 | 0.839 (0.821–0.855); 2,697/4,274 |
| OSA | 0.832 (0.816–0.848); 3,335/4,274 | 0.828 (0.811–0.843); 4,094/4,274 | 0.847 (0.831–0.863); 2,943/4,274 |
| MDD | 0.746 (0.725–0.765); 3,003/4,274 | 0.747 (0.729–0.766); 4,573/4,274 | 0.779 (0.758–0.799); 2,483/4,274 |
| Anxiety | 0.705 (0.683–0.725); 3,561/4,274 | 0.709 (0.690–0.726); 5,645/4,274 | 0.722 (0.701–0.744); 2,944/4,274 |
| Bipolar | 0.774 (0.746–0.803); 1,027/4,274 | 0.772 (0.746–0.798); 1,456/4,274 | 0.775 (0.739–0.808); 792/4,274 |
| ADHD | 0.722 (0.688–0.757); 778/4,274 | 0.736 (0.705–0.767); 1,086/4,274 | 0.729 (0.692–0.767); 656/4,274 |

### XGBoost

| Disease area | Current (primary) | No lookback | Halved lookback |
|---|---|---|---|
| CVD | 0.837 (0.821–0.852); 3,528/4,274 | 0.832 (0.817–0.848); 3,673/4,274 | 0.833 (0.816–0.849); 3,221/4,274 |
| Diabetes | 0.836 (0.820–0.853); 3,040/4,274 | 0.827 (0.809–0.843); 3,214/4,274 | 0.835 (0.816–0.852); 2,697/4,274 |
| OSA | 0.828 (0.812–0.845); 3,335/4,274 | 0.820 (0.804–0.836); 4,094/4,274 | 0.846 (0.829–0.862); 2,943/4,274 |
| MDD | 0.743 (0.721–0.764); 3,003/4,274 | 0.744 (0.726–0.762); 4,573/4,274 | 0.771 (0.750–0.792); 2,483/4,274 |
| Anxiety | 0.713 (0.693–0.733); 3,561/4,274 | 0.703 (0.682–0.720); 5,645/4,274 | 0.704 (0.683–0.725); 2,944/4,274 |
| Bipolar | 0.783 (0.756–0.811); 1,027/4,274 | 0.772 (0.748–0.797); 1,456/4,274 | 0.757 (0.723–0.788); 792/4,274 |
| ADHD | 0.722 (0.689–0.756); 778/4,274 | 0.711 (0.677–0.743); 1,086/4,274 | 0.662 (0.620–0.705); 656/4,274 |

### Logistic regression paired comparisons

| Policy | Disease area | Matched test N | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|
| No lookback | CVD | 726 | +0.004 (-0.000 to +0.008) | 0.050 |
| Halved lookback | CVD | 651 | -0.005 (-0.011 to +0.001) | 0.085 |
| No lookback | Diabetes | 682 | -0.000 (-0.005 to +0.004) | 0.894 |
| Halved lookback | Diabetes | 652 | +0.003 (-0.002 to +0.008) | 0.257 |
| No lookback | OSA | 694 | +0.001 (-0.006 to +0.008) | 0.781 |
| Halved lookback | OSA | 646 | +0.002 (-0.006 to +0.010) | 0.654 |
| No lookback | MDD | 673 | -0.013 (-0.029 to +0.003) | 0.101 |
| Halved lookback | MDD | 605 | +0.012 (-0.002 to +0.026) | 0.095 |
| No lookback | Anxiety | 728 | -0.013 (-0.028 to +0.003) | 0.120 |
| Halved lookback | Anxiety | 631 | +0.012 (-0.000 to +0.025) | 0.058 |
| No lookback | Bipolar | 476 | +0.003 (-0.023 to +0.029) | 0.815 |
| Halved lookback | Bipolar | 450 | -0.004 (-0.028 to +0.020) | 0.755 |
| No lookback | ADHD | 464 | +0.007 (-0.024 to +0.038) | 0.657 |
| Halved lookback | ADHD | 463 | +0.003 (-0.026 to +0.033) | 0.826 |

### XGBoost paired comparisons

| Policy | Disease area | Matched test N | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|
| No lookback | CVD | 726 | +0.009 (-0.002 to +0.020) | 0.123 |
| Halved lookback | CVD | 651 | +0.010 (-0.002 to +0.022) | 0.110 |
| No lookback | Diabetes | 682 | +0.000 (-0.009 to +0.010) | 0.945 |
| Halved lookback | Diabetes | 652 | +0.002 (-0.009 to +0.013) | 0.674 |
| No lookback | OSA | 694 | -0.004 (-0.016 to +0.008) | 0.499 |
| Halved lookback | OSA | 646 | -0.001 (-0.012 to +0.010) | 0.884 |
| No lookback | MDD | 673 | +0.006 (-0.016 to +0.029) | 0.570 |
| Halved lookback | MDD | 605 | +0.001 (-0.021 to +0.023) | 0.916 |
| No lookback | Anxiety | 728 | -0.027 (-0.050 to -0.005) | **0.016** |
| Halved lookback | Anxiety | 631 | -0.006 (-0.027 to +0.015) | 0.565 |
| No lookback | Bipolar | 476 | +0.005 (-0.022 to +0.032) | 0.727 |
| Halved lookback | Bipolar | 450 | -0.013 (-0.049 to +0.024) | 0.491 |
| No lookback | ADHD | 464 | -0.006 (-0.052 to +0.039) | 0.785 |
| Halved lookback | ADHD | 463 | -0.075 (-0.138 to -0.012) | **0.020** |

## 5. Future-diagnosis exploratory analysis

This appendix-only analysis uses the first QC-eligible seven-day streak. Future cases have their first mapped diagnosis after its end and no earlier mapped diagnosis for that condition; controls use their first QC-eligible streak.

### Logistic regression

| Disease area | Future diagnosis |
|---|---|
| CVD | 0.824 (0.793–0.852); 753/4,274 |
| Diabetes | 0.836 (0.807–0.863); 582/4,274 |
| OSA | 0.776 (0.746–0.806); 813/4,274 |
| MDD | 0.763 (0.726–0.796); 742/4,274 |
| Anxiety | 0.774 (0.748–0.798); 1,263/4,274 |
| Bipolar | 0.804 (0.736–0.865); 167/4,274 |
| ADHD | 0.795 (0.737–0.846); 241/4,274 |

### XGBoost

| Disease area | Future diagnosis |
|---|---|
| CVD | 0.832 (0.804–0.860); 753/4,274 |
| Diabetes | 0.825 (0.793–0.857); 582/4,274 |
| OSA | 0.791 (0.762–0.820); 813/4,274 |
| MDD | 0.758 (0.720–0.792); 742/4,274 |
| Anxiety | 0.779 (0.752–0.803); 1,263/4,274 |
| Bipolar | 0.764 (0.688–0.829); 167/4,274 |
| ADHD | 0.744 (0.670–0.809); 241/4,274 |

## 6. Missingness and complete-case assessment

Wearable QC eligibility is applied before streak selection. Missing Fitbit-derived data are not imputed; absence of a QC-eligible streak is structural non-observation. Source demographic and survey variables have zero recorded missingness, so the small incremental loss occurs during analytic-frame construction.

### Cohort flow

| Disease area | Cases selected | Controls selected | Cases after static | Controls after static | Complete-case cases | Complete-case controls | Complete-case total |
|---|---|---|---|---|---|---|---|
| CVD | 3694 | 4635 | 3538 | 4282 | 3528 | 4274 | 7802 |
| Diabetes | 3227 | 4635 | 3048 | 4282 | 3040 | 4274 | 7314 |
| OSA | 3457 | 4635 | 3343 | 4282 | 3335 | 4274 | 7609 |
| MDD | 3122 | 4635 | 3009 | 4282 | 3003 | 4274 | 7277 |
| Anxiety | 3713 | 4635 | 3567 | 4282 | 3561 | 4274 | 7835 |
| Bipolar | 1080 | 4635 | 1031 | 4282 | 1027 | 4274 | 5301 |
| ADHD | 822 | 4635 | 779 | 4282 | 778 | 4274 | 5052 |

### Retained-versus-excluded characteristics

Excluded groups are small and should not be overinterpreted.

| condition | retention_group | n | case_percent | age_mean | bmi_mean | male_percent | nonwhite_percent |
|---|---|---|---|---|---|---|---|
| CVD | excluded_after_static | 18 | 55.55555555555556 | 62.388888888888886 | 30.44151034009151 | 50.0 | 33.33333333333333 |
| CVD | retained_complete_case | 7802 | 45.21917457062292 | 57.55267880030761 | 29.126671026659025 | 36.09330940784415 | 25.711356062548063 |
| Diabetes | excluded_after_static | 16 | 50.0 | 58.6875 | 30.338513828581952 | 37.5 | 31.25 |
| Diabetes | retained_complete_case | 7314 | 41.56412359857807 | 54.24077112387203 | 30.41779484721331 | 31.473885698660105 | 30.708230790265244 |
| OSA | excluded_after_static | 16 | 50.0 | 60.0625 | 32.48676848134365 | 37.5 | 25.0 |
| OSA | retained_complete_case | 7609 | 43.8296753844132 | 54.584045209620186 | 30.61577543667574 | 33.762649494020245 | 27.704034695755027 |
| MDD | excluded_after_static | 14 | 42.85714285714285 | 55.5 | 30.927049913916427 | 35.714285714285715 | 35.714285714285715 |
| MDD | retained_complete_case | 7277 | 41.26700563418992 | 50.95767486601621 | 29.553773013267847 | 25.546241583069943 | 27.621272502404835 |
| Anxiety | excluded_after_static | 14 | 42.85714285714285 | 52.71428571428572 | 30.034977459968008 | 21.428571428571427 | 28.57142857142857 |
| Anxiety | retained_complete_case | 7835 | 45.44990427568602 | 50.2721123165284 | 29.3389126470446 | 24.671346522016595 | 26.47096362476069 |
| Bipolar | excluded_after_static | 12 | 33.33333333333333 | 54.91666666666666 | 32.82530050963754 | 25.0 | 41.66666666666667 |
| Bipolar | retained_complete_case | 5301 | 19.37370307489153 | 50.10564044519902 | 28.7507715663447 | 26.806262969251083 | 29.40954536879834 |
| ADHD | excluded_after_static | 9 | 11.11111111111111 | 55.11111111111112 | 31.62107604883912 | 22.22222222222222 | 44.44444444444444 |
| ADHD | retained_complete_case | 5052 | 15.399841646872526 | 49.17399049881235 | 28.34854382139269 | 26.959619952494062 | 29.473475851148063 |

## 7. Primary-model calibration metric and coefficient stability

Brier skill score (BSS) is reported only for the detailed-race QC seven-day primary model and uses the held-out prevalence as reference with 1,000 percentile-bootstrap intervals.

| Disease area | Logistic BSS (95% CI) | XGBoost BSS (95% CI) |
|---|---|---|
| CVD | 0.663 (0.641–0.685) | 0.648 (0.625–0.671) |
| Diabetes | 0.608 (0.583–0.628) | 0.613 (0.589–0.635) |
| OSA | 0.627 (0.601–0.649) | 0.625 (0.599–0.647) |
| MDD | 0.524 (0.498–0.548) | 0.523 (0.497–0.545) |
| Anxiety | 0.531 (0.506–0.553) | 0.538 (0.514–0.559) |
| Bipolar | 0.327 (0.279–0.369) | 0.328 (0.282–0.368) |
| ADHD | 0.208 (0.156–0.253) | 0.199 (0.150–0.241) |

The deterministic training-partition logistic coefficients are in `results/preindex_streak_duration_qc_detailed_race/logreg_coefficients.csv`.

| Disease area | Features compared | Same direction, all | Direction concordance (%) | FDR-significant, all | Same direction among FDR-significant |
|---|---|---|---|---|---|
| ADHD | 52 | 34 | 65.4 | 6 | 6 |
| Anxiety | 52 | 38 | 73.1 | 8 | 8 |
| Bipolar | 52 | 32 | 61.5 | 2 | 2 |
| CVD | 52 | 32 | 61.5 | 5 | 5 |
| Diabetes | 52 | 24 | 46.2 | 3 | 3 |
| MDD | 52 | 32 | 61.5 | 7 | 7 |
| OSA | 52 | 36 | 69.2 | 5 | 5 |

Calibration curves in the manuscript must be regenerated for this QC/detailed-race primary cohort before finalization.

## 8. QC-matched demographic-only control

The full detailed-race primary model and demographic-only control use the exact same QC-filtered seven-day past-diagnosis cohort and seed-42 split. The control includes age, sex at birth, BMI, and the same four detailed-race indicators; it omits Fitbit, survey, and other clinical covariates.

### Logistic regression

| Disease area | Primary AUROC | Primary BSS | Demographic-only AUROC | Demographic-only BSS |
|---|---|---|---|---|
| CVD | 0.850 (0.834–0.864) | 0.663 (0.641–0.685) | 0.825 (0.808–0.841) | 0.636 (0.613–0.658) |
| Diabetes | 0.832 (0.815–0.846) | 0.608 (0.583–0.628) | 0.801 (0.782–0.818) | 0.576 (0.551–0.597) |
| OSA | 0.832 (0.816–0.848) | 0.627 (0.601–0.649) | 0.814 (0.795–0.829) | 0.608 (0.584–0.629) |
| MDD | 0.746 (0.725–0.765) | 0.524 (0.498–0.548) | 0.671 (0.649–0.693) | 0.469 (0.445–0.491) |
| Anxiety | 0.705 (0.683–0.725) | 0.531 (0.506–0.553) | 0.630 (0.607–0.652) | 0.492 (0.470–0.512) |
| Bipolar | 0.774 (0.746–0.803) | 0.327 (0.279–0.369) | 0.709 (0.675–0.741) | 0.257 (0.223–0.283) |
| ADHD | 0.722 (0.688–0.757) | 0.208 (0.156–0.253) | 0.671 (0.631–0.706) | 0.186 (0.149–0.217) |

### XGBoost

| Disease area | Primary AUROC | Primary BSS | Demographic-only AUROC | Demographic-only BSS |
|---|---|---|---|---|
| CVD | 0.837 (0.821–0.852) | 0.648 (0.625–0.671) | 0.826 (0.810–0.842) | 0.637 (0.614–0.660) |
| Diabetes | 0.836 (0.820–0.853) | 0.613 (0.589–0.635) | 0.801 (0.782–0.817) | 0.576 (0.551–0.599) |
| OSA | 0.828 (0.812–0.845) | 0.625 (0.599–0.647) | 0.812 (0.794–0.828) | 0.608 (0.583–0.630) |
| MDD | 0.743 (0.721–0.764) | 0.523 (0.497–0.545) | 0.670 (0.649–0.694) | 0.469 (0.445–0.491) |
| Anxiety | 0.713 (0.693–0.733) | 0.538 (0.514–0.559) | 0.633 (0.609–0.656) | 0.494 (0.470–0.514) |
| Bipolar | 0.783 (0.756–0.811) | 0.328 (0.282–0.368) | 0.705 (0.672–0.739) | 0.253 (0.216–0.284) |
| ADHD | 0.722 (0.689–0.756) | 0.199 (0.150–0.241) | 0.650 (0.610–0.685) | 0.171 (0.131–0.204) |

### Demographic-only versus primary: Logistic regression

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2,340 | 0.850 | 0.825 | -0.025 (-0.033 to -0.016) | **0.000** |
| Diabetes | 2,194 | 0.832 | 0.801 | -0.030 (-0.041 to -0.020) | **0.000** |
| OSA | 2,282 | 0.832 | 0.814 | -0.019 (-0.028 to -0.010) | **0.000** |
| MDD | 2,183 | 0.746 | 0.671 | -0.075 (-0.095 to -0.056) | **0.000** |
| Anxiety | 2,350 | 0.705 | 0.630 | -0.075 (-0.094 to -0.055) | **0.000** |
| Bipolar | 1,590 | 0.774 | 0.709 | -0.065 (-0.096 to -0.035) | **0.000** |
| ADHD | 1,515 | 0.722 | 0.671 | -0.051 (-0.080 to -0.022) | **0.001** |

### Demographic-only versus primary: XGBoost

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2,340 | 0.837 | 0.826 | -0.011 (-0.020 to -0.001) | **0.034** |
| Diabetes | 2,194 | 0.836 | 0.801 | -0.036 (-0.047 to -0.024) | **0.000** |
| OSA | 2,282 | 0.828 | 0.812 | -0.016 (-0.026 to -0.006) | **0.002** |
| MDD | 2,183 | 0.743 | 0.670 | -0.073 (-0.093 to -0.053) | **0.000** |
| Anxiety | 2,350 | 0.713 | 0.633 | -0.080 (-0.100 to -0.060) | **0.000** |
| Bipolar | 1,590 | 0.783 | 0.705 | -0.078 (-0.107 to -0.048) | **0.000** |
| ADHD | 1,515 | 0.722 | 0.650 | -0.072 (-0.105 to -0.040) | **0.000** |

## Interpretation

The QC-filtered detailed-race rerun retains the same primary scientific framing: the models describe association with recorded, potentially treated prevalent disease among participants meeting a summary-level QC criterion. They are neither incident-disease prediction models nor causal medication analyses. Future-diagnosis results are exploratory and belong in an appendix.
