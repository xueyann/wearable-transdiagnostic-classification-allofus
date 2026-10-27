# QC-filtered seven-day past-diagnosis sensitivity analysis

## Scope and primary definition

The primary analysis uses summary-level Fitbit days meeting **at least 600 HR-zone minutes and 100 steps**, before feature processing and continuous-streak selection. HR-zone minutes are a *provisional summary-level proxy* for observed heart-rate coverage: the available summary field is not validated minute-level coverage and can exceed 1,440 minutes. Thus this is a reproducible summary-data QC sensitivity, not confirmation of the reviewer-requested minute-level adherence rule. See `fitbit_summary_qc.md` and `src/fitbit_summary_qc.py`.

For each disease, cases have at least one mapped diagnosis on or before the selected streak start. The latest qualifying pre-index diagnosis must meet the disease-specific current lookback criterion; the latest eligible continuous streak is selected per case. Controls are lifetime-free of all target diagnoses and relevant ICD exclusions and use their latest eligible streak. The main analysis uses seven consecutive eligible days. Models use the existing deterministic seed (42) and participant split.

**Results/code:** duration: `src/preindex_streak_duration_qc_sensitivity.py`, `results/preindex_streak_duration_qc_sensitivity/`; core sensitivities: `src/seven_day_past_preindex_qc_sensitivities.py`, `results/seven_day_past_preindex_qc_sensitivities/`; lookback: `src/past_lookback_qc_sensitivity.py`, `src/past_lookback_qc_paired_tests.py`, `results/past_lookback_sensitivity_qc_7day_preindex/`; timing: `src/qc_future_diagnosis_exploratory.py`, `results/qc_future_diagnosis_exploratory/`; retention: `src/seven_day_preindex_qc_missingness.py`, `results/seven_day_past_preindex_qc_missingness/`.

Paired comparisons use the two-sided DeLong test for correlated AUROCs on shared held-out participants. Bold P-values are raw P < 0.05; no multiplicity correction was applied.

## 1. Primary QC model and duration sensitivity

Each cell is AUROC (95% percentile-bootstrap CI); cases/controls.

### Logistic regression

| Disease area | 7 days (primary) | 14 days | 21 days |
|---|---|---|---|
| CVD | 0.849 (0.834–0.864); 3,528/4,274 | 0.832 (0.812–0.850); 2,675/3,433 | 0.842 (0.822–0.863); 2,049/2,791 |
| Diabetes | 0.832 (0.815–0.847); 3,040/4,274 | 0.824 (0.803–0.843); 2,136/3,433 | 0.826 (0.802–0.849); 1,595/2,791 |
| OSA | 0.832 (0.815–0.848); 3,335/4,274 | 0.842 (0.824–0.861); 2,485/3,433 | 0.822 (0.801–0.843); 1,859/2,791 |
| MDD | 0.746 (0.724–0.765); 3,003/4,274 | 0.730 (0.705–0.753); 2,170/3,433 | 0.754 (0.726–0.780); 1,610/2,791 |
| Anxiety | 0.703 (0.682–0.723); 3,561/4,274 | 0.726 (0.703–0.747); 2,618/3,433 | 0.676 (0.647–0.704); 1,934/2,791 |
| Bipolar | 0.773 (0.745–0.801); 1,027/4,274 | 0.770 (0.738–0.801); 728/3,433 | 0.781 (0.737–0.818); 509/2,791 |
| ADHD | 0.719 (0.684–0.753); 778/4,274 | 0.732 (0.686–0.769); 532/3,433 | 0.739 (0.687–0.790); 381/2,791 |

### XGBoost

| Disease area | 7 days (primary) | 14 days | 21 days |
|---|---|---|---|
| CVD | 0.839 (0.823–0.855); 3,528/4,274 | 0.828 (0.809–0.846); 2,675/3,433 | 0.833 (0.812–0.853); 2,049/2,791 |
| Diabetes | 0.838 (0.822–0.854); 3,040/4,274 | 0.826 (0.806–0.845); 2,136/3,433 | 0.821 (0.799–0.845); 1,595/2,791 |
| OSA | 0.827 (0.811–0.844); 3,335/4,274 | 0.835 (0.817–0.853); 2,485/3,433 | 0.817 (0.794–0.837); 1,859/2,791 |
| MDD | 0.738 (0.716–0.758); 3,003/4,274 | 0.723 (0.698–0.749); 2,170/3,433 | 0.752 (0.725–0.780); 1,610/2,791 |
| Anxiety | 0.693 (0.672–0.713); 3,561/4,274 | 0.715 (0.691–0.737); 2,618/3,433 | 0.661 (0.632–0.689); 1,934/2,791 |
| Bipolar | 0.782 (0.756–0.809); 1,027/4,274 | 0.755 (0.723–0.787); 728/3,433 | 0.759 (0.719–0.800); 509/2,791 |
| ADHD | 0.719 (0.685–0.752); 778/4,274 | 0.719 (0.671–0.758); 532/3,433 | 0.682 (0.626–0.739); 381/2,791 |

Raw DeLong results for 14-versus-7 and 21-versus-7 days are in `results/preindex_streak_duration_qc_sensitivity/matched_test_delong.csv`; no comparison remained significant after the stored Benjamini–Hochberg adjustment. The duration comparison did not change the substantive interpretation, supporting the prespecified QC seven-day primary definition.

## 2. Multimorbidity

The following is a descriptive, non-exclusive overlap matrix among primary QC case cohorts. Because cases are selected condition-specifically, this is not a single common analytic cohort.

| Primary-case cohort | CVD | Diabetes | OSA | MDD | Anxiety | Bipolar | ADHD |
|---|---|---|---|---|---|---|---|
| CVD | 3528 | 1015 | 1174 | 709 | 718 | 239 | 117 |
| Diabetes | 1015 | 3040 | 1055 | 659 | 634 | 253 | 110 |
| OSA | 1174 | 1055 | 3335 | 1007 | 968 | 325 | 204 |
| MDD | 709 | 659 | 1007 | 3003 | 1608 | 464 | 382 |
| Anxiety | 718 | 634 | 968 | 1608 | 3561 | 446 | 408 |
| Bipolar | 239 | 253 | 325 | 464 | 446 | 1027 | 147 |
| ADHD | 117 | 110 | 204 | 382 | 408 | 147 | 778 |

Single-condition analyses below restrict cases to those with only the target qualifying condition at that selected index.

## 3. Core robustness analyses

### Logistic regression

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.849 (0.834–0.864); 3,528/4,274 | 0.839 (0.819–0.859); 1,483/4,274 | 0.857 (0.837–0.874); 3,528/2,526 | 0.842 (0.825–0.859); 2,391/4,052 | 0.864 (0.848–0.879); 2,526/4,274 |
| Diabetes | 0.832 (0.815–0.847); 3,040/4,274 | 0.771 (0.746–0.796); 1,177/4,274 | 0.838 (0.819–0.857); 3,040/2,526 | 0.822 (0.800–0.842); 2,098/4,052 | 0.845 (0.829–0.863); 2,145/4,274 |
| OSA | 0.832 (0.815–0.848); 3,335/4,274 | 0.766 (0.738–0.792); 968/4,274 | 0.833 (0.816–0.853); 3,335/2,526 | 0.824 (0.803–0.843); 2,028/4,052 | 0.838 (0.824–0.855); 2,797/4,274 |
| MDD | 0.746 (0.724–0.765); 3,003/4,274 | 0.659 (0.615–0.700); 619/4,274 | 0.765 (0.744–0.787); 3,003/2,526 | 0.713 (0.687–0.741); 1,475/4,052 | 0.760 (0.739–0.782); 2,411/4,274 |
| Anxiety | 0.703 (0.682–0.723); 3,561/4,274 | 0.652 (0.618–0.686); 1,129/4,274 | 0.712 (0.689–0.736); 3,561/2,526 | 0.673 (0.646–0.698); 2,018/4,052 | 0.729 (0.708–0.751); 2,654/4,274 |
| Bipolar | 0.773 (0.745–0.801); 1,027/4,274 | 0.698 (0.631–0.761); 255/4,274 | 0.778 (0.748–0.810); 1,027/2,526 | 0.781 (0.742–0.821); 511/4,052 | 0.764 (0.730–0.796); 777/4,274 |
| ADHD | 0.719 (0.684–0.753); 778/4,274 | 0.670 (0.591–0.747); 185/4,274 | 0.737 (0.699–0.773); 778/2,526 | 0.670 (0.624–0.718); 364/4,052 | 0.718 (0.676–0.757); 587/4,274 |

### XGBoost

| Disease area | Primary | Single condition | Stricter comparator | Medication-unexposed | Two diagnosis dates |
|---|---|---|---|---|---|
| CVD | 0.839 (0.823–0.855); 3,528/4,274 | 0.837 (0.816–0.858); 1,483/4,274 | 0.848 (0.828–0.866); 3,528/2,526 | 0.843 (0.827–0.861); 2,391/4,052 | 0.857 (0.841–0.872); 2,526/4,274 |
| Diabetes | 0.838 (0.822–0.854); 3,040/4,274 | 0.791 (0.766–0.816); 1,177/4,274 | 0.845 (0.826–0.863); 3,040/2,526 | 0.825 (0.804–0.844); 2,098/4,052 | 0.837 (0.819–0.855); 2,145/4,274 |
| OSA | 0.827 (0.811–0.844); 3,335/4,274 | 0.765 (0.739–0.791); 968/4,274 | 0.833 (0.815–0.853); 3,335/2,526 | 0.816 (0.796–0.836); 2,028/4,052 | 0.835 (0.819–0.851); 2,797/4,274 |
| MDD | 0.738 (0.716–0.758); 3,003/4,274 | 0.631 (0.588–0.675); 619/4,274 | 0.766 (0.744–0.788); 3,003/2,526 | 0.705 (0.677–0.733); 1,475/4,052 | 0.762 (0.741–0.784); 2,411/4,274 |
| Anxiety | 0.693 (0.672–0.713); 3,561/4,274 | 0.654 (0.622–0.685); 1,129/4,274 | 0.714 (0.689–0.738); 3,561/2,526 | 0.658 (0.633–0.684); 2,018/4,052 | 0.730 (0.709–0.751); 2,654/4,274 |
| Bipolar | 0.782 (0.756–0.809); 1,027/4,274 | 0.638 (0.565–0.707); 255/4,274 | 0.789 (0.758–0.820); 1,027/2,526 | 0.746 (0.704–0.786); 511/4,052 | 0.777 (0.746–0.807); 777/4,274 |
| ADHD | 0.719 (0.685–0.752); 778/4,274 | 0.597 (0.525–0.671); 185/4,274 | 0.733 (0.697–0.770); 778/2,526 | 0.662 (0.615–0.715); 364/4,052 | 0.722 (0.680–0.764); 587/4,274 |

### Single-condition paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1709 | 0.834 | 0.839 | +0.005 (-0.003 to +0.013) | 0.216 |
| Diabetes | 1635 | 0.781 | 0.771 | -0.010 (-0.018 to -0.001) | **0.023** |
| OSA | 1572 | 0.769 | 0.766 | -0.002 (-0.014 to +0.009) | 0.673 |
| MDD | 1447 | 0.652 | 0.659 | +0.007 (-0.014 to +0.028) | 0.521 |
| Anxiety | 1576 | 0.637 | 0.652 | +0.015 (-0.005 to +0.035) | 0.142 |
| Bipolar | 1365 | 0.719 | 0.698 | -0.021 (-0.063 to +0.021) | 0.325 |
| ADHD | 1348 | 0.658 | 0.670 | +0.013 (-0.057 to +0.082) | 0.724 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1709 | 0.818 | 0.837 | +0.019 (+0.008 to +0.031) | **0.001** |
| Diabetes | 1635 | 0.793 | 0.791 | -0.002 (-0.013 to +0.010) | 0.782 |
| OSA | 1572 | 0.767 | 0.765 | -0.002 (-0.014 to +0.011) | 0.791 |
| MDD | 1447 | 0.646 | 0.631 | -0.015 (-0.048 to +0.017) | 0.360 |
| Anxiety | 1576 | 0.630 | 0.654 | +0.025 (-0.003 to +0.053) | 0.078 |
| Bipolar | 1365 | 0.738 | 0.638 | -0.100 (-0.166 to -0.033) | **0.003** |
| ADHD | 1348 | 0.668 | 0.597 | -0.070 (-0.147 to +0.007) | 0.074 |

### Stricter-comparator paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1812 | 0.851 | 0.857 | +0.006 (+0.002 to +0.010) | **0.004** |
| Diabetes | 1669 | 0.833 | 0.838 | +0.005 (-0.001 to +0.010) | 0.087 |
| OSA | 1775 | 0.830 | 0.833 | +0.003 (-0.002 to +0.008) | 0.229 |
| MDD | 1665 | 0.746 | 0.765 | +0.019 (+0.011 to +0.027) | **0.000** |
| Anxiety | 1838 | 0.700 | 0.712 | +0.012 (+0.004 to +0.021) | **0.006** |
| Bipolar | 1051 | 0.776 | 0.778 | +0.003 (-0.008 to +0.013) | 0.610 |
| ADHD | 1000 | 0.718 | 0.737 | +0.018 (+0.003 to +0.034) | **0.018** |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1812 | 0.840 | 0.848 | +0.008 (+0.000 to +0.015) | **0.044** |
| Diabetes | 1669 | 0.843 | 0.845 | +0.002 (-0.005 to +0.010) | 0.528 |
| OSA | 1775 | 0.833 | 0.833 | +0.000 (-0.008 to +0.008) | 0.952 |
| MDD | 1665 | 0.741 | 0.766 | +0.025 (+0.013 to +0.037) | **0.000** |
| Anxiety | 1838 | 0.692 | 0.714 | +0.022 (+0.007 to +0.037) | **0.004** |
| Bipolar | 1051 | 0.791 | 0.789 | -0.002 (-0.017 to +0.013) | 0.768 |
| ADHD | 1000 | 0.727 | 0.733 | +0.007 (-0.014 to +0.028) | 0.532 |

### Medication-unexposed paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1904 | 0.844 | 0.842 | -0.001 (-0.004 to +0.001) | 0.305 |
| Diabetes | 1853 | 0.825 | 0.822 | -0.003 (-0.006 to -0.000) | **0.027** |
| OSA | 1836 | 0.820 | 0.824 | +0.004 (+0.001 to +0.008) | **0.019** |
| MDD | 1660 | 0.716 | 0.713 | -0.003 (-0.011 to +0.004) | 0.362 |
| Anxiety | 1788 | 0.676 | 0.673 | -0.004 (-0.009 to +0.002) | 0.198 |
| Bipolar | 1381 | 0.784 | 0.781 | -0.004 (-0.017 to +0.010) | 0.600 |
| ADHD | 1336 | 0.677 | 0.670 | -0.007 (-0.030 to +0.016) | 0.572 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 1904 | 0.835 | 0.843 | +0.008 (+0.002 to +0.015) | **0.012** |
| Diabetes | 1853 | 0.829 | 0.825 | -0.004 (-0.010 to +0.003) | 0.248 |
| OSA | 1836 | 0.817 | 0.816 | -0.001 (-0.007 to +0.005) | 0.768 |
| MDD | 1660 | 0.712 | 0.705 | -0.007 (-0.021 to +0.006) | 0.279 |
| Anxiety | 1788 | 0.675 | 0.658 | -0.016 (-0.031 to -0.002) | **0.028** |
| Bipolar | 1381 | 0.773 | 0.746 | -0.026 (-0.051 to -0.002) | **0.035** |
| ADHD | 1336 | 0.684 | 0.662 | -0.022 (-0.060 to +0.016) | 0.259 |

### Two-date-confirmation paired comparisons

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 611 | 0.855 | 0.859 | +0.004 (-0.004 to +0.013) | 0.303 |
| Diabetes | 556 | 0.844 | 0.855 | +0.010 (+0.001 to +0.020) | **0.028** |
| OSA | 643 | 0.848 | 0.852 | +0.004 (-0.002 to +0.010) | 0.175 |
| MDD | 613 | 0.758 | 0.753 | -0.005 (-0.014 to +0.004) | 0.289 |
| Anxiety | 654 | 0.703 | 0.704 | +0.001 (-0.009 to +0.011) | 0.839 |
| Bipolar | 438 | 0.787 | 0.777 | -0.010 (-0.041 to +0.020) | 0.517 |
| ADHD | 442 | 0.730 | 0.752 | +0.021 (-0.008 to +0.051) | 0.150 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Sensitivity AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 611 | 0.845 | 0.853 | +0.008 (-0.004 to +0.021) | 0.185 |
| Diabetes | 556 | 0.846 | 0.846 | -0.000 (-0.015 to +0.014) | 0.953 |
| OSA | 643 | 0.834 | 0.840 | +0.006 (-0.006 to +0.017) | 0.337 |
| MDD | 613 | 0.741 | 0.760 | +0.018 (+0.002 to +0.035) | **0.031** |
| Anxiety | 654 | 0.695 | 0.712 | +0.017 (-0.007 to +0.042) | 0.170 |
| Bipolar | 438 | 0.773 | 0.768 | -0.005 (-0.034 to +0.025) | 0.754 |
| ADHD | 442 | 0.744 | 0.748 | +0.004 (-0.042 to +0.049) | 0.879 |

**Definitions.** The stricter comparator excludes controls with any listed relevant medication record. The medication-unexposed analysis excludes cases and controls with a listed medication overlapping `[streak start − 30 days, streak end + 30 days]`; missing end dates are treated as one-day records. Two-date confirmation requires diagnoses on two distinct dates.

## 4. Lookback sensitivity

Cases use the latest eligible seven-day streak under each policy. `no_lookback` accepts any diagnosis before index; `current_rules` is primary; `halved_rules` halves each disease-specific upper limit.

### Logistic regression

| Disease area | Current (primary) | No lookback | Halved lookback |
|---|---|---|---|
| CVD | 0.849 (0.834–0.864); 3,528/4,274 | 0.840 (0.825–0.856); 3,673/4,274 | 0.844 (0.828–0.860); 3,221/4,274 |
| Diabetes | 0.832 (0.815–0.847); 3,040/4,274 | 0.828 (0.810–0.845); 3,214/4,274 | 0.838 (0.821–0.855); 2,697/4,274 |
| OSA | 0.832 (0.815–0.848); 3,335/4,274 | 0.828 (0.810–0.843); 4,094/4,274 | 0.847 (0.831–0.863); 2,943/4,274 |
| MDD | 0.746 (0.724–0.765); 3,003/4,274 | 0.746 (0.727–0.764); 4,573/4,274 | 0.778 (0.758–0.799); 2,483/4,274 |
| Anxiety | 0.703 (0.682–0.723); 3,561/4,274 | 0.709 (0.690–0.726); 5,645/4,274 | 0.721 (0.700–0.742); 2,944/4,274 |
| Bipolar | 0.773 (0.745–0.801); 1,027/4,274 | 0.767 (0.740–0.793); 1,456/4,274 | 0.773 (0.739–0.807); 792/4,274 |
| ADHD | 0.719 (0.684–0.753); 778/4,274 | 0.730 (0.697–0.763); 1,086/4,274 | 0.726 (0.688–0.765); 656/4,274 |

### XGBoost

| Disease area | Current (primary) | No lookback | Halved lookback |
|---|---|---|---|
| CVD | 0.839 (0.823–0.855); 3,528/4,274 | 0.832 (0.817–0.848); 3,673/4,274 | 0.834 (0.817–0.850); 3,221/4,274 |
| Diabetes | 0.838 (0.822–0.854); 3,040/4,274 | 0.828 (0.810–0.844); 3,214/4,274 | 0.835 (0.816–0.852); 2,697/4,274 |
| OSA | 0.827 (0.811–0.844); 3,335/4,274 | 0.821 (0.805–0.837); 4,094/4,274 | 0.844 (0.827–0.860); 2,943/4,274 |
| MDD | 0.738 (0.716–0.758); 3,003/4,274 | 0.745 (0.727–0.764); 4,573/4,274 | 0.773 (0.752–0.794); 2,483/4,274 |
| Anxiety | 0.693 (0.672–0.713); 3,561/4,274 | 0.706 (0.686–0.723); 5,645/4,274 | 0.708 (0.687–0.730); 2,944/4,274 |
| Bipolar | 0.782 (0.756–0.809); 1,027/4,274 | 0.769 (0.744–0.794); 1,456/4,274 | 0.759 (0.724–0.792); 792/4,274 |
| ADHD | 0.719 (0.685–0.752); 778/4,274 | 0.718 (0.684–0.751); 1,086/4,274 | 0.713 (0.674–0.752); 656/4,274 |

### Logistic regression paired comparisons

| Policy | Disease area | Matched test N | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|
| no_lookback | CVD | 726 | +0.004 (+0.001 to +0.008) | **0.026** |
| no_lookback | Diabetes | 682 | +0.001 (-0.003 to +0.005) | 0.631 |
| no_lookback | OSA | 694 | +0.001 (-0.006 to +0.008) | 0.765 |
| no_lookback | MDD | 673 | -0.013 (-0.029 to +0.002) | 0.099 |
| no_lookback | Anxiety | 728 | -0.013 (-0.029 to +0.003) | 0.110 |
| no_lookback | Bipolar | 476 | +0.005 (-0.021 to +0.032) | 0.692 |
| no_lookback | ADHD | 464 | +0.010 (-0.020 to +0.040) | 0.517 |
| halved_rules | CVD | 651 | -0.006 (-0.011 to -0.000) | **0.050** |
| halved_rules | Diabetes | 652 | +0.003 (-0.002 to +0.008) | 0.232 |
| halved_rules | OSA | 646 | +0.001 (-0.007 to +0.009) | 0.782 |
| halved_rules | MDD | 605 | +0.012 (-0.003 to +0.027) | 0.108 |
| halved_rules | Anxiety | 631 | +0.013 (-0.000 to +0.026) | 0.056 |
| halved_rules | Bipolar | 450 | -0.001 (-0.026 to +0.024) | 0.922 |
| halved_rules | ADHD | 463 | +0.004 (-0.026 to +0.034) | 0.799 |

### XGBoost paired comparisons

| Policy | Disease area | Matched test N | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|
| no_lookback | CVD | 726 | +0.006 (-0.005 to +0.016) | 0.294 |
| no_lookback | Diabetes | 682 | -0.002 (-0.012 to +0.008) | 0.672 |
| no_lookback | OSA | 694 | -0.002 (-0.013 to +0.009) | 0.716 |
| no_lookback | MDD | 673 | +0.011 (-0.013 to +0.035) | 0.382 |
| no_lookback | Anxiety | 728 | +0.003 (-0.023 to +0.029) | 0.826 |
| no_lookback | Bipolar | 476 | +0.011 (-0.015 to +0.037) | 0.398 |
| no_lookback | ADHD | 464 | +0.003 (-0.039 to +0.046) | 0.871 |
| halved_rules | CVD | 651 | +0.004 (-0.008 to +0.016) | 0.489 |
| halved_rules | Diabetes | 652 | +0.003 (-0.008 to +0.013) | 0.637 |
| halved_rules | OSA | 646 | +0.000 (-0.011 to +0.011) | 0.960 |
| halved_rules | MDD | 605 | +0.018 (-0.005 to +0.040) | 0.133 |
| halved_rules | Anxiety | 631 | +0.017 (-0.010 to +0.043) | 0.226 |
| halved_rules | Bipolar | 450 | -0.014 (-0.046 to +0.018) | 0.385 |
| halved_rules | ADHD | 463 | -0.007 (-0.061 to +0.047) | 0.798 |

## 5. Future-diagnosis exploratory analysis

This appendix-only analysis is not the primary model. Future cases have their first mapped diagnosis after the end of their first QC-eligible seven-day streak; participants with any earlier mapped diagnosis for that condition are excluded. Controls use their first QC-eligible streak. It therefore answers a distinct, exploratory timing question.

### Logistic regression

| Disease area | Future diagnosis |
|---|---|
| CVD | 0.824 (0.794–0.852); 753/4,274 |
| Diabetes | 0.836 (0.807–0.864); 582/4,274 |
| OSA | 0.777 (0.747–0.807); 813/4,274 |
| MDD | 0.769 (0.733–0.803); 742/4,274 |
| Anxiety | 0.776 (0.751–0.800); 1,263/4,274 |
| Bipolar | 0.793 (0.724–0.858); 167/4,274 |
| ADHD | 0.789 (0.728–0.842); 241/4,274 |

### XGBoost

| Disease area | Future diagnosis |
|---|---|
| CVD | 0.833 (0.806–0.860); 753/4,274 |
| Diabetes | 0.825 (0.793–0.857); 582/4,274 |
| OSA | 0.789 (0.760–0.818); 813/4,274 |
| MDD | 0.768 (0.733–0.801); 742/4,274 |
| Anxiety | 0.789 (0.764–0.813); 1,263/4,274 |
| Bipolar | 0.789 (0.709–0.858); 167/4,274 |
| ADHD | 0.804 (0.746–0.855); 241/4,274 |

## 6. Missingness and complete-case assessment

**Status:** completed with the QC-filtered seven-day past-diagnosis cohort. Outputs are in `results/seven_day_past_preindex_qc_missingness/`; code is `src/seven_day_preindex_qc_missingness.py`.

**Scope:** wearable QC eligibility is applied before streak selection. The flow below then separates the loss from static eligibility and feature complete-case requirements. Missing Fitbit-derived data are not imputed: failure to provide a QC-eligible streak is structural non-observation rather than an ordinary partially missing covariate. The configured person-level demographic variables and all configured survey features have zero missing values in their respective source tables; therefore the incremental complete-case losses below arise after constructing the analytic feature frame, not from recorded missing values in those source columns.

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

The excluded groups are small (9–18 participants per disease-specific frame), so their descriptive means/proportions should not be overinterpreted.

| Disease area | Retention group | n | Cases (%) | Mean age | Mean BMI | Male (%) | Non-White (%) |
|---|---|---|---|---|---|---|---|
| CVD | Excluded after static/feature completeness | 18 | 55.6 | 62.4 | 30.4 | 50.0 | 33.3 |
| CVD | Retained complete case | 7802 | 45.2 | 57.6 | 29.1 | 36.1 | 25.7 |
| Diabetes | Excluded after static/feature completeness | 16 | 50.0 | 58.7 | 30.3 | 37.5 | 31.2 |
| Diabetes | Retained complete case | 7314 | 41.6 | 54.2 | 30.4 | 31.5 | 30.7 |
| OSA | Excluded after static/feature completeness | 16 | 50.0 | 60.1 | 32.5 | 37.5 | 25.0 |
| OSA | Retained complete case | 7609 | 43.8 | 54.6 | 30.6 | 33.8 | 27.7 |
| MDD | Excluded after static/feature completeness | 14 | 42.9 | 55.5 | 30.9 | 35.7 | 35.7 |
| MDD | Retained complete case | 7277 | 41.3 | 51.0 | 29.6 | 25.5 | 27.6 |
| Anxiety | Excluded after static/feature completeness | 14 | 42.9 | 52.7 | 30.0 | 21.4 | 28.6 |
| Anxiety | Retained complete case | 7835 | 45.4 | 50.3 | 29.3 | 24.7 | 26.5 |
| Bipolar | Excluded after static/feature completeness | 12 | 33.3 | 54.9 | 32.8 | 25.0 | 41.7 |
| Bipolar | Retained complete case | 5301 | 19.4 | 50.1 | 28.8 | 26.8 | 29.4 |
| ADHD | Excluded after static/feature completeness | 9 | 11.1 | 55.1 | 31.6 | 22.2 | 44.4 |
| ADHD | Retained complete case | 5052 | 15.4 | 49.2 | 28.3 | 27.0 | 29.5 |

### Variable missingness

- Person variables: `date_of_birth`, `race`, and `sex_at_birth` each had 0% missingness.
- All configured survey indicator variables had 0% missingness (`person_variable_missingness.csv` and `survey_variable_missingness.csv`).

**Interpretation:** distinguish selection due to availability of QC-eligible wearable summaries from the small incremental complete-case loss. Multiple imputation is not indicated for the configured demographic and survey source variables because their measured missingness is zero; it would not recover participants without QC-eligible wearable data.

## 7. Primary-model calibration metric and coefficient stability

Brier skill score (BSS) is reported only for the QC seven-day primary model, not for sensitivity analyses. It uses the held-out prevalence as the reference forecast and 1,000 percentile-bootstrap intervals.

| Disease area | Logistic BSS (95% CI) | XGBoost BSS (95% CI) |
|---|---|---|
| CVD | 0.662 (0.641–0.684) | 0.651 (0.628–0.675) |
| Diabetes | 0.608 (0.583–0.629) | 0.616 (0.591–0.637) |
| OSA | 0.627 (0.601–0.648) | 0.624 (0.598–0.646) |
| MDD | 0.524 (0.499–0.547) | 0.520 (0.493–0.543) |
| Anxiety | 0.530 (0.505–0.552) | 0.520 (0.495–0.543) |
| Bipolar | 0.322 (0.274–0.364) | 0.321 (0.273–0.363) |
| ADHD | 0.205 (0.152–0.250) | 0.204 (0.155–0.246) |

The deterministic training-partition logistic coefficients were refit for 7, 14, and 21 QC-eligible consecutive days and are stored in `results/preindex_streak_duration_qc_sensitivity/logreg_coefficients.csv`. The direction-concordance audit below summarizes all coefficients available under all three duration definitions; the final column is restricted to FDR-significant coefficients in all three models.

| Disease area | Features compared | Same direction, all | Direction concordance (%) | FDR-significant, all | Same direction among FDR-significant |
|---|---|---|---|---|---|
| CVD | 49 | 31 | 63.3 | 5 | 5 |
| Diabetes | 49 | 22 | 44.9 | 3 | 3 |
| OSA | 49 | 34 | 69.4 | 6 | 6 |
| MDD | 49 | 29 | 59.2 | 6 | 6 |
| Anxiety | 49 | 36 | 73.5 | 7 | 7 |
| Bipolar | 49 | 29 | 59.2 | 3 | 3 |
| ADHD | 49 | 31 | 63.3 | 4 | 4 |

Because the manuscript reports calibration curves, those curves must also be regenerated for the QC seven-day primary cohort before the manuscript is finalized; the BSS refresh alone is not a substitute for the plotted calibration assessment.

## 8. QC-matched demographic-only control and detailed-race sensitivity

**Design:** all three models use the exact same QC-filtered seven-day past-diagnosis cohort, complete-case participants, and seed-42 split. The demographic-only control includes age, sex at birth, binary race (non-White versus White), and BMI. The detailed-race model otherwise retains the full primary feature set but replaces `race_nonwhite` with Black or African American, Asian, more than one population, and other/small-or-unknown race indicators; White is the reference category. The latter residual category includes all remaining recorded values (for example, None Indicated, American Indian or Alaska Native, Middle Eastern or North African, Native Hawaiian or Other Pacific Islander, and prefer-not-to-answer/skip values) to avoid unstable sparse indicators.

Results and implementation: `src/qc_primary_control_race_sensitivity.py`, `results/qc_primary_control_race_sensitivity/`. BSS is included here for the primary and demographic-only models because calibration is a main-model quantity, not a general sensitivity-analysis metric.

### Logistic regression

| Disease area | Primary AUROC | Demographic-only AUROC | Primary BSS | Demographic-only BSS | Detailed-race AUROC |
|---|---|---|---|---|---|
| CVD | 0.849 (0.834–0.864) | 0.825 (0.808–0.841) | 0.662 (0.641–0.684) | 0.636 (0.612–0.658) | 0.850 (0.834–0.864) |
| Diabetes | 0.832 (0.815–0.847) | 0.802 (0.783–0.818) | 0.608 (0.583–0.629) | 0.576 (0.552–0.597) | 0.832 (0.815–0.846) |
| OSA | 0.832 (0.815–0.848) | 0.813 (0.796–0.829) | 0.627 (0.601–0.648) | 0.608 (0.584–0.628) | 0.832 (0.816–0.848) |
| MDD | 0.746 (0.724–0.765) | 0.671 (0.648–0.693) | 0.524 (0.499–0.547) | 0.468 (0.444–0.490) | 0.746 (0.725–0.765) |
| Anxiety | 0.703 (0.682–0.723) | 0.631 (0.607–0.652) | 0.530 (0.505–0.552) | 0.491 (0.470–0.510) | 0.705 (0.683–0.725) |
| Bipolar | 0.773 (0.745–0.801) | 0.705 (0.670–0.737) | 0.322 (0.274–0.364) | 0.254 (0.220–0.281) | 0.774 (0.746–0.803) |
| ADHD | 0.719 (0.684–0.753) | 0.665 (0.627–0.699) | 0.205 (0.152–0.250) | 0.184 (0.147–0.214) | 0.722 (0.688–0.757) |

### XGBoost

| Disease area | Primary AUROC | Demographic-only AUROC | Primary BSS | Demographic-only BSS | Detailed-race AUROC |
|---|---|---|---|---|---|
| CVD | 0.839 (0.823–0.855) | 0.825 (0.809–0.841) | 0.651 (0.628–0.675) | 0.636 (0.613–0.659) | 0.837 (0.821–0.852) |
| Diabetes | 0.838 (0.822–0.854) | 0.806 (0.789–0.822) | 0.616 (0.591–0.637) | 0.582 (0.557–0.603) | 0.836 (0.820–0.853) |
| OSA | 0.827 (0.811–0.844) | 0.811 (0.793–0.827) | 0.624 (0.598–0.646) | 0.607 (0.582–0.629) | 0.828 (0.812–0.845) |
| MDD | 0.738 (0.716–0.758) | 0.670 (0.648–0.694) | 0.520 (0.493–0.543) | 0.469 (0.445–0.491) | 0.743 (0.721–0.764) |
| Anxiety | 0.693 (0.672–0.713) | 0.630 (0.605–0.651) | 0.520 (0.495–0.543) | 0.491 (0.468–0.511) | 0.713 (0.693–0.733) |
| Bipolar | 0.782 (0.756–0.809) | 0.699 (0.664–0.732) | 0.321 (0.273–0.363) | 0.251 (0.216–0.282) | 0.783 (0.756–0.811) |
| ADHD | 0.719 (0.685–0.752) | 0.641 (0.601–0.678) | 0.204 (0.155–0.246) | 0.166 (0.124–0.199) | 0.722 (0.689–0.756) |

### Demographic-only versus primary: paired AUROC comparison

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2340 | 0.849 | 0.825 | -0.025 (-0.033 to -0.016) | **0.000** |
| Diabetes | 2194 | 0.832 | 0.802 | -0.030 (-0.041 to -0.020) | **0.000** |
| OSA | 2282 | 0.832 | 0.813 | -0.019 (-0.028 to -0.009) | **0.000** |
| MDD | 2183 | 0.746 | 0.671 | -0.075 (-0.094 to -0.055) | **0.000** |
| Anxiety | 2350 | 0.703 | 0.631 | -0.073 (-0.092 to -0.054) | **0.000** |
| Bipolar | 1590 | 0.773 | 0.705 | -0.068 (-0.100 to -0.036) | **0.000** |
| ADHD | 1515 | 0.719 | 0.665 | -0.053 (-0.084 to -0.022) | **0.001** |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2340 | 0.839 | 0.825 | -0.014 (-0.024 to -0.004) | **0.004** |
| Diabetes | 2194 | 0.838 | 0.806 | -0.032 (-0.043 to -0.021) | **0.000** |
| OSA | 2282 | 0.827 | 0.811 | -0.016 (-0.026 to -0.007) | **0.001** |
| MDD | 2183 | 0.738 | 0.670 | -0.068 (-0.088 to -0.047) | **0.000** |
| Anxiety | 2350 | 0.693 | 0.630 | -0.063 (-0.084 to -0.042) | **0.000** |
| Bipolar | 1590 | 0.782 | 0.699 | -0.083 (-0.113 to -0.053) | **0.000** |
| ADHD | 1515 | 0.719 | 0.641 | -0.077 (-0.111 to -0.044) | **0.000** |

### Detailed-race versus primary: paired AUROC comparison

#### Logistic regression

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2340 | 0.849 | 0.850 | +0.000 (-0.001 to +0.001) | 0.608 |
| Diabetes | 2194 | 0.832 | 0.832 | -0.000 (-0.001 to +0.001) | 0.774 |
| OSA | 2282 | 0.832 | 0.832 | +0.001 (-0.000 to +0.002) | 0.202 |
| MDD | 2183 | 0.746 | 0.746 | +0.000 (-0.002 to +0.003) | 0.834 |
| Anxiety | 2350 | 0.703 | 0.705 | +0.002 (-0.001 to +0.005) | 0.275 |
| Bipolar | 1590 | 0.773 | 0.774 | +0.002 (-0.003 to +0.007) | 0.481 |
| ADHD | 1515 | 0.719 | 0.722 | +0.004 (-0.002 to +0.010) | 0.246 |

#### XGBoost

| Disease area | Matched test N | Primary AUROC | Comparison AUROC | Difference (95% CI) | Raw paired P |
|---|---|---|---|---|---|
| CVD | 2340 | 0.839 | 0.837 | -0.003 (-0.005 to +0.000) | 0.051 |
| Diabetes | 2194 | 0.838 | 0.836 | -0.002 (-0.004 to +0.000) | 0.068 |
| OSA | 2282 | 0.827 | 0.828 | +0.001 (-0.002 to +0.004) | 0.512 |
| MDD | 2183 | 0.738 | 0.743 | +0.005 (-0.000 to +0.010) | 0.067 |
| Anxiety | 2350 | 0.693 | 0.713 | +0.021 (+0.010 to +0.031) | **0.000** |
| Bipolar | 1590 | 0.782 | 0.783 | +0.001 (-0.007 to +0.010) | 0.800 |
| ADHD | 1515 | 0.719 | 0.722 | +0.004 (-0.013 to +0.020) | 0.666 |

**Interpretation:** The demographic-only control is now directly comparable to the QC primary model because it is trained and tested on the same participants. The detailed-race specification is a coding sensitivity: it tests whether collapsing race into a single non-White indicator materially changes discrimination. It does not estimate race-specific performance, and the heterogeneous residual category should not be interpreted as a coherent population.

## Interpretation

The QC filter modestly changes the eligible cohort but retains sufficient sample sizes for all seven disease areas. The primary inference should remain an association with recorded, potentially treated prevalent disease among participants meeting this summary-level QC criterion; it is neither incident-disease prediction nor a causal medication analysis. Future-diagnosis results are exploratory and should be relegated to the appendix.
