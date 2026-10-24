# Reviewer sensitivity-analysis results

## Scope

This memo documents new analyses requested in reviewer feedback. It was prepared before any manuscript edits; the LaTeX manuscript repository was not modified.

## Reproducibility and recovered primary inputs

- Historical primary cohort configuration: `study_cohorts_20260225`, with medication and OSA-device exclusions disabled. This is the user-specified non-`clean_control` analysis.
- Recovered primary input cohort: 9,301 participants after the core demographic completeness step.
- Modelling frame: 9,240 participants after applying the configured feature-complete-case filter. This differs from the manuscript statement that 9,301 participants were used for modelling and should be corrected or reconciled before resubmission.
- Recovered raw cohort counts were close to, but not identical to, the manuscript counts (for example: control 2,841 vs 2,830; CVD 2,306 vs 2,289). This likely reflects an unavailable historical intermediate-data version; it should be disclosed/reconciled rather than silently treated as exact reproduction.
- The static-model pipeline now sets XGBoost `random_state=42` in both direct and cross-validated model fits. Its participant split was already seeded at 42. The reviewer sensitivity script uses seed 42 for the split, XGBoost, and bootstrap resampling.

The fixed-seed primary rerun is in `results/base_mdl_20260225_seed42/`; the reviewer sensitivity artifacts are in `results/reviewer_sensitivity/`.

## Primary-model reproduction

The fixed-seed rerun reproduced the manuscript's reported AUROC point estimates and Fitbit-associated gains. Values below are AUROC for baseline variables / baseline plus Fitbit summary statistics.

| Outcome | Logistic regression | XGBoost | Fitbit AUROC gain (LR / XGB) |
|---|---:|---:|---:|
| CVD | 0.816 / 0.831 | 0.810 / 0.823 | 0.015 / 0.013 |
| Diabetes | 0.789 / 0.811 | 0.786 / 0.812 | 0.022 / 0.026 |
| OSA | 0.806 / 0.825 | 0.804 / 0.828 | 0.019 / 0.024 |
| MDD | 0.663 / 0.740 | 0.669 / 0.741 | 0.077 / 0.071 |
| Anxiety | 0.651 / 0.728 | 0.606 / 0.715 | 0.077 / 0.108 |
| Bipolar | 0.694 / 0.747 | 0.697 / 0.737 | 0.052 / 0.039 |
| ADHD | 0.708 / 0.747 | 0.715 / 0.729 | 0.040 / 0.012 |

Exact primary confidence intervals and Brier skill scores are retained in `results/base_mdl_20260225_seed42/static_mdl_perf.csv` and `core_static_mdl_perf.csv`.

## Multimorbidity: descriptive results

The target-condition count among the 9,240 complete-case participants was:

| Number of target conditions | Participants |
|---:|---:|
| 0 | 2,817 |
| 1 | 3,315 |
| 2 | 1,795 |
| 3 | 860 |
| 4 | 314 |
| 5 | 118 |
| 6 | 18 |
| 7 | 3 |

Thus, 3,315 of 6,423 cases (51.6%) had exactly one target condition and 3,108 (48.4%) had multimorbidity. The largest pairwise overlaps were MDD–anxiety (1,076), CVD–OSA (753), OSA–anxiety (643), OSA–MDD (637), and diabetes–OSA (626). The full matrix is in `results/reviewer_sensitivity/condition_overlap_matrix.csv`.

## Sensitivity analyses

All analyses use the same recovered complete-case feature set, fixed 70/30 participant split, 5-fold XGBoost hyperparameter selection, and 1,000 percentile-bootstrap confidence intervals.

1. **Single-condition analysis:** controls plus cases with exactly one target condition. This isolates the impact of target-condition multimorbidity; it does not exclude medication exposure.
2. **Clean-control analysis:** all cases plus controls with no record of a listed heart-rate- or sleep-altering medication at any time. This is the requested actual clean-control sensitivity; it is distinct from filtering all monitoring windows around medication exposure.
3. **Pre-medication case analysis:** clean controls plus cases whose entire 21-day Fitbit window (`index_date` through `index_date + 20 days`) ended before their first listed medication exposure, or who had no recorded exposure. This is a temporally defined sensitivity analysis, not a causal medication-effect analysis.

| Analysis | Participants | Controls | Multimorbid cases |
|---|---:|---:|---:|
| Single condition | 6,132 | 2,817 | 0 |
| Clean control | 8,114 | 1,691 | 3,108 |
| Pre-medication cases + clean control | 3,442 | 1,691 | 625 |

### AUROC comparison tables (95% bootstrap CI)

The primary-model column is the full baseline-plus-Fitbit model. Each sample size is shown as **cases/controls** and is listed separately for the primary and each sensitivity analysis within the same row.

#### Logistic regression

| Disease area | Sample size (case/control) | Primary model | Single condition | Clean control | Pre-medication |
|---|---|---:|---:|---:|---:|
| CVD | Primary: 2,292/2,817<br>Single: 926/2,817<br>Clean: 2,292/1,691<br>Pre-med: 531/1,691 | 0.831 (0.811 - 0.849) | 0.824 (0.798–0.851) | 0.845 (0.823–0.866) | 0.820 (0.785–0.855) |
| Diabetes | Primary: 1,781/2,817<br>Single: 637/2,817<br>Clean: 1,781/1,691<br>Pre-med: 479/1,691 | 0.811 (0.787 - 0.833) | 0.756 (0.722–0.789) | 0.830 (0.803–0.853) | 0.761 (0.720–0.801) |
| OSA | Primary: 2,125/2,817<br>Single: 570/2,817<br>Clean: 2,125/1,691<br>Pre-med: 493/1,691 | 0.825 (0.803 - 0.845) | 0.757 (0.719–0.793) | 0.835 (0.812–0.857) | 0.803 (0.764–0.838) |
| MDD | Primary: 1,868/2,817<br>Single: 306/2,817<br>Clean: 1,868/1,691<br>Pre-med: 373/1,691 | 0.740 (0.713 - 0.768) | 0.589 (0.530–0.649) | 0.746 (0.715–0.774) | 0.667 (0.613–0.719) |
| Anxiety | Primary: 2,364/2,817<br>Single: 674/2,817<br>Clean: 2,364/1,691<br>Pre-med: 594/1,691 | 0.728 (0.703 - 0.752) | 0.645 (0.606–0.683) | 0.727 (0.700–0.755) | 0.667 (0.619–0.711) |
| Bipolar | Primary: 559/2,817<br>Single: 108/2,817<br>Clean: 559/1,691<br>Pre-med: 84/1,691 | 0.747 (0.705 - 0.788) | 0.709 (0.590–0.802) | 0.753 (0.712–0.796) | 0.567 (0.459–0.674) |
| ADHD | Primary: 471/2,817<br>Single: 94/2,817<br>Clean: 471/1,691<br>Pre-med: 90/1,691 | 0.747 (0.706 - 0.788) | 0.607 (0.489–0.720) | 0.751 (0.709–0.791) | 0.712 (0.603–0.811) |

#### XGBoost

| Disease area | Sample size (case/control) | Primary model | Single condition | Clean control | Pre-medication |
|---|---|---:|---:|---:|---:|
| CVD | Primary: 2,292/2,817<br>Single: 926/2,817<br>Clean: 2,292/1,691<br>Pre-med: 531/1,691 | 0.823 (0.802 - 0.843) | 0.826 (0.800–0.852) | 0.840 (0.819–0.862) | 0.806 (0.772–0.842) |
| Diabetes | Primary: 1,781/2,817<br>Single: 637/2,817<br>Clean: 1,781/1,691<br>Pre-med: 479/1,691 | 0.812 (0.789 - 0.834) | 0.753 (0.719–0.787) | 0.805 (0.779–0.830) | 0.776 (0.732–0.815) |
| OSA | Primary: 2,125/2,817<br>Single: 570/2,817<br>Clean: 2,125/1,691<br>Pre-med: 493/1,691 | 0.828 (0.807 - 0.848) | 0.763 (0.724–0.799) | 0.841 (0.819–0.863) | 0.776 (0.731–0.814) |
| MDD | Primary: 1,868/2,817<br>Single: 306/2,817<br>Clean: 1,868/1,691<br>Pre-med: 373/1,691 | 0.741 (0.715 - 0.766) | 0.594 (0.534–0.653) | 0.736 (0.702–0.763) | 0.637 (0.579–0.693) |
| Anxiety | Primary: 2,364/2,817<br>Single: 674/2,817<br>Clean: 2,364/1,691<br>Pre-med: 594/1,691 | 0.715 (0.690 - 0.740) | 0.631 (0.591–0.669) | 0.713 (0.685–0.741) | 0.674 (0.625–0.722) |
| Bipolar | Primary: 559/2,817<br>Single: 108/2,817<br>Clean: 559/1,691<br>Pre-med: 84/1,691 | 0.737 (0.697 - 0.776) | 0.627 (0.495–0.737) | 0.753 (0.712–0.797) | 0.578 (0.460–0.692) |
| ADHD | Primary: 471/2,817<br>Single: 94/2,817<br>Clean: 471/1,691<br>Pre-med: 90/1,691 | 0.729 (0.685 - 0.770) | 0.581 (0.485–0.679) | 0.733 (0.687–0.776) | 0.733 (0.632–0.822) |

Full AUROC and Brier skill score estimates/CIs are in `results/reviewer_sensitivity/all_sensitivity_performance.csv`.

## Statistical comparison of AUROC

For each disease area and sensitivity definition, we refit the primary and sensitivity models using the pre-specified fixed split. We then evaluated **both** fitted models on the same held-out sensitivity-cohort participants and used a **two-sided paired DeLong test for correlated AUROCs**. The reported difference is sensitivity-trained AUROC minus primary-trained AUROC on that matched test set. Thus, this is a valid paired comparison of models in the restricted target population; it is not a direct test of the displayed overall primary versus sensitivity AUROCs, which were calculated on different (though overlapping) test cohorts.

The table records raw two-sided DeLong p-values and Benjamini–Hochberg false-discovery-rate (FDR) adjusted p-values across all 42 comparisons. No comparison met FDR-adjusted p<0.05. Seven met unadjusted p<0.05; these are exploratory in the context of the multiple comparisons.

Positive differences favor the sensitivity-trained model; negative differences favor the primary-trained model. `n test` is shown as cases/controls.

### Single-condition

| Disease area | n test (case/control) | LogReg difference (95% CI) | Raw p | FDR p | XGBoost difference (95% CI) | Raw p | FDR p |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVD | 272/860 | +0.004 (-0.007 to +0.015) | 0.510 | 0.857 | +0.009 (-0.004 to +0.023) | 0.174 | 0.544 |
| Diabetes | 202/860 | -0.001 (-0.014 to +0.012) | 0.879 | 0.947 | -0.012 (-0.035 to +0.010) | 0.288 | 0.711 |
| OSA | 152/860 | -0.006 (-0.025 to +0.012) | 0.510 | 0.857 | -0.015 (-0.036 to +0.005) | 0.149 | 0.544 |
| MDD | 89/860 | -0.061 (-0.106 to -0.017) | 0.007 | 0.139 | -0.064 (-0.120 to -0.009) | 0.023 | 0.192 |
| Anxiety | 215/860 | -0.007 (-0.037 to +0.024) | 0.672 | 0.889 | +0.003 (-0.036 to +0.043) | 0.868 | 0.947 |
| Bipolar | 28/860 | -0.032 (-0.122 to +0.058) | 0.490 | 0.857 | -0.112 (-0.218 to -0.006) | 0.038 | 0.263 |
| ADHD | 31/860 | -0.054 (-0.172 to +0.064) | 0.369 | 0.816 | -0.048 (-0.157 to +0.062) | 0.392 | 0.823 |

### Clean-control

| Disease area | n test (case/control) | LogReg difference (95% CI) | Raw p | FDR p | XGBoost difference (95% CI) | Raw p | FDR p |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVD | 670/526 | -0.002 (-0.007 to +0.003) | 0.470 | 0.857 | +0.001 (-0.008 to +0.011) | 0.783 | 0.889 |
| Diabetes | 560/526 | +0.001 (-0.005 to +0.007) | 0.765 | 0.889 | -0.022 (-0.035 to -0.008) | 0.002 | 0.089 |
| OSA | 601/526 | -0.002 (-0.008 to +0.003) | 0.435 | 0.857 | -0.000 (-0.010 to +0.010) | 0.981 | 0.981 |
| MDD | 552/526 | -0.001 (-0.011 to +0.010) | 0.909 | 0.955 | -0.011 (-0.027 to +0.005) | 0.181 | 0.544 |
| Anxiety | 723/526 | -0.001 (-0.012 to +0.009) | 0.778 | 0.889 | -0.000 (-0.016 to +0.015) | 0.973 | 0.981 |
| Bipolar | 162/526 | -0.003 (-0.019 to +0.014) | 0.737 | 0.889 | +0.006 (-0.016 to +0.028) | 0.605 | 0.889 |
| ADHD | 156/526 | +0.003 (-0.016 to +0.022) | 0.732 | 0.889 | +0.005 (-0.022 to +0.032) | 0.714 | 0.889 |

### Pre-medication

| Disease area | n test (case/control) | LogReg difference (95% CI) | Raw p | FDR p | XGBoost difference (95% CI) | Raw p | FDR p |
|---|---:|---:|---:|---:|---:|---:|---:|
| CVD | 161/526 | -0.003 (-0.019 to +0.013) | 0.706 | 0.889 | -0.008 (-0.026 to +0.009) | 0.349 | 0.815 |
| Diabetes | 147/526 | -0.020 (-0.044 to +0.004) | 0.101 | 0.472 | +0.008 (-0.017 to +0.032) | 0.548 | 0.885 |
| OSA | 126/526 | -0.004 (-0.020 to +0.012) | 0.623 | 0.889 | -0.035 (-0.063 to -0.007) | 0.014 | 0.192 |
| MDD | 107/526 | -0.027 (-0.068 to +0.013) | 0.181 | 0.544 | -0.056 (-0.112 to -0.001) | 0.048 | 0.288 |
| Anxiety | 171/526 | +0.005 (-0.027 to +0.038) | 0.743 | 0.889 | +0.031 (-0.004 to +0.066) | 0.086 | 0.449 |
| Bipolar | 30/526 | -0.089 (-0.203 to +0.026) | 0.128 | 0.537 | -0.057 (-0.152 to +0.038) | 0.237 | 0.622 |
| ADHD | 25/526 | -0.101 (-0.188 to -0.014) | 0.022 | 0.192 | -0.064 (-0.162 to +0.035) | 0.204 | 0.571 |

The machine-readable results are in `results/reviewer_sensitivity/matched_test_delong_comparisons.csv`; the exact test specification is in `results/reviewer_sensitivity/matched_test_delong_metadata.json`.
## Interpretation for response and manuscript revision

- The primary disease-versus-control cohorts are substantially multimorbid. Single-condition AUROC point estimates are lower for several outcomes, especially MDD, anxiety, and ADHD, but the matched-test DeLong comparisons produced no FDR-significant differences across the 42 tests. The main text should avoid implying that the original models identify disease-specific signatures independent of comorbidity.
- Clean controls improve apparent discrimination across outcomes. This is expected because they are a more restricted comparator group; it should be presented as a sensitivity result, not as the primary estimate or evidence that medication caused the difference.
- The temporal pre-medication analysis has smaller samples, includes only 625 multimorbid cases, and has wider intervals. It supports that discrimination remains above chance for several outcomes but should not be used to make causal claims about treatment effects.
- Six of seven logistic fits in the pre-medication analysis and two in the single-condition analysis required an explicitly labelled, scaled weakly regularized fallback because the manuscript-style unregularized fit did not converge. All clean-control logistic fits converged unregularized. These sparse/separated analyses should be described as sensitivity analyses with this qualification.

## Files and rerun commands

- Analysis code: `src/reviewer_sensitivity_analyses.py`
- Re-run reviewer analyses: `python src/reviewer_sensitivity_analyses.py`
- Re-run fixed-seed primary static models: `python src/base_mdl_fitbit.py`
- Reviewer results: `results/reviewer_sensitivity/`
- Seeded primary results: `results/base_mdl_20260225_seed42/`

No manuscript files were changed.
