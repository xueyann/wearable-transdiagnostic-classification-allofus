# Remaining Revision Items

This document tracks only work that is unresolved before resubmission. Every unresolved item is labelled `[todo]`. Completed results are documented in `reviewer_sensitivity_analysis_7day_past_qc.md` and the manuscript revision changelog.

## Resolved decisions

### [resolved] Consecutive 7-day observation window

Retain seven consecutive QC-eligible days for the primary analysis. Nonconsecutive-day aggregation will not be added because RMSSD and related short-term heart-rate-variability measures require temporally contiguous observations; pooling days separated in time would weaken their physiological interpretability. The 7-day definition retained substantially more participants and showed comparable discrimination to 14- and 21-day windows.

### [resolved] Calibration estimand

Calibration curves and Brier skill scores use held-out test-set predictions only. No separate train-versus-test calibration analysis is required. The remaining task is to regenerate the QC-filtered calibration figure and display its test-set denominators.

### [resolved] Sensitivity-comparison multiplicity framing

The disease-specific paired sensitivity comparisons are reported with raw P values and 95% CIs. No across-disease omnibus multiplicity correction is used because the study’s inferential targets are prespecified disease-specific analyses rather than a pooled disease claim. Benjamini--Hochberg FDR correction is retained for exploratory coefficient-level inference; the stored monitoring-duration comparison additionally used Benjamini--Hochberg adjustment for its prespecified duration comparisons.

### [resolved] Ordinary covariate missingness and imputation

Date of birth, race, sex assigned at birth, and all configured survey indicators had zero measured missingness in their source tables. Complete-case loss after static eligibility was only 9--18 participants per disease-specific analytic frame. Missing QC-eligible wearable runs are structural non-observation, not values amenable to multiple imputation. An MCAR test and multiple-imputation sensitivity analysis are therefore not indicated for the current measured ordinary covariates.

## [resolved] Race and ethnicity specification

- The final primary and sensitivity models use detailed race adjustment: White reference plus Black or African American, Asian, more than one population, and other/small/unknown indicators.
- This specification retains separately estimable groups while combining sparse individual categories and unknown/no-response records.
- The complete detailed-race rerun is incorporated in the main performance table, Multimedia Appendix tables, and manuscript Methods and limitations.
- The reviewer response needs only final proofreading against the updated manuscript language.

## [todo] Diagnosis-recency support

- Add a table or figure showing time from the most recent qualifying diagnosis to the Fitbit index date in clinically interpretable bins.
- Pair this descriptive evidence with the completed no-lookback/current/halved-lookback analysis and disease-course literature.

## [todo] Training/test and data-source balance

- Add a supplementary table comparing training and held-out test participants, including demographics, disease prevalence, and BYOD/WEAR source where available.
- State that this supports internal split comparability only and does not establish external transportability.

## [todo] Derived-ratio edge case

- Verify and document handling of zero or missing denominators in the activity--heart-rate ratio.
- Rerun only if an arbitrary replacement value or unhandled infinite value was used.

## [todo] Fitbit summary-data QC presentation

- Present the distribution and extreme percentiles of retained summary measures plus participant/day attrition attributable to the summary-data QC proxy.
- Continue to call the $\geq$600 heart-rate-zone-minutes plus $\geq$100-steps rule a summary-data QC proxy, not validated wear time, because total heart-rate-zone minutes can exceed 1,440 minutes.

## [todo] Final results assets and manuscript completion

- Regenerate the QC-filtered held-out AUROC and calibration figures; label calibration panels with test-set case/control denominators.
- Add the QC-filtered Fitbit-versus-baseline incremental-performance results to the main table and AUROC plot once the final paired output is available.
- Insert the participant-flow graphic, co-occurrence heat map, sensitivity-summary table, and final coefficient appendix.
- Complete the final reference, language, table/figure, JMIR-style, and highlighted-manuscript audits.
