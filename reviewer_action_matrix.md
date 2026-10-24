# Reviewer action matrix

## Purpose and revision principles

This document separates (1) analyses that have already been completed, (2) analyses
that are needed to address a material threat to validity, (3) optional robustness
analyses, and (4) manuscript or response-letter changes. It is a working plan, not a
response letter.

The manuscript should be reframed as an internal, cross-sectional classification
study of recorded disease-versus-control phenotypes across several domains. It does
not establish disease onset, untreated disease biology, causal effects of wearable
features, or discrimination between the seven diseases.

### Priority calls

| Priority | Action | Why it matters |
|---|---|---|
| Essential | Correct the diagnosis-to-Fitbit temporal definition, then regenerate the primary cohort and results. | The current recency condition accepts a diagnosis recorded after Fitbit monitoring because a negative interval passes the `<= threshold` test. This is a cohort-definition error, not merely a sensitivity analysis. |
| Essential | Report diagnosis-to-index timing before filtering and perform the planned alternative-recency analyses. | Directly answers BN and DD, and makes the label-time assumption transparent. |
| Required | Report missingness and assess the complete-case decision; conduct multiple-imputation sensitivity analysis if missingness is material. | Explicit editorial request. |
| Recommended | Test the 21-day consecutive-window rule with shorter and nonconsecutive windows. | Directly requested by DD and currently lacks empirical justification. |
| Recommended | Produce a train/test table including BYOD versus WEAR status. | Directly requested by DL; this is descriptive, low burden, and should not be represented as external validation. |
| Recommended | Run a stricter EHR-phenotype sensitivity analysis requiring at least two qualifying diagnosis dates/codes. | Directly responds to DD's concern that a single ICD code can misclassify a case. It is more informative than adding medications/labs to the primary phenotype. |
| Decision needed | Decide whether to replicate in the newer CDR version (currently v9) or retain a frozen v8 analysis. | A newer release now exists. A v9 rerun is strongest if feasible; otherwise give a transparent data-freeze rationale and do not imply that v8 is current. |

### Important temporal-cohort correction

For each outcome, diagnoses recorded after the Fitbit index date must be excluded
*before* selecting the most recent qualifying diagnosis. Eligibility should be:

1. at least one qualifying diagnosis on or before the Fitbit index date;
2. recency calculated from the latest such diagnosis; and
3. the selected diagnostic date and time interval retained in the analytic dataset.

The control definition also needs the same temporal reference: a participant should
not be removed from the control cohort solely because of a target diagnosis first
recorded after the Fitbit window. This correction means that the existing primary and
sensitivity results should be treated as provisional until the cohort is rebuilt.

### Completed analyses already available

- Primary-model fixed-seed reproduction.
- Disease-overlap matrix and target-condition count distribution.
- Single-target-condition sensitivity analysis.
- Clean-control sensitivity analysis.
- Pre-medication case sensitivity analysis.

These should be repeated after the temporal-cohort correction if the correction
changes any analytic cohort materially.

## Editorial and content requirements

| Source | Level | Comment / issue | Proposed solution | Action type | Status |
|---|---|---|---|---|---|
| Editor | Major | Follow JARS; use its headings and participant-flow presentation. | Reorganize Methods with the requested reporting headings; replace or supplement attrition table with a JARS-style flow diagram. | Manuscript / figure | To do |
| Editor | Major | Title does not follow JMIR format. | Retitle in the form “Wearable-derived physiological features for disease-versus-control classification across physical and mental health conditions: cross-sectional study.” Refine after final framing. | Manuscript | To do |
| Editor | Major | Abstract is not sufficiently complete. | Expand structured abstract: data source/version, design, eligibility, final analytic N, conditions, models, split, key point estimates with 95% CIs, and calibrated conclusion. | Manuscript | To do |
| Editor | Major | Background, rationale, objective, and recent literature need strengthening. | Add current wearable/EHR and multimorbidity literature; end Introduction with a single, constrained objective. | Manuscript / literature review | To do |
| Editor | Major | Methods need design, setting, sampling, study size, measures/covariates, data sources, and analysis detail. | Add explicit retrospective cross-sectional design; All of Us recruitment and voluntary EHR/wearable sharing; nonprobability/volunteer sample; cohort derivation; no formal power calculation; and all model procedures. | Manuscript | To do |
| Editor | Major | Ethics subsection must address approval, consent, privacy, compensation, and identifiability. | Expand existing subsection to address all five items precisely; confirm the compensation statement from All of Us documentation rather than infer it. | Manuscript | To do |
| Editor | Major | Confidence intervals, statistics reporting, missingness/MCAR, and potential imputation are requested. | Report 95% CIs for all central estimates; add a missing-data table by variable and analytic stage; conduct/describe an appropriate MCAR assessment; if missingness is nontrivial or MCAR is rejected, add a multiple-imputation sensitivity analysis. | New analysis + manuscript | To do |
| Editor | Major | Discussion needs references, structure, calibrated conclusions, and no unreported results. | Rebuild Discussion: main findings, interpretation against literature, limitations, conclusion; move any numerical claims needed for interpretation to Results. | Manuscript | To do |
| Editor | Required | Funding and generative-AI disclosure. | Funding is present; verify JMIR wording. Add a truthful AI-use disclosure in the manuscript or explicitly attest to no use in the response letter, as applicable. | Manuscript / submission | To do |

## Reviewer AV

| Level | Comment | Proposed solution | Action type | Status |
|---|---|---|---|---|
| Major | Participants may belong to multiple disease cohorts; implications for condition-specific interpretation and controls are unclear. | Include the completed condition-overlap matrix and distribution of 0, 1, and 2+ target conditions. Report the completed single-condition sensitivity analysis. Reframe coefficient and performance claims as disease-versus-control associations that may reflect multimorbidity. Define controls temporally when rebuilding the cohort. | Completed analysis + manuscript + temporal cohort correction | Partially complete |
| Minor | Typographic/terminology inconsistencies. | Full copyedit; use “AUROC” consistently and standardize “logistic regression” and “XGBoost.” | Manuscript | To do |

## Reviewer BN

| Level | Comment | Proposed solution | Action type | Status |
|---|---|---|---|---|
| Major | Quantify disease overlap; otherwise patterns may reflect multimorbidity burden. | Same as AV: descriptive overlap matrix, single-condition sensitivity, and cautious interpretation. Do not claim condition-specific signatures independent of comorbidity. | Completed analysis + manuscript | Partially complete |
| Major | Disease-recency windows appear arbitrary; assess sensitivity. | Conduct the temporal correction first. Then report the pre-filter distribution in common bins and run no-upper-limit, uniform 5-year, and—if sample sizes permit—uniform 2-year sensitivity analyses. Do not call a threshold “data based” solely because many diagnoses fall within it. | New analysis | To do |
| Major | Controls may include undiagnosed/subclinical or other long-standing disease, particularly psychiatric disease. | State that EHR-defined controls cannot exclude undiagnosed/subclinical conditions. Present clean-control sensitivity as comparator restriction, not confirmation of true health. The temporal-control correction is also needed. | Completed analysis + manuscript + temporal cohort correction | Partially complete |
| Minor | Typos and lengthy discussion. | Copyedit and condense after restructuring Discussion. | Manuscript | To do |
| Minor | Add subgroup sample sizes to calibration panels. | Add case/control N for each panel/model or include them in a clear legend. | Figure revision | To do |

## Reviewer DD

| Level | Comment | Proposed solution | Action type | Status |
|---|---|---|---|---|
| Major | The manuscript does not adequately address multimorbidity. | Same as AV/BN; no separate multiclass model is necessary if the manuscript is reframed and single-condition sensitivity is reported. | Completed analysis + manuscript | Partially complete |
| Major | A newer All of Us release is available. | If feasible, use CDR v9 as a clearly labelled replication/full rerun. If not feasible, state that v8 was the prespecified frozen source, give its cutoff/version, and acknowledge that v9 may alter sample size and estimates. Do not describe v8 as the current release. | Decision: new analysis or manuscript defense | Decision needed |
| Major | Valid Fitbit days need better quality handling. | First document the fields available in v8 daily summaries and Fitbit's own quality processing. If wear time or an interpretable completeness field is available, add a prespecified quality sensitivity (for example, sufficient intraday observations and exclusion of implausible values). If not available, explicitly state this limitation; do not invent a wear-time cutoff from daily summaries. | Conditional new sensitivity + manuscript | To do |
| Major | One ICD code may be insufficient; consider two codes or richer phenotyping. | Add a sensitivity requiring >=2 qualifying diagnosis dates/codes before the Fitbit index. Explain why medication/laboratory-defined phenotypes are not adopted as the primary definition: they would change the estimand and can themselves encode treatment. | New sensitivity analysis | To do |
| Major | The HR-activity interaction’s exploratory basis is unclear. | Define the interaction exactly, its denominator-zero handling, and the EDA criterion that motivated it. If no pre-specified or reproducible EDA evidence exists, call it an exploratory engineered feature and avoid etiologic interpretation; a supplemental distribution/correlation plot is sufficient. | Manuscript; optional descriptive figure | To do |
| Major | Sex-at-birth restriction and binary non-White category are inadequately justified. | Give counts for excluded/unknown sex-at-birth categories and explain any small-cell or data-availability constraint. Replace “non-White” with disaggregated categories where sample sizes and program policy permit; otherwise state this aggregation is a limitation. No performance subgroup analysis is required unless numbers support it. | Manuscript / descriptive table | To do |
| Major | Potential collinearity in lifestyle variables is unclear. | State which lifestyle variables entered each model. Calculate and report VIF/correlation diagnostics for the complete final predictor set, not only wearable features; if problematic, revise the feature set or state the limited interpretive implications. | Analytic diagnostic + manuscript | To do |
| Major | Compact-feature selection and Table 3 are unclear. | Provide a reproducible selection workflow: candidate features, correlation/VIF rule, decisions based on domain relevance, final features, and removed features. Redesign Table 3 or move its full detail to supplement. | Manuscript / table revision | To do |
| Major | Why logistic regression and XGBoost? | Explain complementary roles: transparent linear benchmark and nonlinear tree model; models were selected for interpretation and comparison, not an exhaustive algorithm competition. Cite methodological sources. No additional algorithms are required. | Manuscript defense | To do |
| Major | Clean cohort has no evident utility in Results. | Report the completed clean-control model sensitivity results in Results and supplement; distinguish it from the pre-medication case analysis. | Completed analysis + manuscript | To do |
| Major | Fitbit timing versus diagnoses is unclear. | Correct future-diagnosis inclusion; document index date, first/latest prior diagnosis, diagnosis-to-index distribution, and alternative recency analyses. | Essential temporal cohort correction + new analysis | To do |
| Major | Justify 21 consecutive days; consider nonconsecutive days. | Compare 7-, 14-, and 21-day consecutive windows, reporting N and core performance. Add one fixed-count nonconsecutive-window definition in a bounded look-back period (for example, 21 valid days within 28 or 30 days), if data extraction allows. Do not choose the final definition by performance alone; balance sample retention and estimate stability. | New sensitivity analysis | To do |
| Major | Main tables/results need more information. | Move a compact table of per-outcome cases/controls, full-model AUROC/BSS with 95% CIs, and Fitbit incremental AUROC/BSS to the main manuscript. Keep secondary metrics and all coefficients in the supplement. | Manuscript / table revision | To do |
| Major | Figure 1 should display AUROC. | Make AUROC the prominent panel; retain BSS as calibration summary or move it to a supplementary figure/table if space is limited. | Figure revision | To do |
| Major | Discussion contains results not shown in Results. | Move or remove each such claim before revising interpretation. | Manuscript | To do |
| Minor | Need a flow diagram. | JARS-style participant flow diagram; retain a compact attrition table only if useful. | Figure revision | To do |
| Minor | WEAR introduced too late. | Define WEAR at first mention. | Manuscript | To do |
| Minor | Table 1 may be text; needs references. | Convert basic cohort/condition rationale to prose or retain only if it conveys necessary information; cite the clinical rationale. | Manuscript / table revision | To do |
| Minor | Explain FDR rather than Bonferroni. | State that FDR controls the expected proportion of false discoveries across correlated exploratory coefficient tests; Bonferroni would target family-wise error and be overly conservative for this purpose. Clearly distinguish this from model-performance comparisons. | Manuscript defense | To do |
| Minor | Undefined acronyms, delayed metric definitions, undefined table asterisks. | Copyedit; define at first use; explain * in every relevant legend. | Manuscript | To do |

## Reviewer DL

| Level | Comment | Proposed solution | Action type | Status |
|---|---|---|---|---|
| Major | Medication and post-diagnosis intervention may drive wearable signals; models classify managed disease states, not raw disease biology. | Make this a central limitation and reframe objective/conclusion accordingly. Report completed pre-medication case and clean-control sensitivities with their smaller samples/wider CIs; do not infer medication causality. The temporal cohort correction is prerequisite. | Completed analysis + manuscript + temporal cohort correction | Partially complete |
| Major | Binary shared-control models do not prove transdiagnostic or disease-specific signatures. | Accept the point and reframe throughout as disease-versus-control evaluation across multiple domains. Do not claim disease-versus-disease discrimination or condition-specific signatures. A multiclass model is not recommended: substantial comorbidity makes mutually exclusive labels artificial and would answer a different question. | Manuscript reframing | To do |
| Major | Internal train/test split does not establish transportability; show split balance and BYOD/WEAR composition. | Add a supplementary train/test table with N, age, sex, race/ethnicity categories, BMI, key lifestyle variables, outcome prevalence, and BYOD/WEAR status. State that balance supports validity of the internal split only, not external generalizability. | New descriptive analysis + manuscript | To do |
| Minor | Ratio of fat-burn minutes to lightly active minutes may divide by zero. | Verify the preprocessing implementation and state precisely whether zero denominators were set missing, excluded, replaced, or otherwise handled. If not handled, correct the feature and rerun all models using it. | Method verification; conditional rerun | To do |
| Minor | “Unknown” lifestyle categories need modeling treatment. | State that they were retained as explicit categorical levels and encoded with the other category indicators; report their prevalence. If any were imputed/dropped instead, say so exactly. | Manuscript | To do |

## Work that is not currently justified as a new analysis

| Proposed work | Reason not to prioritize |
|---|---|
| Multiclass disease-versus-disease model | With 48.4% multimorbidity in the recovered analytic frame, mutually exclusive labels would be artificial. Reframing is more coherent and directly addresses DL. |
| Medication/laboratory-enriched primary phenotype | This changes the scientific question and could increase treatment-selection bias. A two-code sensitivity is a cleaner robustness check. |
| Extensive covariate comparisons across every diagnosis-recency bin | These do not validate clinical activity and would create many underpowered comparisons. Use timing distributions and model-level threshold sensitivities instead. |
| Additional machine-learning algorithms | The reviewer asked for justification, not an algorithm tournament. Logistic regression and XGBoost provide a reasonable interpretable linear/nonlinear comparison. |

## Suggested execution order

1. Resolve the temporal case/control definition and check the zero-denominator feature handling.
2. Rebuild the cohort and repeat the completed sensitivity analyses if needed.
3. Run diagnosis-recency, two-code phenotype, and monitoring-window sensitivities.
4. Generate missingness/MCAR/imputation, train/test balance, and predictor-diagnostic outputs.
5. Decide on CDR v9 replication versus a frozen-v8 response.
6. Rewrite the manuscript and prepare a point-by-point response, using completed analyses only after the cohort definition is final.
