# Draft point-by-point response to editorial and reviewer comments

This is a working content draft for the response letter. It will be converted to the journal-required submission format after manuscript revisions are finalized. Items marked `[todo]` require a final manuscript edit, verification, or result update before submission.

## Editorial comments

### E1. Suggested citations

> “It is on the authors’ decision/discretion to evaluate the suggested citations for their relevance to the MS on revision.”

RESPONSE: No reviewer-supplied scholarly citations were received. The editorial message provided journal-policy and reporting-guidance links only. We have independently updated the manuscript with directly relevant recent literature.

### E2. JARS reporting and headings

> “Please adhere to the JARS guidelines… [and] use all of the headings from the JARS checklist into the manuscript to facilitate review.”

RESPONSE: We will reorganize the Methods section using JARS-aligned headings, including Inclusion and Exclusion, Participant Characteristics, Sampling Procedures, Sample Size, Power, and Precision, Measures and Covariates, and Data Analysis. We will also provide a participant-flow figure aligned with JARS reporting. `[todo: complete manuscript restructuring and flow figure.]`


### E3. Title format

> “Please modify the title to be in line with our journal guidelines for titles (‘Issue or Intervention in Demographic/Disease/Condition: Method/Study Design’).”

RESPONSE: We have revised the title to identify the wearable-derived physiological measures, classification across chronic physical and mental health conditions, and the retrospective cohort study design. We will remove “trans-diagnostic” because the analyses do not aim to distinguish diseases from each other. Proposed title: “Fitbit-Derived Physiological Features for Classifying Chronic Physical and Mental Health Conditions in US Adults: Retrospective Cohort Study.”

### E4. Robust abstract

> “Please make your abstract more robust… In the Methods, provide research design… [and] in the Results, mention the name of the statistical tests, the alpha level, and all point estimates… with confidence intervals.”

RESPONSE: We will expand the structured abstract to report the retrospective cohort design, All of Us data source, final cohort size and eligibility criteria, Fitbit monitoring definition, outcome phenotypes, model-development procedure, and central AUROC and Brier skill-score estimates with 95% CIs. The abstract will also report the relevant paired-comparison procedure and multiplicity adjustment where sensitivity results are summarized. `[todo: revise abstract after final primary and medication-sensitivity results are locked.]`

### E5. Innovation and implications in abstract and discussion

> “In the concluding paragraphs of the abstract and discussion, please mention 1) how the study is innovative, 2) how it is different from existing studies, 3) what it brings to the field, and 4) its implications in the real world.”

RESPONSE: We will align the abstract and Discussion conclusions. The revised text will emphasize the study’s evaluation of routinely collected wearable summaries linked to EHR-recorded disease phenotypes across several physical and mental health domains, while avoiding claims of disease onset prediction, causal inference, or disease-versus-disease discrimination. `[todo: revise abstract and Discussion conclusion in parallel.]`

### E6. Background and recent literature

> “Please provide a more detailed literature background and strengthen the rationale… and cite more scholarly references—more pertinent/related articles in the past 2-3 years could be cited.”

RESPONSE: We will expand the Introduction with recent literature on wearable data linked to EHR phenotypes, disease-state versus incident-disease prediction, multimorbidity, and the interpretation of wearable measures in managed chronic illness. `[todo: complete targeted literature review and revise Introduction.]`

### E7. Clear study aim

> “End your introduction with a clear statement of what the aim/objectives of this paper or study are, or what the hypotheses are.”

RESPONSE: We will end the Introduction with a constrained objective: to evaluate the internal discrimination and incremental value of Fitbit-derived physiological summaries for EHR-recorded disease-versus-control phenotypes across multiple health domains, and to assess robustness to alternative phenotype, comparator, medication, lookback, and monitoring-duration definitions. `[todo: insert finalized objective statement.]`

### E8. Methods completeness, setting, recruitment, and sampling

> “Please cross-check that all of the following items are included in the methods section… Study design… Setting… Participants… Assessments… Data sources… Study size… Data analysis… [and] recruitment procedures.”

RESPONSE: We will add explicit reporting of the retrospective cross-sectional analytic design, All of Us setting and data-release version, relevant data-collection dates, voluntary participation and consented linkage of EHR and Fitbit data, participant eligibility, derived measures, covariates, cohort flow, and analytic procedures. We will characterize this as a nonprobability volunteer sample rather than a random sample and will state that the sample size was determined by eligible linked data rather than a formal prospective power calculation. `[todo: complete Methods restructuring and verify all release-specific wording.]`

### E9. Figure requirements

> “Ensure that Figures comply with the journal’s guidelines.”

RESPONSE: We will review all figures for journal compliance, legibility, accessibility, and appropriate content. `[todo: complete figure audit before resubmission.]`

### E10. Participant flowchart

> “Please use the JARS Participant Flowchart… [to] provide readers with visual information about the participants across each stage of the study.”

RESPONSE: We will replace or supplement the prior attrition presentation with a JARS-style participant-flow diagram that shows the linked-data source population, wearable-data eligibility, phenotype eligibility, complete-case exclusion, and final disease-specific analytic cohorts. `[todo: create and insert flow diagram.]`

### E11. Ethical considerations

> “Add a separate ‘Ethical considerations’ subsection… [addressing] ethics review… informed consent… privacy and confidentiality… compensation… [and] identifiable images.”

RESPONSE: We will add a dedicated Ethical Considerations subsection covering All of Us institutional oversight and participant consent, the permitted secondary use of controlled-access deidentified data, privacy safeguards, compensation information as documented by All of Us, and confirmation that no identifiable participant images are included. `[todo: verify the exact All of Us ethics, consent, and compensation wording.]`

### E12. Confidence intervals and precision

> “Include confidence intervals for all point estimates and interpret findings in light of precision.”

RESPONSE: We have regenerated the primary and completed sensitivity analyses with 1,000 percentile-bootstrap 95% CIs for AUROC and Brier skill score. The revised Results and Discussion will emphasize CIs, smaller restricted-cohort sample sizes, and uncertainty rather than relying only on point estimates. `[todo: propagate 95% CIs to all manuscript tables, figures, and abstract.]`

### E13. Statistical reporting

> “Please cross-check the statistics follow the journal’s Guidelines for Reporting Statistics.”

RESPONSE: We revised the statistical Methods and table/figure legends to define core metrics at first use, identify the 1,000-resample percentile-bootstrap and paired DeLong procedures, and state the two-sided nominal significance threshold. Benjamini--Hochberg FDR correction is used for exploratory coefficient-level analyses with correlated predictors. Disease-specific paired sensitivity comparisons are reported with raw P values and 95\% CIs; the monitoring-duration comparison additionally used the stored Benjamini--Hochberg adjustment across its prespecified duration comparisons. `[todo: complete final statistical-reporting and table-legend audit.]`

### E14. Missing data, MCAR, and multiple imputation

> “Report any missing data and perform the MCAR test to see if data were missing totally at random and consider performing multiple imputations.”

RESPONSE: We completed a missingness and cohort-retention assessment in the QC-filtered seven-day cohort. Date of birth, race, sex assigned at birth, and all configured survey indicators had zero measured missingness in their source tables. The small incremental complete-case exclusions after static eligibility ranged from 9 to 18 participants across disease-specific frames. In contrast, absence of a QC-eligible Fitbit run is structural non-observation, not an ordinary partially missing covariate. Multiple imputation would not recover these participants and was not indicated for the ordinary source variables with zero measured missingness. We will report the flow and retained-versus-excluded descriptive data, while avoiding overinterpretation of the small excluded groups. `[todo: insert final missingness table/flow and complete the associated Methods/Results cross-check.]`

### E15. Discussion references

> “There are statements in the discussion without references. Please include supporting references for all statements.”

RESPONSE: We will audit the Discussion and add references for factual and comparative statements, while removing unsupported speculative language. `[todo: complete reference audit.]`

### E16. Discussion structure and calibrated conclusions

> “Please enhance and structure your Discussion… [and] ensure that conclusions can be drawn based on the methodology and findings.”

RESPONSE: We will restructure the Discussion into: principal findings relative to the stated objective; comparison and interpretation in light of prior literature; limitations; and implications/conclusion. We will remove repetition of numerical Results and explicitly limit conclusions to internal classification of EHR-recorded, potentially treated disease phenotypes. `[todo: rewrite and shorten Discussion.]`

### E17. Funding statement

> “(required) Funding Statement - Add a funding statement, distinct from the Acknowledgements.”

RESPONSE: The manuscript already contains a distinct funding section. We will retain the declared funders and their no-role statement, standardize the heading to “Funding Statement,” and verify that all support for this work is represented. `[todo: verify final wording with all authors.]`

### E18. Generative-AI disclosure

> “(required) Disclose whether generative AI was used in any portion of the manuscript generation.”

RESPONSE: `[todo: insert a truthful generative-AI disclosure in the manuscript, if required by the authors’ actual use, or explicitly attest in the final response letter that no generative AI was used for manuscript generation.]`

### E19. Tracked-changes and clean manuscript files

> “Upload a revised version with tracked changes… in addition to providing a clean version under ‘Revised Ms’.”

RESPONSE: `[todo: prepare and upload both a tracked-changes/highlighted version and a clean revised manuscript at resubmission.]`

### E20. Style, formatting, and word count

> “It is the authors’ responsibility to ensure adherence to JMIR Publications style and formatting, including final word count.”

RESPONSE: `[todo: complete final JMIR style, figure, table, reference, and word-count checks before resubmission.]`

## Reviewer AV

### A1. Multimorbidity and shared controls

> “The manuscript should clarify how multimorbidity was handled in the disease-versus-control models, since participants with multiple diagnoses could belong to more than one diagnostic cohort.”

RESPONSE: We agree. We regenerated a descriptive overlap matrix under the corrected seven-day, pre-index phenotype definition and quantified additional qualifying target conditions within each disease-specific analytic cohort. We also completed a single-condition sensitivity analysis. Participants may contribute to more than one disease-versus-control analysis; therefore, the revised manuscript will describe estimates as associations with recorded disease phenotypes that may reflect multimorbidity, rather than as independent disease-specific signatures. Controls will be described as a deliberately strict, EHR-recorded target-condition-free comparator group, with the limitation that EHR data cannot establish absence of undiagnosed disease. The overlap and single-condition analyses are described in the main Results and Tables S8 and S10.

### A2. English and terminology

> “The manuscript requires careful revision to correct typographical and grammatical errors and ensure consistent terminology, including terms such as ‘AUROC/AUC-ROC’ and the capitalization of model names.”

RESPONSE: `[todo: complete professional copyediting; use ‘AUROC’ consistently and standardize ‘logistic regression’ and ‘XGBoost’ throughout.]`

## Reviewer BN

### B1. Disease overlap and multimorbidity burden

> “The manuscript would benefit from quantifying the overlap between disease groups and present[ing] a comorbidity matrix… [to determine] whether observed physiological patterns reflect specific disorders or shared multimorbidity burden.”

RESPONSE: We agree and completed the requested descriptive overlap analysis and single-condition sensitivity analysis under the QC-filtered pre-index definition. The overlap matrix demonstrates substantial nonexclusive condition membership; for example, 1,608 participants qualified for both MDD and anxiety. Single-condition analyses substantially reduced case counts and often produced lower point estimates, particularly for MDD, bipolar disorder, and ADHD. We will use these analyses to qualify, rather than eliminate, the potential contribution of multimorbidity and will not describe the findings as unique disorder-specific signatures. The overlap and single-condition results are described in the main Results and Tables S8 and S10.

### B2. Diagnosis recency thresholds

> “The disease recency threshold appear arbitrary… why 10 years rather than 5 years for diabetes? why 2 years rather than 1 year for anxiety? how sensitive are results to these thresholds?”

RESPONSE: We agree that the original wording did not sufficiently distinguish a pragmatic EHR phenotype rule from a clinical assertion of active disease. We corrected the temporal definition so that only diagnoses recorded on or before the Fitbit index establish case status and recency is calculated from the most recent qualifying pre-index diagnosis. We completed no-lookback, current disease-specific lookback, and halved-lookback analyses. The paired comparisons showed no consistent pattern of dependence on the selected policy. The revised manuscript will state that the windows balance cohort retention and temporal proximity of documented disease; they do not infer disease onset, remission, or clinical activity. `[todo: add diagnosis-timing distribution and literature rationale.]`

### B3. Control misclassification

> “The control cohort might still have people with undiagnosed, subclinical, or other long standing conditions… especially relevant for psychiatric disorders.”

RESPONSE: We agree. The revised manuscript avoids referring to controls as definitively healthy. Controls are described as participants without recorded target conditions or specified ICD exclusions in their available EHR history. Residual undiagnosed and subclinical disease remains an unavoidable limitation, particularly for psychiatric conditions. We also completed a stricter-comparator sensitivity analysis excluding controls with records of listed relevant medications. This comparator is plausibly generally healthier and less clinically treated because of the excluded medication records, and modestly higher AUROCs are consistent with greater contrast against that selected comparator; they do not establish true health or disease-specific physiology. The clean-control cohort result and its interpretation are reported in the main Results and Table S12.

### B4. Typographical errors

> “Please watch for the typographical errors.”

RESPONSE: `[todo: complete the copyedit described in A2.]`

### B5. Length of the Discussion

> “Some discussion sections are lengthy and can be more condensed to improve readability.”

RESPONSE: We agree. `[todo: restructure and shorten the Discussion as described in E17.]`

### B6. Calibration-panel sample sizes

> “In the Figure 2 calibration curves… reporting subgroup sample sizes directly within the figure panels would be good.”

RESPONSE: `[todo: add disease-specific case/control sample sizes to the calibration-panel labels or figure legend.]`

## Reviewer DD

### D1. Multimorbidity

> “The authors… do not look at the presence of a participant having multiple co-occurring conditions.”

RESPONSE: We agree. We completed a descriptive co-occurrence matrix and an outcome-specific count of additional qualifying target conditions, as well as a single-condition sensitivity analysis. The manuscript will no longer imply that model findings are independent of multimorbidity. The revised Results and Discussion describe these findings with the overlap and single-condition analyses in Tables S8 and S10.

### D2. Newer All of Us release

> “There is now a more recent data release… I would suggest the incorporation of this data in the next iteration of the analysis.”

RESPONSE: `[todo: decide whether a full CDR v9 rerun is feasible. If not, state the frozen CDR v8 version and data cutoff used for this analysis, explain that the analysis was conducted on that locked release, and acknowledge that later releases may yield different sample sizes and estimates.]`

### D3. Fitbit valid-day quality

> “A valid day was one for which summary statistics were available. I would suggest that further exclusions should be made, for example, filtering days based on wear time or for implausible values.”

RESPONSE: We revised the valid-day definition to require at least 10 hours (600 minutes) of wear time per day, approximated as the sum of minutes in Fitbit's out-of-range, fat-burn, cardio, and peak heart-rate zones, and at least 100 steps. This rule is applied before feature engineering, complete-case filtering, and continuous-streak selection. [todo: add final quality-check distribution and flow counts.]

### D4. EHR phenotype confirmation

> “Was presence of 1 related-ICD code sufficient? I would suggest… at least 2 ICD codes, or methods that integrate multiple EHR parameters.”

RESPONSE: We agree that a one-code phenotype may be less specific. We completed a sensitivity analysis requiring two mapped diagnosis records on two distinct calendar dates on or before the Fitbit index. This restriction reduced case counts by approximately 16\% to 29\%, depending on condition, and yielded broadly similar AUROC estimates, with no consistent pattern across conditions or model classes. We retained the inclusive pre-index diagnosis definition as primary because the study targets recorded EHR disease phenotypes rather than only repeatedly documented disease, and because a two-date rule may preferentially select participants with greater health-care use or more established illness. We will report the two-date results as a phenotype-validity sensitivity analysis. The rationale and results are reported in the Methods, Results, and Table S11.

### D5. Exploratory heart-rate/activity interaction

> “What was the exploratory data analysis that is referenced when discussing the interaction terms between heart rate and activity?”

RESPONSE: `[todo: verify and document the exact exploratory rationale, feature construction, and supporting descriptive analysis. If a reproducible prespecified rationale cannot be documented, label the feature as exploratory and avoid etiologic interpretation.]`

### D6. Sex at birth and race categorization

> “Why were only participants with male or female sex at birth included? Also… it’s unclear why [non-White participants] were grouped into one category.”

RESPONSE: Sex was measured as sex assigned at birth, a biological-sex variable in the All of Us data. The primary analyses use male and female categories. Records without either category were very uncommon; their prevalence will be reported, but sparse representation does not support stable disease-specific estimates. This covariate is not intended to represent gender identity. We agree that a White versus non-White indicator is a parsimonious adjustment variable and should not be interpreted as implying that all non-White racialized groups are homogeneous. We retained this coding in the primary disease-specific analyses because several individual self-reported race categories, including Asian participants in some disease cohorts, have sparse case or control counts that would yield unstable coefficients. We will report disaggregated race/ethnicity counts and conduct a sensitivity analysis using the most granular stable race categories available within each disease-specific analytic cohort, combining categories that do not meet a prespecified minimum count in both cases and controls into other/unknown. `[todo: run the expanded-race sensitivity analysis and insert its performance comparison.]`

### D7. Covariate collinearity

> “There still seem to be many covariates that would be related… It would be helpful if the authors could clarify if all these variables were fit in their models.”

RESPONSE: `[todo: report the final predictor set and conduct/report collinearity diagnostics for all model covariates, including lifestyle variables. Revise the feature set or qualify coefficient interpretation if problematic collinearity is identified.]`

### D8. Compact-feature selection and Table 3

> “It is not clear… how exactly the more compact set of features was selected. This should be made more explicit. Table 3 is also confusing.”

RESPONSE: `[todo: document candidate features, correlation/collinearity screening, domain-informed retention criteria, and the final selected feature set; redesign the table or move detailed selection information to the supplement.]`

### D9. Choice of models

> “Why were logistic regression and XGBoost models selected (and not other methods)?”

RESPONSE: We selected logistic regression as an interpretable linear benchmark and XGBoost as a flexible nonlinear tree-based comparator. The aim was not an exhaustive algorithm competition but assessment of whether conclusions were consistent across complementary model classes. `[todo: add this justification and supporting methodological citations.]`

### D10. Utility of the clean cohort

> “The utility of the clean cohort is not clear, especially as it is not really mentioned after the methods.”

RESPONSE: We agree that its purpose was previously unclear. We completed and will report a stricter-comparator sensitivity analysis in which controls had no recorded history of listed sleep-, psychiatric-, or heart-rate-altering medications. This evaluates whether discrimination depends on contrast with a more selected comparator, plausibly generally healthier and less clinically treated because those medication records are excluded. It does not test medication causality or establish true health. AUROCs were modestly higher for several outcomes, supporting the interpretation that comparator selection contributes to discrimination. The purpose and results are reported in the Methods, Results, Discussion, and Table S12.

### D11. Timing of Fitbit measurement and diagnoses

> “How was timing of the Fitbit measurements considered in relation to disease diagnoses?”

RESPONSE: We corrected the temporal cohort definition. A case now requires at least one mapped diagnosis on or before the selected Fitbit streak start and within the disease-specific pre-index lookback window; recency is calculated from the most recent qualifying pre-index diagnosis. Post-index diagnoses do not establish case eligibility. We selected the latest eligible seven-day streak for past-diagnosis cases and retained future-only diagnosis analyses solely as exploratory supplemental analyses, not as incident-disease prediction. `[todo: update Methods, flow diagram, tables, and supplement with the corrected cohort definition and timing distribution.]`

### D12. Consecutive Fitbit days

> “Why did a participant need to have 21 consecutive days of data? … [consider] a certain number of days but not necessarily… consecutive.”

RESPONSE: We completed corrected pre-index sensitivity analyses using 7, 14, and 21 consecutive QC-eligible Fitbit days. Seven days retained substantially more participants (for example, 3,528 CVD cases and 4,274 controls versus 2,049 and 2,791 with 21 days) while maintaining broadly similar discrimination; no duration comparison remained significant after the stored Benjamini--Hochberg adjustment. We selected seven consecutive days to balance retention with a contiguous physiological snapshot, rather than because it yielded the highest point estimate. We did not aggregate nonconsecutive days because RMSSD and related short-term heart-rate-variability measures require temporally contiguous observations; pooling separated days would weaken their physiological interpretability. The duration rationale is in the Methods and the results are reported in Table S14.

### D13. Main-table content

> “Table 5 should have more information, much of the information is in the supplement currently and should be moved to the main text.”

RESPONSE: `[todo: revise the main Results table to include per-outcome cases/controls, full-model AUROC and Brier skill score with 95% CIs, and the incremental value of Fitbit features; retain detailed coefficients and secondary outputs in the supplement.]`

### D14. Figure 1 and AUROC

> “Figure 1 – why not plot the AUROC instead?”

RESPONSE: `[todo: revise Figure 1 so AUROC is the prominent performance display; retain Brier skill score in a complementary panel, table, or supplement as space permits.]`

### D15. Results reported only in Discussion

> “There are results presented in the discussion that were not previously mentioned. All results should be presented in the Results section.”

RESPONSE: `[todo: audit the Discussion, move necessary empirical results to Results or supplement, and remove unsupported numerical claims.]`

### D16. Flow diagram

> “When discussing the inclusion and exclusion criteria a flow diagram would be helpful to include, rather than the Table 4…”

RESPONSE: We agree. `[todo: address through the JARS-style flow diagram described in E11.]`

### D17. WEAR definition

> “The authors introduce the WEAR initiative without first describing what it is.”

RESPONSE: `[todo: define WEAR at first mention and distinguish it from BYOD.]`

### D18. Table 1 and supporting references

> “Not sure Table 1 needs to be a table… it would be helpful to include references to justify these numbers.”

RESPONSE: `[todo: assess whether Table 1 is better presented as concise text; retain only necessary tabular content and add references supporting condition definitions/rationale.]`

### D19. FDR versus Bonferroni

> “Why did the authors select FDR for accounting for multiple comparisons (and not Bonferroni correction)?”

RESPONSE: We will clarify that coefficient-level analyses are exploratory and involve correlated predictors. Benjamini-Hochberg FDR control was selected to limit the expected proportion of false discoveries while retaining reasonable sensitivity in this setting; Bonferroni would control the family-wise error rate but be overly conservative for the exploratory coefficient family. We will distinguish this procedure from the BH adjustment used for the pre-specified family of paired sensitivity comparisons. `[todo: add explanation to Methods and table legends.]`

### D20. Undefined acronyms

> “At times there are acronyms that are not spelled out.”

RESPONSE: `[todo: define all acronyms at first use and ensure consistency across text, tables, figures, and supplement.]`

### D21. Late metric definitions

> “The authors mention AUROC and Brier scores but then don’t explain what these are until later in the paper.”

RESPONSE: `[todo: define AUROC and Brier skill score at their first appearance in Methods and briefly state their interpretation.]`

### D22. Table asterisks

> “The * in tables should be defined in a legend.”

RESPONSE: `[todo: define all symbols and asterisks in the relevant table and figure legends.]`

## Reviewer DL

### L1. Medication and managed disease states

> “The wearable data may therefore be identifying the treatment phenotype rather than the disease phenotype… [discuss] post-diagnosis treatment as a major confounder… [and] classify ‘managed chronic disease states’ rather than raw disease pathology.”

RESPONSE: We agree that the models classify EHR-recorded, potentially treated disease states and cannot isolate untreated disease biology or causal medication effects. We completed a medication-unexposed sensitivity analysis that excluded cases and controls with a listed relevant medication record overlapping the Fitbit interval expanded by 30 days on either side; missing medication end dates were treated as one-day records. AUROC differences were generally small, but medication exclusion substantially reduced case counts, especially for psychiatric conditions, and changes the target population. We therefore retain the inclusive disease-state cohort as primary, present this analysis as a robustness check, and explicitly discuss treatment, disease severity, health-care use, and unrecorded medication as confounding limitations. The medication class list is Table S2 and the medication-unexposed results are Table S13.

### L2. Meaning of “transdiagnostic” and multiclass classification

> “The study lacks a multi-class evaluation or disease-versus-disease modeling… [either] implement a basic multiclass classification model… or explicitly revise the introduction and conclusion to clarify that the study evaluates ‘disease vs. healthy’ phenotypes across multiple domains.”

RESPONSE: We agree that the original terminology could overstate the study’s scope. We will not add a multiclass model because extensive nonexclusive multimorbidity would make mutually exclusive disease labels artificial and would answer a different question. Instead, we will reframe the manuscript as disease-versus-control classification of EHR-recorded phenotypes across multiple physical and mental health domains. We will not claim disease-versus-disease discrimination or unique condition-specific wearable signatures. `[todo: revise title, abstract, Introduction, Results, Discussion, and conclusion consistently.]`

### L3. Internal split and BYOD/WEAR balance

> “An internal test set drawn from the same distribution does not validate the generalizability… [provide] a supplementary demographic breakdown comparing the 70% training set and the 30% test set… particularly… BYOD vs. WEAR.”

RESPONSE: We agree. `[todo: add a supplementary train/test balance table with demographic and lifestyle characteristics, disease prevalence, and BYOD/WEAR composition. The revised text will state that this assessment supports the integrity of the internal split only and does not establish external transportability.]`

### L4. Zero denominator in the activity ratio

> “Please clarify how the data processing handled days or subjects where ‘lightly active minutes’ equaled exactly zero to avoid division-by-zero errors or infinite values.”

RESPONSE: `[todo: verify the preprocessing rule for zero denominators. We will document the exact handling in Methods and, if the ratio was not safely handled, correct it and rerun affected analyses before resubmission.]`

### L5. “Unknown” lifestyle categories

> “Please provide a brief explanation… regarding how these ‘Unknown’ categories were mathematically handled during modeling.”

RESPONSE: `[todo: verify the current encoding and state explicitly whether unknown responses were retained as categorical levels, imputed, or excluded; report their prevalence and ensure the Methods match the implementation.]

### Editorial-revision implementation plan

The following implementation plan will guide the manuscript revision and the subsequent preparation of a highlighted version for resubmission. Completion is tracked in `fitbit_base_model/manuscript_revision_changelog.md`.

1. **Title and abstract:** revise the title to “Fitbit-Derived Physiological Features for Classifying Chronic Physical and Mental Health Conditions in US Adults: Retrospective Cohort Study”; replace the abstract with a structured, approximately 450-word version after the final quality-control and sensitivity-analysis results are locked.
2. **Methods/JARS structure:** reorganize Methods using explicit JARS-aligned headings: Study Design; Setting and Data Sources; Sampling Procedures; Participants (Inclusion and Exclusion Criteria; Participant Characteristics); Sample Size, Power, and Precision; Measures and Covariates; Statistical Analysis; and Ethical Considerations.
3. **Cohort and data quality:** incorporate the corrected pre-index phenotype timing, selected seven-day monitoring window, Fitbit data-quality processing, missingness reporting, and complete-case rationale. Any quality-control sensitivity result will be finalized when the current rerun is complete.
4. **Results and precision:** introduce a JARS-style participant-flow diagram; expand the main performance table with analytic sample sizes, 95% CIs, paired performance differences, and adjusted P values; and report calibration-panel sample sizes.
5. **Interpretation:** reframe the paper as internal classification of EHR-recorded disease states across health domains, rather than incident prediction, causal inference, or disease-versus-disease discrimination; restructure and shorten the Discussion accordingly.
6. **Journal and submission requirements:** audit figures, tables, abbreviations, references, statistics, funding, ethical statements, data availability, and generative-AI disclosure; produce a clean manuscript plus a highlighted/tracked version at submission.

Items that depend on final Fitbit quality-control, medication, or other sensitivity results will remain marked `[todo]` until those results are finalized.

