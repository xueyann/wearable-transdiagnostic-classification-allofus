
# Planned revisioned (written by me)


I have received the following comments from the reviewers for the XGBoost manuscript. Based on the comments, I think here are the major concerns and additional analyses we need:

Multimorbidity and transdiagnostic claim (3 reviewers: AV, BN. DD): both BN and DD suggested a comorbidity matrix; DL suggested a multiclass classification to be truly ‘transdiagnostic’. The tricky parts are:
1. Comorbdity matrix: if we just look at a co-occurrence matrix and present them descriptively,  that is fine, but how can we combine it with the classification task? If we define cohorts based on just single condition, and a separate cohort for each combination, we might end up with too many small cohorts for meaningful interpretation. Perhaps we can keep the current definition, but with additional analysis such as single condition cohort sensitivity analysis, selected comparison with 2+ comorbidity subcohort etc, while presenting a co-occurrence matrix descriptively? Or we just do a sensitivity analysis excluding patients with multicomorbidity in each cohort
2. Multiclass classification: this would fit into transdiagnostic framing better, but it will also make the coefficients harder to interpret. I am not sure whether we want to go with that
3. Overall, it might be better to go with the suggestion to reframe what this paper is; we can rephrase ‘transdiagnostic’ to ‘evaluate disease vs health phenotypes across multiple domains’ like DL suggested, so we avoid changing the original analyses too much; but we can still add the sensitivity analysis with pure single condition cohorts and present a disease co-occurrence matrix descriptively.
Medication confounding (DL): rightly pointed out that our cohort definition is static, not necessarily post-fitbit measurement, therefore patients already on medications might already have an altered heart rate and sleep architecture, therefore we can’t be sure that the effect seen is because of the disease or because of medication. However, this is a tricky topic per se: some epidemiology studies explicitly require prescription of relevant medication alongside ICD code diagnosis to define a disease cohort; if we exclude patients without prescription and only ICD code diagnosis, we might risk losing significant sample size, and the resulting cohort might not actually reflect the real life condition, where most patients are put onto a medication. I guess a way is to look at specific temporal windows where diagnosis has been made (or shortly before diagnosis was made) but before medication was prescribed?
 
Additional/sensitivity analyses to be included:
1. Clean control cohort: need to include actual sensitivity analysis rather than just descriptives
2. Comparison of 7, 14, 21 day streaks as well as sample size. Currently we don’t really have a strong justification for 21 consecutive days, this might also be a good way to justify the choice; also s fixed number of days within a certain time window, not necessarily consecutive
3. Distribution of train/test/ set split to ensure that the distributions are similar in train and test
4. Disease recency definition: DL mentioned that the diagnosis recency filter (e.g 10 years for CVD, 2 years for MDD/anxiety) is a little random, perhaps we can argue in terms of sample size: show a distribution of how recent the diagnoses are?

# Original comments

-------------------
EDITORIAL COMMENTS (Decision: D/major revision)
-------------------

There are critical issues with the manuscript that the authors must address for continued consideration in this journal. If revisions are not sufficient, then the manuscript would be transferred. When submitting a revised manuscript, please let us know your transfer preferences from the suggested JMIR Publications journals below. (Journal list: https://jmir.zendesk.com/hc/en-us/articles/115001442707). Note: Transfer preferences are only used in case of a final decision of rejection from the current journal. If authors do not address the journal transfer preferences, this can lead to delays on further editorial processing. The editor suggests one of the following journals:
- Interactive Journal of Medical Research
- Online Journal of Public Health Informatics

* Editorial and content comments *

In the event that we determine the manuscript is not a good fit for this journal, please let us know if you would be open to transferring it to one of the recommended JMIR Publications journals, and indicate your preferred choice (a list of partner journals can be found here: https://jmir.zendesk.com/hc/en-us/articles/115001442707). Ranking your transfer preferences will not impact the manuscript's evaluation for this journal, but failing to address the transfer options may lead to delays in the editorial process or result in rejection if all transfer options are declined.

Thank you for submitting your manuscript. I have considered the reviewers’ comments and, after my own reading, I would like to provide some additional guidance to help strengthen your manuscript.

It is on the authors’ decision/discretion to evaluate the suggested citations for their relevance to the MS on revision. The decision whether or not to include them in the revised manuscript will not influence further decision making.

Please adhere to the JARS guidelines for reporting the study as it is a comprehensive checklist to ensure important details of the study are reported to enable replication. Please visit https://apastyle.apa.org/jars/quant-table-1.pdf for the checklist.

Please use all of the headings from the JARS checklist into the manuscript to facilitate review.

Method
- Inclusion and Exclusion
- Participant Characteristics
- Sampling Procedures
- Sample Size, Power, and Precision
- Measures and Covariates
- etc...

Please modify the title to be in line with our journal guidelines for titles (“Issue or Intervention in Demographic/Disease/Condition: Method/Study Design”) https://jmir.zendesk.com/hc/en-us/articles/115002943791-What-are-JMIR-s-guidelines-for-article-titles

Please make your abstract more robust. As many readers only read abstracts, JMIR prefers abstracts that are robust and reflective of the manuscript as a whole. Author guidelines allow for up to 450 words in an abstract. Please use the full 450 words available.

What to include in the abstract:
- In the Methods, provide research design (e.g., cross-section, retrospective/prospective cohort), total sample size or per group, some demographic/clinical factors e.g., participants sex, age, diseases, etc., the materials used (e.g., instruments, apparatus), the outcome measures, sampling procedure, and the data-gathering procedures, and if applicable, the source of any secondary data as well as from where participants were recruited, and the selection criteria.
- In the Results, mention the name of the statistical tests, the alpha level, and all point estimates (e.g., differences between groups, single point estimates, etc) with confidence intervals. If presenting %, also include absolute values and confidence intervals.

In the concluding paragraphs of the abstract and discussion, please mention 1) how the study is innovative, 2) how it is different from existing studies, 3) what it brings to the field, and 4) its implications in the real world. Ensure the content in both concluding paragraphs match.

Please provide a more detailed literature background and strengthen the rationale for conducting the study on the topic in the Introduction section and cite more scholarly references - more pertinent/related articles in the past 2-3 years could be cited.

End your introduction with a clear statement of what the aim/objectives of this paper or study are, or what the hypotheses are.

Please, cross-check that all of the following items are included in the methods section:
* Study design
* Setting (Describe the setting, locations, and relevant dates, including periods of recruitment, and data collection)
* Participants (Give the eligibility criteria, and the sources and methods of selection of participants.)
* Assessments (Clearly define all outcomes, exposures, predictors, and potential confounders)
* Data sources (For each variable of interest, give sources of data and details of methods of assessment (measurement).
* Study size (Explain how the study size was arrived at)
* Data analysis (describe all the quantitative and qualitative analysis performed)
* Please provide additional information on the study/clinical setting of recruitment, as well as which recruitment procedures was used (i.e., Convenience Sampling, Random Sampling, Purposive Sampling, Stratified Sampling, Snowball Sampling, Quota Sampling, etc).

Ensure that Figures comply with the journal's guidelines https://support.jmir.org/hc/en-us/articles/360013033312-What-types-of-figures-are-not-permitted-in-JMIR-articles#h_01K7M0296K3WTRCDN5JGQBDHHV

Please use the JARS Participant Flowchart https://apastyle.apa.org/jars/jars-quant-participant-flowchart.pdf this will need to be amended according to the study design. This flowchart will provide readers with visual information about the participants across each stage of the study.

Add a separate “Ethical considerations” subsection in the Methods section and include a statement addressing each of the five following points:
1. Statement regarding human subject research ethics review, exemptions, and approvals. This is necessary even if the study is clearly observations of public behaviors or secondary analyses of research data.
2. Informed consent descriptions or waiver language for studies identified as human subjects research. This also applies to secondary analyses of research data (there should still be consent for the primary data collection), where it should be stated that the original informed consent (or the IRB) allows the secondary analysis without additional consent.
3. Privacy and confidentiality protection description for human subjects research; e.g., a statement that study data are anonymous or de-identified; otherwise, a brief description of the protections in place.
4. Compensation type and amount for human subjects research.
5. Please make sure that no identification of individual participants/users in any images of the manuscript or supplementary material is possible; if including such images is unavoidable, please include a statement that consent has been granted from identifiable individuals, and also upload the relevant consent forms/written communications in the “Other files” field during resubmission.

Include confidence intervals for all point estimates and interpret findings in light of precision.

Please cross-check the statistics follow the journal's Guidelines for Reporting Statistics https://support.jmir.org/hc/en-us/articles/360019690851-Guidelines-for-Reporting-Statistics

Report any missing data and perform the MCAR test to see if data were missing totally at random and consider performing multiple imputations.

There are statements in the discussion without references. Please include supporting references for all statements.

Please enhance and structure your Discussion section as follows:
○ The first paragraph of the discussion should be a brief summary of the main study findings relative to the stated objectives/aims/hypotheses from the last paragraph of the introduction.
○ Following, a more detailed discussion of those findings, including interpretations, implications, and comparisons to existing literature.
○ Next, please include the paragraph(s) on limitations.
○ The last paragraph(s) should be conclusions that don’t just summarize the findings but speak to the broader implications of them.
• Avoid repeating numeric study findings in the discussion except to make direct comparisons to other published literature. Just summarize the findings in plain language.
• Ensure that all statements in the Discussion/Conclusions appropriately match the conclusions that can be drawn, i.e., ensure that conclusions can be drawn based on the methodology and findings of the study. This applies to the Abstract and the manuscript text.

* Formatting/structure comments *

- (required) Funding Statement - Add a funding statement, distinct from the Acknowledgements. If no funding was involved, please state: "The authors declared no financial support was received for this work." More information: https://support.jmir.org/hc/en-us/articles/360015982471-How-should-the-Acknowledgments-section-be-formatted-

- (required) Disclose whether generative AI was used in any portion of the manuscript generation: if used, it is mandatory to include a written statement in the manuscript, even if you disclosed use at the time of submission (e.g., in cover letter or submission form). If none, authors should attest to this in the response to comments. https://support.jmir.org/hc/en-us/articles/13387268671771-Do-you-allow-the-the-use-of-ChatGPT-or-other-generative-language-models-and-how-should-this-be-reported- Authors may optionally use this tool to generate a complete disclosure: https://panbibliotekar.github.io/gaidet-declaration/index.html

*Instructions*

Upload a revised version with tracked changes (or highlighted sections of text where changes were made) as a supplementary file, in addition to providing a clean version under "Revised Ms". See https://jmir.zendesk.com/hc/en-us/articles/115001400448 for details on how to respond to reviewer comments and upload a revised manuscript. Supplying a version of the manuscript with tracked changes facilitates the review process.

Note that it is the authors' responsibility to ensure adherence to JMIR Publications style and formatting, including final word count. See: https://support.jmir.org/hc/en-us/articles/37982552280987

Addressing formatting issues and any remaining comments during the revisions period speeds up the production process in the event of acceptance.

------------------------------------------
External Peer-Review Report(s)
------------------------------------------
------------------------------------------------------
Reviewer AV:
General comments
=============
The paper shows a transdiagnostic analysis of wearable-derived physiological features, aimed at exploring shared and condition-specific patterns across different physical and mental health conditions in the real-world All of Us dataset. The objective addressed is relevant, and the study is supported by a large cohort combining Fitbit data and electronic health records. The methodology appears generally appropriate, and the manuscript is well structured and clearly written overall.

Specific comments
=============
Major comments
---------------------
1. The manuscript should clarify how multimorbidity was handled in the disease-versus-control models, since participants with multiple diagnoses could belong to more than one diagnostic cohort. This may affect the interpretation of condition-specific patterns and model performance, as well as the definition of the control group.

Minor comments
---------------------
2. The English is generally understandable, but the manuscript requires careful revision to correct typographical and grammatical errors and ensure consistent terminology, including terms such as “AUROC/AUC-ROC” and the capitalization of model names.

------------------------------------------------------

Please help us improve the quality of reviewers by providing a rating for each review using the URLs provided below (your ratings are anonymous and only seen by the editor):

TO RATE REVIEWER AV: CLICK ON ONE OF THE FOLLOWING URLS

1. very poor/misleading/wrong statements DO NOT USE REVIEWER AGAIN:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=635648&ro=1&ra=1&au=d61236136c

2. poor review (e.g. very superficial, extremely short):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=635648&ro=1&ra=2&au=d61236136c

3. below average (too short or various weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=635648&ro=1&ra=3&au=d61236136c

4. average (solid review with minor weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=635648&ro=1&ra=4&au=d61236136c

5. excellent / very useful review:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=635648&ro=1&ra=5&au=d61236136c

------------------------------------------------------


------------------------------------------------------
Reviewer BN:
Comments to the authors;
This manuscript comes across as a large-scale trans-diagnostic look at wearable-derived physiological features across seven chronic health conditions, using the All of Us Research Program dataset. It tries to fill an important gap in the literature, but not just by staying with one disease at a time. Instead it kinda moves beyond the usual disease-specific analyses and asks about shared, as well as condition-specific physiological signatures, spanning both physical and mental health disorders. The use of real world Fitbit data that is linked with electronic health records is a noticeable strength,and the analysis shows that wearable-derived features add incremental predictive value even after accounting for demographic and lifestyle variables.
Overall the work feels timely, and in many ways methodologically rigorous,plus it does add novel evidence about how useful wearable-derived measures can be for disease classification. The clearest take-away is that wearable features seem especially informative for mental health conditions, like major depressive disorder and anxiety, where substantial gains in AUROC were seen.
Nevertheless, there are a few interpretational issues that should be addressed before this is ready for publication.
Major comments;
1. The authors claims that participant are from multiple disease sub cohorts. But the analytical implications are not sufficiently addressed. I recommend the manuscript would benefit from the quantifying the overlap between disease group and present a comorbidity matrix may be helpful. Without these analyses it is difficult to determine whether observed physiological patterns reflect specific disorders or shared multimorbidity burden.
2. The disease recency threshold appear aribitrary. For example, the diagnosis recency windows (10 years for CVD/diabetes, 5 years for bipolar disorder, etc.). The soeccfic questions are why 10 years rather than 5 years for diabetes? why 2 years rather than 1 year for anxiety? how sensitive are results to these thresholds? Need more justification in the manuscript for this.
3. The control cohort might still have people with undiagnosed, subclinical, or other long standing conditions that are not really picked up by the exclusion rules, so outcomes can end up being misclassified. This worry could be especially relevant for psychiatric disorders since those are often underdiagnosed, and the possible influence on how the model performs should be discussed.
Minor comments;
1. Please watch for the typographical errors.
"interpretaility to interpretability"
"Axneity to Anxiety"
"Logisitic to Logistic"
"aggregation aggregation algorithms" its duplicated word
2. I feel some discussion sections are lengthy and can be more condensed to improve readability.
3. In the Figure 2 calibration curves may be reporting subgroup sample sizes directly within the figure panels would be good.

------------------------------------------------------

Please help us improve the quality of reviewers by providing a rating for each review using the URLs provided below (your ratings are anonymous and only seen by the editor):

TO RATE REVIEWER BN: CLICK ON ONE OF THE FOLLOWING URLS

1. very poor/misleading/wrong statements DO NOT USE REVIEWER AGAIN:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=667781&ro=1&ra=1&au=3ed4e39510

2. poor review (e.g. very superficial, extremely short):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=667781&ro=1&ra=2&au=3ed4e39510

3. below average (too short or various weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=667781&ro=1&ra=3&au=3ed4e39510

4. average (solid review with minor weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=667781&ro=1&ra=4&au=3ed4e39510

5. excellent / very useful review:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=667781&ro=1&ra=5&au=3ed4e39510

------------------------------------------------------


------------------------------------------------------
Reviewer DD:
General comments
===============
This manuscript by Huang and colleagues explores how adding Fitbit data to machine learning models improves prediction for 7 different health outcomes. The authors do a nice job of exploring the potential to use Fitbit data to detect various diseases. Overall, it is well-written, and this analysis addresses an interesting and important topic. However, several methodological concerns warrant attention before this manuscript can be published.

Specific comments
===============
Introduction
1. One of the limitations the authors present is the lack of focus on multimorbidity. While the authors look at multiple different health conditions, I don’t feel they adequately address this limitation as they do not look at the presence of a participant having multiple co-occurring conditions.
Methods
2. There is now a more recent data release (with more Fitbit data available) in All of Us. I would suggest the incorporation of this data in the next iteration of the analysis.
3. A valid day was one for which summary statistics were available. I would suggest that further exclusions should be made, for example, filtering days based on wear time or for implausible values.
4. How exactly were participants categorized into disease subcohorts using the ICD codes? For example, was presence of 1 related-ICD code sufficient? I would suggest looking into alternative, more robust methods, such as using the presence of at least 2 ICD codes, or methods that integrate multiple EHR parameters (e.g., ICD codes, medications, laboratory values, etc.)
5. What was the exploratory data analysis that is referenced when discussing the interaction terms between heart rate and activity? It is not clear where this is coming from currently.
6. Why were only participants with male or female sex at birth included? Also, the prevalence of non-White participants in All of Us is not that low, so it's unclear why they were grouped into one category.
7. The authors discuss multicollinearity, however, there still seem to be many covariates that would be related (e.g., daily drink count, drinking frequency, binge drinking, etc.) It would be helpful if the authors could clarify if all these variables were fit in their models.
8. It is not clear to me how exactly the more compact set of features was selected. This should be made more explicit. Table 3 is also confusing for this reason.
9. Why were logistic regression and XGBoost models selected (and not other methods)?
10. The utility of the clean cohort is not clear, especially as it is not really mentioned after the methods.
11. How was timing of the Fitbit measurements considered in relation to disease diagnoses?
12. Why did a participant need to have 21 consecutive days of data? I think this loses a large portion of the cohort. The authors could instead still require a certain number of days but not necessarily have them be consecutive.
Results
13. Table 5 should have more information, much of the information is in the supplement currently and should be moved to the main text.
14. Figure 1 – why not plot the AUROC instead? I think this would be a clearer way to show the findings.
Discussion
15. There are results presented in the discussion that were not previously mentioned. All results should be presented in the Results section.

Minor comments
===============
1. When discussing the inclusion and exclusion criteria a flow diagram would be helpful to include, rather than the Table 4 that is currently used and is only referenced in the results.
2. The authors introduce the WEAR initiative without first describing what it is. While they later briefly describe it, this should be discussed at first mention.
3. Not sure Table 1 needs to be a table, the authors could just specify this information in the text. Also, it would be helpful to include references to justify these numbers.
4. Why did the authors select FDR for accounting for multiple comparisons (and not Bonferroni correction)?
5. At times there are acronyms that are not spelled out.
6. The authors mention AUROC and Brier scores but then don’t explain what these are until later in the paper. These should be explained at first mention.
7. The * in tables should be defined in a legend.

------------------------------------------------------

Please help us improve the quality of reviewers by providing a rating for each review using the URLs provided below (your ratings are anonymous and only seen by the editor):

TO RATE REVIEWER DD: CLICK ON ONE OF THE FOLLOWING URLS

1. very poor/misleading/wrong statements DO NOT USE REVIEWER AGAIN:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=840517&ro=1&ra=1&au=554b8caf25

2. poor review (e.g. very superficial, extremely short):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=840517&ro=1&ra=2&au=554b8caf25

3. below average (too short or various weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=840517&ro=1&ra=3&au=554b8caf25

4. average (solid review with minor weaknesses):
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=840517&ro=1&ra=4&au=554b8caf25

5. excellent / very useful review:
https://mhealth.jmir.org/rev/authorRating?ms=98829&re=840517&ro=1&ra=5&au=554b8caf25

------------------------------------------------------


------------------------------------------------------
Reviewer DL:
General comments
=============
A major strength of this study is its utilization of a massive, real-world, longitudinal dataset containing a mix of Bring-Your-Own-Device (BYOD) and study-provided wearable data. The authors demonstrate excellent methodological rigor in their choice of performance metrics, utilizing both discrimination (AUROC) and calibration (Brier Skill Score) to evaluate their models. Furthermore, the study effectively highlights the differing sensitivities of linear and non-linear models to wearable features via SHAP values.

Specific comments
=============
Major comments
---------------------
1. The authors establish a 21-day FitBit monitoring period and match this to the "most recent diagnosis" recorded up to 10 years prior (e.g., 10 years for CVD, 2 years for MDD/Anxiety). Because the diagnoses historically precede the wearable monitoring period, the models are not predicting disease onset; they are identifying patients already living with and likely being treated for these diseases. Medications (e.g., beta-blockers for CVD, SSRIs for MDD, stimulants for ADHD) and lifestyle interventions prescribed post-diagnosis will heavily alter heart rate and sleep architecture. The wearable data may therefore be identifying the treatment phenotype rather than the disease phenotype. While the authors briefly mention medication in the sensitivity analysis for the control cohort, they must explicitly discuss post-diagnosis treatment as a major confounder in the main Limitations section. The framing of the paper should carefully reflect that these models classify "managed chronic disease states" rather than raw disease pathology.
2. The study builds separate binary classification models for each disease versus a shared "Control Cohort". While comparing each condition to a shared healthy baseline is a valid starting point, it does not fully answer the "trans-diagnostic" objective. Since many wearable features (e.g., reduced deep sleep, increased resting heart rate) are shared across morbidities, binary models might simply learn a general "unhealthy vs. healthy" heuristic. The study lacks a multi-class evaluation or disease-versus-disease modeling to prove that these wearable features capture condition-specific signatures. The authors should either implement a basic multiclass classification model (e.g., a single XGBoost model attempting to distinguish the 7 diseases from one another) to evaluate condition-specific signatures, or explicitly revise the introduction and conclusion to clarify that the study evaluates "disease vs. healthy" phenotypes across multiple domains, rather than distinguishing the diseases from each other.
3. The authors use a 70/30 train/test split within the All of Us dataset. The BYOD subset of the All of Us dataset is known to skew toward specific socio-demographic profiles (e.g., higher health literacy, specific income brackets). An internal test set drawn from the same distribution does not validate the generalizability of the models to broader clinical populations. The authors rightly note this in the Limitations section. To strengthen the paper, they should provide a supplementary demographic breakdown comparing the 70% training set and the 30% test set to ensure no inadvertent distribution shifts occurred during the random split, particularly regarding the ratio of BYOD vs. WEAR-provided devices.

Minor comments
---------------------
4. The authors engineered an interaction term: "Ratio of fat burn minutes to lightly active minutes". Please clarify how the data processing handled days or subjects where "lightly active minutes" equaled exactly zero to avoid division-by-zero errors or infinite values.
5. Table 2 outlines several "Unknown" categories for lifestyle variables (e.g., daily drink count, smoke frequency). Please provide a brief explanation in the Methods section regarding how these "Unknown" categories were mathematically handled during modeling (e.g., treated as a separate categorical factor level, imputed, or dropped).

------------------------------------------------------