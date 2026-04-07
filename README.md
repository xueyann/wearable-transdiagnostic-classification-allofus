# wearable-transdiagnostic-classification-allofus

Codes used to generate the results in manuscript titled *Wearable-derived physiological features for trans-diagnostic disease comparison and classification in the All of Us longitudinal real-world dataset*

## Environment and data source 
This project makes use of the All of Us program Controlled Tier version 8. The repository is designed to be run in the All of Us legacy Researcher Workbench environment.

> Running this code outside the Workbench is not possible due to All of Us data access policy. Refer to the [All of Us official policies](https://support.researchallofus.org/hc/en-us/articles/22346942074132-Data-Access-Framework) here: 

## Dependencies
Python version: `3.13.2`<br>
Install requirements: `pip install requirements.txt`

## Project structure / sequence for script run:
### Data pull
1. `src/get_base_cohort.py` **Get base cohort and demographic, used as a basis to create study cohorts later; basic restrictions of having FitBit data and EHR data, share config with get_data.py**
2. `src/get_data.py`
3. `src/get_data_fitbit.py` **Pull fitbit data, both summary and minute-level, and store in GCS bucket**
### Data processing and cohort creation
4. `src/map_phenotype.py` **Maps ICD codes into phecodes; does what phewas package in R does, uses the same mapping, just in python**
5. `src/dp_covariates.py` 
6. `src/dp_survey.py`
7. `src/create_study_cohort.py` 
### Script for modelling
8. `src/base_mdl_fitbit.py` 
9. `src/make_descriptives.py` **Generate descriptives for manuscript**
10. `src/format_outputs.py` **Format outputs for manuscript**


## Disclaimer
This repository does not contain any raw controlled data or intermediate processed data.
All analyses are conducted within a secure environment in accordance with the All of Us Research Program data governance policies.

## Acknowledgement
We gratefully acknowledge All of Us participants for their contributions, without whom this research would not have been possible. We also thank the National Institutes of Health’s All of Us Research Program for making available the participant data examined in this study.
