import pandas as pd
from typing import List
import numpy as np
from collections import Counter
from datetime import datetime
import yaml
import os
import re

def label_parent_phecode(phecodes: List):
    """
    Helper function to label whether elements in list are parents of other elements
    """
    is_parent = {}
    for i in phecodes:
        is_parent[i] = any([j.startswith(i) and j!=i for j in phecodes]) # this check whether the i-th element is a parent of any other element in the list
    return is_parent

# # Test the helper function
# phecodes = ['327.32', '327.3', '327']
# is_parent = label_parent_phecode(phecodes)
# is_parent

def to_float_or_int(num):
    if float(num).is_integer():
        return int(float(num))
    return float(num)

def remove_parent_phecodes(df):
    """
    Function to remove parent phecodes"""
    # remove trailing zero in phecode
    df['phecode_float'] = df['phecode'].apply(lambda x: str(to_float_or_int(x)))
    df_is_parent = df.groupby('person_id').apply(lambda x: pd.Series(label_parent_phecode(x['phecode_float']))).reset_index().rename(columns={0: 'is_parent',
                                    'level_1': 'phecode_float'})
    df = df.merge(df_is_parent, on=['person_id', 'phecode_float'], how='left')
    df = df.loc[df['is_parent']!=True].drop(columns=['is_parent', 'phecode_float'])
    return df

def get_sleep_drugs(fpath_drug, sleep_drugs_list, save_path = None):
    # Load prescription data and filter only sleep drugs
    i = 0
    sleep_drugs = pd.DataFrame()
    fpath = fpath_drug.format(i=i)
    while os.path.exists(fpath):
        print(f"Getting data from {fpath}")
        sleep_drugs_part = pd.read_csv(fpath)
        # Filter down on sleep drug list
        sleep_drugs_part = sleep_drugs_part[['person_id', 
                                            'drug_exposure_start_datetime', 
                                            'drug_exposure_end_datetime',
                                            'standard_concept_name']]
        sleep_drugs_part['drug_name'] = None
        # 1. Prepare a single regex pattern to match any drug in your list
        #    We use re.escape() to handle special chars, and join with '|'.
        pattern = r'(' + '|'.join(re.escape(d.lower()) for d in sleep_drugs_list) + r')'
        # 2. Lowercase your column once
        names_lower = sleep_drugs_part['standard_concept_name'].str.lower()
        # 3. Extract the matching drug into a new column (or NaN if no match)
        sleep_drugs_part['drug_name'] = names_lower.str.extract(pattern, expand=False)
        # 4. Drop rows where there was no match
        sleep_drugs_part = sleep_drugs_part.dropna(subset=['drug_name'])
        sleep_drugs = pd.concat([sleep_drugs, sleep_drugs_part], axis=0)
        # Increment i and fpath
        i += 1
        fpath = fpath_drug.format(i=i)
    
    if save_path:
        sleep_drugs.to_csv(save_path, index=False)
    
    return sleep_drugs

def create_sleep_cohort(config):
    """
    Main function for creating sleep cohort
    """

    ## fetch config
    cdf_path = config['condition_df']
    person_path = config['person_df']
    prescription_path = config['prescription_path']
    sleep_drugs_save_path = config['sleep_drugs_save_path']

    ## Get drugs that affect sleep
    with open(config['drug_list']) as f:
        drug_dict = yaml.safe_load(f)
    # Get list of drugs that affect sleeps
    sleep_drugs = []
    for d in config['sleep_affecting_drug_classes']:
        if d in drug_dict.keys():
            sleep_drugs.extend(drug_dict[d]) 
        else:
            print(f"{d} does not exist as a drug class in {config['drug_list']}")
    # Get prescription df relevant to only sleep-affecting drugs
    sleep_drug_df = get_sleep_drugs(prescription_path, sleep_drugs, sleep_drugs_save_path)
    print('Fetched sleep drug df')

    ## Read person df
    person_df = pd.read_csv(person_path)

    ## Read condition df and merge with person with genomics cohort
    cdf = pd.DataFrame()
    i = 0
    cdf_part_path = cdf_path.format(i=i)
    while os.path.exists(cdf_part_path):
        print(f"Getting data from {cdf_part_path}")
        # Only keep those in cohort to keep the sizes manageable
        cdf_part = pd.read_csv(cdf_part_path).merge(person_df[['person_id']])
        # convert condition start time to datetime
        cdf_part['condition_start_date'] = pd.to_datetime(cdf_part['condition_start_datetime'].str[0:10])
        cdf_part = cdf_part[['person_id', 'group', 'description', 'condition_start_date', 'phecode']].drop_duplicates()
        # concat
        cdf = pd.concat([cdf, cdf_part])
        # increment i and cdf path
        i+=1
        cdf_part_path = cdf_path.format(i=i)
    print('Fetched condition df')
    

    ############### Cohort creation ################
    ## 1. create sleep cohort (OSA and insomnia)
    # Keep only the sleep conditions
    cdf_sleep_full = cdf.loc[cdf['description'].isin(config['sleep_conditions_list'])]
    # Remove parent phecode to only take the most specific diagnosis
    cdf_sleep_full = remove_parent_phecodes(cdf_sleep_full)
    for subcohort in config['subcohorts']:
        # Further filter down to only include the subcohorts (OSA and insomnia)
        cdf_sleep = cdf_sleep_full.loc[cdf_sleep_full['description']==subcohort].copy()
        # Get the index date for each person
        index_df = cdf_sleep.groupby('person_id', as_index = False).apply(lambda x: x.min()['condition_start_date'])\
                        .rename(columns={None: 'condition_start_date'})
        # Merge on person id and index date to get primary diagnosis
        sleep_cohort = index_df.merge(cdf_sleep, on=['person_id', 'condition_start_date'], how = 'inner')\
                        .rename(columns={'condition_start_date': 'index_date', 
                                        'phecode': 'primary_phecode', 
                                        'description': 'primary_diag'})
        sleep_cohort = sleep_cohort[['person_id', 'primary_diag', 'index_date', 'primary_phecode']]
        # TODO create comorbidities?
        # TODO get the date where they get treated with a device/medication
        sleep_cohort.to_csv(config['sleep_cohort_save_path'].format(
                            subcohort=subcohort.lower().replace(' ', '_')), index=False)
        print(f"Saved sleep cohort {subcohort}")

    ## 2. create healthy control cohort
    ## apply exclusion criteria to filter patients based on condition
    person_exclude_conditions = []
    for col, vals in config['exclusion_conditions'].items():
        person_exclude_conditions.extend(cdf.loc[cdf[col].isin(vals), 'person_id'].tolist())
    
    ## control cohort
    control_cohort = person_df.loc[(~person_df['person_id'].isin(person_exclude_conditions)) \
                                & (~person_df['person_id'].isin(sleep_drug_df['person_id']))]
    control_cohort = control_cohort[['person_id']].drop_duplicates()
    control_cohort.to_csv(config['control_cohort_save_path'], index=False)

    print("Saved control cohort")


    return

def main():

    # fetch the configuration
    with open('conf/create_study_cohort.yaml') as f:
         config = yaml.safe_load(f)

    create_sleep_cohort(config['sleep_cohort'])

    
    return


if __name__ == '__main__':
    main()