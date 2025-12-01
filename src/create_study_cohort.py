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
        # this check whether the i-th element is a parent of any other element in the list
        is_parent[i] = any([j.startswith(i) and j!=i for j in phecodes]) 
    return is_parent

def to_float_or_int(num):
    """
    Return int if number if an integer otherwise float
    """
    if float(num).is_integer():
        return int(float(num))
    return float(num)

def remove_parent_phecodes(df):
    """
    Function to remove parent phecodes to only keep the most
    specific diagnosis
    """
    # remove trailing zero in phecode
    df['phecode_float'] = df['phecode'].apply(lambda x: str(to_float_or_int(x)))
    df_is_parent = df.groupby('person_id').apply(lambda x: pd.Series(label_parent_phecode(x['phecode_float']))).reset_index().rename(columns={0: 'is_parent',
                                    'level_1': 'phecode_float'})
    df = df.merge(df_is_parent, on=['person_id', 'phecode_float'], how='left')
    df = df.loc[df['is_parent']!=True].drop(columns=['is_parent', 'phecode_float'])
    return df

def get_excl_drugs(fpath_drug, sleep_drugs_list, save_path = None):
    """
    Function to get drugs to be excluded
    Also includes some engineering because the source value
    of the drug names also include dosage, etc
    """
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

def create_cohort(config):
    """
    Main function for creating cohort
    """

    ## fetch config
    cdf_path = config['condition_df']
    person_path = config['person_df']
    prescription_path = config['prescription_path']

    ## Get drugs that affect sleep or heart rate
    with open(config['drug_list']) as f:
        drug_dict = yaml.safe_load(f)
    # Get list of drugs that affect sleep and heart rate
    excl_drugs = []
    for d in config['excl_drug_classes']:
        if d in drug_dict.keys():
            excl_drugs.extend(drug_dict[d]) 
        else:
            print(f"{d} does not exist as a drug class in {config['drug_list']}")
    # Get prescription df relevant to only sleep-affecting drugs
    excl_drug_df = get_excl_drugs(prescription_path, excl_drugs)
    print('Fetched drugs affecting sleep and heart rate')

    ## TODO get OSA device df
    device_df = pd.read_csv(config['osa_device_df'])

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
    
    ############### Cohort creation ################ TODO change everything
    # Remove parent phecode to only take the most specific diagnosis
    cdf = remove_parent_phecodes(cdf)
    # map condition description into subcohorts
    cdf['subcohort'] = cdf['description'].map(config['condition_map'])
    # separate out patients with a condition
    cohort = cdf.loc[~cohort['subcohort'].isna(), ['person_id', 'subcohort', 'condition_start_date']].copy()
    # TODO copy from make_transformer_dataset: make the value at each position the diagnosis date


    # Create empty dataframe to concat everything
    cohort = pd.DataFrame()
    for cond, subcohort in config['condition_map'].items():

        # Further filter down to only include the subcohort
        cdf_sleep = cdf_sleep_full.loc[cdf_sleep_full['description']==subcohort].copy()
        if cdf_sleep.empty:
            # print(f"Np patient with condition {subcohort}, skipping..")
            # continue
            raise Exception(f"There is no patient with diagnosis {subcohort}, please check the condition list and spelling")
        # Get the index date for each person (earliest diagnosis)

        index_df = cdf_sleep.groupby('person_id', as_index = False).apply(lambda x: x.min()['condition_start_date'])\
                        .rename(columns={None: 'condition_start_date'})
        
        # Merge on person id to get primary diagnosis
        sleep_cohort = index_df.merge(cdf_sleep, on=['person_id', 'condition_start_date'], how = 'inner')\
                        .rename(columns={'condition_start_date': 'index_date', 
                                        'phecode': 'primary_phecode', 
                                        'description': 'primary_diag'})

        # sleep_cohort = index_df.merge(cdf_sleep[['person_id', 'phecode', 'description']], 
        #                               on='person_id', how = 'inner')\
        #                                 .rename(columns={'condition_start_date': 'index_date', 
        #                                                 'phecode': 'primary_phecode', 
        #                                                 'description': 'primary_diag'})

        sleep_cohort = sleep_cohort[['person_id', 'primary_diag', 'index_date']]
        # Rename subcohort
        sleep_cohort['subcohort'] = subcohort_rename
        # TODO create comorbidities?
        # TODO get the date where they get treated with a device/medication
        cohort = pd.concat([cohort, sleep_cohort], axis=0)
        print(f"Created sleep cohort {subcohort_rename}")
    
    # Combine patients with two or more primary diagnoses
    cohort_index = cohort[['person_id', 'index_date']]\
                        .groupby('person_id', as_index=False).min()
    cohort = cohort[['person_id', 'subcohort']]\
                    .groupby('person_id').apply(\
                    lambda x: ' + '.join(sorted(x['subcohort'].unique().tolist())))\
                    .reset_index().rename(columns={0: 'subcohort'})

    cohort = cohort.merge(cohort_index, on = 'person_id')

    ## 2. create healthy control cohort
    ## apply exclusion criteria to filter patients based on condition
    person_exclude_conditions = []
    for col, vals in config['exclusion_conditions'].items():
        person_exclude_conditions.extend(cdf.loc[cdf[col].isin(vals), 'person_id'].tolist())
    
    ## control cohort
    control_cohort = person_df.loc[(~person_df['person_id'].isin(person_exclude_conditions)) \
                                & (~person_df['person_id'].isin(sleep_drug_df['person_id']))]
    control_cohort = control_cohort[['person_id']].drop_duplicates()
    control_cohort['subcohort'] = 'Control'

    cohort = pd.concat([cohort, control_cohort], axis=0)

    print("Created control cohort")

    print(cohort.groupby('subcohort').count())

    cohort_save_path = config['cohort_save_path']
    if not os.path.exists(os.path.dirname(cohort_save_path)):
        os.makedirs(os.path.dirname(cohort_save_path))
    cohort.to_csv(cohort_save_path, index=False)

    return

def main():

    # fetch the configuration
    with open('conf/create_study_cohort.yaml') as f:
         config = yaml.safe_load(f)

    for cohort, cohort_conf in config.items():
        if cohort_conf['run']:
            print(f"Creating {cohort}\n......")
            create_cohort(cohort_conf)

    
    return


if __name__ == '__main__':
    main()