import sys
sys.path.append('./')
import pandas as pd
from typing import List
import numpy as np
from collections import Counter
from datetime import datetime
import yaml
import os
import re
from lib.helpers import find_strict_streaks, filter_sleep_feat_by_streak
import warnings
warnings.filterwarnings("ignore")

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
        # print(f"Getting data from {fpath}")
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

def get_static_features(cohort, # Dataframe, should have index date
                        person_df,
                        person_features,
                        physical_measurement_df,
                        physical_measurement_features):

    ######### Demographic features #########
    # Compute age using sleep data collection date as a reference
    person_df = person_df.merge(cohort[['person_id', 'index_date']])
    person_df['age'] = ((pd.to_datetime(person_df['index_date']) - \
                         pd.to_datetime(person_df['date_of_birth'].str[0:10])).dt.days/365).astype(int)

    # Process the categorical variables so that they are numeric 
    # Race
    person_df.loc[person_df['race']=='White', 'race_white'] = 1
    person_df['race_white'] = person_df['race_white'].fillna(0)
    # Sex at birth
    person_df.loc[person_df['sex_at_birth']=='Male', 'sex_male'] = 1
    person_df['sex_male'] = person_df['sex_male'].fillna(0)

    person_df = person_df[['person_id'] + person_features]
    
    ######### Physical measurement features #########
    # Select one closest to sleep collection date
    physical_measurement_df = physical_measurement_df.merge(cohort)
    physical_measurement_df['diff'] = (pd.to_datetime(physical_measurement_df['index_date']) - \
                                        pd.to_datetime(physical_measurement_df['measurement_date'])).dt.days
    physical_measurement_df = physical_measurement_df.merge(
                                physical_measurement_df[['person_id', 'diff']].groupby('person_id', as_index=False).min(),
                                on = ['person_id', 'diff'])
    physical_measurement_df = physical_measurement_df[['person_id'] + physical_measurement_features]

    # Join everything together
    cohort = cohort.merge(person_df).merge(physical_measurement_df, how = 'left')

    return cohort

def exclude_by_icd(cohort, icd_path):
    # Get ICD codes to exclude
    with open(icd_path) as f:
        icd_excl = yaml.safe_load(f)

    # get a big list of exclusion for ICD 9 and 10
    icd10_excl = []
    for __cached__, v in icd_excl['ICD10'].items():
        icd10_excl.extend(list(v.values()))

    icd9_excl = []
    for __cached__, v in icd_excl['ICD9'].items():
        icd9_excl.extend(list(v.values()))

    # create a flag on whether to exclude these patients
    cohort.loc[(cohort['vocabulary_id']=='ICD10CM') & (cohort['code'].str.startswith(tuple(icd10_excl))), 'excl'] = True 
    cohort.loc[(cohort['vocabulary_id']=='ICD9CM') & (cohort['code'].str.startswith(tuple(icd9_excl))), 'excl'] = True 

    # exclude these patients
    excl_persons = cohort.loc[cohort['excl'] == True, 'person_id']
    cohort = cohort[~cohort['person_id'].isin(excl_persons)]

    cohort = cohort.drop(columns = ['vocabulary_id', 'code', 'excl']).drop_duplicates()

    return cohort

def filter_streak_by_age(streaks, person_df):

    streaks['streak_start'] = pd.to_datetime(streaks['streak_start'])
    streaks['streak_end'] = pd.to_datetime(streaks['streak_end'])
    person_df['date_of_birth'] = pd.to_datetime(person_df['date_of_birth'].str[0:10])
    streaks = streaks.merge(person_df[['person_id', 'date_of_birth']])
    # date after which the person turns 18
    streaks['cutoff'] = streaks['date_of_birth'] + pd.DateOffset(years= 18)
    # case where person turns 18 in the middle of a streak --> break this up TODO check code
    streaks.loc[(streaks['streak_end'] >= streaks['cutoff']) & \
                (streaks['streak_start'] < streaks['cutoff']), 'break_streak'] = True
    streaks.loc[streaks['break_streak'], 'streak_start'] = streaks.loc[streaks['break_streak'], 'cutoff']
    streaks.loc[streaks['break_streak'], 'length'] = (streaks.loc[streaks['break_streak'], 'streak_end'] \
                                                    - streaks.loc[streaks['break_streak'], 'streak_end']).dt.days
    # for the rest --> filter by streak start
    streaks = streaks.loc[streaks['streak_start'] >= streaks['cutoff']]

    return streaks

def create_cohort(config):
    """
    Main function for creating cohort
    """

    ## fetch config
    cdf_path = config['condition_df']
    person_path = config['person_df']
    prescription_path = config['prescription_path']

    ######## Read person df ########
    person_df = pd.read_csv(person_path)

    ############### Read condition df and merge with person df ###############
    cdf = pd.DataFrame()
    i = 0
    cdf_part_path = cdf_path.format(i=i)
    while os.path.exists(cdf_part_path):
        print(f"Getting data from {cdf_part_path}")
        # Only keep those in cohort to keep the sizes manageable
        cdf_part = pd.read_csv(cdf_part_path).merge(person_df[['person_id']])
        # convert condition start time to datetime
        cdf_part['condition_start_date'] = pd.to_datetime(cdf_part['condition_start_datetime'].str[0:10])
        cdf_part = cdf_part[['person_id', 'group', 'description', 'condition_start_date', 
                             'phecode', 'code', 'vocabulary_id']].drop_duplicates()
        # concat
        cdf = pd.concat([cdf, cdf_part])
        # increment i and cdf path
        i+=1
        cdf_part_path = cdf_path.format(i=i)
    print('Fetched condition df')
    
    ############### Cohort creation ################
    # Remove parent phecode to only take the most specific diagnosis
    cdf = remove_parent_phecodes(cdf)
    print('Removed parent phecodes')
    # cdf.to_csv('data/02_intermediate/study_cohorts_20251201/condition_df.csv', index=False)
    # map condition description into subcohorts
    cdf['subcohort'] = cdf['description'].map(config['condition_map'])

    ########## patients with a condition #############
    cohort = cdf.loc[~cdf['subcohort'].isna(), ['person_id', 'subcohort', 'condition_start_date']].drop_duplicates().copy()
    # For each condition, take the earliest date as index date
    cohort = cohort.groupby(['person_id', 'subcohort'], as_index=False).min()
    # Make the value at each position the diagnosis date
    cohort = pd.get_dummies(cohort, columns = ['subcohort'])
    subcohort_cols = [x for x in cohort.columns if x.startswith('subcohort_')]
    for col in subcohort_cols:
        cohort.loc[cohort[col]==True, col] = cohort['condition_start_date'].astype(str)
        cohort.loc[cohort[col]==False, col] = ''
    cohort = cohort.drop(columns = ['condition_start_date'])
    print(cohort.head(5))
    # aggregate
    cohort = cohort.groupby('person_id', as_index=False).sum()

    ########### control cohort ###########
    control_cohort = cdf.loc[~cdf['person_id'].isin(cohort['person_id'])]
    print(f"Control cohort size = {control_cohort['person_id'].nunique()} without any subcohort assignment")
    # # filter by additional exclusion conditions (diagnosis)
    # for col, vals in config['additional_exclusion_map'].items():
    #     control_cohort = control_cohort.loc[~control_cohort[col].isin(vals)]
    # print(f"Control cohort size = {control_cohort['person_id'].nunique()} after filtering for additional exclusion conditions")
    # filter by ICD codes 
    control_cohort = exclude_by_icd(control_cohort, config['icd_excl'])
    # only keep person id column and drop duplicates
    control_cohort = control_cohort[['person_id']].drop_duplicates()
    print(f"Control cohort size = {control_cohort['person_id'].nunique()} after filtering for additional exclusion using ICD codes")

    ## TODO filter by prescription and device 
    ####### Get drugs that affect sleep or heart rate for exclusion in control cohort ########
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
    excl_drug_df = get_excl_drugs(prescription_path, excl_drugs, save_path = config['excl_drug_savepath'])

    # TODO refine filtering criteria
    control_cohort = control_cohort.loc[~control_cohort['person_id'].isin(excl_drug_df['person_id'])]
    print(f"Control cohort size = {control_cohort.shape[0]} after filtering for sleep and HR medications")
    ###### Get OSA device df for exclusion in control cohort ############
    device_df = pd.read_csv(config['osa_device_df'])
    control_cohort = control_cohort.loc[~control_cohort['person_id'].isin(device_df['person_id'])]
    print(f"Control cohort size = {control_cohort.shape[0]} after filtering for OSA device df")


    # get fitbit summary data and filter by streak length
    streaks, fitbit_summary = process_fitbit_summary(config['fitbit_summary_data'],
                                                     person_df,
                                                    min_streak_length = config['min_streak_length'])



    ######## Get index date ##########
    # Get the earliest sleep data collection date as index date for control cohort
    control_cohort = control_cohort.merge(
                    streaks[['person_id', 'streak_start']].groupby('person_id', as_index=False)\
                    .rename(columns = {'streak_start': 'index_date'}))

    # Get the earliest diagnosis date as index date for cohort with diagnosis
    for col in subcohort_cols:
        cohort.loc[cohort[col]=='', col] = np.nan
        cohort[col] = pd.to_datetime(cohort[col], format = '%Y-%m-%d', errors = 'coerce')
    cohort['index_date'] = cohort[subcohort_cols].min(axis=0)

    ###### Concatenate cohort with diagnosis with control cohort ######
    cohort = pd.concat([cohort, control_cohort], axis = 0)
    # select only people with sleep streaks
    cohort = cohort.loc[cohort['person_id'].isin(streaks['person_id'])]
    print(f"Cohort size = {cohort.shape[0]} after filtering for 21 days of sleep streaks")

    #### NOTE optional Get the streak closest to index date but after index date
    streaks = streaks.merge([['person_id', 'index_date']])
    streaks = streaks.loc[streaks['streak_start'] >= streaks['index_date']]
    # select only people with sleep streaks after index date
    cohort = cohort.loc[cohort['person_id'].isin(streaks['person_id'])]
    print(f"Cohort size = {cohort.shape[0]} after filtering for 21 days of sleep streaks after index date")

    # make sure only one line per person
    assert cohort.shape[0] == cohort['person_id'].nunique()

    # print cohort sizess
    for col in subcohort_cols:
        print(f"Subcohort {col.replace('subcohort_', '')}: {cohort.loc[cohort[col].notna()].shape[0]}")
    print(f"Control cohort: {cohort.loc[(cohort[subcohort_cols].isna()).all(axis=1)].shape[0]}")

    ####### Get static features #######
    cohort = get_static_features(cohort,
                                        person_df,
                                        config['person_features'],
                                        pd.read_csv(config['physical_measurement_df']),
                                        config['physical_measurement_features'])
    

    ####### Get time series features #######
    # Find earliest streak and filter fitbit summary data by streak
    streaks = streaks.merge(streaks[['person_id', 'streak_start']].groupby('person_id', as_index=False).min(),
                            on = ['person_id', 'streak_start'])
    fitbit_summary = filter_sleep_feat_by_streak(fitbit_summary, streaks, date_col = 'date')
    # TODO delete, just to inspect the fitbit summary data
    fitbit_summary.to_csv('data/02_intermediate/study_cohorts_20251201/fitbit_summary_filtered.csv')
    # compute summary statistics for the fitbit summary data and merge with cohort
    time_series_features = config['time_series_features']
    fitbit_summary = fitbit_summary[['person_id'] + time_series_features].groupby('person_id').agg(
                            {c: ['mean', 'std'] for c in time_series_features}
                        )
    # flatten multi-index column
    fitbit_summary.columns = [f"{col}_{stat}" for col, stat in fitbit_summary.columns]
    fitbit_summary = fitbit_summary.reset_index()

    cohort = cohort.merge(fitbit_summary, on = 'person_id')

    # TODO get additional sleep features, refer to make_transformer_dataset_fitbit_summary


    ######## Save cohort #######
    cohort_save_path = config['cohort_save_path']
    if not os.path.exists(os.path.dirname(cohort_save_path)):
        os.makedirs(os.path.dirname(cohort_save_path))
    cohort.to_csv(cohort_save_path, index=False)

    return

def process_fitbit_summary(fpath, person_df, min_streak_length = 21, streak_save_path = None):
    # read fitbit summary data and formatting
    df = pd.read_parquet(fpath)
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
    df.loc[df['is_main_sleep']=='false', 'is_main_sleep'] = False
    df.loc[df['is_main_sleep']=='true', 'is_main_sleep'] = True

    # Label non main sleep as nap
    df = df.loc[df['is_main_sleep']==True].merge(
                df.loc[df['is_main_sleep']==False, ['person_id', 'date', 'minute_in_bed']]\
                    .rename(columns = {'minute_in_bed': 'nap_min'}),
                how = 'left')
    
    # fillna for nap min
    df['nap_min'] = df['nap_min'].fillna(0)
    df = df.drop(columns = ['minute_restless', 'is_main_sleep'])
    assert df.shape[0] == df[['person_id', 'date']].drop_duplicates().shape[0]
    # discard some columns further
    df = df.drop(columns = ['calories_bmr', 'elevation', 'floors'])
    df['very_active_minutes'] = df['very_active_minutes'].fillna(0)
    df = df.dropna()
    
    # Find streaks and filter sleep by streaks
    streaks = find_strict_streaks(df, date_col = 'date', min_length = min_streak_length)
    # Filter streaks when >= 18 years old by getting DOB from person df
    streaks = filter_streak_by_age(streaks, person_df)
    # TODO determine which streak to use
    df = filter_sleep_feat_by_streak(df, streaks, date_col = 'date')

    if not streak_save_path:
        streaks.to_csv(streak_save_path, index=False)

    return streaks, df


def get_features(cohort):
    pass

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