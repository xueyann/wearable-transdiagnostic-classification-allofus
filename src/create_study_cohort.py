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
from lib.helpers import find_strict_streaks, filter_sleep_feat_by_streak_memory_efficient, rmssd
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

def get_time_series_sleep_features(pattern,
                                    streaks, # dataframe that contains streak information
                                    sleep_onset_center = 22, # For centering sleep onset wrt a reference time, default = 10pm,
                                    features = None
                                    ):

    # Filter sleep df by streaks
    i = 0
    filtered_sleep_feat = []
    while os.path.exists(pattern.format(i=i)):
        # read the sleep feature data
        sleep_feat = pd.read_csv(pattern.format(i=i)).dropna()
        if features is not None:
            sleep_feat = sleep_feat[['person_id', 'sleep_date'] + features]
        sleep_feat = sleep_feat.dropna()
        # Adjustment on sleep onset time
        sleep_feat['sleep_onset_adjusted'] = sleep_feat['sleep_onset'].astype(float) - sleep_onset_center
        sleep_feat.loc[sleep_feat['sleep_onset_adjusted'] > 12, 'sleep_onset_adjusted'] = 24 - sleep_feat.loc[sleep_feat['sleep_onset_adjusted'] > 12, 'sleep_onset_adjusted'].astype(float)
        sleep_feat.loc[sleep_feat['sleep_onset_adjusted'] < -12, 'sleep_onset_adjusted'] = sleep_feat.loc[sleep_feat['sleep_onset_adjusted'] < -12, 'sleep_onset_adjusted'].astype(float) + 24
        # Drop unnecessary columns
        sleep_feat = sleep_feat.drop(columns = ['sleepless', 'sleep_onset'], errors = 'ignore')
        
        # Filter by streaks
        filtered_sleep_feat.append(filter_sleep_feat_by_streak_memory_efficient(sleep_feat, streaks))
        i += 1
    
    sleep_data = pd.concat(filtered_sleep_feat, axis=0).rename(columns = {'sleep_date': 'date'})

    return sleep_data


def get_static_features(cohort, # Dataframe, should have index date
                        person_df,
                        person_features,
                        physical_measurement_df,
                        physical_measurement_features):

    ######### Demographic features #########
    # Compute age using sleep data collection date as a reference
    # make sure to drop people missing key demographic features
    person_df = person_df.dropna(subset = ['person_id', 'date_of_birth', 'race', 'sex_at_birth'])
    person_df = person_df.merge(cohort[['person_id', 'index_date']])
    person_df['age'] = ((pd.to_datetime(person_df['index_date']) - \
                         pd.to_datetime(person_df['date_of_birth'].astype(str).str[0:10])).dt.days/365).astype(int)

    # Process the categorical variables so that they are numeric 
    # Race
    person_df.loc[person_df['race']!='White', 'race_nonwhite'] = 1
    person_df['race_nonwhite'] = person_df['race_nonwhite'].fillna(0)
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
    physical_measurement_df = physical_measurement_df.dropna()

    # Join everything together
    cohort = cohort.merge(person_df).merge(physical_measurement_df, how = 'inner')
    print(f"Total cohort size after dropping people missing key demographic variables: {cohort.shape[0]}")

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
    streaks['break_streak'] = False
    streaks.loc[(streaks['streak_end'] >= streaks['cutoff']) & \
                (streaks['streak_start'] < streaks['cutoff']), 'break_streak'] = True
    streaks.loc[streaks['break_streak'], 'streak_start'] = streaks.loc[streaks['break_streak'], 'cutoff']
    streaks.loc[streaks['break_streak'], 'length'] = (streaks.loc[streaks['break_streak'], 'streak_end'] \
                                                    - streaks.loc[streaks['break_streak'], 'streak_end']).dt.days
    # for the rest --> filter by streak start
    streaks = streaks.loc[streaks['streak_start'] >= streaks['cutoff']]

    return streaks

def get_excl_drug_minmax(config):
    ####### Get drugs that affect sleep or heart rate for exclusion in control cohort ########
    if config['excl_drug_df_available']:
        excl_drug_df = pd.read_csv(config['excl_drug_savepath'])
    else:
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
        excl_drug_df = get_excl_drugs(config['prescription_path'], 
                                      excl_drugs, 
                                      save_path = config['excl_drug_savepath'])
        # free up some memory
        del excl_drugs

    excl_drug_df['drug_exposure_start_datetime'] = pd.to_datetime(excl_drug_df['drug_exposure_start_datetime'].astype(str).str[0:10])
    excl_drug_df['drug_exposure_end_datetime'] = pd.to_datetime(excl_drug_df['drug_exposure_end_datetime'].astype(str).str[0:10])
    # take care of NA values for either exposure start or end datetime
    excl_drug_df.loc[excl_drug_df['drug_exposure_end_datetime'].isna(), 'drug_exposure_end_datetime'] = \
        excl_drug_df.loc[excl_drug_df['drug_exposure_end_datetime'].isna(), 'drug_exposure_start_datetime']
    excl_drug_df.loc[excl_drug_df['drug_exposure_start_datetime'].isna(), 'drug_exposure_start_datetime'] = \
        excl_drug_df.loc[excl_drug_df['drug_exposure_start_datetime'].isna(), 'drug_exposure_end_datetime']
    
    # take the min and max of drug exposure
    excl_drug_minmax = excl_drug_df[['person_id', 'drug_exposure_start_datetime']]\
                        .groupby('person_id', as_index=False).min().rename(
                        columns = {'drug_exposure_start_datetime': 'drug_min'}).merge(
                        excl_drug_df[['person_id', 'drug_exposure_end_datetime']]\
                        .groupby('person_id', as_index=False).max().rename(
                        columns = {'drug_exposure_end_datetime': 'drug_max'})
    )

    del excl_drug_df
    
    return excl_drug_minmax

def get_cond_df(config, person_df):

    cdf_path = config['condition_df']

    if config['parent_phecode_removed_avail']:
        cdf = pd.read_parquet(config['parent_phecode_removed_df'])
    else: # read raw condition df and remove parent phecode
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
    
        # Remove parent phecode to only take the most specific diagnosis
        cdf = remove_parent_phecodes(cdf)
        print('Removed parent phecodes')
        cdf.to_parquet(config['parent_phecode_removed_df'])
    
    return cdf


def filter_by_recency(cohort, streaks, diag, recency_filter):

    # get minmax of streaks
    streaks_mm = streaks[['person_id', 'streak_start']].groupby('person_id', as_index=False)\
                    .agg(['min', 'max'])
    streaks_mm.columns = ['_'.join(col).replace('person_id_', 'person_id') for col in streaks_mm.columns]
    streaks_mm = streaks_mm[['person_id', 'streak_start_min', 'streak_start_max']]

    # get min amd max of diagnosis date for each condition
    diag_mm = diag.groupby(['person_id', 'subcohort']).agg(['min', 'max'])
    diag_mm.columns = ['_'.join(map(str, col)) for col in diag_mm.columns]
    diag_mm = diag_mm.rename(columns = {'condition_start_date_min': 'diag_min',
                                        'condition_start_date_max': 'diag_max'}).reset_index()
    
    df_t = streaks_mm.merge(diag_mm)

    # create a dataframe with just recent diagnosis
    df_rec = []
    for cond, max_years in recency_filter.items():
        df_rec.append(df_t.loc[(df_t['subcohort']==cond) & (df_t['recency'] <= max_years)])
        # also get the numbers
        n_ori = df_t.loc[(df_t['subcohort']==cond)].shape[0]
        n_filtered = df_t.loc[(df_t['subcohort']==cond) & (df_t['recency'] <= max_years)].shape[0]
        print(f"{cond}: original cohort size = {n_ori}, filtered cohort_size = {n_filtered}")
    df_rec = pd.concat(df_rec)

    # Use recency df to filter cohort
    cohort = cohort.merge(df_rec[['person_id', 'subcohort']])

    return cohort
    


def create_cohort(config):
    """
    Main function for creating cohort
    """

    # create dictionary to record cohort attrition
    attrition = {}

    ## fetch config
    person_path = config['person_df']

    ######## Read person df ########
    person_df = pd.read_csv(person_path)
    # get the min and max of drugs that affect hr and sleep
    excl_drug_minmax = get_excl_drug_minmax(config)
    # get devices prescribed for OSA
    device_df = pd.read_csv(config['osa_device_df'])

    ######## get fitbit summary data and filter by streak length ########
    streaks, fitbit_summary, attrition = process_fitbit_summary(config['fitbit_summary_data'],
                                                    person_df,
                                                    attrition,
                                                    excl_drug_minmax,
                                                    device_df,
                                                    config['filter_by_medication'],
                                                    config['filter_by_osa_device'],
                                                    min_streak_length = config['min_streak_length'],
                                                    streak_save_path = config['streak_save_path'])

    ############### Read condition df and merge with person df ###############
    cdf = get_cond_df(config, person_df)
    # filter cdf by sleep streaks
    cdf = cdf.loc[cdf['person_id'].isin(streaks['person_id'])]

    ############### Cohort creation ################
    # map condition description into subcohorts
    cdf['subcohort'] = cdf['description'].map(config['condition_map'])
    cdf.to_csv(config['diag_alltime_savepath'])

    ########## patients with a condition #############
    cohort = cdf.loc[~cdf['subcohort'].isna(), 
                     ['person_id', 'subcohort', 'condition_start_date']].drop_duplicates().copy()
    ########## create control cohort ##############
    control_cohort = cdf.loc[~cdf['person_id'].isin(cohort['person_id'])]
    print(f"Control cohort size = {control_cohort['person_id'].nunique()} without any lifetime subcohort diagnosis")
    # # filter by additional exclusion conditions (diagnosis)
    # for col, vals in config['additional_exclusion_map'].items():
    #     control_cohort = control_cohort.loc[~control_cohort[col].isin(vals)]
    # print(f"Control cohort size = {control_cohort['person_id'].nunique()} after filtering for additional exclusion conditions")
    # filter by ICD codes 
    control_cohort = exclude_by_icd(control_cohort, config['icd_excl'])
    # only keep person id column and drop duplicates
    control_cohort = control_cohort[['person_id']].drop_duplicates()
    print(f"Control cohort size = {control_cohort['person_id'].nunique()} after filtering for additional exclusion using ICD codes")

    ######### diagnosis recency filter #########
    # Include only recent diagnosis
    cohort = filter_by_recency(cohort, streaks, cdf, config['recency_filter'])
    # For each condition, take the earliest date as index date
    cohort = cohort.groupby(['person_id', 'subcohort'], as_index=False).min()
    # Make the value at each position the diagnosis date
    cohort = pd.get_dummies(cohort, columns = ['subcohort'])
    subcohort_cols = [x for x in cohort.columns if x.startswith('subcohort_')]
    for col in subcohort_cols:
        cohort.loc[cohort[col]==True, col] = cohort['condition_start_date'].astype(str)
        cohort.loc[cohort[col]==False, col] = ''
    cohort = cohort.drop(columns = ['condition_start_date'])
    # aggregate
    cohort = cohort.groupby('person_id', as_index=False).sum()

    ########### control cohort ###########
    

    ########### the original strict filtering criteria ##############
    # control_cohort = control_cohort.loc[~control_cohort['person_id'].isin(excl_drug_df['person_id'])]
    # print(f"Control cohort size = {control_cohort.shape[0]} after filtering for sleep and HR medications")
    ###### Get OSA device df for exclusion in control cohort ############
    # device_df = pd.read_csv(config['osa_device_df'])
    # control_cohort = control_cohort.loc[~control_cohort['person_id'].isin(device_df['person_id'])]
    # print(f"Control cohort size = {control_cohort.shape[0]} after filtering for OSA device df")

    
    del cdf
    del device_df

    ###### Concatenate subcohorts with diagnosis with control cohort ######
    cohort = pd.concat([cohort, control_cohort], axis = 0)

    ######## Get index date ##########
    # Get the earliest sleep data collection date as index date for all cohort
    cohort = cohort.merge(streaks[['person_id', 'streak_start']].groupby('person_id', as_index=False).min()\
                    .rename(columns = {'streak_start': 'index_date'}))

    # # Get the earliest diagnosis date as index date for cohort with diagnosis
    # for col in subcohort_cols:
    #     cohort.loc[cohort[col]=='', col] = np.nan
    #     cohort[col] = pd.to_datetime(cohort[col], format = '%Y-%m-%d', errors = 'coerce')
    # cohort['index_date'] = cohort[subcohort_cols].min(axis=1)


    # # select only people with sleep streaks
    # cohort = cohort.loc[cohort['person_id'].isin(streaks['person_id'])]
    # print(f"Cohort size = {cohort.shape[0]} after filtering for 21 days of sleep streaks")
    attrition['either a diagnosis assigned or in control cohort'] = cohort.shape[0]

    #### NOTE optional Get the streak closest to index date but after index date
    if config['streak_overlap']:
        streaks = streaks.merge(cohort[['person_id', 'index_date']])
        streaks = streaks.loc[streaks['streak_end'] >= streaks['index_date']]
        streaks = streaks
        # select only people with sleep streaks after index date
        cohort = cohort.loc[cohort['person_id'].isin(streaks['person_id'])]
        attrition['sleep streak overlap with formal diagnosis'] = cohort.shape[0]
        print(attrition)

    # make sure only one line per person
    assert cohort.shape[0] == cohort['person_id'].nunique()

    # print cohort sizes
    for col in subcohort_cols:
        print(f"Subcohort {col.replace('subcohort_', '')}: {cohort.loc[cohort[col].notna()].shape[0]}")
    print(f"Control cohort: {cohort.loc[(cohort[subcohort_cols].isna()).all(axis=1)].shape[0]}")

    ####### Get static features #######
    cohort = get_static_features(cohort,
                            person_df,
                            config['person_features'],
                            pd.read_csv(config['physical_measurement_df']),
                            config['physical_measurement_features'])
    attrition['Have all key demographic features'] = cohort.shape[0]
    print(attrition)
    
    ######## Save cohort together with static features #######
    cohort_save_path = config['cohort_save_path']
    if not os.path.exists(os.path.dirname(cohort_save_path)):
        os.makedirs(os.path.dirname(cohort_save_path))
    cohort.to_csv(cohort_save_path, index=False)
    # save attrition
    pd.Series(attrition).to_csv(config['attrition_savepath'], index=False)

    return cohort, streaks, fitbit_summary


def get_time_series_features(cohort, streaks, fitbit_summary, config):
    """
    Get time series features, multiple rows per person to preserve the temporal element
    
    :param cohort: Description
    :param streaks: Description
    :param fitbit_summary: Description
    :param config: Description
    """

    # select only person ids inside the cohort
    streaks = streaks.loc[streaks['person_id'].isin(cohort['person_id'])]
    
    ####### Get time series features #######
    # Find earliest streak and filter fitbit summary data by streak
    streaks = streaks.merge(streaks[['person_id', 'streak_start']].groupby('person_id', as_index=False).min(),
                            on = ['person_id', 'streak_start'])
    fitbit_summary = filter_sleep_feat_by_streak_memory_efficient(fitbit_summary, streaks, date_col = 'date')

    # include just time series features
    time_series_features = config['time_series_features']
    print(fitbit_summary.columns)
    fitbit_summary = fitbit_summary[['person_id', 'date'] + time_series_features]

    ####### Get additional sleep features #######
    if config['get_additional_sleep_feat']:

        sleep_df = get_time_series_sleep_features(config['additional_sleep_features_path_pattern'],
                                                streaks, # dataframe that contains streak information
                                                # For centering sleep onset wrt a reference time, default = 10pm,
                                                sleep_onset_center = config['sleep_onset_center'], 
                                                features = ['sleep_onset'] + \
                                                    [x for x in config['additional_sleep_features'] if x != 'sleep_onset_adjusted'])
        # sleep_df = pd.read_csv(config['additional_sleep_features_path'])
        # sleep_df = sleep_df[['person_id', 'sleep_date'] + config['additional_sleep_features']].rename(
        #     columns = {'sleep_date': 'date'}
        # )
        fitbit_summary = fitbit_summary.merge(sleep_df, on = ['person_id', 'date'], how = 'left')
        # check for any na values
        fitbit_summary.loc[fitbit_summary.isna().any(axis=1)].to_csv('data/02_intermediate/study_cohorts_20251201/additional_sleep_feat_na.csv')

    # save time series features
    fitbit_summary.to_csv(config['time_series_features_savepath'], index=False)
    # get sumstats
    fitbit_summary = get_fitbit_sumstats(fitbit_summary)

    fitbit_summary.to_csv(config['time_series_features_sumstats_savepath'], index=False)
    
    return

def get_fitbit_sumstats(fitbit_summary, time_series_features, 
                                        rmssd_features = ['minute_in_bed',
                                                        'deep_perc',
                                                        'rem_perc',
                                                        'wake_perc']):
    ######### compute summary stats and save #########
    fitbit1 = fitbit_summary[['person_id'] + time_series_features].groupby('person_id').agg(
                            {c: ['mean', 'std'] for c in time_series_features})
    fitbit2 = fitbit_summary.groupby('person_id')[rmssd_features].agg(rmssd)
    # flatten multi-index column
    fitbit1.columns = [f"{col}_{stat}" for col, stat in fitbit1.columns]
    fitbit2.columns = [f"{col}_rmssd" for col in fitbit2.columns]
    fitbit1 = fitbit1.reset_index()
    fitbit2 = fitbit2.reset_index()

    return fitbit1.merge(fitbit2)

def process_fitbit_summary(fpath, person_df, attrition, excl_drug_minmax, osa_device_df,
                           filter_by_medication = True, filter_by_osa_device = True,
                           min_streak_length = 21, streak_save_path = None):
    # read fitbit summary data and formatting
    df = pd.read_parquet(fpath)
    # some feature engineering
    df['sleep_efficiency'] = df['minute_asleep']/df['minute_in_bed']
    df['deep_perc'] = df['minute_deep']/df['minute_in_bed']
    df['rem_perc'] = df['minute_rem']/df['minute_in_bed']
    df['wake_perc'] = df['minute_wake']/df['minute_in_bed']
    df['fat_burn_hr_to_light_active_ratio'] = df['fat_burn_min']/df['lightly_active_minutes']
    df['cardio_hr_to_fairly_active_ratio'] = df['cardio_min']/df['fairly_active_minutes']
    df['peak_hr_to_very_active_ratio'] = df['peak_min']/df['very_active_min']

    attrition['Having fitbit data'] = df['person_id'].nunique()
    # filter for df in person df
    df = df.loc[df['person_id'].isin(person_df['person_id'])]
    attrition['fitbit_and_ehr'] = person_df['person_id'].nunique()
    print(attrition)

    # discard some columns that we are not going to use as features
    df = df.drop(columns = ['calories_bmr', 'elevation', 'floors', 'minute_restless'])
    print('read fitbit summary')
    df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
    df.loc[df['is_main_sleep']=='false', 'is_main_sleep'] = False
    df.loc[df['is_main_sleep']=='true', 'is_main_sleep'] = True

    print('converted main sleep false to nap')

    # Label non main sleep as nap
    df = df.loc[df['is_main_sleep']==True].merge(
                df.loc[df['is_main_sleep']==False, ['person_id', 'date', 'minute_in_bed']]\
                    .rename(columns = {'minute_in_bed': 'nap_min'}),
                how = 'left')
    df = df.drop(columns = ['is_main_sleep'])

    print('labelled condition df')
    
    # fillna for nap min and active minutes
    df['nap_min'] = df['nap_min'].fillna(0)
    df['very_active_minutes'] = df['very_active_minutes'].fillna(0)
    df = df.dropna()

    # at this point we should have just one row per person
    assert df.shape[0] == df[['person_id', 'date']].drop_duplicates().shape[0]
    
    # Find streaks and filter sleep by streaks
    streaks = find_strict_streaks(df, date_col = 'date', min_length = min_streak_length)

    attrition[f'{min_streak_length} days of FitBit data streak'] = streaks['person_id'].nunique()
    print(attrition)
    
    # Filter streaks when >= 18 years old by getting DOB from person df
    streaks = filter_streak_by_age(streaks, person_df)
    attrition['sleep data on or after 18 years old'] = streaks['person_id'].nunique()
    print(attrition)
    # Filter streaks before prescription of any HR or sleep affecting medication
    if filter_by_medication:
        streaks = streaks.merge(excl_drug_minmax[['person_id', 'drug_min']].drop_duplicates(), how = 'left')
        streaks = streaks.loc[(streaks['streak_end'] <= streaks['drug_min']) |
                            (streaks['drug_min'].isna())]
        streaks = streaks.drop(columns = ['drug_min'])
        attrition['sleep streaks before or entirely after prescriptions affecting HR or sleep'] = streaks['person_id'].nunique()

    # filter streaks before prescription of OSA devices
    if filter_by_osa_device:
        osa_device_df['osa_device_date'] = pd.to_datetime(osa_device_df['osa_device_date'].astype(str).str[0:10], 
                                                        errors='coerce')
        osa_device_df = osa_device_df.dropna()
        streaks = streaks.merge(osa_device_df[['person_id', 'osa_device_date']], how = 'left')
        streaks = streaks.loc[(streaks['streak_end'] <= streaks['osa_device_date']) |
                            streaks['osa_device_date'].isna()]
        streaks = streaks.drop(columns = ['osa_device_date'])
        attrition['sleep streaks before prescriptions of OSA devices'] = streaks['person_id'].nunique()
        print(attrition)
 
    # Filter df by streaks
    
    df = filter_sleep_feat_by_streak_memory_efficient(df, streaks, date_col = 'date')

    if streak_save_path:
        streaks.to_csv(streak_save_path, index=False)

    return streaks, df, attrition


def get_features(cohort):
    pass

def main():

    # fetch the configuration
    with open('conf/create_study_cohort.yaml') as f:
         config = yaml.safe_load(f)

    for cohort_name, cohort_conf in config.items():
        if cohort_conf['run']:
            print(f"Creating {cohort_name}\n......")
            cohort, streaks, fitbit_summary = create_cohort(cohort_conf)
            get_time_series_features(cohort, streaks, fitbit_summary, cohort_conf)
    
    return


if __name__ == '__main__':
    main()