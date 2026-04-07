import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import numpy as np
from scipy.stats import circvar, circmean
from scipy.stats import zscore


def robust_z_scores(values):
    """
    Compute robust z-scores using median and MAD.
    values: array-like (e.g., sleep durations for one person)
    Returns: array of robust z-scores
    """
    # values = np.array(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    if mad == 0:  # handle case where all values are identical
        return np.zeros_like(values)
    
    robust_z = 0.6745 * (values - median) / mad
    return robust_z

def get_label_df(sleep_streaks = None,
                 conditions = ['diabetes', 'cvd', 'osa', 'mdd', 'anxiety', 'adhd', 'bipolar'],
                 path_pattern = 'data/02_intermediate/study_cohorts_20250902/{cond}_cohort_20250902.csv'):
    
    label_dfs = []
    # Get conditions
    for cond in conditions:
        label_dfs.append(pd.read_csv(path_pattern.format(cond = cond)))
    df = pd.concat(label_dfs).drop_duplicates().dropna()
    # Add back the control cohort
    df = pd.concat([df, label_dfs[0].loc[label_dfs[0]['subcohort']=='Control']])
    # one hot encoding
    df = pd.get_dummies(df, columns = ['subcohort'])
    df = df.reset_index(drop=True)

    # Set index date as value if patient belongs to that cohort
    for col in [x for x in df.columns if x != 'person_id']:
        df.loc[df[col]==True, col] = df['index_date']
        df.loc[df[col]==False, col] = ''
    df = df.drop(columns=['index_date'])
    # control column is redundant, drop
    df = df.drop(columns=['subcohort_Control'])
    # Aggregate
    df = df.groupby('person_id', as_index=False).sum()
    # Ensure all the person ids have sleep streaks (optional)
    if sleep_streaks is not None:
        df = df.merge(sleep_streaks)

    # cast all date columns
    subcohort_cols = [x for x in df.columns if x.startswith('subcohort_')]
    if sleep_streaks is not None:
        date_cols = subcohort_cols+['streak_start', 'streak_end']
    date_cols = subcohort_cols
    for col in date_cols:
        df.loc[df[col]=='', col] = np.nan
        df[col] = pd.to_datetime(df[col], errors="coerce", format = '%Y-%m-%d')
    # Label whether the streak is after all diagnoses
    if sleep_streaks is not None:
        df['streak_after'] = df['streak_start'] >= df[subcohort_cols].max(axis=1)

    return df

def get_circ_stats(sleep_onsets):
    # Convert hours -> radians
    angles = (sleep_onsets / 24.0) * 2 * np.pi

    # Circular mean & variance
    mean_angle = circmean(angles, high=2*np.pi, low=0)
    R = 1 - circvar(angles, high=2*np.pi, low=0)
    # Circular SD
    circ_sd = np.sqrt(-2*np.log(R))

    # Express circular SD in hours
    circ_sd_hours =  circ_sd / (2*np.pi) * 24

    # Convert mean_angle back to hour
    mean_hour = mean_angle / (2*np.pi) * 24

    return angles, R, \
        mean_angle, mean_hour, \
        circ_sd, circ_sd_hours

def plot_polar_sleep_onset(angles, R, mean_angle, ax):
    ax.scatter(angles, np.ones_like(angles), c="blue", s=50, alpha=0.7)
    # mean direction vector
    ax.arrow(mean_angle, 0, 0, R, width=0.03, color="red", alpha=0.7)
    ax.set_theta_zero_location("N")  # midnight at top
    ax.set_theta_direction(-1)       # clockwise
    # ---- Custom hour ticks ----
    hours = np.arange(0, 24, 3)  # ticks every 3 hours
    angles_ticks = hours / 24.0 * 2 * np.pi
    ax.set_xticks(angles_ticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])

    # ---- Hide radial ticks ----
    ax.set_yticklabels([])   # remove labels
    ax.yaxis.set_ticks([])   # remove tick marks too

    ax.set_title("Sleep onset times (polar clock plot)")

    return ax


def flag_anomaly_single_feature(ts, cutoff):
    z = zscore(ts)
    robust_z = zscore(ts)
    return pd.concat(ts[z<-cutoff], ts[z>cutoff]), pd.concat([robust_z<-cutoff], ts[robust_z>cutoff])

def flag_anomaly(df, feature, cutoff=3):

    print(feature)
    # Find the z score
    z = zscore(df[feature])
    robust_z = robust_z_scores(df[feature])
    print('Z score')
    print(df[feature][z<-cutoff])
    print(df[feature][z>cutoff])
    print('Robust z score')
    print(df[feature][z<-cutoff])
    print(df[feature][z>cutoff])
    print('='*50)

    return
   

def find_strict_streaks(df, date_col = 'sleep_date', min_length = 7):
    """
    Function to find the continuous streaks of data collection
    """

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(['person_id', date_col])

    # Create a helper column: index per person
    df['day_index'] = df.groupby('person_id').cumcount()

    # Compute "date - index" per person
    df['streak_key'] = df[date_col] - pd.to_timedelta(df['day_index'], unit='D')

    # Group by (person_id, streak_key) to identify streaks
    streaks = (
        df.groupby(['person_id', 'streak_key'])
        .agg(
            streak_start=(date_col, 'min'),
            streak_end=(date_col, 'max'),
            length=(date_col, 'count')
        )
        .reset_index()
    )

    streaks = streaks.drop(columns=['streak_key'])

    # Filter by minimum length
    streaks = streaks[streaks['length'] >= min_length].reset_index(drop=True)

    return streaks

def find_strict_streaks_p(df, date_col = 'date', min_length = 21):
    """
    Function to find the continuous streaks of data collection, but just for one person
    for efficiency
    """
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates() # just in case

    # Create a helper column: simple sequential index
    df['day_index'] = range(len(df))

    # Compute "date - index" per person
    df['streak_key'] = df[date_col] - pd.to_timedelta(df['day_index'], unit='D')

    # Group by (person_id, streak_key) to identify streaks
    streaks = (
        df.groupby('streak_key')
        .agg(
            streak_start=(date_col, 'min'),
            streak_end=(date_col, 'max'),
            length=(date_col, 'count')
        )
        .reset_index()
    )

    streaks = streaks.drop(columns=['streak_key'])

    # Filter by minimum length
    streaks = streaks[streaks['length'] >= min_length].reset_index(drop=True)

    return streaks

def filter_sleep_feat_by_streak_memory_efficient(sleep_feat_df, streaks, date_col = 'sleep_date'):

    sleep_feat_df[date_col] = pd.to_datetime(sleep_feat_df[date_col])
    streaks['streak_start'] = pd.to_datetime(streaks['streak_start'])
    streaks['streak_end'] = pd.to_datetime(streaks['streak_end'])

    # expand streak df
    streaks[date_col] = streaks.apply(lambda x: list(pd.date_range(x['streak_start'], x['streak_end'])), axis = 1)
    streaks = streaks.explode(date_col)
    streaks = streaks[['person_id', date_col]]

    # filter sleep feature df
    sleep_feat_df = sleep_feat_df.merge(streaks)

    return sleep_feat_df


def filter_sleep_feat_by_streak(sleep_feat_df, streaks, date_col = 'sleep_date'):
    """
    Function to filter sleep features to only keep those days appearing in streaks df
    """
    # Step 0: Ensure date columns are datetime
    sleep_feat_df[date_col] = pd.to_datetime(sleep_feat_df[date_col])
    streaks['streak_start'] = pd.to_datetime(streaks['streak_start'])
    streaks['streak_end'] = pd.to_datetime(streaks['streak_end'])

    # Step 2: Merge on patient_id (this will duplicate rows for each patient where multiple intervals exist)
    merged_df = sleep_feat_df.merge(
        streaks,
        on='person_id',
        how='inner'
    )

    # Step 3: Filter where sleep_date falls inside the interval
    filtered_df = merged_df[
        (merged_df[date_col] >= merged_df['streak_start']) &
        (merged_df[date_col] <= merged_df['streak_end'])
    ].copy()

    # Step 4: Keep interval_id and whatever columns you need
    filtered_df = filtered_df.drop(columns=['streak_start', 'streak_end'])

    # Now `filtered_df` has each sleep record matched with its corresponding `interval_id`

    return filtered_df
