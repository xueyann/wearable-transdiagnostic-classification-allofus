from itertools import islice
import os
import pandas as pd
# For slicing an iterable

def chunked(iterable, size):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

def read_txt(fpath):
    with open(fpath) as f:
        content = f.read()
    return content

class DefaultDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def get_files_in_dir(directory_path, pattern):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if str(file).startswith(pattern):
                file_paths.append(os.path.join(root, file))
    # sort the file paths to ensure reproduciability 
    file_paths.sort()
    return file_paths

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


def make_tracker():
    len_dict = {}
    def track_len(func):
        # len_dict = {}
        def inner(df, *args, **kwargs):
            len_dict[func.__name__] = df['date'].nunique()
            return func(df, *args, **kwargs)
        track_len.len_dict = len_dict
        return inner
    return track_len, len_dict
