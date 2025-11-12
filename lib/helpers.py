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