from itertools import islice
import os
import pandas as pd
import numpy as np
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

def format_p(p, alpha = 0.05):
    if p < 0.001:
        return r"$<0.001*$"
    elif p <= alpha:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

def rmssd(x):
    x = np.asarray(x)
    diff = np.diff(x)
    return np.sqrt(np.mean(diff**2))