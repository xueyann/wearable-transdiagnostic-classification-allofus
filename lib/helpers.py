from itertools import islice
import os
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