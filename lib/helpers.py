from itertools import islice
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