
flatten = lambda l: [item for sublist in l for item in sublist]
remove_duplicates = lambda l:list(dict.fromkeys(l))

DEFAULT_MAX_DEPTH=4
DEFAULT_MAX_PROOFS=8
