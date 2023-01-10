''' From Nathaniel's NELLIE codebase.'''

from .read_data import *
from .utils import *

DEFAULT_MAX_DEPTH=4
DEFAULT_MAX_PROOFS=8

flatten = lambda l: [item for sublist in l for item in sublist]
remove_duplicates = lambda l:list(dict.fromkeys(l))