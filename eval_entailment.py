import numpy as np
from datasets import load_dataset
from src.filters import SBERTFilter, CrossEncoderEntailmentFilter

DSET_NAME = "snli"
FILTER_NAMES = [CrossEncoderEntailmentFilter]

# load dataset
def load_data(dset_name):
	dset = load_dataset(dset_name)
	return dset

# load filter(s)
def load_filters(filter_names):
	filters = []
	for f in filter_names:
		x = f()
		filters.append(x)
	return filters

# run filtering
def retrieve_preds(data, filters):
	# returns array of 1s and 0s: 1 == entailment
	arr = [postprocess(f(data)) for f in filters]
	return any(arr, axis=0)

def postprocess(data):
	# NOT IMPLEMENTED
	return data

# score retrieved candidates
def score_preds(y_data, preds):
	# accuracy or PR?
	acc = np.mean([1 if x == y else 0 for (x,y) in zip(y_data, preds)])
	p = np.mean([1 if x == y else 0 for (x,y) in zip(y_data, preds) if y == 1])
	r = np.mean([1 if x == y else 0 for (x,y) in zip(y_data, preds) if x == 1])
	return acc, p, r 

# run script
if __name__ == "__main__":
	dset = load_data(DSET_NAME)
	filters = load_filters(FILTER_NAMES)
	preds = retrieve_preds(dset, filters)
	acc, p, r = score_preds(dset, preds)