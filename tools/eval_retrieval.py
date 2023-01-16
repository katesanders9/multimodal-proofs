import json
import numpy as np
from src.dialogue_retrieval import RetrievalEngine

DATA_DIR = '/srv/local2/ksande25/NS_data/TVQA/'
RANGE = 0, 30

# load hypotheses
def load_data(fn):
	with open(DATA_DIR + fn, 'r') as f:
		data = json.load(f)
	return data

def load_ground_truth(h, s, m):
	'''{h_id: [dialogue,]}'''
	maps = {}
	for item in h:
		i = item['id']
		v, ts = m[i]['vid'], m[i]['ts']
		d = s[v]
		m = get_dialogue(d, ts)
		maps[i] = m
	return maps

def get_dialogue(d_list, ts):
	ts = ts.split("-")
	start = float(ts[0])
	end = float(ts[1])
	return [d for d in d_list if in_between(start, end, d['start'], d['end'])]

def get_time(t):
	r = t[0] * 60 * 60 + t[1] * 60 + t[2] + t[3] / 1000
	return r

def in_between(s, e, t_s, t_e):
	return ((t_s > s) and (t_s < e)) or ((t_e > e) and (t_e < e))

def get_pr(gt, pred):
	'''
	P = true positive / (true positive + false positive)
	R = true positive / (true positive + false negative)
	'''
	tp = np.sum([p in pred if p in gt])
	fp = np.sum([p in pred if not p in gt])
	fn = np.sum([g in gt if not g in pred])
	return float(tp) / (tp + fp), float(tp) / (tp + fn)


if __name__ == "__main__":
	hypotheses = load_data('hypotheses.jsonl')
	dialogue = load_data('tvqa_subtitles_all.jsonl')
	mapping = load_data('qa_metadata.jsonl')
	ground_truth = load_ground_truth(hypotheses, dialogue, mapping)
	engine = RetrievalEngine()
	p, r = [], []
	for h in hypotheses[RANGE[0]:RANGE[1]]:
		d_pred = RetrievalEngine.retrieve_dialogue(h['h'], mapping[h['id']]['vid'], metric=None, threshold=None)
		d_gt = ground_truth[h['id']]
		prec, rec = get_pr(d_gt, d_pred)
		p.append(prec)
		r.append(rec)
	print('Precision: {}   Recall: {}'.format(np.mean(p), np.mean(r)))
