import os
os.environ['TRANSFORMERS_CACHE'] = "/srv/local2/ksande25/huggingface_cache/"
import json
import numpy as np
from tqdm import tqdm
from src.dialogue_retrieval import RetrievalEngine

DATA_DIR = '/srv/local2/ksande25/NS_data/TVQA/'
RANGE = 50, 100

# load hypotheses
def load_data(fn):
	with open(DATA_DIR + fn, 'r') as f:
		data = json.load(f)
	return data

def load_ground_truth(h, s, m):
	'''{h_id: [dialogue,]}'''
	maps = {}
	for item in h:
		i = str(item['id'])
		v, ts = m[i]['vid'], m[i]['ts']
		d = s[v]
		x = get_dialogue(d, ts)
		maps[i] = x
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
	t_s = get_time(t_s)
	t_e = get_time(t_e)
	return ((t_s > s) and (t_s < e)) or ((t_e > e) and (t_e < e))

def get_pr(gt, pred):
	'''
	P = true positive / (true positive + false positive)
	R = true positive / (true positive + false negative)
	'''
	if len(pred) < 1:
		return 0, 0
	gt_d = [x['text'] for x in gt]
	p_d = [x['text'] for x in pred]
	f = lambda x, l: any([y == x for y in l])
	tp = np.sum([1 for p in p_d if f(p,gt_d)])
	fp = np.sum([1 for p in p_d if not f(p, gt_d)])
	fn = np.sum([1 for g in gt_d if not f(g, p_d)])
	if (tp == 0 and fn == 0):
		print(gt)
	return float(tp) / (tp + fp), float(tp) / (tp + fn)


if __name__ == "__main__":
	print("Loading data...")
	hypotheses = load_data('hypotheses.jsonl')
	dialogue = load_data('tvqa_subtitles_all.jsonl')
	mapping = load_data('qa_metadata.jsonl')
	ground_truth = load_ground_truth(hypotheses, dialogue, mapping)
	print("Data loaded.")
	print("Loading retrieval engine...")
	engine = RetrievalEngine(threshold=0.97)
	print("Engine loaded.")
	p, r = [], []
	print("Retrieving dialogue...")
	for i in tqdm(range(RANGE[0],RANGE[1])):
		h = hypotheses[i]
		d_gt = ground_truth[str(h['id'])]
		if len(d_gt) > 0:
			d_pred = engine.retrieve_dialogue(h['h'], mapping[str(h['id'])]['vid'], dialogue, QA_flag=False, metric=None, threshold=None)
			print(h['h'])
			print([x['text'] for x in d_gt])
			print([x['text'] for x in d_pred])
			1/0
			prec, rec = get_pr(d_gt, d_pred)
			p.append(prec)
			r.append(rec)
	print("Dialogue retrieved.")
	print('Precision: {}   Recall: {}'.format(np.mean(p), np.mean(r)))
