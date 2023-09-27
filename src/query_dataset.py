import json
import numpy as np

DATA_DIR = '/srv/local2/ksande25/NS_data/TVQA/'

class Query(object):
	def __init__(self, h, clip, time):
		self.hypothesis = h['h']
		self.id = h['id']
		self.clip = clip
		self.time = time

class TVQADataSet(object):
	def __init__(self):
		self.hypotheses = self.load_data('hypotheses.jsonl')
		self.dialogue = self.load_data('tvqa_subtitles_all.jsonl')
		self.mapping = self.load_data('qa_metadata.jsonl')
		self.ground_truth = self.load_ground_truth(self.hypotheses, self.dialogue, self.mapping)
		self.queries = None
		self.qa = None

		self.h_to_q()

	def __len__(self):
		return len(self.hypotheses)

	def h_to_q(self):
		self.queries = {}
		for h in self.hypotheses:
			q = Query(h, self.get_clip(h['id']), self.get_time_h(h['id']))
			self.queries[h['id']] = q

	def get_dialogue_q(self, q, all_lines=False):
		hid = q.id
		if not all_lines:
			return self.ground_truth[str(hid)]
		clip = self.mapping[str(hid)]['vid']
		return self.dialogue[clip]

	def get_clip(self, hid):
		return self.mapping[str(hid)]['vid']

	def get_time_h(self, hid):
		return self.mapping[str(hid)]['ts']

	def load_data(self, fn):
		fn = DATA_DIR + fn
		data = self.load_jsonl(fn)
		return data

	def load_ground_truth(self, h, s, m):
		'''{h_id: [dialogue,]}'''
		maps = {}
		for item in h:
			i = str(item['id'])
			v, ts = m[i]['vid'], m[i]['ts']
			d = s[v]
			x = self.get_dialogue(d, ts)
			maps[i] = x
		return maps

	def get_dialogue(self, d_list, ts):
		ts = ts.split("-")
		start = float(ts[0])
		end = float(ts[1])
		return [d for d in d_list if self.in_between(start, end, d['start'], d['end'])]

	def get_time(self, t):
		r = t[0] * 60 * 60 + t[1] * 60 + t[2] + t[3] / 1000
		return r

	def in_between(self, s, e, t_s, t_e):
		t_s = self.get_time(t_s)
		t_e = self.get_time(t_e)
		return ((t_s > s) and (t_s < e)) or ((t_e > e) and (t_e < e))

	def load_jsonl(self, fn):
		with open(fn) as f:
			x = []
			for l in f:
				x.append(json.loads(l))
		return x[0]

	def load_qa(self):
		train = self.load_jsonl(DATA_DIR + '/tvqa_train.jsonl')
		val = self.load_jsonl(DATA_DIR + '/tvqa_val.jsonl')
		test = self.load_jsonl(DATA_DIR + '/tvqa_test_public.jsonl')
		self.qa = train + val + test

	def get_qa(self, q):
		if self.qa:
			res = [x for x in self.qa if x['qid'] == q.id]
			return res[0] if res else None
		return None
