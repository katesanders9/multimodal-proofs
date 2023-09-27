import os
import json

path = '/srv/local2/ksande25/NS_data/TVQA/'
shows = {'Castle': 'castle', 'House M.D.': 'house', 'How I Met You Mother': 'met', 'The Big Bang Theory': 'bbt', "Grey's Anatomy": 'grey', 'Friends': 'friends'}

# load dset
def load_dataset(path):
	with open(path + 'tvqa_subtitles_all.jsonl', 'r') as f:
		transcripts = [json.loads(x) for x in f][0] # 21794
	with open(path + 'tvqa_qa_release/tvqa_train.jsonl', 'r') as f:
		train = [json.loads(x) for x in f] # 122039
	with open(path + 'tvqa_qa_release/tvqa_val.jsonl', 'r') as f:
		val = [json.loads(x) for x in f] # 15253
	with open(path + 'tvqa_qa_release/tvqa_test_public.jsonl', 'r') as f:
		test = [json.loads(x) for x in f] # 7623
	return transcripts, train, val, test

# load qa pair
def load_qa_pair(data, index):
	item = data[index]
	ans = item['a' + str(item['answer_idx'])]
	return [shows[item['show_name']], item['vid_name'], item['q'], ans]
