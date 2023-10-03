import os
import json

path = '/srv/local2/ksande25/NS_data/TVQA/'
shows = {'Castle': 'castle', 'House M.D.': 'house', 'How I Met You Mother': 'met', 'The Big Bang Theory': 'bbt', "Grey's Anatomy": 'grey', 'Friends': 'friends'}

# load dset
def load_dataset(path=path):
    with open(path + 'tvqa_subtitles_all.jsonl', 'r') as f:
        transcripts = [json.loads(x) for x in f][0] # 21794
    with open(path + 'tvqa_qa_release/tvqa_train.jsonl', 'r') as f:
        train = [json.loads(x) for x in f] # 122039
    with open(path + 'tvqa_qa_release/tvqa_val.jsonl', 'r') as f:
        val = [json.loads(x) for x in f] # 15253
    with open(path + 'tvqa_qa_release/tvqa_test_public.jsonl', 'r') as f:
        test = [json.loads(x) for x in f] # 7623
    return transcripts, train, val, test

def get_dialogue(data_pt, t):
    ts = data_pt['ts']
    start, end = [float(i) for i in ts.split('-')]
    out = []
    for i in range(len(t)):
        if to_seconds(t[i]['start']) > start and to_seconds(t[i]['end']) < end:
            out.append(i)
    return out

def to_seconds(time):
    return time[0] * 60 * 60 + time[1] * 60 + time[2] + time[3] / 1000

# text - 0 vision - 1 both - 2
def load_labels():
    return np.load('val_labels.npy')