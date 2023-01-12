import os
import json
import pysrt

from datasets import load_dataset


QA_FN = "/srv/local2/ksande25/NS_data/TVQA/tvqa_qa_release/tvqa_train.jsonl"
SUB_DIR = "/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles/"
SUB_FN = "/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl"

def load_qa():
    '''
    "a0": "A martini glass",
    "a1": "Nachos",
    "a2": "Her purse",
    "a3": "Marshall's book",
    "a4": "A beer bottle",
    "answer_idx": 4,
    "q": "What is Robin holding in her hand when she is talking to Ted about Zoey?",
    "qid": 7,
    "ts": "1.21-8.49",
    "vid_name": "met_s06e22_seg01_clip_02"
    '''
    data = []
    for line in open(qa_fn, 'r'):
        data.append(json.loads(line))
    return data

def load_subs():
    files = os.listdir(SUB_DIR)
    data = []
    for i in tqdm(range(len(files))):
        f = files[i]
        sub_data = pysrt.open(os.path.join(SUB_DIR, f))
        i = 0
        for d in sub_data:
            sub = {}
            sub['file'] = f[:-4]
            sub['id'] = i
            sub['text'] = d.text
            sub['start'] = [d.start.hours, d.start.minutes, d.start.seconds, d.start.milliseconds]
            sub['end'] = [d.end.hours, d.end.minutes, d.end.seconds, d.end.milliseconds]
            subs.append(sub)
            i += 1
        data[f[:-4]] = subs
    return data

def preprocess_subs():
    data = load_subs()
    with open(SUB_FN, 'w') as f:
        json.dump(data, f)

def load_datasets():
    qa = load_dataset(QA_FN)
    subs = load_dataset(SUB_FN)
    return qa, subs


if __name__ == "__main__":
    preprocess_subs()