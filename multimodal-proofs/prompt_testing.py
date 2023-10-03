from utils.text_utils import *
from text import TextGen
from dataset import Dataset
from text import Retriever

path = '../../'

data = Dataset(path)
model = TextGen()
data.set_data('val')
r = Retriever()

def load(i):
	dp = data.load_data_sample(i)
	t = [x['text'] for x in dp[4]]
	q = dp[2]
	a = dp[3]
	ans = list(data.data[i].values())[:5]
	return q, a, ans, t

def l(i):
	d = load(i)
	r.set_transcript(d[3])
	return d

def run(h,d,p1,p2):
    d = [remove_breaks(x) for x in d]
    d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
    d = '\n'.join(d)
    prompt = p1 + p2.format(h=h,d=d)
    s = model.model(prompt)
    return s

def run2(h,p1,p2):
    prompt = p1 + p2.format(h=h)
    s = model.model(prompt)
    return s

def run3(a,b,c,p1,p2):
    prompt = p1 + p2.format(a=a,b=b,c=c)
    s = model.model(prompt)
    return s