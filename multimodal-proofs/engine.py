import os
import json
from vqa import VisionModel
from text import TextGen, NLI, Retriever, LineRetriever


os.environ['TRANSFORMERS_CACHE'] = '/srv/local1/ksande25/cache/huggingface'
path = '/srv/local2/ksande25/NS_data/TVQA/'

class Engine(object):

	def __init__(self):
		with open(path + 'tvqa_subtitles_all.jsonl', 'r') as f:
			self.transcripts = [json.loads(x) for x in f][0]
		self.max_steps = 3
		self.vision = VisionModel()
		self.retrieval = Retriever()
		self.line_retrieval = LineRetriever()
		self.nli = NLI()
		self.generator = TextGen()

		self.show = None
		self.clip = None

	def set_clip(self, show, clip):
		self.show = show
		self.clip = clip
		self.vision.set_clip(show, clip)
		self.t = self.transcripts[clip]
		self.retrieval.set_transcript([x['text'] for x in self.t])
		self.line_retrieval.set_transcript([x['text'] for x in self.t])

	# call vision
	def call_vision(self, h):
	    q, qa = self.generator.toq(h)
	    names = q2v(q, self.t)
	    out = self.vision(qa, names)
	    return out

	# recursive loop
	def query(self, h, k):
	    d = self.retrieval(h)
	    if d:
	        l = self.line_retrieval(h, d)
	        s, x = [], []
	        for line in l:
	            s += self.generator.inference(h, d, l)
	            x += self.nli(s, h, th=0)
	        if not x and k < self.max_steps - 1:
	            h1, h2 = self.generator.branch(h, d)
	            x1 = self.query(h1, k+1)
	            x2 = self.query(h2, k+1)
	            x = x1 + x2
	        else:
	            return None
	    else:
	        x = self.call_vision(h, t)
	    return x

	# full pipeline
	def main(self, q, a):
	    h = self.generator.declaritivize(q, a)
	    proof = self.query(h, 0)
	    return proof
