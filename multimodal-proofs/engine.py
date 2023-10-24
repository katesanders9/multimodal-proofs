import os
import json
from vqa import VisionModel
from text import TextGen, NLI, Retriever, LineRetriever, RetrieverBM25
from utils.text_utils import *


os.environ['TRANSFORMERS_CACHE'] = '/srv/local1/ksande25/cache/huggingface'
path = '/srv/local2/ksande25/NS_data/TVQA/'

class Engine(object):

    def __init__(self, max_steps=3, temp=0.2):
        self.cache = []
        with open(path + 'tvqa_subtitles_all.jsonl', 'r') as f:
            self.transcripts = [json.loads(x) for x in f][0]
        self.max_steps = max_steps
        self.vision = VisionModel()
        self.retrieval = Retriever()
        self.line_retrieval = LineRetriever(2)
        self.bm25 = RetrieverBM25()
        self.nli = NLI()
        self.generator = TextGen(temp=temp)

        self.show = None
        self.clip = None

    def set_clip(self, show, clip):
        self.show = show
        self.clip = clip
        self.generator.model.set_clip(clip)
        self.vision.set_clip(show, clip)
        self.t = self.transcripts[clip]
        self.bm25.set_transcript([x['text'] for x in self.t])
        self.retrieval.set_transcript([x['text'] for x in self.t])
        self.line_retrieval.set_transcript([x['text'] for x in self.t])

    def save_cache(self):
        self.generator.model.cache.save()

    # call vision
    def call_vision(self, h):
        q, qa = self.generator.toq(h)
        names = q2v(q, self.t)
        out = self.vision(qa, names)
        return out

    # recursive loop
    def query(self, h, q, k):
        print("DEPTH " + str(k))
        print("H: " + h)
        print("-----")
        print("Text eval...")
        x = None
        d = self.retrieval(h)
        if d:
            # l = self.line_retrieval(h, d)
            s = list(self.cache)
            # for line in l:
            c = [(d,i) for i in self.generator.inference(q, d)]
            s += c
            self.cache += c
            x, check = self.nli(s, h, thresh=1.0, c_check=True)
            if check:
                return [h, None, 0]
            if x:
                x = max(x, key=lambda y: y[1])
                return [h, x[0], x[1] / 4.0]
            elif not x and k < self.max_steps - 1:
                #print("Vision eval...")
                v = None
                #v = self.call_vision(h)
                if v:
                    x = max(x, key=lambda y: y[2])
                    return [h, x[0], x[2]]
                #elif k == 0:
                else:
                    h1, h2 = self.generator.branch_a(h, d)
                    q1 = self.generator.toq(h1)
                    q2 = self.generator.toq(h2)
                    x1 = self.query(h1, q1, k+1)
                    x2 = self.query(h2, q2, k+1)
                    x = [x1, x2]
                #else:
                #    h1, h2 = self.generator.branch_a(h, d)
                #    x1 = self.query(h1, k+1)
                #    x2 = self.query(h2, k+1)
                #    xa = [x1, x2]
                #    h1, h2 = self.generator.branch_b(h, d)
                #    x1 = self.query(h1, k+1)
                #    x2 = self.query(h2, k+1)
                #    xb = [x1, x2]
                #    if not xa[0] or not xa[1]:
                #        x = xb
                #    elif not xb[0] or not xb[1]:
                #        x = xa
                #    else:
                #        x = max([xa, xb], key=lambda y: y[0][2] * y[1][2])
                if not x[0] or not x[1]:
                    return [h, x, 0]
                return [h, x, x[0][2] * x[1][2]]
            else:
                return None
        else:
            #print("Aborted. Vision eval...")
            #x = self.call_vision(h)
            x = None
            if x:
                x = max(x, key=lambda y: y[2])
                return [h, x[0], x[2]]
            else:
                return [h, None, 0]
            return [h, x[0], x[1]]

    # full pipeline
    def main(self, q, a):
        self.cache = []
        h = self.generator.declarativize(q, a)
        proof = self.query(h, 0)
        return proof
