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
    def query(self, h, k):
        print("DEPTH " + str(k))
        print("H: " + h)
        print("-----")
        print("Text eval...")
        x = None
        d = self.retrieval(h)
        if d:
            s, x = self.cache, self.nli(self.cache, h)
            c = [(d,i) for i in self.generator.inference(h, d)]
            s += c
            self.cache += c
            x += self.nli(s, h)
            if x:
                x = max(x, key=lambda y: y[1])
                return [h, x[0], x[1]]
            elif not x and k < self.max_steps - 1:
                print("Vision eval...")
                v = self.call_vision(h)
                if v:
                    x = max(x, key=lambda y: y[2])
                    return [h, x[0], x[2]]
                elif k == 0:
                    h1, h2 = self.generator.branch_a(h, d)
                    x1 = self.query(h1, k+1)
                    x2 = self.query(h2, k+1)
                    x = [x1, x2]
                else:
                    h1, h2 = self.generator.branch_a(h, d)
                    x1 = self.query(h1, k+1)
                    x2 = self.query(h2, k+1)
                    xa = [x1, x2]
                    h1, h2 = self.generator.branch_b(h, d)
                    x1 = self.query(h1, k+1)
                    x2 = self.query(h2, k+1)
                    xb = [x1, x2]
                    if not xa[0] or not xa[1]:
                        x = xb
                    else:
                        x = max([xa, xb], key=lambda y: y[0][2] * y[1][2])
                if not x[0] or not x[1]:
                    return None
                return [h, x, x[0][2] * x[1][2]]
            else:
                return None
        else:
            print("Aborted. Vision eval...")
            x = self.call_vision(h)
            if x:
                x = max(x, key=lambda y: y[2])
                return [h, x[0], x[2]]
            else:
                return None
            return [h, x[0], x[1]]

    # full pipeline
    def main(self, q, a):
        h = self.generator.declarativize(q, a)
        proof = self.query(h, 0)
        return proof
