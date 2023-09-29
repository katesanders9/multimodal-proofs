import os
import json
import numpy as np
import openai
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from prompts import *
from utils.text_utils import *

class GPT(object):

    def __init__(self, temp=0.5, model="gpt-3.5-turbo", max_tokens=1000, preamble=[]):
        with open('keys.txt', 'r') as f:
            self.key = f.read().strip()
        self.temp = temp
        self.model = model
        self.max_tokens = max_tokens
        self.preamble = preamble
        self.set_key()

    def set_key(self):
        openai.api_key = self.key

    def __call__(self, message):
        response = openai.ChatCompletion.create(
          model = self.model,
          temperature = self.temp,
          max_tokens = self.max_tokens,
          messages = self.preamble + [
            {"role": "user", "content": message}
          ]
        )
        return response['choices'][0]['message']['content']


class NLI(object):
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/nli-distilroberta-base')

    def __call__(self, inferences, hypothesis, thresh=0):
        if not inferences:
            return []
        inputs = [(x[1],hypothesis) for x in inferences]
        scores = self.model.predict(inputs)
        filtered = [inferences[i] for i in range(len(scores)) if scores[i][1] > thresh]
        return filtered


class Retriever(object):
    def __init__(self, n=6):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
        self.n = n

    def set_transcript(self, transcript):
        self.transcript = transcript

    def __call__(self, hypothesis, thresh=0):
        samples = sample_h(self.transcript, self.n, hypothesis)
        scores = self.model.predict([s[1] for s in samples])
        if thresh and not any([s > thresh for s in scores]):
            return None
        d = samples[np.argmax(scores)][0]
        return d

class RetrieverBM25(Retriever):
    def __init__(self, n=6):
        self.n = n

    def set_transcript(self, transcript):
        self.transcript_og = transcript
        transcript = [x[1][1] for x in sample_h(transcript, self.n, 'h')]
        self.transcript = [[x.lower() for x in doc.replace('\n',' ').split(" ")] for doc in transcript]
        self.model = BM25Okapi(self.transcript)

    def __call__(self, hypothesis, thresh=0):
        samples = sample_h(self.transcript_og, self.n, hypothesis)
        hypothesis = [h.lower() for h in hypothesis.split(" ")]
        scores = self.model.get_scores(hypothesis)
        if thresh and not any([s > thresh for s in scores]):
            return None
        d = samples[np.argmax(scores)][0]
        return d

class LineRetriever(Retriever):

    def __call__(self, hypothesis, transcript):
        samples = [(hypothesis,x) for x in transcript]
        scores = list(self.model.predict(samples))
        scores_c = list(scores)
        inds = []
        for i in range(self.n):
            top = max(scores_c)
            inds.append(top)
            scores_c.remove(top)
        inds = [scores.index(x) for x in inds]
        return inds


class TextGen(object):
    def __init__(self,temp=0.5):
        self.model = GPT(temp)
        self.h = None
        self.d = None

    def declaritivize(self, q, a):
        prompt = declaritivize_prompt.format(q=q,a=a)
        h = self.model(prompt)
        if h.startswith('"'):
            h = h[1:-1]
        return h

    def inference(self, h, d, l):
        d = [remove_breaks(x) for x in d]
        d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
        d = '\n'.join(d)
        prompt = inference_preamble + inference_prompt.format(l=l, h=h, d=d)
        s = self.model(prompt)
        return list(json.loads(s).values())

    def branch(self, h, d):
        d = '\n'.join(d)
        prompt =  branch_preamble + branch_prompt.format(h=h, d=d)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def toq(self, h):
        prompt = toq_prompt.format(h=h)
        qa = self.model(prompt)
        qa = qa.split('"')
        qa, qaa = qa[1], qa[3]
        return qa, qaa

