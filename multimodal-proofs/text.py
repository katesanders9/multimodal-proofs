import os
import json
import numpy as np
import openai
from sentence_transformers import CrossEncoder
from prompts import *
from utils.text_utils import *

class GPT(object):
    self.temp = 0.5
    self.model = "gpt-3.5-turbo"
    self.max_tokens = 1000
    self.preamble = []

    def __init__(self):
        with open('keys.txt', 'r') as f:
            self.key = f.read().strip()

    def set_key(self, key):
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


class NLI(object)
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/nli-distilroberta-base')

    def __call__(self, inferences, hypothesis, thresh=0):
        inputs = [(x,hypothesis) for x in inferences]
        scores = self.model.predict(inputs)
        filtered = [inputs[i][1] for i in range(len(scores)) if scores[i][1] > thresh]
        return filtered


class Retriever(object):
    def __init__(self, n=6):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
        self.n = n

    def set_transcript(self, transcript):
        self.transcript = transcript

    def __call__(self, hypothesis):
        samples = [(hypothesis,x) for x in self.transcript]
        scores = list(self.model.predict(samples))
        scores_c = list(scores)
        inds = []
        for i in range(n):
            top = max(scores_c)
            inds.append(top)
            scores_c.remove(top)
        inds = [scores.index(x) for x in inds]
        return inds


class LineRetriever(Retriever):
    def __call__(self, hypothesis, transcript, thresh=0):
        samples = sample_h(transcript, self.n, hypothesis)
        scores = self.model.predict([s[1] for s in samples])
        if not any([s > thresh for s in scores]):
            return None
        d = samples[np.argmax(scores)][0]
        return d


class TextGen(object):
    def __init__(self):
        self.model = GPT()
        self.h = None
        self.d = None

    def set_hypothesis(self, h):
        self.h = h

    def set_dialogue(self, d):
        self.d = d

    def declaritivize(self, q, a):
        prompt = declaritivize_prompt.format(q=q,a=a)
        h = self.model(prompt)
        if h.startswith('"'):
            h = h[1:-1]
        return h

    def inference(self, l):
        d = [remove_breaks(x) for x in d]
        d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
        d = '\n'.join(d)
        prompt = inference_prompt.format(l=l, h=self.h, d=self.d)
        s = self.model(prompt)
        return list(json.loads(s).values())

    def branch(self):
        d = '\n'.join(d)
        prompt =  branch_prompt.format(h=self.h, d=self.d)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def toq(self):
        prompt = toq_prompt.format(h=self.h)
        qa = self.model(prompt)
        qa = qa.split('"')
        qa, qaa = qa[1], qa[3]
        return qa, qaa

