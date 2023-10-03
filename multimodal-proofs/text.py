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
        f_scores = [inferences[i] for i in range(len(scores)) if scores[i][1] > thresh]
        return list(zip(filtered, f_scores))


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

        # prompts
        self.declarativize_prompt = declarativize_prompt
        self.inference_preamble = inference_preamble
        self.inference_prompt = inference_prompt
        self.branch_a_preamble = branch_a_preamble
        self.branch_a_prompt = branch_a_prompt
        self.branch_b_preamble = branch_b_preamble
        self.branch_b_prompt = branch_b_prompt
        self.verify_i_preamble = verify_i_preamble
        self.verify_i_prompt = verify_i_prompt
        self.verify_b_preamble = verify_b_preamble
        self.verify_b_prompt = verify_b_prompt

        self.toq_prompt = toq_prompt

    def update_gpt_param(self, param, value):
        if param == 'temp':
            self.model.temp = value
        if param == 'model':
            self.model.model = value
        if param == 'max_tokens':
            self.model.max_tokens = value
        if param == 'preamble':
            self.model.preamble = value

    def declarativize(self, q, a):
        prompt = self.declarativize_prompt.format(q=q,a=a)
        h = self.model(prompt)
        if h.startswith('"'):
            h = h[1:-1]
        return h

    def inference(self, h, d, l):
        d = [remove_breaks(x) for x in d]
        d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
        d = '\n'.join(d)
        prompt = self.inference_preamble + self.inference_prompt.format(l=l, h=h, d=d)
        s = self.model(prompt)
        return list(json.loads(s).values())

    def branch_a(self, h, d):
        d = '\n'.join(d)
        prompt =  self.branch_a_preamble + self.branch_a_prompt.format(h=h, d=d)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def branch_b(self, h, d):
        d = '\n'.join(d)
        prompt =  self.branch_b_preamble + self.branch_b_prompt.format(h=h, d=d)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def verify_inf(self, d, s):
        # s = [a, b, c, ...]
        d = '\n'.join(d)
        prompt =  self.verify_i_preamble + self.verify_i_prompt.format(d=d, s=s)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def verify_branch(self, h, s):
        # s = [[a, b], [a, b], ...]
        prompt =  self.verify_b_preamble + self.verify_b_prompt.format(h=h, s=s)
        hp = self.model(prompt)
        return list(json.loads(hp).values())

    def toq(self, h):
        prompt = self.toq_prompt.format(h=h)
        qa = self.model(prompt)
        qa = qa.split('"')
        qa, qaa = qa[1], qa[3]
        return qa, qaa

    def load_prompt(self, name):
        with open('test_prompts.json','r') as f:
            p = json.load(f)
        return p[name]