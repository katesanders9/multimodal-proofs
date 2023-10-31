import os
import json
import numpy as np
import openai
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from prompts import *
from utils.text_utils import *
from cache import Cache

class GPT(object):

    def __init__(self, temp=0.5, model="gpt-3.5-turbo", max_tokens=1000, preamble=[], cache_path='/srv/local2/ksande25/NS_data/TVQA/new_cache/'):
        with open('keys.txt', 'r') as f:
            self.key = f.read().strip()
        self.temp = temp
        self.model = model
        self.max_tokens = max_tokens
        self.preamble = preamble
        self.cache = Cache(cache_path)
        self.count = 0
        self.clip = None
        self.set_key()

    def set_key(self):
        openai.api_key = self.key

    def set_clip(self, clip):
        self.clip = clip

    def __call__(self, message):
        c = self.check_cache(message)
        if c:
            return c
        else:
            self.count += 1
            response = openai.ChatCompletion.create(
              model = self.model,
              temperature = self.temp,
              max_tokens = self.max_tokens,
              messages = self.preamble + [
                {"role": "user", "content": message}
              ]
            )
            out = response['choices'][0]['message']['content']
            self.cache.add(self.clip, message, out)
            self.cache.save()
            return out

    def check_cache(self, message):
        if not self.clip in self.cache.data.keys():
            return None
        elif message in [x[0] for x in self.cache.data[self.clip]]:
            return [m[1] for m in self.cache.data[self.clip] if m[0] == message][0]
        else:
            return None


class NLI(object):
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/nli-distilroberta-base')

    def __call__(self, inferences, hypothesis, thresh=0, c_check=False):
        if not inferences:
            return []
        inputs = [(x[1],hypothesis) for x in inferences]
        out = self.model.predict(inputs)
        contras = [s[0] for s in out]
        flags = [True if c > thresh else False for c in contras]
        scores = [s[1] for s in out]
        filtered = [inferences[i] for i in range(len(scores)) if scores[i] > thresh]
        f_scores = [scores[i] for i in range(len(scores)) if scores[i] > thresh]
        if c_check:
            return list(zip(filtered, f_scores)), any(flags)
        else:
            return list(zip(filtered, f_scores)), False

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
        self.inference_preamble = data_inf_preamble #inference_preamble_c
        self.inference_prompt = data_inf_prompt #inference_prompt_c
        self.branch_a_preamble = branch_a_preamble
        self.branch_a_prompt = branch_a_prompt
        self.branch_b_preamble = branch_b_preamble
        self.branch_b_prompt = branch_b_prompt
        self.verify_i_preamble = verify_i_preamble
        self.verify_i_prompt = verify_i_prompt
        self.verify_b_preamble = verify_b_preamble
        self.verify_b_prompt = verify_b_prompt

        self.toq_prompt = toq_prompt
        self.toq_preamble = toq_preamble

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

    def inference(self, h, d, l=0, remove_one=False):
        d = [remove_breaks(x) for x in d]
        d1 = d
        d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
        d = '\n'.join(d)
        text = d1[l]
        # prompt = inference_preamble_z1.format(l=l, x=text) + inference_preamble_e + inference_prompt_z.format(h=h, d=d, l=l)
        prompt = self.inference_preamble + self.inference_prompt.format(h=h,d=d)
        s = self.model(prompt)
        if remove_one:
            return list(json.loads(s).values())[1:]
        else:
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
        prompt = self.toq_preamble + self.toq_prompt.format(h=h)
        qa = self.model(prompt)
        return list(json.loads(qa).values())

    def load_prompt(self, name):
        with open('test_prompts.json','r') as f:
            p = json.load(f)
        return p[name]
