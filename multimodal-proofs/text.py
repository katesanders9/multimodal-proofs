import os
import json
import numpy as np
import openai
from sentence_transformers import CrossEncoder

with open('keys.txt', 'r') as f:
    openai.api_key = f.read().strip()

os.environ['TRANSFORMERS_CACHE'] = '/srv/local1/ksande25/cache/huggingface'

TRANSCRIPT_DB = '/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl'
max_steps = 3

entailment_definition = 'Textual entailment is defined as a directional relation between two text fragments, called text (t, the entailing text), and hypothesis (h, the entailed text), so that a human being, with common understanding of language and common background knowledge, can infer that h is most likely true on the basis of the content of t.'

nli = CrossEncoder('cross-encoder/nli-distilroberta-base')
msm = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

with open(TRANSCRIPT_DB, 'r') as f:
    transcripts = [json.loads(x) for x in f]
    transcripts = transcripts[0]

# query gpt
def gpt(message, temp=0.5):
    response = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo",
      temperature = temp,
      max_tokens = 1000,
      messages = [
        {"role": "user", "content": message}
      ]
    )
    return response['choices'][0]['message']['content']

# declaritivize
def declaritivize(q, a):
    prompt = 'Combine the following question-answer pair into a single declarative statement. \nQUESTION: "' + q + '"\nANSWER: "' + a + '"\nSTATEMENT: '
    h = gpt(prompt)
    if h.startswith('"'):
        h = h[1:-1]
    return h

def sample_h(x, n, q):
    out = []
    for i in range(len(x) - n + 1):
        t = []
        for j in range(n):
            t.append(x[i+j])
        out.append((t,(q,'\n'.join(t))))
    return out

# sample text
def sample_text(h, t, n=6, thresh=0):
    samples = sample_h(t, n, h)
    scores = msm.predict([s[1] for s in samples])
    if not any([s > thresh for s in scores]):
        return None
    d = samples[np.argmax(scores)][0]
    return d

# retrieve dialogue
def retrieve_lines(h, d, n):
    samples = [(h,x) for x in d]
    scores = list(msm.predict(samples))
    scores_c = list(scores)
    inds = []
    for i in range(n):
        top = max(scores_c)
        inds.append(top)
        scores_c.remove(top)
    inds = [scores.index(x) for x in inds]
    return inds

def remove_breaks(l):
    while '\n' in l:
        ind = l.index('\n')
        l = l[:ind] + ' ' + l[ind+1:]
    return l

# generate inferences
def inference(h, d, l):
    d = [remove_breaks(x) for x in d]
    d = ['(' + str(i) + ') ' + x for i, x in enumerate(d)]
    d = '\n'.join(d)
    prompt = entailment_definition + ' \n Write a set of five hypotheses that relate to the FACT and are specifically entailed by dialogue line (' + str(l) + ') in JSON format, i.e. {"1": "<answer here>", "2": "<answer here>", ...} and nothing else. \n FACT: "' + h +'" \n DIALOGUE: \n ```\n' + d + '\n``` \n LINE (' + str(l) + ') ENTAILMENTS:'
    s = gpt(prompt)
    start = s.index('{')
    end = s.index('}')
    return list(json.loads(s).values())

# filter inferences
def filter_inf(s, h, th=0):
    inputs = [(x,h) for x in s]
    scores = nli.predict(inputs)
    filtered = [inputs[i][1] for i in range(len(scores)) if scores[i][1] > th]
    return filtered

# hypothesis branching
def branch(h, d):
    d = '\n'.join(d)
    prompt =  entailment_definition + ' \n Write two facts that are entailed by the dialogue that, together, make the hypothesis true. Write your answer in JSON format, i.e. {"1": "<fact 1>", "2": "<fact 2>"} and nothing else..\n HYPOTHESIS: "' + h + '"\n DIALOGUE:\n ```\n' + d + '\n ```\nFACTS:'
    hp = gpt(prompt)
    return list(json.loads(hp).values())

# to question
def h2q(h):
    prompt = 'Convert the following statement into a "yes" or "no" question, and then rewrite the question with the names replaced with "person" or "people".\nSTATEMENT: "' + h + '"\nQUESTION: '
    qa = gpt(prompt)
    qa = qa.split('"')
    qa, qaa = qa[1], qa[3]
    return qa, qaa

def q2v(q, t):
    names = [x for x in t if x.startswith('(') and ':)' in x]
    names = [x[x.index('('):x.index(':')+1].lower() for x in ts]
    ql = q.lower()
    q_names = [n for n in names if n in ql]
    return q_names

# call vision
def call_vision(show, clip, h, t):
    q, qa = h2q
    names = q2v(q, t)
    out = run(show, clip, qa, names, fps=2, rate=3)
    return out

# recursive loop
def query(h, t, k, show, clip):
    tp = [x['text'] for x in t]
    d = sample_text(h, tp, n=6)
    if d:
        l = retrieve_lines(h, d, 2)
        s, x = [], []
        for line in l:
            s += inference(h, d, l)
            x += filter_inf(s, h, th=0)
        if not x and k < max_steps - 1:
            h1, h2 = branch(h, d)
            x1 = query(h1, t, k+1, show, clip)
            x2 = query(h2, t, k+1, show, clip)
            x = x1 + x2
        else:
            return None
    else:
        x = call_vision(show, clip, h, t)
    return x

# full pipeline

def main(q, a, show, clip):
    h = declaritivize(q, a)
    t = transcripts[clip]
    proof = query(h, t, 0, show, clip)
    print_proof(proof)