'''FLAN-T5-XL hypothesis generation script. Works with hf_dataset.py.'''

import os
os.environ['TRANSFORMERS_CACHE'] = "/srv/local2/ksande25/huggingface_cache/"
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = 'google/flan-t5-xl'
QA_FN = "~/TVQA/tvqa_qa_release/tvqa_train.jsonl"
OUT_FN = "~/TVQA/out.json"
ICL_PROMPT = '''
Combine the question/answer pair into a single declarative statement:
Q: Why is Castle vexed after he reads the note?
A: Castle believed he will see blood in the kitchen.
Castle is vexed after he reads the note because Castle believed he will see blood in the kitchen.

Combine the question/answer pair into a single declarative statement:
Q: What is Robin holding in her hand when she is talking to Ted about Zoey?
A: A beer bottle
Robin is holding a beer bottle in her hand when she is talking to Ted about Zoey.

Combine the question/answer pair into a single declarative statement:
'''

class HypothesisGenerator:
    def __init__(self, model_name, prompt):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.prompt = prompt

    def encode_input(self, query):
        full_query = self.prompt + query
        tokenized = self.tokenizer(full_query, max_length=512, truncation=True, return_tensors="pt").input_ids
        return tokenized

    def forward_pass(self, encoded_query):
        out = self.model.generate(encoded_query)
        return out

    def decode_output(self, tokens):
        out = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return out

    def inference(self, query):
        x = self.encode_input(query)
        x = self.forward_pass(x)
        x = self.decode_output(x)
        return x

    def load_QA_data(self, json_fn):
        data = []
        for line in open(json_fn, 'r'):
            item = json.loads(line)
            record = {}
            record['id'] = item['qid']
            record['q'] = item['q']
            record['a'] = item['a' + str(item['answer_idx'])]
            record['vid_id'] = item['vid_name']
            record['time'] = item['ts']
            data.append(record)
        return data

    def gen_hypothesis(self, record):
        query = self.prompt + "Q: " + record['q'] + " A: " + record['a']
        hypothesis = self.inference(query)
        return hypothesis

    def gen_from_json(self, json_fn):
        data = self.load_QA_data(json_fn)
        results = []
        for record in data:
            out = {}
            out['id'] = record['id']
            out['h'] = self.gen_hypothesis(record)
            results.append(out)
        with open(OUT_FN, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    generator = HypothesisGenerator(MODEL_NAME, ICL_PROMPT)
    generator.gen_from_json(QA_FN)

