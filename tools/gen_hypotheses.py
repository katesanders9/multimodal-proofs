import os
import json
import random
import datasets
import argparse
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed, T5ForConditionalGeneration, AutoTokenizer, pipeline

from src.utils import read_jsonl
from src.dataset import EntailmentbankDatasetReader, ARC

set_seed(42)
parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=['entailmentbank', 'worldtree', 'worldtree_1_0'])
parser.add_argument("--model_path", default="models/entailer_QA2D")
parser.add_argument("--model_name", default="t5_combined_eb")
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--do_sample", action='store_true')
parser.add_argument("--num_beams", type=int, default=10)
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--use_setup', action='store_true')
parser.add_argument('--eval_splits_only', action='store_true')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--worker_id', type=int, default=0)

args = parser.parse_args()

reader = EntailmentbankDatasetReader(use_gpt_declaratives=False)

arc = ARC.load_arc_dataset()

ebdf = arc.merge(reader.question_df[['id', 'hypothesis', 'split']], left_on='id', right_on='id')

if args.use_setup:
    import spacy
    nlp = spacy.load("en_core_web_sm")

def extract_setup(x):
    parse=nlp(x)
    sentences = [str(s) for s in parse.sents]
    return ' '.join(sentences[:-1]), sentences[-1]

def _extract_qa2d_prompt(row):

    if args.use_setup:
        prompt = '$hypothesis$ ; $setup$ = {setup} ; $question$ = {last_question} ; $answer = {choice}'
        setup, last_question = extract_setup(row['question'])
        return prompt.format(setup=setup, last_question=last_question, choice=row['choice'])
    else:
        prompt = '$hypothesis$ ; $question$ = {question} ; $answer = {choice}'
        return prompt.format(**row)


if args.dataset == 'entailmentbank':
    eval_qa2d: pd.DataFrame = ebdf.query("split != 'train'")
else:
    raise NotImplementedError()

if args.eval_splits_only:
    eval_qa2d = eval_qa2d.query("split != 'train'")

model = T5ForConditionalGeneration.from_pretrained(args.model_path)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained("t5-large")
pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=model.device)


do_sample = args.do_sample

results_path = f"data/{args.dataset}/{args.dataset}_{args.model_name}_declarativized"


partial_results = f"{results_path}.jsonl"
if os.path.exists(partial_results):
    pres_df = pd.DataFrame(read_jsonl(partial_results))
else:
    pres_df = None

if args.num_workers > 1:
    full_results = partial_results
    partial_results = partial_results.replace(".jsonl", f"_{args.worker_id}.jsonl")
    if os.path.exists(partial_results):
        pres_df = pd.concat([pres_df, pd.DataFrame(read_jsonl(partial_results))])


def declarativize(row):
    if pres_df is not None:
        matches = pres_df.query(f"id == '{row.id}' and label == '{row.label}'")
        if not matches.empty:
            completion = matches.iloc[0]['declarativized']
            return completion

    prompt = _extract_qa2d_prompt(row)
    N=(args.num_beams if not do_sample else 1)
    completions = pipe(prompt, max_length=100,
                       do_sample=args.do_sample, num_beams=N,
                       num_return_sequences=N)

    completion = completions[0]['generated_text'].split('=')[-1].strip()
    print(f"QUESTION: {row.question}\nANSWER: {row['choice']}\nDECLARATIVIZED:{completion}")
    rdict = deepcopy(row.to_dict())
    rdict['declarativized'] = completion
    with open(partial_results, 'a') as f:
        f.write(json.dumps(rdict))
        f.write("\n")
    return completion


tqdm.pandas()

if args.dry_run:
    print(eval_qa2d.head(5).progress_apply(declarativize, axis=1))
else:
    if args.num_workers == 1:
        eval_qa2d['declarativized'] = eval_qa2d.progress_apply(declarativize, axis=1)
        eval_qa2d.to_csv(f"{results_path}.tsv", sep='\t')
    else:
        wid = args.worker_id
        partial_df = eval_qa2d.iloc[wid::args.num_workers].copy()
        partial_df['declarativized'] = partial_df.progress_apply(declarativize, axis=1)
        partial_df.to_csv(f"{results_path}_{wid}.tsv", sep='\t')
