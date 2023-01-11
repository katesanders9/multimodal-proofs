import copy
from pprint import pprint

import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
import os
from utils import flatten
from timers import Timer
from functools import partial

# os.environ['RAYON_RS_NUM_CPUS'] = "1"

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from datasets import load_dataset
from tqdm import tqdm

csqa = load_dataset("json", data_files=dict(train="data/csqa2/CSQA2_train.json",
                                            dev="data/csqa2/CSQA2_dev.json"))

model_name = 'google/flan-t5-xl'
model_basename = model_name.split("/")[-1]

def init_model(device_id):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    pipe = pipeline('text2text-generation', model=model, device=device_id, tokenizer=tokenizer)
    generate.pipe = pipe


BATCH_SIZE = 256 if 'xl' not in model_name else 64


def generate(inputs, **kwargs):
    _kwargs = copy.deepcopy(kwargs)
    if "batch_size" not in _kwargs:
        _kwargs['batch_size'] = BATCH_SIZE
    if 'max_length' not in _kwargs:
        _kwargs['max_length'] = 100
    return generate.pipe(inputs, **_kwargs)


ZERO_SHOT_PROMPT = """\
Answer the following yes/no question by reasoning step-by-step.

{question}\
"""

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    n_gpus = torch.cuda.device_count()

    print("constructing world...")
    print(n_gpus)

    with Timer("Init workers"):
        init_pool = multiprocessing.Pool(processes=n_gpus)
        init_pool.map(init_model, list(range(n_gpus)))


    def splitlist(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


    def parallel_generate(prompts, **kwargs):
        split_prompts = splitlist(prompts, n_gpus)
        print(kwargs)
        with Timer("pool.map"):
            res = init_pool.map(partial(generate, **kwargs), split_prompts)

        with Timer("post process"):
            outputs = flatten(res)

        return outputs


    def answer(df, **kwargs):
        NUM_PREDS = 64
        RET_SEQ = 16
        preds = []

        gen_kwargs = dict(
            top_p=0.95, do_sample=True,
            num_return_sequences=RET_SEQ, batch_size=int(BATCH_SIZE/RET_SEQ)
        )
        pprint(gen_kwargs)
        for _ in tqdm(list(range(int(NUM_PREDS / RET_SEQ)))):
            _preds = parallel_generate(df.question.apply(lambda x: ZERO_SHOT_PROMPT.format(question=x)).tolist(),
                                      **gen_kwargs)

            preds.extend([dict(id=_id, **g) for (_id, preds_i) in zip(tr.id, _preds) for g in preds_i])

        _df = pd.DataFrame(preds)
        _df['pred'] = _df.generated_text.apply(lambda x: x.split()[-1].strip("."))
        _df = _df.merge(df, on="id")
        _df['correct'] = (_df.pred  == _df.answer)
        return _df

    tr = csqa['dev'].to_pandas()
    df = answer(tr)
    df.to_csv(f"csqa_{model_basename}.tsv", sep='\t', index=False)

    breakpoint()

    print(tr.head())
    init_pool.close()
    init_pool.join()
