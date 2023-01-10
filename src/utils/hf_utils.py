''' From Nathaniel's NELLIE codebase.'''

import logging
from itertools import product
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer, AutoModelForSequenceClassification,
)
import re

from src.utils import flatten
from src.utils.random_mask import random_mask, MASK


@dataclass
class EngineArguments:
    drg_model_path: str = field(default=None)
    ev_model_path: str = field(default=None)
    eqasc_index_path: str = field(default='data/eqasc/sbert_support_fact_indexed_qasc')
    worldtree_index_path: str = field(default='data/worldtree/sbert_support_fact_indexed_worldtree_extended')
    test_set: str = field(default="entailmentbank")
    split: str = field(default='dev')
    results_dir: str = field(default=None)
    use_eqasc_retrieval: bool = field(default=False)
    use_generator_templates: bool = field(default=True)
    use_generator_retrieval: bool = field(default=True)
    second_vanilla_generate_call: bool = field(default=False)
    outputs_per_template : int = field(default=1)
    templates_top_k : int = field(default=1000)
    templates_max_generation_calls : int = field(default=1)
    templates_greedy_generation : bool = field(default=True)
    templates_stochastic_generation : bool = field(default=True)
    filter_confidence : float = field(default=0.8)
    use_single_h_entailment_scorer : bool = field(default=True)
    search_type : str = field(default='dfs')
    max_retrieval_set_size: int = field(default=15)
    seed: int = field(default=1234)
    worker_id: int = field(default=0)
    num_workers: int = field(default=1)
    time_limit: int = field(default=300)
    max_num_proofs: int = field(default=8)
    one_proof_on_recursions: bool = field(default=False)
    dry_run: bool = field(default=False)
    search_pdfs_dir: str = field(default=None)
    max_eval_examples: int = field(default=None)
    debug_id: str = field(default=None)
    debug_generator: bool = field(default=False)
    use_gpt_declarativizer: bool = field(default=True)
    use_delete_lists : bool = field(default=False) # for eb recall eval


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_files: List[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_files: List[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
                    "a jsonlines file."
        },
    )
    test_files: List[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    test_label: str = field(
        default='label'
    )
    task_type: str = field(
        default='ev'
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
                    "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
                    "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        # if self.train_files is None and self.validation_files is None:
        #     raise ValueError("Need a training/validation file.")

        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class GenerationArguments:
    output_dir: str = field(default=None)
    task: str = field(default="drg")
    generation_max_length: Optional[int] = field(
        default=60,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                    "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                    "to the `num_beams` value of the model configuration."
        },
    )
    # seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of generation."})
    batch_size: int = field(default=16)
    top_p: Optional[float] = field(default=0.95)
    top_k: Optional[int] = field(default=None)
    num_candidates: Optional[int] = field(default=340)
    # candidate_gpu_cap : Optional[int] = field(default=600)
    filter_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to finetuned T5 checkpoint for scoring candidate outputs"
        }
    )
    filter_batch_size: int = field(default=64)


FIRST_GPU_MEMORY = "4GB"
SUCCESSIVE_GPU_MEMORY = "21GB"
import torch


def auto_load_model(model_args=None, path=None):
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    assert not (model_args is not None and path is not None)

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f'{free_in_GB - 2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = dict(
        (i, FIRST_GPU_MEMORY if i == 0 else SUCCESSIVE_GPU_MEMORY)
        for i in range(n_gpus)
    )

    if path:
        config = AutoConfig.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast_tokenizer=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(path, config=config, revision="main",
                                                      # max_memory=max_memory
                                                      )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            # max_memory=max_memory,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def auto_load_classification_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer

class PromptPattern:
    def __init__(self, pattern):
        self.pattern = pattern
        self.fmt = re.sub(r'<[^>]+>', "{}", self.pattern)
        self.fmtdict = re.sub(r'<([^>]+)>', "{\\1}", self.pattern)
        self.re = re.sub(r'<[^>]+>', "(.+)", self.pattern)

    def format(self, **argdict):
        return self.fmtdict.format(**argdict)


FACT2_GEN_PROMPT = PromptPattern("rule generation hypothesis: <hypothesis> fact1: <fact1>")
FACT12_GEN_PROMPT = PromptPattern("rule generation hypothesis: <hypothesis>")
FACT12_GEN_PROMPT_WITH_RELS = PromptPattern("rule generation hypothesis: <hypothesis> "
                                            "fact1 relation: <fact1_rel> fact2 relation: <fact2_rel>")
FACT12_GEN_PROMPT_WITH_TEMPLATES = PromptPattern("rule generation hypothesis: <hypothesis> "
                                                 "fact1 template: <fact1_template> fact2 template: <fact2_template>")
EV_PROMPT = PromptPattern("eqasc fact1: <fact1> fact2: <fact2> hypothesis: <hypothesis>")
FACT12_OUTPUT_PATTERN = PromptPattern("fact1: <fact1> fact2: <fact2>")
MNLI_PROMPT = PromptPattern("mnli hypothesis: <hypothesis> premise: <premise>")
SUPPORT_FACT_PROMPT = PromptPattern("support fact: hypothesis: <hypothesis> fact1: <fact1>")


def process_classifier_input(ex, prompt=EV_PROMPT):
    return prompt.fmt.format(*ex[:3])


def classifier_preprocess_function(examples, tokenizer, data_args, prompt=EV_PROMPT):
    padding = "max_length" if data_args.pad_to_max_length else False
    df = pd.DataFrame(examples.data)

    def _process_input(row):
        inputs = [
            prompt.format(hypothesis=row['hypothesis'], fact1=row['fact1'], fact2=row['fact2']),
            prompt.format(hypothesis=row['hypothesis'], fact1=row['fact2'], fact2=row['fact1']),
        ]

        targets = [row["label"], row["label"]]

        return (inputs, targets)

    inputs, targets = [flatten(l) for l in zip(*df.apply(_process_input, axis=1))]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def generator_preprocess_function(examples, tokenizer, data_args, add_relation_tokens=False,
                                  add_masked_templates=False):
    padding = "max_length" if data_args.pad_to_max_length else False
    assert not (add_relation_tokens and add_masked_templates), "cannot have both relation token and templates (for now)"

    def _process_input(row):
        # only generate from entailments
        if row['label'] != 'entailment': return [], []

        if add_relation_tokens:  # (F1, F2) = T5(H, F1_TOK, F2_TOK)
            fact1_relations = ['ANY', row['fact1_rel']] if row['fact1_rel'] else ["ANY"]
            fact2_relations = ['ANY', row['fact2_rel']] if row['fact2_rel'] else ["ANY"]
            inputs = []
            targets = []
            for order in [
                [fact1_relations, fact2_relations, [row['fact1']], [row['fact2']]],
                [fact2_relations, fact1_relations, [row['fact2']], [row['fact1']]]
            ]:
                for (f1rel, f2rel, f1tgt, f2tgt) in product(*order):
                    # inputs.append(f"{hypothesis_prompt} fact1 relation: {f1rel} fact2 relation: {f2rel}")
                    inputs.append(FACT12_GEN_PROMPT_WITH_RELS.format(hypothesis=row['hypothesis'],
                                                                     fact1_rel=f1rel, fact2_rel=f2rel))
                    targets.append(FACT12_OUTPUT_PATTERN.format(fact1=f1tgt, fact2=f2tgt))
        elif add_masked_templates:  # (F1, F2) = T5(H, F1_TOK, F2_TOK)
            fact1_templates = [MASK, *[random_mask(row['fact1']) for _ in range(2)]]
            fact2_templates = [MASK, *[random_mask(row['fact2']) for _ in range(2)]]
            inputs = []
            targets = []
            for order in [
                [fact1_templates, fact2_templates, [row['fact1']], [row['fact2']]],
                [fact2_templates, fact1_templates, [row['fact2']], [row['fact1']]]
            ]:
                for (f1template, f2template, f1tgt, f2tgt) in product(*order):
                    # inputs.append(f"{hypothesis_prompt} fact1 relation: {f1rel} fact2 relation: {f2rel}")
                    inputs.append(FACT12_GEN_PROMPT_WITH_TEMPLATES.format(hypothesis=row['hypothesis'],
                                                                          fact1_template=f1template,
                                                                          fact2_template=f2template))
                    targets.append(FACT12_OUTPUT_PATTERN.format(fact1=f1tgt, fact2=f2tgt))
        else:
            inputs, targets = (
                [
                    FACT12_GEN_PROMPT.format(hypothesis=row['hypothesis']),
                    FACT12_GEN_PROMPT.format(hypothesis=row['hypothesis'])
                ], [
                    FACT12_OUTPUT_PATTERN.format(fact1=row['fact1'], fact2=row['fact2']),
                    FACT12_OUTPUT_PATTERN.format(fact1=row['fact2'], fact2=row['fact1'])
                ]
            )

        return (inputs, targets)

    df = pd.DataFrame(examples.data)

    inputs, targets = [flatten(l) for l in zip(*df.apply(_process_input, axis=1))]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    # model_inputs['ids'] = examples['id']
    return model_inputs


def generator_postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels
