import re
import sys
import time
from copy import copy, deepcopy
from itertools import product
from typing import List, Optional

import torch
import numpy as np

from src.rule_filters import RuleFilter, EVScorer
from src.utils import remove_duplicates, flatten
from src.utils.hf_utils import FACT12_GEN_PROMPT, FACT2_GEN_PROMPT, FACT12_OUTPUT_PATTERN, FACT12_GEN_PROMPT_WITH_RELS, \
    FACT12_GEN_PROMPT_WITH_TEMPLATES
from problog.logic import list2term, Term, make_safe, unquote, term2str, is_list, term2list, Constant
from problog.extern import problog_export

from transformers import PrefixConstrainedLogitsProcessor

import logging

from src.utils.random_mask import MASK
from src.utils.worldtree_utils import WT_TEMPLATES

logger = logging.getLogger(__name__)


# logger.setLevel(logging.INFO)


class RuleGenerator(torch.nn.Module):

    def __init__(self, model, tokenizer, gen_kwargs=None, filters=None, scorer=None, filter_confidence=0.8, update_db=False):
        super(RuleGenerator, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.filters: List[RuleFilter] = filters
        self.filter_confidence = filter_confidence
        self.scorer: Optional[EVScorer] = scorer
        self.gen_kwargs = gen_kwargs
        self.batch_size = 256
        self.update_db = update_db
        self.prediction_cache = {}

        # max return sequences per batch (that would fit on a gpu using T5 large as DRG)
        self.candidate_gpu_cap = 330 if torch.cuda.device_count() == 2 else 250
        self.max_ooms = 5

    def cuda(self, **kwargs):
        self.model.cuda(**kwargs)

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def generate(self, model_input_sequences: List[str], prefixes: List[str] = None, **kwargs) -> List[List[str]]:

        gkwargs = deepcopy(self.gen_kwargs)
        for k, v in kwargs.items():
            if k in gkwargs:
                gkwargs[k] = v

        input_ids = self.tokenizer(model_input_sequences, padding=True, return_tensors='pt').input_ids.to(
            self.model.device)

        if prefixes:
            assert len(prefixes) in [1, len(model_input_sequences)]
            # with self.tokenizer.as_target_tokenizer():
            prefix_ids = self.tokenizer(text_target=prefixes, padding=True, return_tensors='pt').input_ids
            prefix_ids = prefix_ids[:, :-1]
            prefix_ids[prefix_ids == self.tokenizer.eos_token_id] = 0
            prefix_lengths = (prefix_ids != 0).sum(axis=1)

            # model_input_sequence = FACT12_GEN_PROMPT.format(hypothesis=hypothesis)

            def next_allowed_token(batch_id, in_prefix):
                _batch_id = 0 if len(prefixes) == 1 else batch_id
                return ([prefix_ids[_batch_id, in_prefix.shape[0] - 1].item()]
                        if prefix_lengths[_batch_id].item() >= in_prefix.shape[0]
                        else torch.arange(self.model.vocab_size))

            logit_processor = PrefixConstrainedLogitsProcessor(next_allowed_token,
                                                               num_beams=gkwargs['num_return_sequences'])
        else:
            logit_processor = None

        ooms = 0
        generated_tokens = None

        while generated_tokens is None and ooms <= self.max_ooms:
            try:
                generated_tokens = self.model.generate(
                    input_ids, logits_processor=[logit_processor] if prefixes else [],
                    **gkwargs
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    ooms += 1
                    if ooms == self.max_ooms:
                        raise e
                    self._log_oom(e)
                    old_num_outputs = gkwargs['num_return_sequences']
                    new_num_outputs = int(old_num_outputs * .9)
                    logger.warning(
                        f"attempting to recover from OOM, decreasing num return sequences by 10% "
                        f"({old_num_outputs} -> {new_num_outputs})"
                    )
                    gkwargs['num_return_sequences'] = new_num_outputs
                    torch.cuda.empty_cache()
            except Exception as e:
                print(e)
        if generated_tokens is None:
            return [[] for _ in model_input_sequences]
        else:
            generated_tokens = generated_tokens.cpu().numpy()
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_pred_sets = np.array(decoded_preds).reshape(-1, gkwargs['num_return_sequences']).tolist()
            decoded_pred_sets = [remove_duplicates(s) for s in decoded_pred_sets]
            logger.debug(decoded_pred_sets[0][:20])
            return decoded_pred_sets

    def demo(self, hypothesis, fact1, fact1_rel, fact2_rel):
        return self(*[Term(f"'{x}'") for x in [hypothesis, fact1, fact1_rel, fact2_rel]])

    def demo_infill(self, prompt):
        decoded_pred_sets = self.generate([prompt])
        return decoded_pred_sets

    def forward(self, hypothesis: Term, fact1: Term, fact1_rel: Term, fact2_rel: Term):
        # TODO if end up using this base class for prediction, make sure it uses the pred cache for bfs
        [fact1, fact1_rel, fact2_rel] = [unquote(term2str(x)) for x in [fact1, fact1_rel, fact2_rel]]
        hypothesis = unquote(term2str(hypothesis))
        generating_single_fact = fact1 != 'none'

        # model_input_sequence = FACT2_GEN_PROMPT.format(hypothesis=hypothesis, fact1=fact1)
        # model_input_sequence = FACT12_GEN_PROMPT.format(hypothesis=hypothesis, fact1_rel=fact1_rel)
        model_input_sequence = FACT12_GEN_PROMPT_WITH_RELS.format(hypothesis=hypothesis, fact1_rel=fact1_rel,
                                                                  fact2_rel=fact2_rel)

        decoded_pred_sets = self.generate([model_input_sequence],
                                          prefixes=[f"fact1: {fact1} fact2:"] if generating_single_fact else None)

        preds = []
        linfo = {}
        linfo['h'] = hypothesis
        for pred in decoded_pred_sets[0]:
            try:
                pfact1, pfact2 = re.findall(FACT12_OUTPUT_PATTERN.re, pred)[0]
                preds.append(dict(hypothesis=hypothesis, fact1=pfact1, fact2=pfact2))
            except:
                continue
        if generating_single_fact:
            linfo['fact_1'] = fact1
        linfo['generated'] = len(preds)
        if self.filters:
            for filter in self.filters:
                if not preds: continue
                preds = filter.filter_candidates(preds, threshold=self.filter_confidence)
        linfo['filtered'] = len(preds)
        logger.info(f"DRG: {linfo}")

        if self.scorer:
            scores = self.scorer.score_candidates(preds)
        else:
            scores = [1.0 for _ in preds]

        ret = list2term([
            list2term([Term(make_safe(pred['fact1'])), Term(make_safe(pred['fact2'])), Constant(score)])
            for (pred, score) in zip(preds, scores)
        ])

        return ret

    def add_links_to_db(self, h_str, preds, scores, first_facts_retrieved=False):
        if not preds:
            trm = Term.from_string(f"no_expansions('{h_str}')")
            problog_export.database += trm
        for (pred, score) in zip(preds, scores):
            trm = Term.from_string(f"drg_link('{h_str}', '{pred['fact1']}', '{pred['fact2']}', {score})")
            problog_export.database += trm
            if first_facts_retrieved:
                trm = Term.from_string(f"is_fact('{pred['fact1']}')")
                problog_export.database += trm


class TemplateConditionedRuleGenerator(RuleGenerator):
    def __init__(self, *args, use_templates=True, outputs_per_template=1,
                 templates_top_k=1000, templates_max_generation_calls=10,
                 templates_greedy_generation=True,
                 templates_stochastic_generation=True,
                 second_vanilla_generate=False,
                 debug=False, **kwargs):
        super(TemplateConditionedRuleGenerator, self).__init__(*args, **kwargs)
        self.outputs_per_template = outputs_per_template
        if not use_templates:
            self.outputs_per_template = 0
        self.templates_max_generation_calls = templates_max_generation_calls
        self.templates_greedy_generation = templates_greedy_generation
        self.templates_stochastic_generation = templates_stochastic_generation
        self.second_vanilla_generate = second_vanilla_generate
        self.debug = debug
        self.wt_templates = [wtt.replace("<mask>", MASK) for wtt in
                             flatten(WT_TEMPLATES(topk=templates_top_k).values())]

    def predict(self, hypothesis: str, fact1s: List[str]):
        sttime = time.time()

        generating_single_fact = fact1s != ['none']
        linfo = {}
        linfo['h'] = hypothesis
        if generating_single_fact:
            linfo['n_fact_1'] = len(fact1s)

        if generating_single_fact:
            # order = [[MASK], self.wt_templates]  # use templates for fact 2 only
            order = [[MASK], [MASK]]  # use templates for neither
        else:
            order = [self.wt_templates, [MASK]]  # use templates for fact 1 only

        templated_model_input_sequences = [
            FACT12_GEN_PROMPT_WITH_TEMPLATES.format(hypothesis=hypothesis,
                                                    fact1_template=f1t,
                                                    fact2_template=f2t)
            for f1t, f2t in product(*order)
        ]

        no_template_input_sequence = FACT12_GEN_PROMPT_WITH_TEMPLATES.format(hypothesis=hypothesis,
                                                                             fact1_template=MASK,
                                                                             fact2_template=MASK)
        # try:
        no_template_decoded_pred_sets = self.generate(
            [no_template_input_sequence for _ in fact1s],
            prefixes=[f"fact1: {f1} fact2:" for f1 in fact1s] if generating_single_fact else None,
            num_return_sequences=min(self.gen_kwargs['num_return_sequences'],
                                     int(self.candidate_gpu_cap / len(fact1s))))

        if self.second_vanilla_generate and not generating_single_fact:
            # generate as many outputs as templates would have provided
            second_set = self.generate(
                [no_template_input_sequence for _ in fact1s],
                prefixes=None,
                num_return_sequences=self.candidate_gpu_cap)
            no_template_decoded_pred_sets = [
                flatten(ls) for ls in
                zip(no_template_decoded_pred_sets,
                    second_set)
            ]

        # except Exception as e:
        #     print(e)

        gen1time = time.time()

        linfo['vanilla_gen_time'] = f"{gen1time - sttime:.2f}"

        if generating_single_fact:
            greedy_vanilla_st_time = time.time()
            greedy_no_template_decoded_pred_sets = self.generate(
                [no_template_input_sequence for _ in fact1s],
                prefixes=[f"fact1: {f1} fact2:" for f1 in fact1s] if generating_single_fact else None,
                num_beams=1, do_sample=False,
                num_return_sequences=1
            )
            linfo['greedy_vanilla_gen_time'] = f"{time.time() - greedy_vanilla_st_time:.2f}"
        else:
            greedy_no_template_decoded_pred_sets = [[] for _ in fact1s]

        if generating_single_fact or not self.outputs_per_template:
            decoded_pred_sets = [
                flatten(ls) for ls in
                zip(no_template_decoded_pred_sets,
                    greedy_no_template_decoded_pred_sets)
            ]
            gen2time = time.time()
        else:
            template_gen_times = []
            template_conditioned_decoded_pred_sets = []

            ## compute cap according to templates max generation
            max_num_calls = self.templates_max_generation_calls
            templates_per_call = len(templated_model_input_sequences) / max_num_calls
            max_outputs_per_template = int(self.candidate_gpu_cap / templates_per_call)
            actual_outputs_per_template = min(max_outputs_per_template, self.outputs_per_template)
            batch_size = int(self.candidate_gpu_cap / actual_outputs_per_template)
            linfo['template_batch_size'] = batch_size
            linfo['outputs_per_template'] = actual_outputs_per_template
            # for f1 in fact1s:
            if self.templates_stochastic_generation:
                for i in range(0, len(templated_model_input_sequences),
                               batch_size):
                    in_batch = templated_model_input_sequences[i:i + batch_size]
                    tsg_sttime = time.time()
                    template_conditioned_decoded_pred_sets.append(
                        self.generate(
                            in_batch,
                            # prefixes=[f"fact1: {f1} fact2:"] if generating_single_fact else None,
                            prefixes=None,
                            # top_p=self.gen_kwargs['top_p'],
                            num_return_sequences=actual_outputs_per_template)
                    )
                    template_gen_times.append(f"{time.time() - tsg_sttime:.2f}")
            if self.templates_greedy_generation:
                tgg_sttime = time.time()
                greedy_template_outputs = self.generate(
                    templated_model_input_sequences,
                    num_beams=1, do_sample=False,
                    prefixes=None,
                    num_return_sequences=1)
                template_gen_times.append(f"{time.time() - tgg_sttime:.2f}")
                linfo['template_greedy_gen_time'] = template_gen_times[-1]
                template_conditioned_decoded_pred_sets.append(
                    greedy_template_outputs
                )
            decoded_pred_sets = [
                flatten(ls) for ls in
                zip(no_template_decoded_pred_sets,
                    *template_conditioned_decoded_pred_sets)
            ]
            gen2time = time.time()
            linfo['template_gen_time_total'] = f"{gen2time - gen1time:.2f}"
            linfo['template_gen_times'] = template_gen_times
        preds = []

        for bid, predset in enumerate(decoded_pred_sets):
            for pred in predset:
                try:
                    pfact1, pfact2 = re.findall(FACT12_OUTPUT_PATTERN.re, pred)[0]
                    preds.append(dict(hypothesis=hypothesis, fact1=pfact1, fact2=pfact2, batch_id=bid))
                except:
                    continue

        linfo['generated'] = len(preds)
        if self.filters:
            stfilters = time.time()
            linfo['filter_times'] = []
            for filter in self.filters:
                if not preds: continue
                preds = filter.filter_candidates(preds, threshold=self.filter_confidence)
                curr_time = time.time()
                linfo['filter_times'].append(f"{curr_time - stfilters:0.2f}")
                stfilters = curr_time

            linfo['filtered'] = len(preds)
        if self.scorer:
            scores = self.scorer.score_candidates(preds)
            # breakpoint()
        else:
            scores = [1.0 for _ in preds]

        logger.info(f"DRG: {linfo}")
        if self.debug:
            breakpoint()
        if preds:
            [preds, scores] = [list(l) for l in zip(*sorted(zip(preds,scores), key=lambda x: -x[1]))]

        return preds, scores

    def forward(self, hypothesis: Term, fact1: Term, fact1_rel: Term, fact2_rel: Term):
        [fact1_rel, fact2_rel] = [unquote(term2str(x)) for x in [fact1_rel, fact2_rel]]
        if is_list(fact1):
            fact1s = [unquote(term2str(x)) for x in term2list(fact1)]
        else:
            fact1s = [unquote(term2str(fact1))]
        hypothesis = unquote(term2str(hypothesis))
        cache_key = str((hypothesis, sorted(fact1s)))

        if cache_key in self.prediction_cache:
            preds, scores = self.prediction_cache[cache_key]
        else:
            preds, scores = self.predict(hypothesis, fact1s)
            self.prediction_cache[cache_key] = (preds, scores)

        ret = list2term([
            list2term([Term(make_safe(pred['fact1'])), Term(make_safe(pred['fact2'])), Constant(score)])
            for (pred, score) in zip(preds, scores)
        ])
        # print(ret)
        if self.update_db:
            self.add_links_to_db(hypothesis, preds, scores, first_facts_retrieved=(fact1s != ['none']))

        return ret
