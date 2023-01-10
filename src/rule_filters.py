''' From Nathaniel's NELLIE codebase. Necessary for fact_retrieval.py.'''

from typing import List

import datasets
import torch
from problog.logic import Term, unquote, term2str, make_safe, is_list, term2list
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import paired_cosine_distances
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq, pipeline
)

from src.utils.hf_utils import EV_PROMPT, MNLI_PROMPT, auto_load_model
from src.utils.pattern_utils import InferencePattern
from src.utils.sbert_utils import SBERT, ENTAILMENT_CROSS_ENCODER, EV_CROSS_ENCODER, EV_UNIFICATION_ENCODER


class RuleFilter(torch.nn.Module):

    def filter_candidates(self, candidates: List[dict], **kwargs) -> List[dict]:
        """
        Applies filter to candidate list (dict of hypothesis, fact1 and fact2) and returns remaining candidates

        :param candidates:
        :return:
        """
        raise NotImplementedError()


class EVClassificationFilter(RuleFilter):
    def __init__(self, model, tokenizer, batch_size=32):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100
        )

    @property
    def device(self):
        return self.model.device

    def filter_candidates(self, candidates: List[dict], threshold: float = .80) -> List[dict]:
        filter_inputs = [EV_PROMPT.format(**c) for c in candidates]
        filter_inputs = self.tokenizer(filter_inputs, max_length=100, padding=False, truncation=True)
        filter_dataset = datasets.arrow_dataset.Dataset.from_dict(filter_inputs)
        eval_dataloader = DataLoader(filter_dataset, collate_fn=self.data_collator, batch_size=self.batch_size)
        labels = []
        for f_batch in eval_dataloader:
            generation_output = self.model.generate(
                f_batch["input_ids"].to(self.device),
                attention_mask=f_batch["attention_mask"].to(self.device),
                num_return_sequences=2, num_beams=2,
                output_scores=True, return_dict_in_generate=True,
                max_new_tokens=5
            )
            sequence_scores = torch.exp(generation_output.sequences_scores).view(-1, 2)
            classification_scores = sequence_scores[:, 0] / sequence_scores.sum(axis=1)
            generated_tokens = generation_output.sequences
            decoded_filter_labels = self.tokenizer.batch_decode(
                generated_tokens.view(-1, 2, generated_tokens.shape[-1])[:, 0], skip_special_tokens=True)
            labels.extend([1 if label == 'entailment' and cscore > threshold else 0
                           for (label, cscore) in zip(decoded_filter_labels, classification_scores)])

        filtered_candidates = [cand for (cand, label) in zip(candidates, labels) if label]
        return filtered_candidates



class RegexEVFilter(RuleFilter):
    def __init__(self):
        super().__init__()
        self.pdicts = [
            dict(hypothesis="(a |an )?<X> is a (kind of)? <Y>",
                 fact1="(a |an |the )?<X> (is|are) (a kind of|a type of|an example of)? <Z>(s)?",
                 fact2="(a |an |the )?<Y> (is|are) (a kind of|a type of|an example of)? <Z>(s)?"),
            dict(hypothesis="(a |an )?<X> is a (kind of)? <Y>",
                 fact1="(a |an |the )?<X> (is|are) (a kind of|a type of|an example of)? <Z>(s)?",
                 fact2="(a |an |the )?<Z> (is|are) (a kind of|a type of|an example of)? <Y>(s)?"),
            dict(hypothesis="<X>",
                 fact1="(a |an |the )?<Y> (is|are) a (kind of|type of)? (object|material)",
                 fact2="an (object|material) <Z>"),

        ]
        self.patterns = [InferencePattern(p) for p in self.pdicts]

    def filter_candidates(self, candidates: List[dict], **kwargs) -> List[dict]:
        if not candidates:
            return candidates
        assert all(["hypothesis" in c and 'fact1' in c and 'fact2' in c for c in candidates])

        filtered_candidates = []
        for cand in candidates:
            if any(pat.match(cand) for pat in self.patterns):
                continue
            flipped_cand = dict(hypothesis=cand['hypothesis'], fact1=cand['fact2'], fact2=cand['fact1'])
            if any(pat.match(flipped_cand) for pat in self.patterns):
                continue
            filtered_candidates.append(cand)

        return filtered_candidates


class SBERTSimilarityFilter(RuleFilter):
    # rules out duplicates via SBERT thresholding

    def __init__(self, model=None, batch_size=32, threshold=0.85, multi_gpu=False):
        super().__init__()
        self.model = model if model is not None else SBERT()
        self.batch_size = batch_size
        self.threshold = threshold
        self.multi_gpu = multi_gpu

    def encode_sentences(self, sentences: List[str]):
        if self.multi_gpu:
            pool = self.model.start_multi_process_pool()
            encoded_sentences = self.model.encode_multi_process(sentences, pool, batch_size=self.batch_size)
            self.model.stop_multi_process_pool(pool)
        else:
            encoded_sentences = self.model.encode(sentences, batch_size=self.batch_size, show_progress_bar=False)

        return encoded_sentences

    def filter_candidates(self, candidates: List[dict], **kwargs) -> List[dict]:
        fact1_list = [c['fact1'] for c in candidates]
        fact2_list = [c['fact2'] for c in candidates]

        fact1_embs = self.encode_sentences(fact1_list)
        fact2_embs = self.encode_sentences(fact2_list)

        fact1_sims = cos_sim(fact1_embs, fact1_embs)
        fact2_sims = cos_sim(fact2_embs, fact2_embs)
        both_over_threshold = (torch.logical_and(fact1_sims > self.threshold, fact2_sims > self.threshold))

        ret_idx = []
        for c_idx in range(len(candidates)):
            if not any([both_over_threshold[r_idx, c_idx].item() for r_idx in ret_idx]):
                ret_idx.append(c_idx)
        return [candidates[i] for i in ret_idx]


class Seq2SeqEntailmentFilter(RuleFilter):
    def __init__(self, model, tokenizer, batch_size=16):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100
        )

    @property
    def device(self):
        return self.model.device

    def filter_candidates(self, candidates: List[dict], threshold: float = .7) -> List[dict]:
        if not candidates:
            return candidates
        assert all(["hypothesis" in c and 'premise' in c for c in candidates])
        filter_inputs = [MNLI_PROMPT.format(**c) for c in candidates]
        filter_inputs = self.tokenizer(filter_inputs, max_length=100, padding=False, truncation=True)
        filter_dataset = datasets.arrow_dataset.Dataset.from_dict(filter_inputs)
        eval_dataloader = DataLoader(filter_dataset, collate_fn=self.data_collator, batch_size=self.batch_size)
        filtered_candidates = []
        for f_batch in eval_dataloader:
            generation_output = self.model.generate(
                f_batch["input_ids"].to(self.device),
                attention_mask=f_batch["attention_mask"].to(self.device),
                num_return_sequences=2, num_beams=2,
                output_scores=True, return_dict_in_generate=True,
                max_new_tokens=5
            )
            sequence_scores = torch.exp(generation_output.sequences_scores).view(-1, 2)
            classification_scores = sequence_scores[:, 0] / sequence_scores.sum(axis=1)
            generated_tokens = generation_output.sequences
            decoded_filter_labels = self.tokenizer.batch_decode(
                generated_tokens.view(-1, 2, generated_tokens.shape[-1])[:, 0], skip_special_tokens=True)

            batch_filtered_candidates = [cand for (c_idx, (cand, label)) in
                                         enumerate(zip(candidates, decoded_filter_labels))
                                         if label == 'entailment' and classification_scores[c_idx] > threshold]

            filtered_candidates.extend(batch_filtered_candidates)

        return filtered_candidates

    def forward(self, hypothesis: Term, premise: Term):
        hypothesis = unquote(term2str(hypothesis))
        premise = unquote(term2str(premise))
        _candidates = [dict(hypothesis=hypothesis, premise=premise)]
        _filtered = self.filter_candidates(_candidates)
        if _filtered:
            return Term(make_safe("entailment"))
        else:
            return Term(make_safe("contradiction"))


class ClassificationEntailmentFilter(RuleFilter):
    def __init__(self, model, tokenizer, batch_size=128):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pipeline = None

    @property
    def device(self):
        return self.model.device

    def filter_candidates(self, candidates: List[dict], threshold: float = .7) -> List[dict]:
        if not candidates:
            return candidates
        assert all(["hypothesis" in c and 'premise' in c for c in candidates])
        if self.pipeline is None:
            self.pipeline = pipeline('text-classification',
                                     model=self.model, tokenizer=self.tokenizer,
                                     batch_size=self.batch_size, device=self.model.device)

        preds = self.pipeline([dict(text=cand['premise'], text_pair=cand['hypothesis']) for cand in candidates])

        filtered_candidates = [
            cand for (pred, cand) in zip(preds, candidates)
            if pred['label'] == 'entailment' and pred['score'] > threshold
        ]

        return filtered_candidates


CONFIDENCE = lambda x: max(x - 0.5, 0) * 2


class EVScorer:
    def score_candidates(self, candidates: List[dict]) -> List[float]:
        raise NotImplementedError

class EntailmentScorer:
    def score_candidates(self, candidates: List[dict]) -> List[float]:
        raise NotImplementedError

class CrossEncoderEntailmentFilter(RuleFilter, EntailmentScorer):
    # entailment filter using sentence_transformer cross encoder
    # updates scores as well

    def __init__(self, model=None, batch_size=128):
        super().__init__()
        self.model = model if model is not None else ENTAILMENT_CROSS_ENCODER()
        self.batch_size = batch_size

    def filter_candidates(self, candidates: List[dict], threshold: float = .7) -> List[dict]:
        if not candidates:
            return candidates
        assert all(["hypothesis" in c and 'premise' in c for c in candidates])
        scores = self.model.predict([(c['premise'], c['hypothesis']) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False, apply_softmax=True)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = []

        for scores_i, score_max in zip(scores, scores.argmax(axis=1)):
            raw_label = label_mapping[score_max]
            if raw_label == 'entailment' and scores_i[score_max].item() < threshold:
                raw_label = 'neutral'
            labels.append(raw_label)

        filtered_candidates = [cand for label, cand in zip(labels, candidates) if label == 'entailment']

        return filtered_candidates

    def score_candidates(self, candidates: List[dict]) -> List[float]:
        if not candidates:
            return []
        assert all(["hypothesis" in c and 'premise' in c for c in candidates])
        scores = self.model.predict([(c['premise'], c['hypothesis']) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False, apply_softmax=True)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        return_scores = []

        for scores_i, score_max in zip(scores, scores.argmax(axis=1)):
            raw_label = label_mapping[score_max]
            if raw_label != 'entailment':
                return_scores.append(0)
            else:
                return_scores.append(CONFIDENCE(scores_i[score_max].item()))

        return return_scores

class SBERTEVScorer(EVScorer):
    def __init__(self, model=None, batch_size=128):
        super().__init__()
        self.model = model if model is not None else EV_UNIFICATION_ENCODER()
        self.batch_size = batch_size

    def score_candidates(self, candidates: List[dict]) -> List[float]:
        """
        for filtering a list of generated candidate decompositions
        :param candidates:
        :return:
        """
        if not candidates: return []

        assert all(["hypothesis" in c and 'fact1' in c and 'fact2' in c for c in candidates])

        embeddings1 = self.model.encode([c['hypothesis'] for c in candidates],
                                        batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        embeddings2 = self.model.encode(['. '.join([c['fact1'], c['fact2']]) for c in candidates],
                                        batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True)
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        # normalize confidence
        confidence_scores = [CONFIDENCE(cs) for cs in cosine_scores]

        return confidence_scores

    def forward(self, hypothesis: Term, facts: Term):
        if is_list(facts):
            facts = [unquote(term2str(x)) for x in term2list(facts)]
        else:
            facts = [unquote(term2str(facts))]
        hypothesis = unquote(term2str(hypothesis))
        embeddings = self.model.encode(facts, batch_size=self.batch_size,
                                       show_progress_bar=False, convert_to_numpy=True)

        return None


class CrossEncoderEVFilter(RuleFilter, EVScorer):
    # entailment filter using sentence_transformer cross encoder
    # updates scores as well

    def __init__(self, model=None, batch_size=128):
        super().__init__()
        self.model = model if model is not None else EV_CROSS_ENCODER()
        self.batch_size = batch_size

    def filter_candidates(self, candidates: List[dict], threshold: float = .8) -> List[dict]:
        if not candidates:
            return candidates
        assert all(["hypothesis" in c and 'fact1' in c and 'fact2' in c for c in candidates])
        scores = self.model.predict([('. '.join([c['fact1'], c['fact2']]), c['hypothesis']) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False, apply_softmax=True)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = []

        for scores_i, score_max in zip(scores, scores.argmax(axis=1)):
            raw_label = label_mapping[score_max]
            if raw_label == 'entailment' and scores_i[score_max].item() < threshold:
                raw_label = 'neutral'
            labels.append(raw_label)

        filtered_candidates = [cand for label, cand in zip(labels, candidates) if label == 'entailment']

        return filtered_candidates

    def score_candidates(self, candidates: List[dict]) -> List[float]:
        if not candidates:
            return []
        assert all(["hypothesis" in c and 'fact1' in c and 'fact2' in c for c in candidates])
        scores = self.model.predict([('. '.join([c['fact1'], c['fact2']]), c['hypothesis']) for c in candidates],
                                    batch_size=self.batch_size, show_progress_bar=False, apply_softmax=True)
        label_mapping = ['contradiction', 'entailment', 'neutral']
        return_scores = []

        for scores_i, score_max in zip(scores, scores.argmax(axis=1)):
            raw_label = label_mapping[score_max]
            if raw_label != 'entailment':
                return_scores.append(0)
            else:
                return_scores.append(CONFIDENCE(scores_i[score_max].item()))

        return return_scores

if __name__ == "__main__":
    from datasets import load_dataset

    mnli_data = load_dataset("multi_nli")
    dev = mnli_data['validation_matched']
    pos_dev = dev.filter(lambda x: x['label'] == 0)
    pos_dev = [dict(premise=x['premise'], hypothesis=x['hypothesis']) for x in pos_dev]
    neg_dev = dev.filter(lambda x: x['label'] == 2)
    neg_dev = [dict(premise=x['premise'], hypothesis=x['hypothesis']) for x in neg_dev]

    ev_model_path = 'exp/train_ev/Baseline.baseline/outdir'
    ev_model, ev_tokenizer = auto_load_model(path=ev_model_path)
    t5_model, tokenizer = auto_load_model(path='t5-base')
    t5_model.cuda()
    ev_model.cuda()

    t5_filter = Seq2SeqEntailmentFilter(model=t5_model, tokenizer=tokenizer, batch_size=64)
    ev_filter = Seq2SeqEntailmentFilter(model=ev_model, tokenizer=ev_tokenizer, batch_size=64)

    for threshold in [0.01]:
        f_pos = t5_filter.filter_candidates(pos_dev, threshold=threshold)
        print(f"T5\t{threshold}\t{len(f_pos)}/{len(pos_dev)}")
        f_neg = t5_filter.filter_candidates(neg_dev, threshold=threshold)
        print(f"T5\t{threshold}\t{len(f_neg)}/{len(neg_dev)}")

        f_pos = ev_filter.filter_candidates(pos_dev, threshold=threshold)
        print(f"EV\t{threshold}\t{len(f_pos)}/{len(pos_dev)}")
        f_neg = ev_filter.filter_candidates(neg_dev, threshold=threshold)
        print(f"EV\t{threshold}\t{len(f_neg)}/{len(neg_dev)}")
