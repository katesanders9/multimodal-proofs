''' From Nathaniel's NELLIE codebase. Includes text_utils.py, sbert_utils.py, retrieval_utils.py, print_utils.py, and pattern_utils.py.'''

import os
import string
import re
import torch
import pprint
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

from src.utils.sbert_utils import SBERT, SBERT_MSMARCO


SBERT = lambda: SentenceTransformer('all-MiniLM-L6-v2')
ENTAILMENT_CROSS_ENCODER = lambda: CrossEncoder("brtx_exp/train_scitail/training_scitail-2022-08-10_20-51-16")
EV_CROSS_ENCODER = lambda: CrossEncoder("bfs/train_cross_encoder_ev/NegExPerPos.4+Seed.4321+UseEQASC.yes/outdir")
EV_CROSS_ENCODER_2 = lambda: CrossEncoder("bfs/train_cross_encoder_ev/NegExPerPos.0+Seed.1234+UseEQASC.yes/outdir")
EV_UNIFICATION_ENCODER = lambda: SentenceTransformer("bfs/train_sbert_ev/NegExPerPos.2+Seed.42+UseEQASC.yes/outdir")
SBERT_MSMARCO = lambda: SentenceTransformer('msmarco-distilbert-base-v4')



class InferencePattern:
    def __init__(self, pattern):
        self.patterns = pattern
        self.regexes = {k: re.compile(re.sub(r'(<[^>]+>)', "(?P\\1.+)", v)) for (k,v) in self.patterns.items()}

    def match(self, cand_dict):
        def _pull_matches(pat, text):
            result = pat.match(text)
            if result:
                return result.groupdict()
            else:
                return None

        matches = {}

        for k,reg in self.regexes.items():
            if k not in cand_dict:
                raise Exception(f"match key {k} not in candidate dict {cand_dict}")
            matchdict = _pull_matches(reg, cand_dict[k])
            # print(matchdict)
            if matchdict is None:
                return False

            for (mkey, mtext) in matchdict.items():
                if mkey in matches and mtext != matches[mkey]:
                    return False
                else:
                    matches[mkey] = mtext


        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.patterns)


class RetrievalEncoder(torch.nn.Module):
    def encode_queries(self, query_list, batch_size=64):
        raise NotImplementedError()

    def encode_facts(self, fact_list, batch_size=64):
        raise NotImplementedError()

    @classmethod
    def build_encoder(cls, name):
        if name in ['sbert', 'sbert_msmarco']:
            return SBERTRetrievalEncoder(sbert_model=name)
        elif name == 'sbert_support_fact':
            encoder = SentenceTransformer("brtx_exp/train_retrieval/output/msmarco-distilbert-base-v4-support-fact")
            return SBERTRetrievalEncoder(sbert_model=encoder)
        else:
            raise NotImplementedError()


class SBERTRetrievalEncoder(RetrievalEncoder):
    def __init__(self, sbert_model=None):
        super(SBERTRetrievalEncoder, self).__init__()
        self.sbert : Optional[SentenceTransformer] = None
        if sbert_model is None:
            self.sbert = SBERT()
        elif type(sbert_model) == str:
            if sbert_model == 'sbert':
                self.sbert = SBERT()
            elif sbert_model == 'sbert_msmarco':
                self.sbert = SBERT_MSMARCO()
        else:
            self.sbert = sbert_model
        self.embedding_size = self.sbert.get_sentence_embedding_dimension()

    def encode_facts(self, fact_list, batch_size=64):
        encoded_facts = self.sbert.encode(fact_list, batch_size=batch_size, show_progress_bar=False)
        faiss.normalize_L2(encoded_facts)
        return encoded_facts

    def encode_queries(self, query_list, batch_size=64):
        return self.encode_facts(query_list, batch_size)


class MultiProcessSentenceTransformer(torch.nn.Module):
    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        pool = self.model.start_multi_process_pool()
        encoded_data = self.model.encode_multi_process(sentences, pool=pool, batch_size=batch_size)
        self.model.stop_multi_process_pool(pool)

        if convert_to_tensor:
            encoded_data = torch.tensor(encoded_data)

        return encoded_data



def normalize_text(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.replace("  ", ' ')
    return s.strip()


pp = pprint.PrettyPrinter(width=120, compact=True)
ppr = lambda s: pp.pprint(s)