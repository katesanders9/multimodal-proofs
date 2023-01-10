''' From Nathaniel's NELLIE codebase.'''

from typing import List, Union, Optional
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import os
from src.utils.sbert_utils import SBERT, SBERT_MSMARCO


class RetrievalEncoder(torch.nn.Module):
    def encode_queries(self, query_list, batch_size=64):
        raise NotImplementedError()

    def encode_facts(self, fact_list, batch_size=64):
        raise NotImplementedError()

    @classmethod
    def build_encoder(cls, name):
        if name in ['sbert', 'sbert_msmarco']:
            return SBERTRetrievalEncoder(sbert_model=name)
        # elif name == 'bert':
        #     return BERTRetrievalEncoder()
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

