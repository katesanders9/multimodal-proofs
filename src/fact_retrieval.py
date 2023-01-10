''' From Nathaniel's NELLIE codebase. Defines the main fact retrieval engine class.'''

import logging
import os
import time
import faiss
import torch
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Optional
from traitlets import Int

from problog.extern import problog_export
from problog.logic import Term, list2term, make_safe, term2str, Constant, unquote, term2list

from src.rule_filters import EntailmentScorer
from src.utils import read_sentences, normalize_text, flatten
from src.utils.retrieval_utils import RetrievalEncoder, SBERTRetrievalEncoder

logger = logging.getLogger(__name__)


class FactRetrievalEngine(torch.nn.Module):

    def __init__(self, index=None, data=None, encoder_model=None, entailment_filters: List = None,
                 support_filters: List = None, max_support_facts: Int = 10,
                 allow_first_fact_retrieval=True, save_entailments=False,
                 scorer=None):
        super(FactRetrievalEngine, self).__init__()
        self.encoder: RetrievalEncoder = \
            encoder_model if (encoder_model is not None and type(encoder_model) != 'str') \
                else SBERTRetrievalEncoder(sbert_model=encoder_model)
        self.data = data if data is not None else []
        embedding_size = self.encoder.embedding_size if self.encoder is not None else 768
        self.max_support_facts = max_support_facts
        self.index: faiss.IndexIDMap = (
            index if index is not None
            else faiss.IndexIDMap(faiss.IndexFlatIP(embedding_size))
        )
        self.entailment_filters = entailment_filters
        self.support_filters = support_filters
        self.scorer: Optional[EntailmentScorer] = scorer
        self.result_cache = {}
        self.prediction_cache = {}
        self.save_entailment_results = save_entailments
        self.allow_first_fact_retrieval = allow_first_fact_retrieval

    def cuda(self, device=None):
        self.encoder.cuda(device=device)

    def save(self, file):
        faiss.write_index(self.index, file + ".index")
        np.save(file + ".data", self.data)

    @classmethod
    def from_file(cls, file, read_data=True, model=None, entailment_filters: List = None,
                  support_filters: List = None, **kwargs):
        assert os.path.exists(file + ".index")
        index = faiss.read_index(file + ".index")
        data = None
        if read_data:
            data = np.load(file + ".data.npy", allow_pickle=True)
        return cls(index=index, data=data, encoder_model=model, entailment_filters=entailment_filters,
                   support_filters=support_filters, **kwargs)

    def __getitem__(self, item):
        return self.data[item]

    def index_to_gpu(self):
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def add_to_index(self, embeddings):
        self.index.add_with_ids(embeddings, np.arange(self.index.ntotal, self.index.ntotal + embeddings.shape[0]))

    def _search(self, query_list: List[str], topk=100, threshold: float = 0):  # -> List[Tuple[Float, Int, str]]
        """

        :param query_list: list of string queries
        :param topk: how many closest neighbors to return
        :param threshold: similarity threshold
        :return: list of ids and similarity scores
        """
        logger.debug(f"querying db for {query_list}")
        query_list = [normalize_text(t) for t in query_list]
        _query_set = list(set([q for q in query_list if q not in self.result_cache]))
        if _query_set:
            query_vectors = self.encoder.encode_queries(_query_set)
            logger.debug(f"searching index for query vectors of shape {query_vectors.shape}")
            similarities, indices = self.index.search(query_vectors, k=topk)
            retset = []
            for qset_idx, (scores_i, idx_i) in enumerate(zip(similarities, indices)):
                ret_i = []  # td
                for score_ij, idx_ij in zip(scores_i, idx_i):
                    if score_ij < threshold: break
                    ret_i.append((score_ij, idx_ij))
                self.result_cache[_query_set[qset_idx]] = ret_i
                retset.append(ret_i)
        else:
            retset = []

        ret = [retset[_query_set.index(query)]
               if query in _query_set else
               [(sc, i) for (sc, i) in self.result_cache[query] if sc >= threshold]
               for query in query_list]
        return ret

    def add_links_to_db(self, h_strs, search_results, mode=None):
        for h_str, search_results_i in zip(h_strs, search_results):
            if search_results_i:
                res = search_results_i[0]
                support_fact = res['sentence']
                score = res['score']
                if mode == 'entailment':
                    trm = Term.from_string(f"entailed('{h_str}', '{support_fact}', {score})")
                    problog_export.database += trm
                trm = Term.from_string(f"is_fact('{support_fact}')")
                problog_export.database += trm

    def postprocess_search_results(self, search_results: List[List[Dict]], mode='unification') -> Term:
        if mode in ['entailment', 'support_fact']:
            ret = []
            for search_results_i in search_results:
                if mode == 'entailment':
                    search_results_i = search_results_i[:1]
                if mode == 'support_fact':
                    search_results_i = search_results_i[:self.max_support_facts]
                ret.append(list2term([list2term([Term(make_safe(res['sentence'])), Constant(res['score'])])
                                      for res in search_results_i]))
            ret = list2term(ret)
        else:
            raise NotImplementedError()

        return ret

    def forward(self, hypotheses: Term, mode: Term, threshold: Constant) -> Term:
        sttime = time.time()
        cache_key = str(hypotheses)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        hypotheses = [unquote(term2str(h)) for h in term2list(hypotheses)]
        mode = unquote(term2str(mode))
        if type(threshold) == Constant:
            threshold = threshold.value
        elif type(threshold) == torch.Tensor:
            threshold = threshold.item()

        if mode == 'support_fact' and not self.allow_first_fact_retrieval:
            search_results = [[] for _ in hypotheses]
        else:
            search_results = self.search(hypotheses, threshold=threshold)

        time_after_search = time.time()
        linfo = dict(n_hypothesis=len(hypotheses),
                     mode=mode,
                     search_time=f"{time_after_search - sttime:.2f}")

        if mode == 'support_fact':
            filters_to_use = self.support_filters
        elif mode == 'entailment':
            filters_to_use = self.entailment_filters
        else:
            raise NotImplementedError()

        linfo['retrieved'] = sum(len(s) for s in search_results)
        flattened_search_results = flatten(search_results)
        if filters_to_use:
            filter_inputs = []
            for i, search_results_i in enumerate(search_results):
                for (_, result) in enumerate(search_results_i):
                    filter_inputs.append(
                        dict(idx=len(filter_inputs), res_index=i,
                             hypothesis=hypotheses[i], premise=result['sentence'])
                    )
            linfo['filter_time'] = []
            for filter in filters_to_use:
                f_sttime = time.time()
                filter_inputs = filter.filter_candidates(filter_inputs)
                linfo['filter_time'].append(f"{time.time() - f_sttime:.2f}")

            if self.scorer is not None and mode == 'entailment' and any(filter_inputs):
                scores = self.scorer.score_candidates(filter_inputs)
                for sc, finput in zip(scores, filter_inputs):
                    finput['score'] = sc

            new_results = []
            for (i, original_search_result) in enumerate(search_results):
                sr_i = []
                for inp in filter_inputs:
                    if inp['res_index'] == i:
                        sr_i.append(flattened_search_results[inp['idx']])
                        if 'score' in inp:
                            sr_i[-1]['score'] = inp['score']

                new_results.append(sr_i)

            search_results = new_results
            linfo['filtered'] = sum(len(s) for s in search_results)

        logger.info(f"Retrieval: {linfo}")
        if self.save_entailment_results:
            self.add_links_to_db(hypotheses, search_results, mode)

        ret = self.postprocess_search_results(search_results, mode=mode)
        self.prediction_cache[cache_key] = ret
        return ret

    def search(self, *args, **kwargs) -> List[List[Dict]]:
        raise NotImplementedError()



if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = ArgumentParser()
    parser.add_argument("data", choices=['qasc', 'wt'], default='wt')
    parser.add_argument('--encoder_type', choices=['sbert', 'bert', 'sbert_msmarco', 'sbert_support_fact'],
                        default='sbert_support_fact')
    parser.add_argument("--add-eb-facts", action='store_true')
    parser.add_argument("--full-unroll-slotfills", action='store_true')
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    encoder = RetrievalEncoder.build_encoder(args.encoder_type)

    if args.data == 'qasc':
        engine = SentenceFactRetrievalEngine(encoder_model=encoder, data=None)
        engine.cuda()
        engine.index_data_from_file('data/eqasc/QASC_Corpus/QASC_Corpus.txt', progbar=True, batch_size=16384,  # 1024
                                    max_sentences=None, debug=args.debug)
        print(engine.data[:10])
        print(engine.index.ntotal)
        print(engine.search(['a hawk uses a beak to hunt']))
        engine.save(f'data/eqasc/{args.encoder_type}_indexed_qasc')

    else:
        raise NotImplementedError()
