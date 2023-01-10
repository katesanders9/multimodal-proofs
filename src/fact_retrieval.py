import json
import logging
import os
import time
from argparse import ArgumentParser
from glob import glob
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from problog.extern import problog_export
from problog.logic import Term, list2term, make_safe, term2str, Constant, unquote, term2list
from traitlets import Int

from src.rule_filters import EntailmentScorer
from src.utils import read_sentences, normalize_text, flatten
from src.utils.retrieval_utils import RetrievalEncoder, SBERTRetrievalEncoder
from src.utils.worldtree_utils import WorldTreeFact, WT_ID_COL, WorldTree

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
                    # print(score_ij, idx_ij)
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


class SentenceFactRetrievalEngine(FactRetrievalEngine):
    def __init__(self, *args, **kwargs):
        super(SentenceFactRetrievalEngine, self).__init__(*args, **kwargs)
        if kwargs['data'] is not None and kwargs['index'] is None:
            self.index_data(self.data)

    def index_data(self, data: List[str], batch_size: int = 64):
        encoded_data = self.encoder.encode_facts(data, batch_size=batch_size)
        self.add_to_index(encoded_data)

    def index_data_from_file(self, file, batch_size: int = 64, **kwargs):
        new_data = []
        for i, text in enumerate(read_sentences(file, **kwargs)):
            new_data.append(normalize_text(text))
            if kwargs['debug'] and i > 10000:
                break
        self.data.extend(new_data)
        self.index_data(new_data, batch_size=batch_size)

    def search(self, *args, **kwargs) -> List[List[Dict]]:
        search_result = self._search(*args, **kwargs)
        return [[dict(score=sim_score, id=id, sentence=self[id]) for sim_score, id in search_result_i]
                for search_result_i in search_result]


class WorldTreeFactRetrievalEngine(FactRetrievalEngine):
    def __init__(self, worldtree_path="data/worldtree/tablestore/v2.1/tables/",
                 data=None, include_eb_added_facts=False,
                 fully_unroll_slotfills=False,
                 **kwargs):
        super(WorldTreeFactRetrievalEngine, self).__init__(data=data, **kwargs)
        self.wt_tables = {}
        self.column_names = ['id', 'table_name', 'sentence']
        self.include_eb_added_facts = include_eb_added_facts
        self.fully_unroll_slotfills = fully_unroll_slotfills
        for f in glob(os.path.join(worldtree_path, '*.tsv')):
            _df = pd.read_csv(f, sep='\t')
            if _df.columns[-1].startswith('Unn'): _df = _df[_df.columns[:-1]]
            self.wt_tables[f.split("/")[-1].split('.')[0]] = _df

        if data is None:
            self.data = self.get_sentences()
        if type(self.data) == np.ndarray:
            self.data = pd.DataFrame(self.data, columns=self.column_names)

        # for gold tree reconstruction
        self.exclude_list = None

    def id_lookup(self, idstr):
        for (k, wdf) in self.wt_tables.items():
            match = wdf[wdf[WT_ID_COL] == idstr]
            if not match.empty:
                return k, match.iloc[0, :]

        return None

    def reverse_lookup(self, factstr):
        matches = self.data.query(f"sentence == '{factstr}'")
        if matches.shape[0] < 1:
            # print(f"fact {factstr} not found in factbase!!!")
            return ''
        else:
            return matches.iloc[0].get("id", None)

    def set_exclude_list(self, exclude_list):
        self.exclude_list = exclude_list

    def clear_exclude_list(self):
        self.exclude_list = None

    def get_sentences(self, entailmentbank_path="data/entailmentbank"):  # columns: id, table_name, nl
        ret = []
        if self.fully_unroll_slotfills:
            wt = WorldTree()
            ret = wt.to_lookup_corpus()
        else:
            for (tid, table) in self.wt_tables.items():
                ret.extend(
                    table.apply(lambda x: (x[WT_ID_COL], tid, WorldTreeFact(None, x).nl), axis=1).tolist()
                )
        if self.include_eb_added_facts:
            eb_supporting_facts = json.load(
                open(os.path.join(entailmentbank_path, 'supporting_data/worldtree_corpus_sentences_extended.json')))
            ret.extend(
                [(k, "ET_addon", normalize_text(v))
                 for (k, v) in eb_supporting_facts.items()
                 if "ET_addon" in k]
            )

        df = pd.DataFrame(ret, columns=self.column_names)
        # df.set_index('id', inplace=True)
        return df

    def index_data(self, batch_size: int = 64):
        encoded_data = self.encoder.encode_facts(self.data['sentence'], batch_size=batch_size)
        self.add_to_index(encoded_data)

    def index_data_from_file(self, file, batch_size: int = 64, **kwargs):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.data.loc[item, :]

    def search(self, *args, **kwargs) -> List[List[Dict]]:
        search_result = self._search(*args, **kwargs)

        def _allowed(uuid):
            return self.exclude_list is None or uuid not in self.exclude_list

        return [[dict(score=sim_score, **dict(self[id])) for sim_score, id in search_result_i
                 if _allowed(self[id]['id'])]
                for search_result_i in search_result]


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
        # embed(user_ns=locals())
        engine.save(f'data/eqasc/{args.encoder_type}_indexed_qasc')

    elif args.data == 'wt':
        engine = WorldTreeFactRetrievalEngine(encoder_model=encoder, include_eb_added_facts=args.add_eb_facts,
                                              fully_unroll_slotfills=args.full_unroll_slotfills)
        engine.index_data()
        print(engine.data[:10])
        print(engine.index.ntotal)
        print(engine.search(['a hawk is a kind of bird']))
        save_path = f'data/worldtree/{args.encoder_type}_indexed_worldtree'
        if args.add_eb_facts:
            save_path += '_extended'
        if args.full_unroll_slotfills:
            save_path += '_unrolled'
        engine.save(save_path)
        # embed(user_ns=locals())

    else:
        raise NotImplementedError()
