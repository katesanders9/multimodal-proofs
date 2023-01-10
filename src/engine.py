''' From Nathaniel's NELLIE codebase. Defines the main NELLIE class for tree generation.'''

import sys
import logging
import copy
import graphviz
import textwrap
import torch
from nltk import Tree
from typing import List
from collections import defaultdict

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.query import Query
from deepproblog.semiring import Result
from problog.extern import problog_export
from problog.logic import term2list, Term, unquote, term2str
from problog.library.record import recorded

from src.dataset import query_from_hypothesis
from src.lm_engine import LMEngine
from src.rule_generators import TemplateConditionedRuleGenerator
from src.utils.hf_utils import auto_load_model, auto_load_classification_model
from src.utils.utils import RetrievalEncoder
from src.rule_filters import EVClassificationFilter, SBERTSimilarityFilter, \
    CrossEncoderEntailmentFilter, CrossEncoderEVFilter, RegexEVFilter, ClassificationEntailmentFilter


logging.basicConfig(
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class NELLIE:

    def __init__(self, args, generation_args):
        self.args = args
        self.generation_args = generation_args

        problog_program = ''.join(open("src/problog/problog_rules.pl", 'r'))

        logger.info("initializing SBERT")
        retrieval_encoder = RetrievalEncoder.build_encoder("sbert_support_fact")

        logger.info("loading Entailment model")
        deberta_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        deberta_nli_model, deberta_nli_tokenizer = auto_load_classification_model(deberta_model_name)
        deberta_entailment_filter = ClassificationEntailmentFilter(model=deberta_nli_model,
                                                                   tokenizer=deberta_nli_tokenizer,
                                                                   batch_size=128)
        ce_entailment_filter = CrossEncoderEntailmentFilter()

        logger.info("loading DRG model")
        drg_model, drg_tokenizer = auto_load_model(path=args.drg_model_path)

        logger.info("loading EV model")
        ev_model, ev_tokenizer = auto_load_model(path=args.ev_model_path)

        ce_ev_filter = CrossEncoderEVFilter()
        t5_ev_filter = EVClassificationFilter(model=ev_model, tokenizer=ev_tokenizer, batch_size=32)
        sbert_filter = SBERTSimilarityFilter()
        regex_filter = RegexEVFilter()

        gen_kwargs = {
            "max_new_tokens": (generation_args.generation_max_length
                               if generation_args.generation_max_length is not None
                               else drg_model.config.max_length),
            "num_beams": generation_args.generation_num_beams,
            "top_p": generation_args.top_p,
            "top_k": generation_args.top_k,
            "num_return_sequences": generation_args.num_candidates,
            "do_sample": True
        }

        drg_args = dict(model=drg_model, tokenizer=drg_tokenizer,
                        gen_kwargs=gen_kwargs, use_templates=args.use_generator_templates,
                        outputs_per_template=args.outputs_per_template,
                        templates_max_generation_calls=args.templates_max_generation_calls,
                        templates_top_k=args.templates_top_k,
                        debug=args.debug_generator,
                        second_vanilla_generate=args.second_vanilla_generate_call,
                        filters=[regex_filter,
                                 ce_ev_filter,
                                 t5_ev_filter,
                                 sbert_filter],
                        scorer=ce_ev_filter,
                        filter_confidence=args.filter_confidence)
        drg = TemplateConditionedRuleGenerator(**drg_args)

        self.cached_drg_args = drg_args
        self.drg = drg
        drg_network = Network(drg, "drg_model")

        # load any FAISS indices here

        networks = [drg_network,
                    ]

        if args.one_proof_on_recursions:
            problog_program += """
            max_1_on_recursive_calls(1).
            """

        n_gpus = torch.cuda.device_count()

        if n_gpus == 1:
            for m in [deberta_nli_model,
                      drg_model,
                      ev_model
                      ]:
                m.eval()
                m.cuda()

        elif n_gpus == 2:
            for m in [drg_model]:
                m.eval()
                m.cuda(1)

            for m in [deberta_nli_model,
                      ev_model
                      ]:
                m.eval()
                m.cuda(0)

        elif n_gpus == 3:
            for m in [drg_model]:
                m.eval()
                m.cuda(1)

            for m in [deberta_nli_model]:
                m.eval()
                m.cuda(0)

            for m in [ev_model]:
                m.eval()
                m.cuda(2)

        model = Model(program_string=problog_program, networks=networks, load=False)
        self.lm_engine = LMEngine(model)
        model.set_engine(self.lm_engine, cache=True)
        self.model = model

    def reload_drg(self, **kwargs):
        new_args = copy.copy(self.cached_drg_args)
        for k in kwargs:
            if k not in new_args:
                raise Exception(f"kwarg {k} not accepted by nellie DRG")
            new_args[k] = kwargs[k]

        self.drg = TemplateConditionedRuleGenerator(**new_args)

    def query_model(self, query_strs, depth=2, max_num_proofs=10, time_limit=3600 * 24, pretty_print=False,
                    prove_predicate='timeout_bfs_prove_no_cut', exclude_fact_ids=None):
        if type(query_strs) == str:
            query_strs = [query_strs]
        queries = [query_from_hypothesis(query_str, prove_predicate=prove_predicate, depth=depth,
                                         max_num_proofs=max_num_proofs, time_limit=time_limit)
                   for query_str in query_strs]
        logger.info(f"running queries {query_strs}")
        if exclude_fact_ids is not None:
            print(exclude_fact_ids)
            self.retriever.set_exclude_list(exclude_fact_ids)
        r = self.model.solve(queries)
        if exclude_fact_ids is not None:
            self.retriever.clear_exclude_list()

        if pretty_print:
            for res_i in r:
                try:
                    for result_term in sorted(res_i, key=lambda term: -term.args[2].value):
                        try:
                            proof_tree = Tree.fromlist(term2list(result_term.args[1]))
                            proof_tree.pprint(margin=250, indent=3)
                        except:
                            breakpoint()
                        print("      ")
                    print("========================")
                except:
                    continue

        return r

    def fact_lookup(self, query):
        return self.retriever.search([query])

    def fact_decomposition(self, hypothesis, fact1s=None, **kwargs):
        """for out-of-loop eval of decomps"""
        backup_gkwargs = copy.deepcopy(self.drg.gen_kwargs)
        for k, v in kwargs.items():
            self.drg.gen_kwargs[k] = v

        preds, scores = self.drg.predict(hypothesis=hypothesis, fact1s=fact1s if fact1s else ['none'])
        self.drg.gen_kwargs = backup_gkwargs
        return preds, scores

    def solve(self, queries: List[Query]) -> List[Result]:
        return self.model.solve(queries)

    def create_dfs_tree(self, h_str, save_render_path=None):
        db = problog_export.database
        dot = graphviz.Graph()
        dot.attr(rankdir="LR", splines="ortho")
        dot.attr('node', shape='box', style='filled')

        seen_nodes = set()
        ndid = [0]

        def _get_matches(term_str):
            search_term = Term.from_string(term_str)
            nodekey = db.find(search_term)
            try:
                node = db.get_node(nodekey)
            except:
                return []
            matches = node.children.find(search_term.args)
            ret = []

            for match_id in matches:
                match_term = db.get_node(match_id)
                ret.append([unquote(term2str(x)) for x in match_term.args])

            return ret

        def _add_node(nstr, **kwargs):
            dot.node(nstr, label='\n'.join(textwrap.wrap(nstr, width=50)), **kwargs)

        def _get_color(term):
            if _get_matches(f"entailed('{term}',_,_)"):
                return "green"
            elif _get_matches(f"is_fact('{term}')"):
                return "yellow"
            elif recorded(Term.from_string(f"'{term}'")):
                return "darkolivegreen1"
            elif _get_matches(f"drg_link('{term}',_,_,_)"):
                return "lightblue2"
            elif _get_matches(f"no_expansions('{term}')"):
                return "lightred"
            else:
                return "lightgrey"

        _add_node(h_str, rank='1', color=_get_color(h_str))
        recursions = defaultdict(int)

        def _recursive_construct(_hstr, rank):
            matches = _get_matches(f"drg_link('{_hstr}', _,_,_)")
            for [h, f1, f2, sc] in matches:
                ndid[0] += 1
                recursions[rank] += 1
                sc = float(sc)
                nid = 'n' + str(ndid[0])
                dot.node(nid, label=f"{sc:.2f}", shape='diamond',
                         height="0.25", width="0.25",
                         rank=f"{rank + 1}")
                dot.edge(_hstr, nid)

                if f1 not in seen_nodes:
                    _add_node(f1, rank=f"{rank + 2}", color=_get_color(f1))
                    seen_nodes.add(f1)
                    _recursive_construct(f1, rank=rank + 2)
                dot.edge(nid, f1)

                if f2 not in seen_nodes:
                    _add_node(f2, rank=f"{rank + 2}",
                              color=_get_color(f2))
                    seen_nodes.add(f2)
                    _recursive_construct(f2, rank=rank + 2)
                dot.edge(nid, f2, weight="10")

        _recursive_construct(h_str, rank=1)

        if save_render_path:
            dot.render(save_render_path)
        breakpoint()

        return dot
