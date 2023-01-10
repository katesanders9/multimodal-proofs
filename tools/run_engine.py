import json
import logging
import os
import sys
import time
from argparse import ArgumentError
from typing import List

import pandas as pd
from deepproblog.semiring import Result
from nltk.tree import *
from problog.logic import term2list, term2str, unquote, is_variable
from transformers import HfArgumentParser, set_seed
from tqdm import tqdm

from src.dataset import WorldTreeDatasetReader, QuestionDataset, EntailmentbankDatasetReader, Question, \
    OBQADatasetReader
from src.engine import NELLIE
from src.metrics import compute_qa_metrics
from src.utils import read_jsonl
from src.utils.hf_utils import GenerationArguments, EngineArguments

logger = logging.getLogger(__name__)


# Setup logging

def run_eval(args, generation_args):
    set_seed(args.seed)

    print(args)
    print(generation_args)

    if args.test_set == "entailmentbank":
        reader = EntailmentbankDatasetReader(use_gpt_declaratives=args.use_gpt_declarativizer)
    elif args.test_set == 'openbookqa':
        reader = OBQADatasetReader()
    else:
        assert args.test_set in ['worldtree', 'worldtree_1_0']
        reader = WorldTreeDatasetReader(use_gpt_declaratives=args.use_gpt_declarativizer, worldtree_data_path=f"data/{args.test_set}")

    dataset: QuestionDataset = reader.make_dataset(args.split, as_questions=True)
    # if args.dry_run: breakpoint()
    nellie = NELLIE(args, generation_args)

    PREDICATE_MAP = {
        'dfs': 'timeout_prove',
        'bfs': 'timeout_bfs_prove',
        'bfs_no_cut': "timeout_bfs_prove_no_cut"
    }
    prove_predicate = PREDICATE_MAP.get(args.search_type, None)
    if prove_predicate is None:
        raise ArgumentError(args.search_type, f"search type {args.search_type} does not exist!")
    # friction = query_model("frictional force between two sticks causes them to increase in temperature", depth=3, time_limit=60)

    if args.dry_run:
        # breakpoint()
        # from src.utils.worldtree_utils import WorldTree
        # wt = WorldTree()
        # print(wt.to_lookup_corpus()[:100])

        # hawk = query_model("a hawk uses a beak to catch prey", depth=2)
        # eruption = query_model("eruptions cause plants to die", depth=3, max_num_proofs=20)
        test_h = [
            "smoke from volcanic eruptions can decrease resources in an area by decreasing the availability of sunlight",
            # "a person can exert mechanical energy to push a block",
            # "frictional force between two sticks causes them to increase in temperature",
            # "plants use chlorophyll to produce sugar"
            # "a volcanic eruption may have happened to cause the cold weather and acid rain in europe"
            # "brown color can be used to indicate the soil is high in nutrients",
            # "sand dunes are formed by wind erosion and deposition",
            # "burning a leaf with fire causes a chemical change",
            # "astronomers and biologists both use optical instruments to make scientific discoveries",
            # "a line graph can be used to show yearly water usage in the usa",
            # "planting trees is a kind of human action that slows the rate of soil erosion"

        ]
        # for f in nellie.fact_decomposition('chlorophyll is used by plants to produce carbohydrates')[0]:
        #     print(f['fact1'] , ' , ' , f['fact2'])
        # for th in test_h:
        nellie.drg.update_db = True
        nellie.retriever.save_entailment_results = True
        test_res = nellie.query_model(test_h, depth=5, max_num_proofs=20, time_limit=args.time_limit, pretty_print=True,
                                      prove_predicate=prove_predicate)

        for h in test_h:
            filename = '_'.join(h.split()[:-3])
            nellie.create_dfs_tree(h, save_render_path=os.path.join(args.search_pdfs_dir, filename))

        exit(0)

    # query_model(["a bird uses a beak to catch prey"], 2)
    # query_model("penguins are predatory hunters", 3)

    def postprocess_predictions(question: Question, result: List[Result]):
        _res = []
        correct_h = question.correct_hypothesis_text
        for i, res_i in enumerate(result):
            hyp_i = unquote(term2str([trm for trm in res_i][0].args[0]))
            is_correct_answer = (i == question.correct_idx)
            success_terms = [term for (term, v) in res_i.result.items() if v]
            proofs = []
            res_item = dict(id=question.id, hypothesis=hyp_i, time=res_i.ground_time, label=int(is_correct_answer))
            if question.difficulty is not None:
                res_item['difficulty'] = question.difficulty
            if not success_terms:
                res_item.update(dict(score=0.0, proofs=proofs))
            else:
                def _get_score(trm):
                    score_term = trm.args[2]
                    if is_variable(score_term):
                        return 0
                    else:
                        return score_term.value

                if args.results_dir:
                    with open(os.path.join(args.results_dir, "proofs.txt"), 'a') as w:
                        print(
                            f"======== Proofs of \"{hyp_i}\" ({question.id}, {'RIGHT' if is_correct_answer else 'WRONG'}) ========",
                            file=w)
                        for result_term in sorted(res_i, key=lambda term: -_get_score(term)):
                            try:
                                proof_tree = Tree.fromlist(term2list(result_term.args[1]))
                                proof_tree.pprint(margin=175, indent=5, stream=w)
                                treestr = proof_tree.pformat()
                                proofs.append(treestr)
                            except:
                                breakpoint()
                            print("", file=w)
                        print("========================", file=w)
                else:
                    for result_term in sorted(res_i, key=lambda term: -_get_score(term)):
                        try:
                            proof_tree = Tree.fromlist(term2list(result_term.args[1]))
                            treestr = proof_tree.pformat()
                            proofs.append(treestr)
                        except:
                            breakpoint()
                res_item.update(dict(score=max([_get_score(term) for term in success_terms]),
                                     proofs=proofs))
            _res.append(res_item)

        df = pd.DataFrame(_res)

        df['mc_pred'] = (df.score == max(df.score)).astype(int) if any(df.score) else 0
        if not any(df['label']):
            breakpoint()
        return df

    eb_test = pd.DataFrame(read_jsonl("brtx_exp/preproc_ev/Baseline.baseline/out/entailmentbank_test.jsonl"))
    eb_test = eb_test[eb_test['fact1_rel'] != '']
    eb_test.drop_duplicates(subset='hypothesis', keep='first', inplace=True)
    # for (i,row) in eb_test.iterrows():
    #     res = query_model(row['hypothesis'], depth=1, max_num_proofs=1)
    #
    #     embed(user_ns=locals())

    eval_start_time = time.time()
    results = []
    for qidx, question in enumerate(tqdm(dataset)):
        if args.debug_id and question.id != args.debug_id:
            print(f"skipping question {question.id} because not debug id {args.debug_id}")
            continue
        if qidx % args.num_workers != args.worker_id:
            continue

        if args.dry_run and len(results) > 1:
            break
        if args.max_eval_examples and len(results) == args.max_eval_examples:
            break

        queries = question.to_queries(time_limit=args.time_limit, max_num_proofs=args.max_num_proofs,
                                      prove_predicate=prove_predicate)

        res = []
        for q in queries:
            res.extend(nellie.solve([q]))
        results.append(postprocess_predictions(question, res))

    logger.info("finished evaluation loop")
    resdf = pd.concat(results)

    metrics = compute_qa_metrics(resdf, time_limit=args.time_limit)
    metrics['total_runtime'] = time.time() - eval_start_time
    if args.results_dir:
        resdf.to_pickle(os.path.join(args.results_dir, "results.pkl"))
        resdf.to_csv(os.path.join(args.results_dir, "results.tsv"), sep='\t')
        resdf.to_json(os.path.join(args.results_dir, "results.json"), orient="records")
        with open(os.path.join(args.results_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, sort_keys=True, indent=2)
    return metrics


if __name__ == "__main__":
    inargs, ingeneration_args = HfArgumentParser((EngineArguments, GenerationArguments)).parse_args_into_dataclasses()

    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    metrics = run_eval(inargs, ingeneration_args)
