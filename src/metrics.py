''' From Nathaniel's NELLIE codebase.'''

import copy
import pandas as pd

from src.fact_retrieval import WorldTreeFactRetrievalEngine
from src.utils import read_jsonl, flatten, normalize_text
from nltk.tree import Tree

eb_test_orig = pd.DataFrame(read_jsonl(f"data/entailmentbank/dataset/task_1/test.jsonl"))

def compute_qa_metrics(df, time_limit=90, include_difficulty_eval=True, include_groundable_eval=False):
    def _eval_df(df):
        metrics = {}
        answered = []
        correct = []
        outscored = []
        for q, subdf in df.groupby("id"):
            answered.append(int(subdf['mc_pred'].sum() > 0))
            correct.append(subdf[subdf.label == 1].iloc[0]['mc_pred'])
            outscored.append(
                subdf[subdf.label == 1].iloc[0]['mc_pred'] == 0 and
                subdf[subdf.label == 1].iloc[0]['score'] > 0
            )
        metrics['n_problems'] = len(correct)
        metrics['n_answered'] = pd.Series(answered).mean()
        metrics['correct'] = pd.Series(correct).mean()
        metrics['outscored'] = pd.Series(outscored).mean()
        if 'time' in df.columns:
            metrics['timeout_rate'] = df.time.apply(
                lambda x: x >= time_limit).mean()

            metrics['correct_timeout_rate'] = df.query('label == 1').time.apply(
                lambda x: x >= time_limit).mean()

        rdf = pd.DataFrame({'answered': answered, 'correct': correct})
        metrics['proof_precision'] = (df[df.score > 0].label == 1).mean()
        metrics['proof_recall'] = (df[df.label == 1].score > 0).mean()
        metrics['answer_recall'] = df[df.label == 1].mc_pred.mean()
        metrics['answer_precision'] = df[df.mc_pred == 1].label.mean()
        return metrics

    metrics = _eval_df(df)
    if include_difficulty_eval:
        if 'difficulty' not in df.columns:
            from src.dataset import ARC
            arc = ARC.load_arc_dataset()
            df['difficulty'] = df.id.apply(lambda _id: arc.query(f"id == '{_id}'").iloc[0]['difficulty'])
        for diff, subdf in df.groupby('difficulty'):
            diff = diff.lower()
            diff_metrics = _eval_df(subdf)
            metrics[f"{diff}_correct"] = diff_metrics['correct']
    if include_groundable_eval:
        from src.utils import normalize_text

        eb_test_orig = pd.DataFrame(flatten(
            [read_jsonl(f"data/entailmentbank/dataset/task_1/{split}.jsonl") for split in ['train', 'dev', 'test']]))
        groundable_hypothesis_ids = list(eb_test_orig[eb_test_orig.apply(eb_ex_is_groundable, axis=1)].id)
        gdf = df[df.id.apply(lambda x: x in groundable_hypothesis_ids)]
        groundable_metrics = compute_qa_metrics(gdf, time_limit=90, include_difficulty_eval=True)
        metrics['groundable_results'] = groundable_metrics
        metrics['n_groundable'] = groundable_metrics['n_problems']
        metrics['groundable_correct'] = groundable_metrics['correct']
        metrics['groundable_proof_recall'] = groundable_metrics['proof_recall']
        metrics['groundable_proof_precision'] = groundable_metrics['proof_precision']

    return metrics
