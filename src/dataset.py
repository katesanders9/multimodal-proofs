from typing import List
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)

def import_check(package):
    import importlib
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

if import_check('deepproblog'):
    from deepproblog.dataset import Dataset
    from deepproblog.query import Query

if import_check('problog'):
    try:
        from problog.logic import Term, Var, Constant, term2list
    except:
        print("failed to import problog items!!")

from src.utils import DEFAULT_MAX_DEPTH, flatten, read_jsonl, DEFAULT_MAX_PROOFS, normalize_text
import pandas as pd
import os
import numpy as np
from datasets import load_dataset


class ARC:
    arc = None

    @staticmethod
    def load_arc_dataset():
        if ARC.arc is None:
            exs = []
            for difficulty in ['easy', 'challenge']:
                arc_dset = load_dataset('ai2_arc', f'ARC-{difficulty.title()}')
                for split in ['train', 'test', 'validation']:
                    for ex in arc_dset[split]:
                        for choice, label in zip(ex['choices']['text'], ex['choices']['label']):
                            exs.append(dict(
                                arc_split=split,
                                id=ex['id'],
                                choice=choice,
                                label=label,
                                correct=label == ex['answerKey'],
                                question=ex['question'],
                                difficulty=difficulty
                            ))
            ARC.arc = pd.DataFrame(exs)
        return ARC.arc



def query_from_hypothesis(h_str, prove_predicate="timeout_prove",
                          depth=DEFAULT_MAX_DEPTH,
                          max_num_proofs=DEFAULT_MAX_PROOFS,
                          time_limit=60 * 60 * 24,
                          label=1):
    if not time_limit:
        prove_predicate = 'prove'
    query_args = [prove_predicate, Term(f"'{h_str}'"), Var("X"), Var("Y"),
                  Constant(depth), Constant(max_num_proofs)]
    if time_limit: query_args.append(Constant(time_limit))
    query = Query(Term(*query_args), p=float(label))
    return query


if import_check('deepproblog'):
    class HypothesisDataset(Dataset):
        def __init__(self, hypotheses):
            self.hypotheses = hypotheses

        def to_query(self, i: int) -> Query:
            return self.hypotheses[i].to_query()

        def __len__(self):
            return len(self.hypotheses)

        def __repr__(self):
            return ', '.join(x.id for x in self.hypotheses)

        def __getitem__(self, item):
            return self.hypotheses[item]


class QuestionDataset:
    def __init__(self, questions):
        super(QuestionDataset, self).__init__()
        self.questions: List[Question] = questions

    def __getitem__(self, item):
        return self.questions[item]

    def get_queries(self, i):
        return self.questions[i].to_queries()

    def __len__(self):
        return len(self.questions)

    def __repr__(self):
        return ', '.join(x.id for x in self.questions)

    def to_hypothesis_dataset(self):
        return HypothesisDataset(flatten([q.hypotheses for q in self.questions]))


class Hypothesis:
    def __init__(self, text, id, label, question_text):
        self.text = text
        self.id = id
        self.label = label
        self.question_text = question_text

    def __repr__(self):
        return f'Hypothesis({self.id}, {self.text} , {self.label})'

    def to_query(self, **kwargs):
        return query_from_hypothesis(self.text, label=self.label, **kwargs)


class Question:
    def __init__(self, hypotheses: List[Hypothesis], question_id: str, correct_idx: int, difficulty: int):
        self.hypotheses = hypotheses
        self.correct_idx = correct_idx
        self.id = question_id
        self.difficulty = difficulty

    def to_queries(self, **kwargs):
        return [h.to_query(**kwargs) for h in self.hypotheses]

    def __repr__(self):
        return f'Question({self.id}, {self.hypotheses})'

    @property
    def correct_hypothesis_text(self):
        return self.hypotheses[self.correct_idx].text


class QuestionDatasetReader:
    def __init__(self):
        self.question_df = None

    def make_dataset(self, split, as_questions=False):
        sdf = self.question_df.query(f"split == '{split}'")

        def _get_question(row):
            hypotheses = [
                Hypothesis(
                    normalize_text(row[f"answer{i}"]),
                    row['questionID'] + f"_{i}",
                    float(row['AnswerKey'] == row[f"key{i}"]),
                    normalize_text(row['question'])
                )
                for i in range(1, 6)
                if row[f'key{i}'] is not None
            ]
            return Question(hypotheses,
                            question_id=row['questionID'],
                            correct_idx=[i for (i, h) in enumerate(hypotheses) if h.label][0],
                            difficulty=row.get('arcset', None))

        question_dataset = QuestionDataset(sdf.apply(_get_question, axis=1).to_list())

        if as_questions:
            return question_dataset
        else:
            return question_dataset.to_hypothesis_dataset()


class WorldTreeDatasetReader(QuestionDatasetReader):
    def __init__(self, worldtree_data_path='data/worldtree', arc_path='data/ARC', use_gpt_declaratives=True):
        super().__init__()

        # read and reformat opt-declarativized eb hypotheses
        wt_declarativized = pd.read_csv(
            # os.path.join(worldtree_data_path, "worldtree_gpt-j-6B_declarativized.tsv"),
            os.path.join(worldtree_data_path, "worldtree_t5_combined_eb_declarativized.tsv"),
            sep='\t', index_col=0
        ).drop_duplicates(subset=['id', 'label'])
        _eb = []
        for i, subdf in wt_declarativized.groupby('id'):
            newrow = dict()
            for j, (_, row) in enumerate(subdf.iterrows()):
                newrow[f'key{j + 1}'] = row['label']
                newrow[f'answer{j + 1}'] = row['declarativized'].strip()
            for k in ['split', 'question']:
                newrow[k] = subdf[k].iloc[0]
            newrow['id'] = i
            _eb.append(newrow)

        wt_declarativized = pd.DataFrame(_eb)

        # TODO worldtree gpt2 declarativized is split by ARC, not worldtree
        arc_df = pd.read_csv(os.path.join(arc_path, "ARC-ALL+declaratives-QA2D.csv"),
                             encoding='unicode_escape', engine='python')

        wt_tables = []
        for split in ['train', 'dev', 'test']:
            _df = pd.read_csv(f'{worldtree_data_path}/questions/questions.{split}.tsv', sep='\t')
            if 'QuestionID' not in _df.columns: _df['QuestionID'] = _df['questionID']
            _df['split'] = split
            _df = _df.merge(arc_df, left_on='QuestionID', right_on='questionID', suffixes=('', '_y'))
            if 'AnswerKey' not in _df.columns:
                _df["AnswerKey"] = _df["AnswerKey_x"]
            if 'question' not in _df.columns:
                _df["question"] = _df["question_x"]

            if split == 'test' and use_gpt_declaratives:
                _opt_decl_df = _df[
                    [c for c in _df.columns if (c == 'id') or (c not in wt_declarativized.columns)]
                ].merge(wt_declarativized, how='left', left_on='QuestionID', right_on='id').fillna(np.nan).replace(
                    [np.nan], [None])
                wt_tables.append(_opt_decl_df)
            else:
                wt_tables.append(_df)

            # wt_tables.append(_df)
        self.question_df = pd.concat(wt_tables)


class EntailmentbankDatasetReader(QuestionDatasetReader):
    def __init__(self, entailmentbank_data_path='data/entailmentbank', arc_path='data/ARC', use_gpt_declaratives=True):
        super().__init__()
        arc_df = pd.read_csv(os.path.join(arc_path, "ARC-ALL+declaratives-QA2D.csv"),
                             encoding='unicode_escape', engine='python')

        # read and reformat opt-declarativized eb hypotheses
        eb_declarativized = pd.read_csv(
            # os.path.join(entailmentbank_data_path, "entailmentbank_gpt-j-6B_declarativized.tsv"),
            os.path.join(entailmentbank_data_path, "entailmentbank_t5_combined_eb_declarativized.tsv"),
            sep='\t', index_col=0
        ).drop_duplicates(subset=['id', 'label'])
        _eb = []
        for i, subdf in eb_declarativized.groupby('id'):
            newrow = dict()
            for j, (_, row) in enumerate(subdf.iterrows()):
                newrow[f'key{j + 1}'] = row['label']
                newrow[f'answer{j + 1}'] = row['declarativized'].strip()
            for k in ['split', 'question']:
                newrow[k] = subdf[k].iloc[0]
            newrow['id'] = i
            _eb.append(newrow)

        eb_declarativized = pd.DataFrame(_eb)

        eb_tables = []
        for split in ['train', 'dev', 'test']:
            _df = pd.DataFrame(read_jsonl(f"{entailmentbank_data_path}/dataset/task_1/{split}.jsonl"))
            _df['split'] = split

            _df = _df.merge(arc_df, left_on='id', right_on='questionID', suffixes=('', '_y'))

            if 'AnswerKey' not in _df.columns:
                _df["AnswerKey"] = _df["AnswerKey_x"]
            if 'question' not in _df.columns:
                _df["question"] = _df["question_x"]

            if split != 'train' and use_gpt_declaratives:
                _opt_decl_df = _df[
                    [c for c in _df.columns if (c == 'id') or (c not in eb_declarativized.columns)]
                ].merge(eb_declarativized, how='left', on='id').fillna(np.nan).replace([np.nan], [None])
                eb_tables.append(_opt_decl_df)
            else:
                eb_tables.append(_df)
        self.question_df = pd.concat(eb_tables)


class OBQADatasetReader(QuestionDatasetReader):

    def __init__(self, obqa_path='data/openbookqa'):
        super().__init__()

        self.df = pd.read_csv(
            os.path.join(obqa_path, "openbookqa_gpt-j-6B_declarativized.tsv"), sep='\t', index_col=0
        ).drop_duplicates(subset=['id', 'label'])

    def make_dataset(self, split, as_questions=False):
        sdf = self.df.query(f"split == '{split}'")

        def _get_question(subdf):
            hypotheses = [
                Hypothesis(
                    normalize_text(row[f"declarativized"]),
                    row['id'] + f"_{i}",
                    float(row['correct']),
                    normalize_text(row['question'])
                )
                for i, (_, row) in enumerate(subdf.iterrows())
            ]
            return Question(hypotheses,
                            question_id=subdf.id.iloc[0],
                            correct_idx=[i for (i, h) in enumerate(hypotheses) if h.label][0],
                            difficulty=0)

        question_dataset = QuestionDataset(sdf.groupby('id').apply(_get_question).to_list())

        if as_questions:
            return question_dataset
        else:
            return question_dataset.to_hypothesis_dataset()


if __name__ == "__main__":
    reader = WorldTreeDatasetReader()
    dataset: QuestionDataset = reader.make_dataset('dev', as_questions=True)
    from IPython import embed

    embed(user_ns=locals())
