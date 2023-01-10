''' From Nathaniel's NELLIE codebase. Contains dataset structuring classes and methods.'''

from typing import List
from src.utils import DEFAULT_MAX_DEPTH, flatten, DEFAULT_MAX_PROOFS, normalize_text


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




if __name__ == "__main__":
    reader = QuestionDatasetReader()
    dataset: QuestionDataset = reader.make_dataset('dev', as_questions=True)
    from IPython import embed

    embed(user_ns=locals())
