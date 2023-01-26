import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from src.dialogue_retrieval import RetrievalEngine

"""Sentences should take form [query, candidate]"""

ENTAILMENT_CROSS_ENCODER = lambda: CrossEncoder("cross-encoder/qnli-electra-base")

class RuleFilter(torch.nn.Module):
    """Skeleton class"""
    def filter_candidates(self, candidates: List[dict], **kwargs) -> List[dict]:
        """
        Applies filter to candidate list (dict of hypothesis, fact1 and fact2) and returns remaining candidates

        :param candidates:
        :return:
        """
        raise NotImplementedError()

class SBERTFilter(RuleFilter):
    """Filter by cosine similarity of SBERT embeddings"""
    def __init__(self, threshold=0.95):
        super().__init__()
        self.metric = lambda x, y: cosine_similarity(Tensor(x).reshape(1,-1), Tensor(y).reshape(1,-1))
        self.model = RetrievalEngine(metric=self.metric)
        self.threshold = threshold
        return None

    def filter_candidates(self, sentences):
        if not sentences:
            return sentences
        encodings = self.encode_candidates(sentences)
        sim = [self.model.compute_similarity(q, s) for (q, s) in encodings]
        return [e for (s, e) in zip(sim, sentences) if s >= self.threshold]

    def encode_candidates(self, sentences):
        return [self.model.encode_sentences(x) for x in sentences]


class CrossEncoderEntailmentFilter(RuleFilter):
    """From NELLIE: src/rule_filters.py"""
    def __init__(self, model=None, batch_size=128):
        super().__init__()
        self.model = model if model is not None else ENTAILMENT_CROSS_ENCODER()
        self.batch_size = batch_size

    def filter_candidates(self, candidates: List[dict], threshold: float = .7) -> List[dict]:
        if not candidates:
            return candidates
        scores = self.model.predict([(c[1], c[0]) for c in candidates],
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