''' From Nathaniel's NELLIE codebase.'''

from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch

SBERT = lambda: SentenceTransformer('all-MiniLM-L6-v2')
ENTAILMENT_CROSS_ENCODER = lambda: CrossEncoder("brtx_exp/train_scitail/training_scitail-2022-08-10_20-51-16")
EV_CROSS_ENCODER = lambda: CrossEncoder("bfs/train_cross_encoder_ev/NegExPerPos.4+Seed.4321+UseEQASC.yes/outdir")
EV_CROSS_ENCODER_2 = lambda: CrossEncoder("bfs/train_cross_encoder_ev/NegExPerPos.0+Seed.1234+UseEQASC.yes/outdir")
EV_UNIFICATION_ENCODER = lambda: SentenceTransformer("bfs/train_sbert_ev/NegExPerPos.2+Seed.42+UseEQASC.yes/outdir")
SBERT_MSMARCO = lambda: SentenceTransformer('msmarco-distilbert-base-v4')


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
