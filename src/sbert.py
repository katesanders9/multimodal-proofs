import json
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader, Dataset

TRAIN_JSON = '/srv/local2/ksande25/NS_data/TVQA/sbert_data_filtered.jsonl'

class Triplets(Dataset):
    def __init__(self, df, train=True):
        self.is_train = train
        self.transform = None

        sentences = preprocess(df,train)
        
        if self.is_train:
            self.chunks = sentences['hypotheses']
            self.dialogue_chunks = sentences['dialogue']
            self.ids = list([x['id'] for x in self.chunks])
        else:
            self.chunks = sentences
         
    def __len__(self):
        if self.is_train:
            return len(self.chunks)
        else:
            return len(self.chunks)
    
    def __getitem__(self, item):
        anchor = self.chunks[item]
        
        if self.is_train:
            positive_list = self.dialogue_chunks[str(anchor['id'])]

            positive = random.choice(positive_list)

            neg_ids = list(self.ids)
            neg_ids.remove(anchor['id'])
            neg_id = random.choice(neg_ids)
            
            negative_list = self.dialogue_chunks[str(neg_id)]
            negative = random.choice(negative_list)
            
            return InputExample(texts=[anchor['h'], positive['text'], negative['text']])
        
        else:
            return InputExample(texts=[anchor['text']])

def preprocess(data_file, train):
    data = []
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data

print("Loading data...")
data = Triplets(TRAIN_JSON, train=True)
dataset = DataLoader(data, batch_size=32, shuffle=True)
print("Loading model...")
sbert_model = SentenceTransformer('all-mpnet-base-v2')
loss = TripletLoss(model=sbert_model)

print("Training model...")
sbert_model.fit(train_objectives=[(dataset, loss)], epochs=5,output_path='sbert_tvqa', show_progress_bar=True)
