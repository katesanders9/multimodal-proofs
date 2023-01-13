import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader

TRAIN_JSON = '/srv/local2/ksande25/NS_data/TVQA/sbert_data.jsonl'

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
        	return len(self.chunks) + len(self.dialogue_chunks)
        else:
        	return len(self.chunks)
    
    def __getitem__(self, item):
        anchor = self.chunks[item]
        
        if self.is_train:
            positive_list = self.dialogue_chunks[anchor['id']]

            positive = random.choice(positive_list)

            neg_ids = list(self.ids)
            neg_ids.remove(anchor['id'])
            neg_id = random.choice(neg_ids)
            
            negative_list = self.dialogue_chunks[neg_id]
            negative = random.choice(negative_list)
            
            return anchor['h'], positive['text'], negative['text']
        
        else:
            return anchor['text']

def preprocess(data_file, train):
	data = []
    for line in open(data_file, 'r'):
        data.append(json.loads(line))
    return data


sbert_model = SentenceTransformer('all-mpnet-base-v2')
data = Triplets(TRAIN_JSON, train=True)
dataset = DataLoader(data, batch_size=batch_size, shuffle=True)

loss = TripletLoss(model=sbert_model)
sbert_model.fit(train_objectives=[(dataset, loss)], epochs=5,output_path='sbert_tvqa')