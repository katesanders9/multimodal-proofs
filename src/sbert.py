# sentence-transformers==1.0.4, torch==1.7.0.
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader

# Load pre-trained model - we are using the original Sentence-BERT for this example.
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Set up data for fine-tuning 
sentence_reader = LabelSentenceReader(folder='~/tsv_files')
data_list = sentence_reader.get_examples(filename='recipe_bot_data.tsv')
triplets = triplets_from_labeled_dataset(input_examples=data_list)
finetune_data = SentencesDataset(examples=triplets, model=sbert_model)
finetune_dataloader = DataLoader(finetune_data, shuffle=True, batch_size=16)

# Initialize triplet loss
loss = TripletLoss(model=sbert_model)

# Fine-tune the model
sbert_model.fit(train_objectives=[(finetune_dataloader, loss)], epochs=4,output_path='bert-base-nli-stsb-mean-tokens-recipes')