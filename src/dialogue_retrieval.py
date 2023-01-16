import os
os.environ['TRANSFORMERS_CACHE'] = "/srv/local2/ksande25/huggingface_cache/"
from torch import cosine_similarity, Tensor
from sentence_transformers import SentenceTransformer
from src.flan import HypothesisGenerator, MODEL_NAME, ICL_PROMPT

class RetrievalEngine:
	def __init__(self, threshold=0.8, init_generator=False):
		self.generator = None
		if init_generator:
			self.generator = self.initialize_generator()
		self.encoder = self.load_encoder('sbert_tvqa_3')
		self.threshold = threshold
		self.metric = lambda x, y: cosine_similarity(Tensor(x).reshape(1,-1), Tensor(y).reshape(1,-1))

	def initialize_generator(self):
		gen = HypothesisGenerator(MODEL_NAME, ICL_PROMPT)
		return gen

	def load_encoder(self, model_name):
		return SentenceTransformer(model_name)

	def generate_hypothesis(self, question, answer):
		query = "Q: " + question + " A: " + answer
		return self.generator.inference(query)

	def encode_sentences(self, sentences):
		return self.encoder.encode(sentences)

	def compute_similarity(self, query, sentences, metric):
		encoded = self.encode_sentences([query] + sentences)
		q_encoded = encoded[0]
		s_encoded = encoded[1:]
		sim_scores = [metric(q_encoded, s).item() for s in s_encoded]
		return sim_scores

	def retrieve_dialogue(self, query, clip_name, dset, QA_flag=False, metric=None, threshold=None):
		dialogue = dset[clip_name]
		if not metric:
			metric = self.metric
		if not threshold:
			threshold = self.threshold
		if QA_flag:
			query = self.generate_hypothesis(query[0], query[1])
		s_scores = self.compute_similarity(query, dialogue, metric)
		zipped = zip(dialogue, s_scores)
		n = 3
		#z = sorted(zipped, key=lambda x: x[1])[(-1*n):]
		#r = [d for d, s in z]
		#print([s for d, s in z])
		return [d for d, s in zipped if s > threshold]

def load_dialogue(json_fn):
	with open(json_fn, 'r') as f:
		data = json.load(f)
	return data

if __name__ == "__main__":
	engine = RetrievalEngine()
	dialogue_dataset = load_dialogue('/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl')
