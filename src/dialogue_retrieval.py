from sklearn.metrics.pairwise import cosine_similarity

from flan.flan import HypothesisGenerator, MODEL_NAME, ICL_PROMPT

class RetrievalEngine:
	def __init__(self):
		self.generator = initialize_generator()
		self.encoder = load_encoder('sbert_tvqa')
		self.threshold = 0.8
		self.metric = cosine_similarity

	def initialize_generator(self):
		gen = HypothesisGenerator(MODEL_NAME, ICL_PROMPT)
		return gen

	def load_encoder(self, model_name):
		return SentenceTransformer(model_name)

	def generate_hypothesis(self, question, answer):
		query = "Q: " + question + " A: " + answer
		return self.generator.inference(self, query)

	def encode_sentences(self, query):
		return self.encoder.encode(sentences)

	def compute_similarity(self, query, sentences, metric):
		encoded = self.encode_sentences([query] + sentences)
		q_encoded = encoded[0]
		s_encoded = encoded[1:]
		sim_scores = [metric(q_encoded, s) for s in s_encoded]
		return sim_scores

	def retrieve_dialogue(self, query, clip_name, metric=None, threshold=None):
		dialogue = dialoge_dset[clip_name]
		if not metric:
			metric = self.metric
		if not threshold:
			threshold = self.threshold
		if len(query) > 1:
			query = generate_hypothesis(query[0], query[1])
		s_scores = compute_similarity(query, dialogue, metric)
		zipped = zip(dialogue, s_scores)
		return [d for d, s in zipped if s > threshold]

def load_dialogue(json_fn):
	with open(json_fn, 'r') as f:
		data = json.load(f)
	return data

if __name__ == "__main__":
	engine = RetrievalEngine()
	dialogue_dataset = load_dialogue('/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl')
