from src.query_dataset import Query, TVQADataSet

class ProofGen(object):

	def __init__(self):
		self.max_depth
		self.filters = [RuleFilter(), ]
		self.retriever = RetrievalEngine()
		self.vision = VisionModule()

	def prove(self, statement, dialogue, depth):
		candidates = self.retrieval(statement, dialogue)
		evidence = self.filter(statement, candidates)
		if evidence:
			return evidence
		if depth == self.max_depth:
			return None
		# branch
		branches = branch()
		evidence_b = None
		for b in branches:
			evidence_a, new_statement = b
			evidence_b = prove(new_statement, dialogue, depth+1)
			if evidence_b:
				return [evidence_a, new_statement, evidence_b]
		return None


	def retrieval(self, statement, dialogue):
		candidates = self.retriever.retrieve()
		return candidates

	def filter(self, statement, candidates):
		for f in self.filters:
			candidates = f.filter_candidates(statement, candidates)
		return candidates

	def branch(self):
		pass

	def query_vision(self, frame_path, query):
		return self.vision.forward(frame_path, query)

