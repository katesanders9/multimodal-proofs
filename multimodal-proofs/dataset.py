class Dataset(object):
	def __init__(self):
		self.transcripts, self.train, self.val, self.test = load_dataset()

	def set_data(self, type):
		if type=='train':
			self.data = self.train
		elif type=='val':
			self.data = self.val
		else:
			self.data = self.test

	# load qa pair
	def load_qa_pair(self, index):
		item = self.data[index]
		ans = item['a' + str(item['answer_idx'])]
		return [shows[item['show_name']], item['vid_name'], item['q'], ans]

	def load_data_sample(self, index):
		show, clip, q, a = self.load_qa_pair(index)
		t = self.transcript[clip]
		return clip, q, a, t, get_dialogue(data[index], t)

	def print_data(self, index):
		clip, q, a, t, tp = load_data_sample(self.data, self.transcript, index)
		if tp:
			[print(t[x]['text']) for x in range(0, tp[0])]
			print('*****')
			[print(t[x]['text']) for x in tp]
			print('*****')
			[print(t[x]['text']) for x in range(tp[-1]+1, len(t))]
		else:
			[print(x['text']) for x in t]
		print('\nCLIP: ' + clip)
		print('Q: ' + q)
		print('A: ' + a)
		return