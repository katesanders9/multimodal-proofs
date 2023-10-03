# dialogue sampling
from engine import Engine
from dataset import Dataset
from tqdm import tqdm

e = Engine(2)
data = Dataset()
data.set_data('val')

start=40
end=100

f1, f2 = {}, {}
for i in tqdm(range(start,end)):
	try:
		x = data.load_data_sample(i)
		if not x[5]:
			continue
		e.set_clip(x[0],x[1])
		h=e.generator.declaritivize(x[2],x[3])
		d1=e.retrieval(h)
		d2=e.bm25(h)
		a=[i for i in range(len(x[4])) if x[4][i]['text'] == d1[0]][0]
		b=[i for i in range(len(x[4])) if x[4][i]['text'] == d1[-1]][0]
		c=[i for i in range(len(x[4])) if x[4][i]['text'] == d2[0]][0]
		d=[i for i in range(len(x[4])) if x[4][i]['text'] == d2[-1]][0]
		f=x[5][0]
		g=x[5][-1]
		a=list(range(a,b+1))
		b=list(range(c,d+1))
		c=list(range(f,g+1))
		f1[i] = sum([1 for z in a if z in c])/min(len(c),6)
		f2[i] = sum([1 for z in b if z in c])/min(len(c),6)
	except:
		print(i)

def check(i):
	show, clip, q, a, t, tp = data.load_data_sample(i)
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
# inference gen + hypothesis branching + entailment filtering



# VQA



# full pipeline


