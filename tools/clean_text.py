# SOURCES

# FRIENDS: http://friends.tktv.net/
# THE BIG BANG THEORY: https://bigbangtrans.wordpress.com/
# HOW I MET YOUR MOTHER: https://transcripts.foreverdreaming.org/viewforum.php?f=177 # s2e1
# GREY'S ANATOMY: https://greysanatomy.fandom.com/
# HOUSE M.D.: https://clinic-duty.livejournal.com/12225.html
# CASTLE: https://dustjackets.fandom.com/wiki/Transcripts

import json

def load_dialogue(fn):
	fn = DATA_DIR + fn
	with open(fn) as f:
		x = []
		for l in f:
			x.append(json.loads(l))
		return x[0]

def grab_clip(show, season, episode, clip):
	key = show + '_s' + str(season).zfill(2) + 'e' + str(episode).zfill(2) + '_seg02_clip_' + str(clip).zfill(2)
	d = dialogue[key]
	r = []
	for i in d:
		t = i['text']
		if ':' in t:
			a = t.split(':')[0]
			b = t[len(a)+1:]
		else:
			a = ''
			b = t
		if b.startswith('-'):
			b = b[1:]
		r.append([a,b])
	return r

def remove_p(x):
	flag = 1
	while flag:
		flag = 0
		s = -1
		for i in range(len(x)):
			if s < 0 and x[i] == '(':
				s = i
			elif s >= 0 and x[i] == ')':
				x = x[:s] + x[i+1:]
				flag = 1
				break
	return x

def clean_script(x, show):
	r = []
	if show == 'friends':
		for i in x:
			if ': ' in i and not i.startswith('SCENE'):
				a = i.split(':')[0]
				b = i[len(a)+1:]
				b = remove_p(b)
				r.append([a,b])

	if show == 'met':
		for i in x:
			if ':' in i:
				a = i.split(':')[0]
				b = i[len(a)+1:]
				b = remove_p(b)
				r.append([a,b])

	if show == 'bang':
		for i in x:
			if ':' in i and not i.startswith('Scene'):
				a = i.split(':')[0]
				b = i[len(a)+1:]
				b = remove_p(b)
				r.append([a,b])

	if show == 'house':
		for i in x:
			if ':' in i:
				a = i.split(':')[0]
				b = i[len(a)+1:]
				b = remove_p(b)
				r.append([a,b])

	if show == 'grey':
		for i in x:
			if ':' in i:
				a = i.split(':')[0]
				b = i[len(a)+1:]
				b = remove_p(b)
				r.append([a,b])

	if show == 'castle':
		i = 0
		while i < len(x):
			if x[i].isupper() and x[i+1]:
				a = x[i]
				b = x[i+1]
				b = remove_p(b)
				r.append([a,b])

	return r


def load_text(fn):
	with open(fn,'r') as f:
		x = f.read()
	return x.split('\n')

