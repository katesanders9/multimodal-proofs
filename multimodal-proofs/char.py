import os
import json
from tqdm import tqdm
import numpy as np
import PIL.Image as Image
from deepface import DeepFace
from sklearn.cluster import DBSCAN as db
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from itertools import permutations as perm
from deepface.commons import functions, realtime, distance as dst

frames_path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/'
data_path = '/srv/local2/ksande25/NS_data/TVQA/chars/'
with open('/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl', 'r') as f:
    transcripts = [json.loads(x) for x in f][0]


# return {im_name: [[id, label, face_area], ...]}
def pred_chars(show, clip, method='pairwise', rate=2,th=0.5):
    data = load_data(show, clip,rate,th) # {id: {im_name, feats, facial_area, confidence}}
    data = filter_spans(data, h=50, conf=.8)
    pred_clusters = calc_clusters(data, show, clip, method=method, th=.25) # output format of {id: cluster_id, ...}
    pred_labels = calc_labels(data, pred_clusters, clip, rate) # [(name, cluster_id), ...]
    labels = {}
    for x in pred_labels:
        labels[x[0]] = x[1][1:-2]
    out = {}
    images = list(set([d['im_name'] for d in data.values()]))
    for i in images:
        out[i] = []
    for i in list(data.keys()):
        if pred_clusters[i] in labels.keys():
            out[data[i]['im_name']] += [[i, labels[pred_clusters[i]], data[i]['facial_area']]]
        else:
            out[data[i]['im_name']] += [[i, None, data[i]['facial_area']]]
    return out

# load retinanet outputs, gt clusters, and labels as dict. 
def load_data(show, clip,rate,th):
    # format should be {id: {im_name, face, facial_area, confidence, cluster, label}}
    if os.path.isfile(data_path + clip + '_' + str(rate) + '_' + str(th) + '_char.npy'):
        extra = np.load(data_path + clip + '_' + str(rate) + '_' + str(th) + '_char.npy', allow_pickle=True).item()
    else:
        extra = retinanet(show,clip,rate)
    return extra

def filter_spans(data, h=50, conf=.8):
    new_data = {}
    i = 0
    for k in list(data.keys()):
        item = data[k]
        if item['facial_area']['h'] > h and item['confidence'] > conf:
            new_data[i] = item
            i += 1
    return new_data

def calc_clusters(d, show, clip, method='pairwise', th=.5, dist=.25):
    # output format of {id: cluster_id, ...}
    if method == 'pairwise':
        clusters = {}
        remaining = list(d.keys())
        ind = 0
        while remaining:
            m = max(remaining, key=lambda k: d[k]['facial_area']['w'])
            clusters[m] = ind
            remaining.remove(m)
            m_im = d[m]['feats']
            matches = []
            for r in remaining:
                out = dst.findCosineDistance(m_im, d[r]['feats'])
                if out < dist:
                    matches.append(r)
            for r in matches:
                clusters[r] = ind
                remaining.remove(r)
            ind += 1
        return clusters
    elif method == 'unsupervised':
        pass
    pass

# gt, pred, cluster, name, clip
def calc_labels(gt, pred, clip, rate):
    vals = {}
    for l in list(pred.values()):
        if not l in vals.keys():
            vals[l] = 0
        vals[l] += 1
    cs = []
    if len(vals.keys()) < 9:
        cs = list(vals.keys())
    else:
        remaining = list(vals.keys())
        while len(cs) < 8:
            nxt = max(remaining, key=lambda r:vals[r])
            cs.append(nxt)
            remaining.remove(nxt)
    num_clusters = len(cs)
    print(num_clusters)
    aligned = align(clip, gt, rate)
    names = get_labels(aligned)
    if len(names) <= num_clusters:
        pp = list(perm(cs))
        print(len(pp))
        qq = names
        pp = [list(zip(e, qq)) for e in pp]
    else:
        print('Warning: More names than clusters')
        pp = list(perm(names))
        qq = range(num_clusters)
        pp = [list(zip(qq, e)) for e in pp]
    z = np.argmax([sum([calc_p(gt, pred, q, l, aligned) for q,l in ps]) for ps in pp])
    return pp[z]

def align(clip, gt,rate):
    subs = transcripts[clip]
    subs = [(i['id'], i['start'], i['end']) for i in subs]
    ims = sorted(list(set([x['im_name'] for x in gt.values()])))
    ald = {}
    for i in range(len(ims)):
        t = i * rate / 3
        m, sn = t // 60, t % 60
        al = [s[0] for s in subs if (s[1][1] <= m and s[1][2] + s[1][3] / 1000 <= sn) and (s[2][1] >= m and s[2][2] + s[2][3] / 1000 >= sn)]
        ald[ims[i]] = al
    al = {}
    for i in list(gt.keys()):
        if ald[gt[i]['im_name']]:
            al[i] = ald[gt[i]['im_name']]
        else:
            al[i] = None
    subs = transcripts[clip]
    out = {}
    for i in list(gt.keys()):
        if al[i]:
            text =  subs[al[i][0]]['text']
            if ')' in text:
                end = text.index(')')
                out[i] = text[:end+1]
            else:
                out[i] = None
        else:
            out[i] = None
    return out

def get_labels(n):
    all_vals = [i for i in list(n.values()) if i]
    return list(set(all_vals))

def calc_p(gt, pred, cluster, name, aligned):
    n = aligned
    count, corr = 0, 0
    for i in list(n.keys()):
        if n[i] and name.lower() in n[i].lower():
            count += 1
            if pred[i] == cluster:
                corr += 1
    return corr / count if count else 0


# utils

# x = DeepFace.extract_faces(img_path='/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/grey_frames/grey_s03e20_seg02_clip_14/00001.jpg', enforce_detection= False,detector_backend='retinaface')

# {id: {im_name, feats, facial_area, confidence}}
def retinanet(show, clip, rate, th=.5):
    im_path = frames_path + show + '_frames/' + clip +'/'
    ims = os.listdir(im_path)
    ims = sorted(ims)
    ims = ims[::rate]
    d = {}
    for i in tqdm(ims):
        d[i] = DeepFace.extract_faces(img_path=im_path + i,enforce_detection= False,detector_backend='retinaface')
    ind = 0
    out = {}
    for i in tqdm(list(d.keys())):
        for r in d[i]:
            out[ind] = {'im_name': i, 'feats': extract(show, clip, i, r['facial_area'], th), 'facial_area': r['facial_area'], 'confidence': r['confidence']}
            ind += 1
    np.save(data_path + clip + '_' + str(rate) + '_' + str(th) + '_char.npy', out)
    return out

def extract(show, clip, im_name, area, th=.5):
    nd = {}
    image = np.array(Image.open(frames_path+show+'_frames/'+clip+'/'+im_name))
    a = area
    w2 = int(a['w']*th)
    h2 = int(a['h']*th)
    x = max(0, a['x'] - w2)
    y = max(0, a['y'] - h2)
    w = a['w']+w2
    h = a['h']+h2
    im2 = image[y:y+h, x:x+w]
    im2 = DeepFace.represent(img_path = im2, enforce_detection=False,detector_backend='skip', model_name="VGG-Face")[0]["embedding"]
    return im2