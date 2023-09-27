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

path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/'
with open('/srv/local2/ksande25/NS_data/TVQA/tvqa_subtitles_all.jsonl', 'r') as f:
    transcripts = [json.loads(x) for x in f]

transcripts = transcripts[0]

# eval this method on this data
def eval(show, clip, method='pairwise'):
    gt, data, gt_labels = load_data(show, clip) # {id: {im_name, face, facial_area, confidence, cluster, label}}
    gt = filter_spans(gt, h=50, conf=.8)
    data = filter_spans(data, h=50, conf=.8)
    gt_clusters = {} # {id: cluster_id, ...}
    for k in list(gt.keys()):
        gt_clusters[k] = gt[k]['cluster']
    gt_labels = [(gt_labels[k], k) for k in list(gt_labels.keys())]
    pred_clusters = calc_clusters(data, show, clip, path, method=method, th=.25) # output format of {id: cluster_id, ...}
    pred_labels = calc_labels(data, pred_clusters, clip, 2) # [(name, cluster_id), ...]
    mapping = align_clusters(gt_clusters, pred_clusters) # gt -> pred
    c_p, c_r = calc_cluster_pr(mapping, gt_clusters, pred_clusters)
    l_p, l_r = calc_label_pr(mapping, gt_clusters, pred_clusters, gt_labels, pred_labels)
    return c_p, c_r, l_p, l_r

def calc_cluster_pr(mapping, gt_clusters, pred_clusters):
    c_p, c_r = [], []
    for c in list(mapping.keys()):
        match, count, miss = 0, 0, 0
        for i in list(gt_clusters.keys()):
            if gt_clusters[i] == c:
                count += 1
                if mapping[c] == pred_clusters[i]:
                    match += 1
            elif mapping[c] == pred_clusters[i]:
                miss += 1
        c_p.append(match / (match + miss))
        c_r.append(match / count)
    c_p = np.mean(c_p)
    c_r = np.mean(c_r)
    return c_p, c_r

def calc_label_pr(mapping, gt_clusters, pred_clusters, gt_labels, pred_labels):
    prld = {}
    gtld = {}
    for p in pred_labels:
        prld[p[0]] = p[1]
    for p in gt_labels:
        gtld[p[1]] = p[0]
    l_p, l_r = [], []
    for x in gt_labels:
        if x[1] == 0:
            continue
        l = x[0]
        match, count, miss = 0, 0, 0
        for i in list(gt_clusters.keys()):
            if gtld[gt_clusters[i]] == l:
                count += 1
                if pred_clusters[i] in prld.keys() and l in prld[pred_clusters[i]].lower():
                    match += 1
            elif pred_clusters[i] in prld.keys() and l in prld[pred_clusters[i]].lower():
                miss += 1
        if match + miss:
            l_p.append(match / (match + miss))
        if count:
            l_r.append(match / count)
    l_p = np.mean(l_p)
    l_r = np.mean(l_r)
    return l_p, l_r

def align_clusters(gt, pred):
    # return {gt_cluster: pred_cluster} mapping
    clusters = list(set(gt.values()))
    pred_num = max(pred.values())
    mapping = {}
    if 0 in clusters:
        clusters.remove(0)
    for c in clusters:
        count = [0 for _ in range(pred_num+1)]
        for k in list(gt.keys()):
            if gt[k] == c:
                count[pred[k]] += 1
        mapping[c] = np.argmax(count)
    return mapping


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

# load retinanet outputs, gt clusters, and labels as dict. 
def load_data(show, clip):
    # format should be {id: {im_name, face, facial_area, confidence, cluster, label}}
    retina = np.load(clip + '.npy', allow_pickle=True).item()
    clusters = np.load(clip + '_anns.npy', allow_pickle=True)
    labels = np.load(clip + '_dict.npy', allow_pickle=True).item()
    extra = np.load(clip + '_extra.npy', allow_pickle=True).item()
    ind = 0
    new = {}
    new2 = {}
    ims = list(retina.keys())
    for i in list(retina.keys()):
        for r in retina[i]:
            item = {'im_name': i, 'face': None, 'facial_area': None, 'confidence': None, 'cluster': None, 'label': None}
            item['face'] = r['face']
            item['facial_area'] = r['facial_area']
            item['confidence'] = r['confidence']
            item['cluster'] = clusters[ind]
            item['label'] = labels[clusters[ind]]
            new[ind] = item
            new2[ind] = {'im_name': i, 'face': r['face'], 'facial_area': r['facial_area'], 'confidence': r['confidence']}
            ind += 1
    for i in list(extra.keys()):
        if not i in ims:
            for r in extra[i]:
                new2[ind] = {'im_name': i, 'face': r['face'], 'facial_area': r['facial_area'], 'confidence': r['confidence']}
                ind += 1
    return new, new2, labels

def filter_spans(data, h=50, conf=.8):
    new_data = {}
    i = 0
    for k in list(data.keys()):
        item = data[k]
        if item['facial_area']['h'] > h and item['confidence'] > conf:
            new_data[i] = item
            i += 1
    return new_data

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

def calc_p_counts(gt, pred, cluster, name, aligned):
    n = aligned
    count, corr = 0, 0
    for i in list(n.keys()):
        if n[i] and name.lower() in n[i].lower():
            count += 1
            if pred[i] == cluster:
                corr += 1
    return corr, count

def calc_clusters(d, show, clip, path, method='pairwise', th=.5, dist=.25):
    # output format of {id: cluster_id, ...}
    ims = np.load(clip + '_feats.npy', allow_pickle=True).item()
    if method == 'pairwise':
        clusters = {}
        remaining = list(ims.keys())
        ind = 0
        while remaining:
            m = max(remaining, key=lambda k: d[k]['facial_area']['w'])
            clusters[m] = ind
            remaining.remove(m)
            m_im = ims[m]
            matches = []
            for r in remaining:
                out = dst.findCosineDistance(m_im, ims[r])
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

def retinanet(show, clip, fps):
    path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/' + show + '_frames/' + clip +'/'
    ims = load_ims(path, fps)
    d = {}
    for i in tqdm(ims):
        d[i] = DeepFace.extract_faces(img_path=path + i,enforce_detection= False,detector_backend='retinaface')
    return d

def load_ims(im_path, fps):
    ims = os.listdir(im_path)
    ims = sorted(ims)
    ims = ims[::fps]
    return ims

def extract(d,show,clip,path,th):
    nd = {}
    for k in tqdm(list(d.keys())):
        image = np.array(Image.open(path+show+'_frames/'+clip+'/'+d[k]['im_name']))
        a = d[k]['facial_area']
        w2 = int(a['w']*th)
        h2 = int(a['h']*th)
        x = max(0, a['x'] - w2)
        y = max(0, a['y'] - h2)
        w = a['w']+w2
        h = a['h']+h2
        im2 = image[y:y+h, x:x+w]
        im2 = DeepFace.represent(img_path = im2, enforce_detection=False,detector_backend='skip', model_name="VGG-Face")[0]["embedding"]
        nd[k] = im2
    return nd