# /srv/local1/ksande25/vilt/vilt_vqa.ckpt
import os 
import numpy as np
import json
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection
from tqdm import tqdm

# video preprocessing
# -------------------

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/'
data_path = '/srv/local2/ksande25/NS_data/TVQA/chars/'

# run character recognition
def char_rec(show, clip, rate=fps):
    return pred_chars(show, clip, fps=fps)
    # return {im_name: [[id, label, face_area], ...]}


def align_boxes(face_list, box_list):
    new = {}
    for im in list(box_list.keys()):
        entry = []
        boxes = box_list[im]
        faces = face_list[im]
        for f in faces:
            box = get_overlap(f, boxes)
            if box:
                entry += [f[:2] + box]
            else:
                entry += [[list(f)]]
        new[im] = entry
    return new


def get_overlap(f, boxes):
    os = [[b] + [overlapping(f,b)] for b in boxes]
    return max(os, key=lambda x: x[1])[0]


def overlapping(f, b):
    f_dims = [f[2]['x'], f[2]['y'], f[2]['w'], f[2]['h']]
    b_dims = b.tolist()
    overlap_x = min(f_dims[0] + f_dims[2], b_dims[0] + b_dims[2]) - max(f_dims[0], b_dims[0])
    overlap_y = min(f_dims[1] + f_dims[3], b_dims[1] + b_dims[3]) - max(f_dims[1], b_dims[1])
    if overlap_x <= 0 or overlap_y <= 0:
        return -1, -1
    return overlap_x * overlap_y #, b_dims[2] * b_dims[3]


# sample frames
def get_ims(show, clip, fps):
    im_path = path + show + '_frames/' + clip +'/'
    ims = os.listdir(im_path)
    ims = sorted(ims)
    ims = ims[::fps]
    return ims


def load_ims(show, clip, fps):
    im_names = get_ims(show, clip, fps)
    images = {}
    for i in im_names:
        images[i] = Image.open(path+show+'_frames/'+clip+'/'+i)
    return images


# run yolo-ing
def yolo(show, clip, fps, thresh=.5):
    ims = load_ims(show, clip, fps)
    d = {}
    for image in tqdm(list(ims.keys())):
        inputs = image_processor(images=ims[image], return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([ims[image].size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        preds = list(zip(results["scores"], results["labels"], results["boxes"]))
        preds = [[a.detach().numpy().item(), model.config.id2label[b.detach().numpy().item()], list(c.detach().numpy())] for a, b, c in preds]
        d[image] = [p[2] for p in preds if p[1] == 'person']
    np.save(data_path + show + '_' + clip + '_' + str(fps) + '_boxes.npy', d)
    return d


# video querying
# --------------------

# load video info
def load_video(show, clip, fps, th):
    if not os.path.isfile(data_path + clip + '_' + str(fps) + '_' + str(th) + '_char.npy'):
        faces = char_rec(show, clip, fps)
    else:
        faces = np.load(data_path + clip + '_' + str(fps) + '_' + str(th) + '_char.npy', allow_pickle=True).item()
    if not os.path.isfile(data_path + show + '_' + clip + '_' + str(fps) + '_boxes.npy'):
        boxes = yolo(show, clip, fps)
    else:
        boxes = np.load(data_path + show + '_' + clip + '_' + str(fps) + '_boxes.npy', allow_pickle=True).item()
    ims = load_ims(show, clip, fps)
    return faces, boxes, ims


# filter frames
def filter_ims(d, names, rate=3):
    oi = {}
    ims = list(d.keys())[::rate]
    for i in ims:
        if all([name in [x[1] for x in d[i]] for name in names]):
            oi[i] = d[i]
    return oi


# query vision model
def query(q, im):
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    ans = model.config.id2label[idx]
    score = logits[idx]
    return ans, score

def crop(im, dims):
    return im[dims[1]:dims[1]+dims[3], dims[0]:dims[0]+dims[2]]

def load(people, names, crop=True):
    # cropped image dict of {im_name: cropped_im}
    cropped = {}
    # iterate
    if not crop:
        for im_name in list(people.keys()):
            im = np.array(Image.open(path+show+'_frames/'+clip+'/'+d[k]['im_name']))
            cropped[im_name] = im
        return cropped
    for im_name in list(people.keys()):
        im = np.array(Image.open(path+show+'_frames/'+clip+'/'+d[k]['im_name']))
        if len(names) == len(people[im_name]):
            cropped[im_name] = im 
        else:
            good_boxes = [p[2] for p in people[im_name] if p[1] in names]
            bad_boxes = [p[2] for p in people[im_name] if not p[1] in names]
            # min x, y necessary
            minn = [min([p[0] for p in good_boxes]), min([p[1] for p in good_boxes])]
            # max x, y necessary
            maxn = [max([p[0] for p in bad_boxes]), max([p[1] for p in bad_boxes])]
            # max min x, y unnecessary
            minu = [max([p[0]+p[2] for p in good_boxes]), max([p[1]+p[3] for p in good_boxes])]
            # min max x, y unnecessary
            maxu = [min([p[0]+p[2] for p in good_boxes]), min([p[1]+p[3] for p in good_boxes])]

            x = min(minu[0], minn[0])
            y = min(minu[1], minn[1])
            x2 = max(maxu[0], maxn[0])
            y2 = max(maxu[1], maxn[1])
            w = x2 - x
            h = y2 - y 
            cropped[im_name] = crop(im, [x, y, w, h])
    return cropped


def run(show, clip, qp, q_names, fps=2, rate=3):
    faces, boxes, ims = load_video(show, clip, fps, 0.5)
    people = align_boxes(faces, boxes)
    # {im_name: [[id, label, person_area], ...]}
    people = filter_ims(people, q_names)
    ims = load(people, q_names, crop=True)
    res = [[im] + query(qp, im) for im in ims]
    res = [x for x in res if x[1] == 'yes']
    return max(res, key= lambda x: x[2])[0] if res else None
