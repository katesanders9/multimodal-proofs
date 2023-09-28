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
from utils.vision_utils import *
from char import CharacterEngine

# video preprocessing
# -------------------


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


class ObjectRecognition(object):
    def __init__(self):
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/'
        self.data_path = '/srv/local2/ksande25/NS_data/TVQA/chars/'
        self.show = None
        self.clip = None
        self.rate = 2
        self.thresh = .9

    def set_rate(self, rate):
        self.rate = rate

    def set_thresh(self, thresh):
        self.thresh = thresh

    def set_clip(self, show, clip):
        self.show = show
        self.clip = clip

    def load_ims(self):
        im_names = get_ims(self.show, self.clip, self.rate)
        images = {}
        for i in im_names:
            images[i] = Image.open(self.path + self.show + '_frames/' + self.clip + '/' + i)
        return images

    # run yolo-ing
    def yolo(self):
        ims = load_ims()
        d = {}
        for image in tqdm(list(ims.keys())):
            inputs = image_processor(images=ims[image], return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([ims[image].size[::-1]])
            results = self.image_processor.post_process_object_detection(outputs, threshold=self.thresh, target_sizes=target_sizes)[0]
            preds = list(zip(results["scores"], results["labels"], results["boxes"]))
            preds = [[a.detach().numpy().item(), self.model.config.id2label[b.detach().numpy().item()], list(c.detach().numpy())] for a, b, c in preds]
            d[image] = [p[2] for p in preds if p[1] == 'person']
        np.save(self.data_path + self.show + '_' + self.clip + '_' + str(fps) + '_boxes.npy', d)
        return d


class VisionModel(object):
    def __init__(self):
        self.objectRec = ObjectRecognition()
        self.charRec = CharacterEngine()
        self.data_path = '/srv/local2/ksande25/NS_data/TVQA/chars/'
        self.path = '/srv/local2/ksande25/NS_data/TVQA/frames/frames_hq/'

    def set_clip(self, show, clip):
        self.show = show
        self.clip = clip
        self.charRec.set_clip(show, clip)
        self.objectRec.set_clip(show, clip)

    def set_rate(self, rate)
        self.rate = rate
        self.charRec.set_rate(self.rate)

    def set_char_thresh(self, thresh):
        self.char_thresh = thresh
        self.charRec.set_thresh(thresh)

    def set_obj_thresh(self, thresh):
        self.obj_thresh = thresh
        self.objectRec.set_thresh(thresh)

    # load video info
    def load_video(self):
        if not os.path.isfile(self.data_path + self.clip + '_' + str(self.rate) + '_' + str(self.char_thresh) + '_faces.npy'):
            faces = self.charRec()
        else:
            faces = np.load(self.data_path + self.clip + '_' + str(self.rate) + '_' + str(self.char_thresh) + '_faces.npy', allow_pickle=True).item()
        if not os.path.isfile(self.data_path + self.show + '_' + self.clip + '_' + str(self.rate) + '_boxes.npy'):
            boxes = yolo()
        else:
            boxes = np.load(self.data_path + self.show + '_' + self.clip + '_' + str(self.rate) + '_boxes.npy', allow_pickle=True).item()
        ims = load_ims()
        return faces, boxes, ims

    # sample frames
    def get_ims(self):
        im_path = self.path + self.show + '_frames/' + self.clip +'/'
        ims = os.listdir(im_path)
        ims = sorted(ims)
        ims = ims[::self.rate]
        return ims


    def load_ims(self):
        im_names = get_ims()
        images = {}
        for i in im_names:
            images[i] = Image.open(self.path+self.show+'_frames/'+self.clip+'/'+i)
        return images


    # query vision model
    def query(self, q, im):
        encoding = processor(im, q, return_tensors="pt")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        ans = self.model.config.id2label[idx]
        logits = logits.reshape(logits.shape[-1])
        score = logits[idx]
        return [ans, score]



    def load(self, people, names, ims, cropping=True):
        # cropped image dict of {im_name: cropped_im}
        cropped = {}
        # iterate
        if not cropping:
            for im_name in list(people.keys()):
                im = np.array(ims[im_name])
                cropped[im_name] = im
            return cropped
        for im_name in list(people.keys()):
            im = np.array(ims[im_name])
            if len(names) == len(people[im_name]):
                cropped[im_name] = im 
            else:
                good_boxes = [p[2:] for p in people[im_name] if p[1].lower() in names]
                bad_boxes = [p[2:] for p in people[im_name] if not p[1].lower() in names]
                # min x necessary
                x1 = min([p[0] for p in good_boxes])
                # max x necessary
                x2 = max([p[2] for p in good_boxes])
                a1 = [p[2] for p in bad_boxes if p[2] < x1]
                a2 = [p[0] for p in bad_boxes if p[0] > x2]
                b1 = max(a1) if a1 else 0
                b2 = min(a2) if a2 else im.shape[1]
                if not any([p[2] > x2 and p[2] < b2 for p in bad_boxes]):
                    x2 = b2
                if not any([p[0] < x1 and p[0] > b1 for p in bad_boxes]):
                    x1 = b1
                cropped[im_name] = crop(im, [int(x1), int(x2)])
        return cropped

    def __call__(self, qp, q_names, rate=1):
        faces, boxes, ims = self.load_video()
        people = align_boxes(faces, boxes)
        # {im_name: [[id, label, person_area], ...]}
        people = filter_ims(people, q_names,rate)
        ims2 = self.load(people, q_names, ims, cropping=True)
        res = [[im] + self.query(qp, im) for im in ims2.values()]
        res = [x for x in res if x[1] == 'yes']
        return max(res, key= lambda x: x[2])[0] if res else None
