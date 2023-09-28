def get_overlap(f, boxes):
    outs = [[b] + [overlapping(f,b)] for b in boxes]
    return max(outs, key=lambda x: x[1])[0]

def overlapping(f, b):
    f_dims = [f[2]['x'], f[2]['y'], f[2]['w'], f[2]['h']]
    b_dims = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
    overlap_x = min(f_dims[0] + f_dims[2], b_dims[0] + b_dims[2]) - max(f_dims[0], b_dims[0])
    overlap_y = min(f_dims[1] + f_dims[3], b_dims[1] + b_dims[3]) - max(f_dims[1], b_dims[1])
    if overlap_x <= 0 or overlap_y <= 0:
        return -1
    return overlap_x * overlap_y #, b_dims[2] * b_dims[3]

def filter_ims(d, names, rate=3):
    oi = {}
    ims = list(d.keys())[::rate]
    for i in ims:
        if all([name in [x[1].lower() if x[1] else None for x in d[i]] for name in names]):
            oi[i] = d[i]
    return oi

# crop image horizontally to x1, x2 dimensions
def crop(im, dims):
    return im[:, dims[0]:dims[1]]

def align_boxes(face_list, box_list):
    new = {}
    ims = [im for im in list(box_list.keys()) if im in face_list.keys()]
    for im in ims:
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