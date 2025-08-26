import time
import pdb
import fire
from glob import glob
import os
from . import kitti_common as kitti
import numpy as np
import csv

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0 or len(lines[0]) < 15:
        content = []
    else:
        content = [line.strip().split(' ') for line in lines]

    annotations['name'] = np.array([x[0] for x in content])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros([len(annotations['bbox'])])
    return annotations

def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1,
             metric='R40'):
    
    from .eval import get_coco_eval_result, get_official_eval_result
    # dt_labels = glob(os.path.join(result_path, "segment-*", "*.txt"), recursive=True, )
    # gt_labels = glob(os.path.join(label_path, "segment-*", "*.txt"), recursive=True, )
    dt_annos = []
    # for dt_label in dt_labels:
    #     dt_annos.append(get_label_anno(dt_label))
    # dt_annos = kitti.get_label_annos(result_path)
    # if score_thresh > 0:
    #     dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    lines = []
    with open(label_split_file, 'r') as file:
        lines = [line.strip() for line in file]
    # val_image_ids = _read_imageset_file(label_split_file)
    val_image_ids = lines
    gt_annos = []
    for idx in val_image_ids:
        parent_path = os.path.dirname(label_path)
        parent_path = os.path.dirname(parent_path)
        label_filename = parent_path + '/' + idx + '.txt'
        gt_annos.append(get_label_anno(label_filename))
    for idx in val_image_ids:
        parent_path = os.path.dirname(result_path)
        parent_path = os.path.dirname(parent_path)
        label_filename = parent_path + '/' + idx + '.txt'
        dt_annos.append(get_label_anno(label_filename))
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    # gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class, metric=metric)

def generate_kitti_3d_detection(prediction, predict_txt):

    ID_TYPE_CONVERSION = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
    }

    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)

def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()

if __name__ == '__main__':
    fire.Fire()