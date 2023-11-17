from tqdm import tqdm
from PIL import Image
import numpy as np
import itertools
import os
import copy


class Vertex:
    def __init__(self, model, class_name, confidence, bbox, prob_vector=None):
        self.model = model
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.prob_vector = prob_vector

    def area(self):
        return (self.bbox[0] - self.bbox[2]) * (self.bbox[1] - self.bbox[3])

    def iou(self, v):
        return bb_intersection_over_union(self.bbox, v.bbox)

    def overlap(self, v, iou_thresh=0.50):
        return bb_intersection_over_union(self.bbox, v.bbox) >= iou_thresh

    def to_tuple(self):
        return (self.class_name, self.confidence, *self.bbox)


class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.iou = bb_intersection_over_union(self.v1.bbox, self.v2.bbox)

    def __hash__(self):
        return {self.v1, self.v2}


class Clique:
    def __init__(self, v=None):
        self.vs = []
        self.models = set()
        if v is not None:
            self.vs.append(v)
            self.models.add(v.model)

    def update(self, c):
        for v in c.vs:
            self.vs.append(v)
            self.models.add(v.model)

    def winning_vertex(self, max_votes):
        # Find the winning class with the largest number of votes
        # TODO: refinement
        meta = {}
        for v in self.vs:
            if v.class_name not in meta:
                meta[v.class_name] = []
            meta[v.class_name].append(v)

        outputs = []
        for class_name, votes in meta.items():
            if len(votes) < 2:
                continue
            confidences = [v.confidence for v in votes]
            # weighted bounding box
            bbox = np.average(np.asarray([v.bbox for v in votes]), axis=0, weights=confidences)
            # averge scores
            confidence = np.mean(confidences)
            outputs.append(Vertex('_'.join([str(v.model) for v in votes]), class_name, confidence, bbox))
        return outputs


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum
    # of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def soft_non_maximum_suppression(vs):
    vs = sorted(vs, key=lambda v: -v.confidence)
    vs_nms = []
    for v in vs:
        v_overlap = None
        for v_ in vs_nms:
            if v.class_name == v_.class_name and v.overlap(v_):
                v_overlap = v_
                break
        if v_overlap is None:  # No overlapped bounding box with the same class is found
            vs_nms.append(v)
        else:  # Compute adjusted confidence score
            adjusted_confidence = v.confidence * (1 - bb_intersection_over_union(v.bbox, v_overlap.bbox))
            if adjusted_confidence >= 0.05:
                v.confidence = adjusted_confidence
                vs_nms.append(v)
    return vs_nms


def load_detections(fname, min_conf=0.05):
    out = []
    with open(fname, 'r') as f:
        for values in f.readlines():
            values = values.strip().split(' ')
            if len(values) == 0:
                continue
            tup = (values[0], *map(float, values[1:]))
            if tup[1] >= min_conf:
                out.append(tup)
    return out


def load_test_file_paths(filename):
    fpaths = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                if len(line.split(' ')) > 1:
                    line = line.split(' ')[0]
                fpaths.append(line)
    return fpaths

def graph_fusion(fname, attack_dirs):
    skip = False

    # Step 1: Add each detection as a vertex colored by model
    model2vertexes, vertex2clique = {}, {}
    low_confidence_vertexes = []
    for model in models:
        fpath_txt = os.path.join(attack_dirs[model], fname)
        if not os.path.exists(fpath_txt):
            skip = True
            break
        detections = load_detections(fpath_txt)
        model2vertexes[model] = []
        for obj in detections:
            v = Vertex(model, obj[0], obj[1], obj[2:])
            if v.confidence >= thresholds_default[model]:
                c = Clique(v)
                model2vertexes[model].append(v)
                vertex2clique[v] = c
            else:
                low_confidence_vertexes.append(v)
    if skip:
        return None, None

    # Step 2: Add an edge between any two vertexes if they overlap with IOU > 0.50
    edges = []
    for m1, m2 in itertools.combinations(model2vertexes.keys(), r=2):
        for v1, v2 in itertools.product(model2vertexes[m1], model2vertexes[m2]):
            if v1.overlap(v2):
                edges.append(Edge(v1, v2))

    # Step 3: Sort edges by IOU (weight) in descending order
    edges = sorted(edges, key=lambda edge: -edge.iou)

    # Step 4: Merge cliques by a greedy approach
    for edge in edges:
        clique1 = vertex2clique[edge.v1]
        clique2 = vertex2clique[edge.v2]
        if clique1.models.isdisjoint(clique2.models):
            clique1.update(clique2)
            for v in clique2.vs:
                vertex2clique[v] = clique1
    return low_confidence_vertexes, set(vertex2clique.values())


import mxnet as mx

def getEnsembleResults(det_ids_list, det_scores_list, det_bboxes_list, sampleId, netIds, thresholds_default = None):
    if not thresholds_default:
        thresholds_default = {j:0.5 for j in netIds}

    model2vertexes, vertex2clique = {}, {}
    low_confidence_vertexes = []
    for j in netIds:
        valid_pred = np.where(det_ids_list[j][sampleId].flat >=0)[0]
        valid_ids = det_ids_list[j][sampleId][0][valid_pred].squeeze(1)
        valid_scores = det_scores_list[j][sampleId][0][valid_pred].squeeze(1)
        valid_bboxes = det_bboxes_list[j][sampleId][0][valid_pred]
        model2vertexes[j] = []

        for i in range(len(valid_ids)):
            v = Vertex(j, valid_ids[i], valid_scores[i], valid_bboxes[i])
            if v.confidence >= thresholds_default[j]:
                c = Clique(v)
                model2vertexes[j].append(v)
                vertex2clique[v] = c
            else:
                low_confidence_vertexes.append(v)
    edges = []
    for m1, m2 in itertools.combinations(model2vertexes.keys(), r=2):
        for v1, v2 in itertools.product(model2vertexes[m1], model2vertexes[m2]):
            if v1.overlap(v2):
                edges.append(Edge(v1, v2))

    edges = sorted(edges, key=lambda edge: -edge.iou)

    for edge in edges:
        clique1 = vertex2clique[edge.v1]
        clique2 = vertex2clique[edge.v2]
        if clique1.models.isdisjoint(clique2.models):
            clique1.update(clique2)
            for v in clique2.vs:
                vertex2clique[v] = clique1

    cliques = set(vertex2clique.values())
    vertexes = copy.deepcopy(low_confidence_vertexes)

    if cliques is None:  # cliques is None means some detection files are missing
        return [], [], []
    # Decision-making process for each clique
    vertexes += list(itertools.chain.from_iterable([c.winning_vertex(max_votes=len(netIds)) for c in cliques]))

    # Apply non-maximum suppression to finalize detection results and output
    bboxes = []
    ids = []
    scores = []
    for v in soft_non_maximum_suppression(vertexes):
        ids.append(v.class_name)
        scores.append(v.confidence)
        bboxes.append(v.bbox)

    scores = np.array(scores)
    idx = scores.argsort()[::-1]
    bboxes = np.array(bboxes)[idx]
    scores = scores[idx]
    ids = np.array(ids)[idx]
    bboxes = mx.nd.array(bboxes[np.newaxis, ...])
    scores = mx.nd.array(scores[np.newaxis, ..., np.newaxis])
    ids = mx.nd.array(ids[np.newaxis, ..., np.newaxis])

    return ids, scores, bboxes

