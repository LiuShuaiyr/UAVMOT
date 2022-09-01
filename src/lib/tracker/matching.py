import lap
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from lib.tracking_utils import kalman_filter
import math


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    :param cost_matrix:
    :param thresh:
    :return:
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), \
               tuple(range(cost_matrix.shape[0])), \
               tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


# TODO: using GIOU, DIOU, CIOU... to replace IOU

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    # 默认计算两个特征向量之间的夹角余弦
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf,
                cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean,
                                             track.covariance,
                                             measurements,
                                             only_position,
                                             metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix
def local_relation_fuse_motion(cost_matrix,
                tracks,
                detections,
                only_position=False,
                lambda_=0.98):
    """
    :param kf:
    :param cost_matrix:
    :param tracks:
    :param detections:
    :param only_position:
    :param lambda_:
    :return:
    """
    if cost_matrix.size == 0:
        return cost_matrix

    gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    structure_distance = structure_similarity_distance(tracks,
                                                       detections)
    cost_matrix = lambda_ * cost_matrix + (1 - lambda_) * structure_distance

    return cost_matrix
def structure_similarity_distance(tracks, detections):
    track_structure = structure_representation(tracks)
    detection_structure = structure_representation(detections,mode='detection')
    cost_matrix = np.maximum(0.0, cdist(track_structure, detection_structure, metric="cosine"))

    return cost_matrix
def angle(v1, v2):
    # dx1 = v1[2] - v1[0]
    # dy1 = v1[3] - v1[1]
    # dx2 = v2[2] - v2[0]
    # dy2 = v2[3] - v2[1]
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0]
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def structure_representation(tracks,mode='trcak'):
    local_R =400
    structure_matrix =[]
    for i, track_A in enumerate(tracks):
        length = []
        index =[]
        for j, track_B in enumerate(tracks):
            # print(track_A.mean[0:2])
            if mode =="detection":
                pp = list(
                    map(lambda x: np.linalg.norm(np.array(x[0] - x[1])), zip(track_A.to_xyah()[0:2], track_B.to_xyah()[0:2])))
            else:
                pp=list(map(lambda x: np.linalg.norm(np.array(x[0] - x[1])), zip(track_A.mean[0:2],track_B.mean[0:2])))
            lgt = np.linalg.norm(pp)
            if lgt < local_R and lgt >0:
                length.append(lgt)
                index.append(j)

        if length==[]:
            v =[0.0001,0.0001,0.0001]

        else:
            max_length = max(length)
            min_length = min(length)
            if max_length == min_length:
                v = [max_length, min_length, 0.0001]
            else:
                max_index = index[length.index(max_length)]
                min_index = index[length.index(min_length)]
                if mode == "detection":
                    v1 = tracks[max_index].to_xyah()[0:2] - track_A.to_xyah()[0:2]
                    v2 = tracks[min_index].to_xyah()[0:2] - track_A.to_xyah()[0:2]
                else:
                    v1 = tracks[max_index].mean[0:2] - track_A.mean[0:2]
                    v2 = tracks[min_index].mean[0:2] - track_A.mean[0:2]

                include_angle = angle(v1, v2)
                v = [max_length, min_length, include_angle]

        structure_matrix.append(v)

    return np.asarray(structure_matrix)