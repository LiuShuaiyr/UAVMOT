import numpy as np
import math
import itertools
from scipy.spatial.distance import cdist
import copy

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
def Vector(A, B):
    # a = A.tlbr[0]
    A_xy = [0.5*(A.tlbr[0] + A.tlbr[2]), 0.5*(A.tlbr[1] + A.tlbr[3])]
    B_xy = [0.5*(B.tlbr[0] + B.tlbr[2]), 0.5*(B.tlbr[1] + B.tlbr[3])]
    v=list(map(lambda x: x[0] - x[1], zip(B_xy, A_xy)))
    return v

def local_relation_filtering(tracks, detections, reference_track, reference_detect, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    matches = []
    BC_tracks = tracks.copy()
    tracks.remove(reference_track)
    BC_detections = detections.copy()
    detections.remove(reference_detect)

    # reference_track_id = BC_tracks.index(reference_track)
    # reference_detect_id = BC_detections.index(reference_detect)
    # matches.append([reference_track_id, reference_detect_id])

    t=1
    while t<15:
        track_BC_points = list(itertools.permutations(tracks, 2))
        for i in range(len(track_BC_points)):
            track_BC_point = track_BC_points[i]
            track_B_point = track_BC_point[0]
            track_C_point = track_BC_point[1]
            track_AB = Vector(reference_track, track_B_point)
            track_AC = Vector(reference_track, track_C_point)
            track_BC = Vector(track_B_point, track_C_point)
            track_ang_BAC = angle(track_AB, track_AC)
            track_ang_ABC = 180-angle(track_AB, track_BC)
            track_ang_length = [track_ang_BAC,track_ang_ABC,np.linalg.norm(np.array(track_AB)), np.linalg.norm(np.array(track_AC)),np.linalg.norm(np.array(track_BC))]
            # print(track_ang_length)
            if detections==[]:
                break
            if len(detections)<2:
                # print(detections)
                AB = Vector(reference_detect, detections[0])
                det_length = np.linalg.norm(np.array(AB))
                track_length = np.linalg.norm(np.array(track_AB))
                error = np.abs(det_length-track_length)
                # print(error)
                if error < 15:
                    reference_track_B_id = BC_tracks.index(track_B_point)
                    detect_B_id = BC_detections.index(detections[0])
                    matches.append([reference_track_B_id, detect_B_id])
                    detections.remove(detections[0])
                    tracks.remove(track_B_point)

                # print("将最后一个没有匹配的匹配")
            else:
                det_BC_points = list(itertools.permutations(detections, 2))
                cost = []

                for i in range(len(det_BC_points)):
                    BC_point = det_BC_points[i]
                    B_point = BC_point[0]
                    C_point = BC_point[1]
                    AB = Vector(reference_detect, B_point)
                    AC = Vector(reference_detect, C_point)
                    BC = Vector(B_point, C_point)
                    ang_BAC = angle(AB, AC)
                    ang_ABC = 180 - angle(AB, BC)
                    det_ang_length = [ang_BAC, ang_ABC, np.linalg.norm(np.array(AB)), np.linalg.norm(np.array(AC)),
                                      np.linalg.norm(np.array(BC))]
                    # print(det_ang_length)
                    cost_matrix = np.maximum(0.0, cdist([track_ang_length], [det_ang_length], metric))
                    cost.append(cost_matrix[0][0])
                if det_BC_points == []:
                    break

                # print(min(cost))
                if min(cost) < 0.001:
                    id = cost.index(min(cost))
                    detect_B_point = det_BC_points[id][0]
                    detect_C_point = det_BC_points[id][1]
                    reference_track_B_id = BC_tracks.index(track_B_point)
                    reference_track_C_id = BC_tracks.index(track_C_point)
                    detect_B_id = BC_detections.index(detect_B_point)
                    detect_C_id = BC_detections.index(detect_C_point)
                    matches.append([reference_track_B_id, detect_B_id])
                    matches.append([reference_track_C_id, detect_C_id])

                    # reference_track = track_B_point
                    # reference_detect = detect_B_point

                    detections.remove(detect_B_point)
                    detections.remove(detect_C_point)
                    tracks.remove(track_B_point)
                    tracks.remove(track_C_point)
                    # track_BC_points = list(itertools.permutations(tracks, 2))
                    break


        t=t+1






    u_track =list(range(0,len(BC_tracks)))
    u_detect = list(range(0, len(BC_detections)))
    if matches==[]:
        print("**")
    else:
        matches = np.asarray(matches)
        for i in range(len(matches[:, 0])):
            # print(matches[i, 0])
            u_track.remove(matches[i, 0])

        for i in range(len(matches[:, 1])):
            # print(matches[i, 1])
            u_detect.remove(matches[i, 1])



    # u_detect = BC_detections.index(detections[0])




    # for i in range(len(detections)):
    #     B_point = detections[i]
    #     C_detections.remove(B_point)
    #     for j in range(len(C_detections)):
    #         C_point = C_detections[j]
    #         AB = Vector(reference_detect, B_point)
    #         AC = Vector(reference_detect, C_point)
    #         ang = angle(AB, AC)


    # cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    # if cost_matrix.size == 0:
    #     return cost_matrix

    # det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    # 默认计算两个特征向量之间的夹角余弦
    # Nomalized features
    # cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return matches,u_track,u_detect