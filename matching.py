import threading

import cv2
import numpy as np


def cosine_distance(a, b):
    a_vec = np.asarray(a, dtype=np.float32)
    b_vec = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) + 1e-6
    return 1.0 - float(np.dot(a_vec, b_vec) / denom)


class FaceRegistry:
    def __init__(self, embedding_dim=128):
        self.lock = threading.Lock()
        self.embedding_dim = embedding_dim
        self.known_ids = []
        self._id_to_index = {}
        self.known_embeddings = np.empty((0, embedding_dim), dtype=np.float32)

    def load(self, ids, embeddings):
        with self.lock:
            self.known_ids = list(ids)
            self._id_to_index = {pid: i for i, pid in enumerate(self.known_ids)}
            embs = np.asarray(embeddings, dtype=np.float32)
            if embs.size == 0:
                self.known_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
            else:
                self.known_embeddings = embs.reshape(len(self.known_ids), -1)

    def upsert(self, person_id, embedding):
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        with self.lock:
            idx = self._id_to_index.get(person_id)
            if idx is not None:
                self.known_embeddings[idx] = emb[0]
                return

            self._id_to_index[person_id] = len(self.known_ids)
            self.known_ids.append(person_id)
            if self.known_embeddings.size == 0:
                self.known_embeddings = emb
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, emb])

    def match(self, embedding, threshold):
        with self.lock:
            if self.known_embeddings.size == 0:
                return None, None

            query = np.asarray(embedding, dtype=np.float32)
            embs = self.known_embeddings
            query_norm = np.linalg.norm(query) + 1e-6
            emb_norms = np.linalg.norm(embs, axis=1) + 1e-6
            similarities = np.dot(embs, query) / (emb_norms * query_norm)
            distances = 1.0 - similarities

            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])
            if best_dist <= threshold:
                return self.known_ids[best_idx], best_dist
        return None, None


def face_quality(face_bgr):
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(sharpness)
    except cv2.error:
        return 0.0


def classify_pose_action(kpts, kpt_conf, bbox):
    # COCO keypoints: 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
    def _valid(idx):
        return kpt_conf[idx] > 0.3

    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = w / h

    if aspect > 1.3:
        return "lying", 0.5

    if _valid(5) and _valid(9) and kpts[9][1] < kpts[5][1] - 10:
        return "arms_raised", 0.6
    if _valid(6) and _valid(10) and kpts[10][1] < kpts[6][1] - 10:
        return "arms_raised", 0.6

    if _valid(11) and _valid(13) and _valid(15):
        hip = kpts[11]
        knee = kpts[13]
        ankle = kpts[15]
        v1 = hip - knee
        v2 = ankle - knee
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if angle < 120:
            return "sitting", 0.5
        if angle > 160:
            return "standing", 0.5

    return None, None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def assign_actions_from_pose(pose_results, person_boxes, person_ids, record_action_fn):
    if pose_results is None:
        return
    r0 = pose_results[0]
    if r0.boxes is None or r0.keypoints is None:
        return

    p_boxes = r0.boxes.xyxy.cpu().numpy().astype(np.int32)
    kpts = r0.keypoints.xy.cpu().numpy()
    kconf = r0.keypoints.conf.cpu().numpy()

    for i, pose_box in enumerate(p_boxes):
        best_iou = 0.0
        best_idx = -1
        for j, det_box in enumerate(person_boxes):
            score = iou(pose_box, det_box)
            if score > best_iou:
                best_iou = score
                best_idx = j
        if best_idx == -1 or best_iou < 0.2:
            continue
        if best_idx >= len(person_ids):
            continue
        person_id = person_ids[best_idx]
        if person_id is None:
            continue
        label, conf = classify_pose_action(kpts[i], kconf[i], pose_box)
        if label:
            record_action_fn(person_id, label, conf, "pose")
