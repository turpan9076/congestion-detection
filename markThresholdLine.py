# -*- coding: utf-8 -*-

import os
import glob
import time
import re
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
import tensorflow as tf
import tensorflow_hub as hub

# ================================
# ユーザー入力
# ================================
TARGET_POINT = "C00495"
START_DATETIME = "20251113 0710"
END_DATETIME   = "20251114 0700"
FOLDER_NAME = "20251113"

MODEL_HANDLE = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
TARGET_LABELS = ["Land vehicle", "Vehicle", "Car", "Bus", "Truck", "Van", "Motorcycle"]
MIN_SCORE = 0.1

LINE_COLOR = (255, 0, 0)
LINE_WIDTH = 4

# ================================
# 日付時刻パース
# ================================
def parse_file_datetime(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d{8})_(\d{4})\.jpg$", name)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")
    return dt


def is_in_range(file_dt, start_dt, end_dt):
    if start_dt <= end_dt:
        return start_dt <= file_dt <= end_dt
    if file_dt.date() == start_dt.date() or file_dt.date() == end_dt.date():
        return file_dt >= start_dt or file_dt <= end_dt
    return False


start_dt = datetime.strptime(START_DATETIME, "%Y%m%d %H%M")
end_dt   = datetime.strptime(END_DATETIME,   "%Y%m%d %H%M")

INPUT_DIR = f"./input/{FOLDER_NAME}/{TARGET_POINT}"
OUTPUT_BASE = f"./input_with_line/{FOLDER_NAME}/{TARGET_POINT}"
os.makedirs(OUTPUT_BASE, exist_ok=True)

print("Loading model...")
rcnn_detector = hub.load(MODEL_HANDLE).signatures['default']
print("Model loaded.")

# ---------- 補助関数 ----------
def load_img_tensor(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def iou(box1, box2):
    y1, x1, y2, x2 = box1
    y1_, x1_, y2_, x2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h
    box1_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    box2_area = max(0.0, (x2_ - x1_)) * max(0.0, (y2_ - y1_))
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def merge_overlapping_detections(boxes, scores, class_names, target_labels, iou_threshold=0.5, min_score=0.1):

    def iou_local(b1, b2):
        return iou(b1, b2)

    def is_contained(b1, b2):
        return (b1[0] <= b2[0] and b1[1] <= b2[1] and
                b1[2] >= b2[2] and b1[3] >= b2[3])

    def center_distance(b1, b2):
        cy1 = (b1[0] + b1[2]) / 2
        cx1 = (b1[1] + b1[3]) / 2
        cy2 = (b2[0] + b2[2]) / 2
        cx2 = (b2[1] + b2[3]) / 2
        return np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2)

    def similar_size(b1, b2, ratio=0.35):
        A = (b1[2] - b1[0]) * (b1[3] - b1[1])
        B = (b2[2] - b2[0]) * (b2[3] - b2[1])
        if A == 0 or B == 0:
            return False
        r = min(A, B) / max(A, B)
        return r >= ratio

    def is_duplicate(b1, b2):
        if iou_local(b1, b2) > iou_threshold:
            return True
        if is_contained(b1, b2) or is_contained(b2, b1):
            return True
        if center_distance(b1, b2) < 0.05 and similar_size(b1, b2):
            return True
        return False

    N = boxes.shape[0]

    labels = []
    for cn in class_names:
        if isinstance(cn, bytes):
            labels.append(cn.decode("utf-8"))
        else:
            labels.append(str(cn))

    candidate_idx = [
        i for i in range(N)
        if labels[i] in target_labels and scores[i] >= min_score
    ]

    if len(candidate_idx) == 0:
        return np.zeros((0,4)), np.zeros((0,)), np.array([])

    m = len(candidate_idx)
    visited = [False] * m

    cand_boxes = [boxes[i] for i in candidate_idx]
    cand_scores = [scores[i] for i in candidate_idx]

    merged_indices = []

    for i in range(m):
        if visited[i]:
            continue

        stack = [i]
        comp = []

        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue

            visited[cur] = True
            comp.append(cur)

            for j in range(m):
                if not visited[j]:
                    if is_duplicate(cand_boxes[cur], cand_boxes[j]):
                        stack.append(j)

        best_local = max(comp, key=lambda idx: cand_scores[idx])
        merged_indices.append(candidate_idx[best_local])

    merged_indices_sorted = sorted(merged_indices, key=lambda i: scores[i], reverse=True)

    return (
        boxes[np.array(merged_indices_sorted)],
        scores[np.array(merged_indices_sorted)],
        class_names[np.array(merged_indices_sorted)]
    )


# 外れ値除去
def remove_outliers_iqr(values):
    if len(values) == 0:
        return [], None, None
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)].tolist()
    return filtered, float(lower), float(upper)


# ---------- ファイル抽出 ----------
search_pattern = os.path.join(INPUT_DIR, TARGET_POINT + "_*.jpg")
all_candidates = sorted(glob.glob(search_pattern))

target_files = []

for path in all_candidates:
    file_dt = parse_file_datetime(path)
    if file_dt and is_in_range(file_dt, start_dt, end_dt):
        target_files.append(path)

print(f"Found {len(target_files)} files in range.")

# ---------- 検出 ----------
per_image_detections = {}
all_ymaxs = []

for img_path in target_files:
    try:
        img_tensor = load_img_tensor(img_path)
        converted_img = tf.image.convert_image_dtype(img_tensor, tf.float32)[tf.newaxis, ...]
        result = rcnn_detector(converted_img)
        result = {k: v.numpy() for k, v in result.items()}
    except Exception as e:
        print(f"Detection failed: {img_path}, {e}")
        continue

    boxes = result.get("detection_boxes", np.zeros((0,4)))
    scores = result.get("detection_scores", np.zeros((0,)))
    class_entities = result.get("detection_class_entities", np.array([], dtype=object))

    boxes_final, scores_final, classes_final = merge_overlapping_detections(
        boxes, scores, class_entities, TARGET_LABELS,
        iou_threshold=0.5, min_score=MIN_SCORE
    )

    for b in boxes_final:
        ymax = float(b[2])
        all_ymaxs.append(ymax)

    per_image_detections[img_path] = {
        "boxes": boxes_final,
        "scores": scores_final,
        "classes": classes_final
    }


if len(all_ymaxs) == 0:
    print("No detections.")
    raise SystemExit(0)

filtered, _, _ = remove_outliers_iqr(all_ymaxs)
if len(filtered) == 0:
    global_bottom_norm = float(min(all_ymaxs))
else:
    global_bottom_norm = float(min(filtered))

print(f"Global bottom = {global_bottom_norm:.6f}")

# ---------- 描画 ----------
for img_path in target_files:
    try:
        with Image.open(img_path).convert("RGB") as im:
            w, h = im.size
            y_px = int(round(global_bottom_norm * h))
            y_px = max(0, min(h-1, y_px))
            draw = ImageDraw.Draw(im)
            draw.line([(0, y_px), (w, y_px)], fill=LINE_COLOR, width=LINE_WIDTH)
            save_path = os.path.join(OUTPUT_BASE, os.path.basename(img_path))
            im.save(save_path)
    except Exception as e:
        print(f"Draw failed: {img_path}, {e}")

print("Done.")
