import os
import time
import json
import requests
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# =========================
# 設定
# =========================

BASE_URL = "https://www.seishiga.kkr.mlit.go.jp/himeji/pic"
INPUT_DIR = "./input"
OUTPUT_JSON = "./output_congestion.json"
CAMERA_JSON = "./demo.json"

TARGET_MINUTES = {0, 10, 20, 30, 40, 50}

# =========================
# CAMERA_INFO
# =========================

with open(CAMERA_JSON, "r", encoding="utf-8") as f:
    camera_json = json.load(f)

CAMERA_INFO = {
    cam["id"]: {
        "name": cam["name"],
        "lat": cam["lat"],
        "lon": cam["lon"]
    }
    for cam in camera_json["cameras"]
}

CAMERA_URLS = {
    cam_id: f"{BASE_URL}/{cam_id}.jpg"
    for cam_id in CAMERA_INFO.keys()
}

# =========================
# Faster R-CNN
# =========================

rcnn_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
rcnn_detector = hub.load(rcnn_handle).signatures["default"]

TARGET_LABELS = ["Land vehicle", "Vehicle", "Car", "Bus", "Truck", "Van", "Motorcycle"]

LANE_INFO = {
    "C00452": {"lanes": 4, "range": 240},
    "C00453": {"lanes": 4, "range": 340},
    "C00454": {"lanes": 5, "range": 160},
    "C00456": {"lanes": 4, "range": 490},
    "C00457": {"lanes": 4, "range": 140},
    "C00460": {"lanes": 4, "range": 140},
    "C00461": {"lanes": 6, "range": 310},
    "C00463": {"lanes": 6, "range": 270},
    "C00464": {"lanes": 6, "range": 210},
    "C00465": {"lanes": 5, "range": 210},
    "C00466": {"lanes": 4, "range": 420},
    "C00467": {"lanes": 4, "range": 130},
    "C00468": {"lanes": 4, "range": 150},
    "C00469": {"lanes": 4, "range": 80},
    "C00470": {"lanes": 4, "range": 200},
    "C00471": {"lanes": 4, "range": 180},
    "C00472": {"lanes": 4, "range": 120},
    "C00475": {"lanes": 4, "range": 320},
    "C00477": {"lanes": 4, "range": 200},
    "C00479": {"lanes": 4, "range": 1100},
    "C00481": {"lanes": 4, "range": 160},
    "C00483": {"lanes": 4, "range": 180},
    "C00486": {"lanes": 4, "range": 120},
    "C00487": {"lanes": 4, "range": 270},
}

# =========================
# ユーティリティ
# =========================

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def calc_congestion(cam_id, count):
    info = LANE_INFO[cam_id]
    return (count / (info["lanes"] * (info["range"] / 1000))) / 20

# =========================
# ダウンロード
# =========================

def download_image(cam_id, url):
    os.makedirs(os.path.join(INPUT_DIR, cam_id), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(INPUT_DIR, cam_id, f"{cam_id}_{ts}.jpg")

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

    return path

# =========================
# 検知
# =========================

def detect_vehicles(image_path):
    img = load_img(image_path)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    result = rcnn_detector(img)
    result = {k: v.numpy() for k, v in result.items()}

    scores = result["detection_scores"]
    classes = result["detection_class_entities"]

    count = 0
    for s, c in zip(scores, classes):
        if s >= 0.1 and c.decode("ascii") in TARGET_LABELS:
            count += 1

    return count

# =========================
# 時刻待機
# =========================

def wait_until_next():
    now = datetime.now()
    next_min = ((now.minute // 10) + 1) * 10
    if next_min >= 60:
        target = (now + timedelta(hours=1)).replace(minute=0, second=0)
    else:
        target = now.replace(minute=next_min, second=0)
    time.sleep((target - now).total_seconds())

# =========================
# メイン
# =========================

def main():
    times = []
    congestion_history = {cid: [] for cid in CAMERA_INFO.keys()}

    while True:
        now = datetime.now()
        if now.minute in TARGET_MINUTES and now.second < 5:

            time_label = now.strftime("%Y/%m/%d %H:%M")
            times.append(time_label)

            for cam_id, url in CAMERA_URLS.items():
                try:
                    img_path = download_image(cam_id, url)
                    count = detect_vehicles(img_path)
                    congestion = calc_congestion(cam_id, count)
                except Exception as e:
                    print(cam_id, e)
                    congestion = None

                congestion_history[cam_id].append(congestion)

            # JSON 出力
            cameras_out = []
            for cam_id, info in CAMERA_INFO.items():
                cameras_out.append({
                    "id": cam_id,
                    "name": info["name"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "congestion": congestion_history[cam_id]
                })

            output = {
                "times": times,
                "cameras": cameras_out
            }

            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"Saved {time_label}")
            wait_until_next()
        else:
            wait_until_next()

if __name__ == "__main__":
    main()
