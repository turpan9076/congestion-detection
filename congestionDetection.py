import os
import glob
import time
from datetime import datetime
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

# 定義
output_dir_name = "output_with_line/20251113"

# モデル読み込み
rcnn_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
rcnn_detector = hub.load(rcnn_handle).signatures['default']
ssd_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
ssd_detector = hub.load(ssd_handle).signatures['default']

def load_img(path):
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

def merge_overlapping_detections(boxes, scores, class_names,target_labels, iou_threshold=0.5, min_score=0.1):
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
        A = (b1[2]-b1[0])*(b1[3]-b1[1])
        B = (b2[2]-b2[0])*(b2[3]-b2[1])
        if A == 0 or B == 0:
            return False
        r = min(A, B) / max(A, B)
        return r >= ratio

    def is_duplicate(b1, b2):
        # IoU が threshold を超える
        if iou_local(b1, b2) > iou_threshold:
            return True

        # 完全包含
        if is_contained(b1, b2) or is_contained(b2, b1):
            return True

        # 中心が非常に近い & サイズが近い
        if center_distance(b1, b2) < 0.05:
            if similar_size(b1, b2, ratio=0.35):
                return True

        return False

    N = boxes.shape[0]
    labels = [cn.decode("ascii") for cn in class_names]

    candidate_idx = [
        i for i in range(N) 
        if labels[i] in target_labels and scores[i] >= min_score
    ]

    if len(candidate_idx) == 0:
        return (np.zeros((0,4), dtype=boxes.dtype), 
                np.zeros((0,), dtype=scores.dtype), 
                np.array([], dtype=class_names.dtype))

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

        # グループの中でスコア最大のものを採用
        best_local = max(comp, key=lambda idx: cand_scores[idx])
        best_global = candidate_idx[best_local]
        merged_indices.append(best_global)

    merged_indices_sorted = sorted(merged_indices, key=lambda i: scores[i], reverse=True)

    boxes_final = boxes[np.array(merged_indices_sorted)]
    scores_final = scores[np.array(merged_indices_sorted)]
    class_names_final = class_names[np.array(merged_indices_sorted)]

    print(f"[merge] original: {len(candidate_idx)}, after merge: {len(merged_indices)}")

    return boxes_final, scores_final, class_names_final

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=6, display_str=""):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    if display_str:
        try:
            bbox = font.getbbox(display_str)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(display_str)

        margin = 5
        label_top = top - text_height - 2*margin
        if label_top < 0:
            label_top = bottom + margin
        draw.rectangle([(left, label_top), (left + text_width, label_top + text_height + 2*margin)], fill=color)
        draw.text((left + margin, label_top + margin), display_str, fill="black", font=font)

def draw_boxes(image, boxes, class_names, scores, max_boxes=200, min_score=0.1):
    colors = [
        'red',
        'lime',
        'blue',
        'yellow',
        'cyan',
        'magenta',
        'orange',
        'deeppink',
        'springgreen',
        'gold',
        'hotpink',
        'dodgerblue',
        'orangered',
        'greenyellow',
        'aqua',
        'fuchsia',
        'coral',
        'lightgreen',
        'royalblue',
        'tomato'
    ]
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
    count = min(boxes.shape[0], max_boxes)
    for i in range(count):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str=display_str)
    return np.array(image_pil)

def run_detector_rcnn_with_merge(path, location_id, model_name="Faster R-CNN"):
    img = load_img(path)
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = rcnn_detector(converted_img)
    end_time = time.time()

    result = {key: value.numpy() for key, value in result.items()}
    print(f"[{model_name}] Inference time: {end_time - start_time:.2f}s")

    boxes = result.get("detection_boxes", np.zeros((0,4)))
    scores = result.get("detection_scores", np.zeros((0,)))
    class_entities = result.get("detection_class_entities", np.array([], dtype=object))

    target_labels = ["Land vehicle", "Vehicle", "Car", "Bus", "Truck", "Van", "Motorcycle"]

    boxes_final, scores_final, class_names_final = merge_overlapping_detections(
        boxes, scores, class_entities, target_labels,
        iou_threshold=0.5, min_score=0.1
    )

    # カウント
    counts = {label: 0 for label in target_labels}
    total_count = 0
    for lbl in class_names_final:
        label = lbl.decode("ascii")
        if label in counts:
            counts[label] += 1
            total_count += 1

    print(f"\n[{model_name}] Detection counts after merge:")
    for label, cnt in counts.items():
        if cnt > 0:
            print(f"  {label}: {cnt}")
    print(f"  Total: {total_count}")

    # # 混雑判定
    # traffic_threshold = (71074 / (2*24)) / 70 * 0.27 
    # congestion_text = "Congested" if total_count > traffic_threshold else "Not congested"
    # print(f"[{model_name}] {congestion_text}")
    # congestion_text = "N/A"  # 混雑判定が無効化されている場合

    # 描画
    image_with_boxes = draw_boxes(img.numpy(), boxes_final, class_names_final, scores_final)
    image_pil = Image.fromarray(np.uint8(image_with_boxes)).convert("RGB")

    # # 混雑テキストを画像に追加（コメントアウト）
    # draw = ImageDraw.Draw(image_pil)
    # try:
    #     font = ImageFont.truetype("arial.ttf", 60)
    # except IOError:
    #     font = ImageFont.load_default()
    
    # try:
    #     bbox = draw.textbbox((0, 0), congestion_text, font=font)
    #     text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # except AttributeError:
    #     text_w, text_h = font.getsize(congestion_text)

    # margin = 10
    # draw.rectangle([margin, margin, margin+text_w+20, margin+text_h+20], fill="yellow")
    # draw.text((margin+10, margin+10), congestion_text, fill="black", font=font)

    # 保存
    basename = os.path.basename(path)
    output_dir = f"./{output_dir_name}/{location_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, basename)

    image_pil.save(output_path)
    print(f"[{model_name}] Saved: {output_path}\n")
    
    # カウント結果を返す
    return basename, counts, total_count, congestion_text

def parse_filename_datetime(filename):
    basename = os.path.basename(filename)
    parts = basename.split("_")
    if len(parts) < 3:
        return None
    
    date_part = parts[1]
    time_part = parts[2].split(".")[0]
    
    try:
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        return datetime(year, month, day, hour, minute)
    except (ValueError, IndexError):
        return None

def filter_files_by_datetime_range(files, start_datetime_str, end_datetime_str):
    try:
        start_year = int(start_datetime_str[:4])
        start_month = int(start_datetime_str[4:6])
        start_day = int(start_datetime_str[6:8])
        start_hour = int(start_datetime_str[9:11])
        start_minute = int(start_datetime_str[11:13])
        start_dt = datetime(start_year, start_month, start_day, start_hour, start_minute)
    except (ValueError, IndexError):
        print(f"Error: Invalid start datetime format: {start_datetime_str}")
        return []
    
    try:
        end_year = int(end_datetime_str[:4])
        end_month = int(end_datetime_str[4:6])
        end_day = int(end_datetime_str[6:8])
        end_hour = int(end_datetime_str[9:11])
        end_minute = int(end_datetime_str[11:13])
        end_dt = datetime(end_year, end_month, end_day, end_hour, end_minute)
    except (ValueError, IndexError):
        print(f"Error: Invalid end datetime format: {end_datetime_str}")
        return []
    
    filtered = []
    for f in files:
        file_dt = parse_filename_datetime(f)
        if file_dt and start_dt <= file_dt <= end_dt:
            filtered.append(f)
    
    return filtered

if __name__ == "__main__":
    # 処理したい地点IDのリスト (C00452からC00495まで)
    LOCATION_IDS = [f"C{i:05d}" for i in range(489, 496)]  # C00452 ~ C00495
    
    START_DATETIME = "20251113_0710"
    END_DATETIME = "20251114_0700"
    
    input_base_dir = "./input_with_line/20251113"
    
    # 全地点の結果を格納する辞書
    all_location_results = {}
    
    for location_id in LOCATION_IDS:
        print(f"\n{'='*80}")
        print(f"Processing Location: {location_id}")
        print(f"{'='*80}\n")
        
        # 各地点のファイルを取得
        input_files = sorted(glob.glob(f"{input_base_dir}/{location_id}/*.jpg"))
        filtered_files = filter_files_by_datetime_range(input_files, START_DATETIME, END_DATETIME)

        print(f"Found {len(filtered_files)} files to process for {location_id} (from {START_DATETIME} to {END_DATETIME})")
        
        if len(filtered_files) == 0:
            print(f"No files found for {location_id}. Skipping...\n")
            continue
        
        # 結果を格納するリスト
        results = []
        
        for f in filtered_files:
            print(f"Processing {f}")
            basename, counts, total_count, congestion_text = run_detector_rcnn_with_merge(f, location_id)
            
            # 結果を辞書に格納
            result_row = {
                'Location': location_id,
                'Filename': basename,
                'DateTime': parse_filename_datetime(f).strftime('%Y-%m-%d %H:%M') if parse_filename_datetime(f) else 'Unknown',
                **counts,  # ラベル別のカウントを展開
                'Total': total_count,
                'Status': congestion_text
            }
            results.append(result_row)
        
        # DataFrameを作成
        df = pd.DataFrame(results)
        
        # 列の並び順を指定
        target_labels = ["Land vehicle", "Vehicle", "Car", "Bus", "Truck", "Van", "Motorcycle"]
        column_order = ['Location', 'Filename', 'DateTime'] + target_labels + ['Total', 'Status']
        df = df[column_order]
        
        # 地点ごとのExcelファイルに保存
        output_excel_path = f"./{output_dir_name}/vehicle_counts_{location_id}.xlsx"
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"\nVehicle count results for {location_id} saved to: {output_excel_path}")
        
        # 全体の結果に追加
        all_location_results[location_id] = df
    
    # 全地点を統合したExcelファイルを作成
    if all_location_results:
        combined_df = pd.concat(all_location_results.values(), ignore_index=True)
        combined_output_path = "./vehicle_counts_all_locations.xlsx"
        combined_df.to_excel(combined_output_path, index=False, engine='openpyxl')
        print(f"\n{'='*80}")
        print(f"Combined results for all locations saved to: {combined_output_path}")
        print(f"{'='*80}\n")
    else:
        print("\nWARNING: No results to save. No files were processed.")
        print("Please check the debug information above.")