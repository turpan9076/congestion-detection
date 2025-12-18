import os
import glob
from xml.etree.ElementTree import Element, SubElement, ElementTree
from PIL import Image

# ==== 設定 ====
DATASET = "val"
# YOLO形式アノテーションのディレクトリ
YOLO_LABEL_DIR = f"./dataset/{DATASET}/labels"     # ← trainまたはvalに変更
# 対応する画像のディレクトリ
IMAGE_DIR = f"./dataset/{DATASET}/images"          # ← trainまたはvalに変更
# 出力先ディレクトリ
OUTPUT_DIR = f"./dataset/{DATASET}/annotations"    # 保存先

# クラス名リスト（classes.txt と同じ順番で）
CLASSES = ["Car", "Bus", "Truck", "Van", "Motorcycle", "Vehicle"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 変換処理 ====
txt_files = glob.glob(os.path.join(YOLO_LABEL_DIR, "*.txt"))
print(f"Found {len(txt_files)} YOLO annotation files.")

for txt_path in txt_files:
    base = os.path.basename(txt_path)
    image_name = os.path.splitext(base)[0] + ".jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    if not os.path.exists(image_path):
        print(f"Warning: image not found for {image_name}, skipping.")
        continue

    # 画像サイズを取得
    with Image.open(image_path) as img:
        width, height = img.size

    # XMLルート
    annotation = Element("annotation")
    SubElement(annotation, "folder").text = os.path.basename(IMAGE_DIR)
    SubElement(annotation, "filename").text = image_name
    SubElement(annotation, "path").text = os.path.abspath(image_path)

    source = SubElement(annotation, "source")
    SubElement(source, "database").text = "Unknown"

    size = SubElement(annotation, "size")
    SubElement(size, "width").text = str(width)
    SubElement(size, "height").text = str(height)
    SubElement(size, "depth").text = "3"

    SubElement(annotation, "segmented").text = "0"

    # YOLOラベルを読み取り
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, w, h = map(float, parts)
        class_id = int(class_id)

        # 正規化座標 → ピクセル座標
        xmin = int((x_center - w / 2) * width)
        ymin = int((y_center - h / 2) * height)
        xmax = int((x_center + w / 2) * width)
        ymax = int((y_center + h / 2) * height)

        # 範囲補正
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width - 1, xmax)
        ymax = min(height - 1, ymax)

        # オブジェクト情報をXMLに追加
        obj = SubElement(annotation, "object")
        SubElement(obj, "name").text = CLASSES[class_id]
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"

        bndbox = SubElement(obj, "bndbox")
        SubElement(bndbox, "xmin").text = str(xmin)
        SubElement(bndbox, "ymin").text = str(ymin)
        SubElement(bndbox, "xmax").text = str(xmax)
        SubElement(bndbox, "ymax").text = str(ymax)

    # XMLを保存
    xml_path = os.path.join(OUTPUT_DIR, os.path.splitext(base)[0] + ".xml")
    ElementTree(annotation).write(xml_path, encoding="utf-8")

    print(f"Converted: {base} → {os.path.basename(xml_path)}")

print("Completion of YOLO to VOC conversion.")
