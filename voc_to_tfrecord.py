import os
import glob
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# ==== 設定 ====
DATASET = "val"
IMAGE_DIR = f"./dataset/{DATASET}/images"          # 画像フォルダ
ANNOTATION_DIR = f"./dataset/{DATASET}/annotations" # XMLフォルダ
OUTPUT_PATH = f"./{DATASET}.record"                 # 出力TFRecord
CLASSES = ["Car", "Bus", "Truck", "Van", "Motorcycle", "Vehicle"]  # クラス名リスト

# ==== 補助関数 ====
def class_text_to_int(label):
    if label in CLASSES:
        return CLASSES.index(label) + 1  # idは1から
    else:
        return None

def create_tf_example(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(image_path):
        print(f"Warning: image not found: {image_path}")
        return None

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = tf.io.decode_jpeg(encoded_jpg)
    height = int(root.find("size/height").text)
    width = int(root.find("size/width").text)

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for member in root.findall("object"):
        class_name = member.find("name").text
        xmin = int(member.find("bndbox/xmin").text)
        ymin = int(member.find("bndbox/ymin").text)
        xmax = int(member.find("bndbox/xmax").text)
        ymax = int(member.find("bndbox/ymax").text)

        # 正規化座標に変換
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

        classes_text.append(class_name.encode("utf8"))
        classes.append(class_text_to_int(class_name))

    if not classes:
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example


# ==== 変換処理 ====
def main():
    xml_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))
    print(f"Found {len(xml_files)} XML annotation files.")

    with tf.io.TFRecordWriter(OUTPUT_PATH) as writer:
        for xml_file in tqdm(xml_files):
            tf_example = create_tf_example(xml_file)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    print(f"TFRecord saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
