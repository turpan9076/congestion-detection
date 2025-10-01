import os
import glob
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def draw_horizontal_line(path, y_normalized=0.2373):
    """
    画像を読み込み、y=y_normalizedの位置に横線を引いて保存
    """
    # 画像読み込み
    img = load_img(path)
    image_pil = Image.fromarray(np.uint8(img.numpy())).convert("RGB")
    
    # 画像サイズ取得
    img_width, img_height = image_pil.size
    
    # y座標をピクセル値に変換
    y_line = int(y_normalized * img_height)
    
    # 横線を描画
    draw = ImageDraw.Draw(image_pil)
    draw.line([(0, y_line), (img_width, y_line)], fill="red", width=3)
    
    print(f"Horizontal line drawn at y={y_line} pixels (normalized: {y_normalized})")
    
    # 保存
    basename = os.path.basename(path)
    output_dir = "./output_line"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, basename)
    
    image_pil.save(output_path)
    print(f"Saved: {output_path}\n")

if __name__ == "__main__":
    # 入力ファイルを時刻でフィルタ（0512 ～ 0804）
    input_files = sorted(glob.glob("./input/C00452_20250918_*.jpg"))
    filtered_files = []
    for f in input_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) < 3:
            continue
        timestamp_part = parts[2].split(".")[0]
        try:
            hhmm = int(timestamp_part)
        except ValueError:
            continue
        if 512 <= hhmm <= 804:
            filtered_files.append(f)

    print(f"Found {len(filtered_files)} files to process")
    for f in filtered_files:
        print(f"Processing {f}")
        draw_horizontal_line(f)