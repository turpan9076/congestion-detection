from pathlib import Path
from PIL import Image, ImageDraw

# 設定
for i in range(6):
    cam_id = ["C00489","C00490","C00491","C00492","C00493","C00495"]
    global_bottom_norm = [0.090462,0.106624,0.333397,0.182597,0.278652,0.253794]

    SRC_DIR = Path(f"./output/20260101/{cam_id[i]}")   # 元画像フォルダ
    DST_DIR = Path(f"./output_with_line/20260101/{cam_id[i]}") # 保存先フォルダ

    COLOR = (255, 0, 0)    # 赤
    WIDTH = 4              # 線の太さ

    DST_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in SRC_DIR.glob("*.jpg"):
        with Image.open(img_path) as img:
            w, h = img.size
            y_px = int(round(global_bottom_norm[i] * h))
            y_px = max(0, min(h - 1, y_px))

            draw = ImageDraw.Draw(img)
            draw.line([(0, y_px), (w, y_px)], fill=COLOR, width=WIDTH)

            dst_path = DST_DIR / img_path.name
            img.save(dst_path)
