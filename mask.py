from PIL import Image, ImageDraw
import glob
import os

# 入力フォルダと出力フォルダ
input_dir = "./input/20251113/C00469"
output_dir = "./input/20251113/C00469_masked"
os.makedirs(output_dir, exist_ok=True)

# 長方形のサイズ（px）
rect_width = 700   # 横幅
rect_height = 750  # 高さ

for path in glob.glob(input_dir + "/*.jpg"):
    img = Image.open(path)
    w, h = img.size

    draw = ImageDraw.Draw(img)

    # 左下の長方形の位置
    x1 = 0
    y1 = h - rect_height
    x2 = rect_width
    y2 = h

    draw.rectangle([x1, y1, x2, y2], fill="black")

    filename = os.path.basename(path)
    img.save(os.path.join(output_dir, filename))

print("完了しました。")
