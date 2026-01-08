from PIL import Image, ImageDraw
import glob
import os

# 入力フォルダと出力フォルダ
input_dir = "./input/20260101/C00469_masked"
output_dir = "./input/20260101/C00469_masked"
os.makedirs(output_dir, exist_ok=True)

# 三角形の3点を指定
# 例: 左下を覆う三角形
triangle_points = [
    (1920, 600),   # 点1
    (1920, 290), # 点2
    (800, 290)     # 点3
]

for path in glob.glob(input_dir + "/*.jpg"):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)

    # 三角形を描画
    draw.polygon(triangle_points, fill="black")

    filename = os.path.basename(path)
    img.save(os.path.join(output_dir, filename))

print("完了しました。")
