import os
import pathlib
import subprocess

# models ディレクトリのチェックとクローン
if "models" in pathlib.Path.cwd().parts:
    while "models" in pathlib.Path.cwd().parts:
        os.chdir('..')
elif not pathlib.Path('models').exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tensorflow/models"], check=True)

# research ディレクトリへ移動
os.chdir("models/research")

# .proto ファイルのコンパイル
subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], shell=True, check=True)

# setup.py をコピーしてインストール
subprocess.run(["cp", "object_detection/packages/tf2/setup.py", "."], shell=True, check=True)
subprocess.run(["python", "-m", "pip", "install", "."], check=True)
