import kagglehub
import os

# 设置下载路径为当前目录
current_dir = r"d:\MarkLab\YOLO_Learn"
os.environ['KAGGLEHUB_CACHE'] = current_dir

# Download Nature3: Leaf, Flower, and Fruit Detection dataset
path = kagglehub.dataset_download("se00n00/nature3-leaf-flower-and-fruit-detection")

print("Path to dataset files:", path)
