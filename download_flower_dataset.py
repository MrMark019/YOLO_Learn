import os
from pathlib import Path

import kagglehub


repo_root = Path(__file__).resolve().parent
os.environ["KAGGLEHUB_CACHE"] = str(repo_root)

# Download Nature3: Leaf, Flower, and Fruit Detection dataset
path = kagglehub.dataset_download("se00n00/nature3-leaf-flower-and-fruit-detection")

print("Path to dataset files:", path)
