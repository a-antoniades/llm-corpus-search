from datasets import load_dataset
CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
import os
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Load the c4 dataset
dataset = load_dataset("c4", "en", cache_dir=CACHE_DIR)