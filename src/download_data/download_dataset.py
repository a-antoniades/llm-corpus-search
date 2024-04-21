from datasets import load_dataset
import os
from argparse import ArgumentParser

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"

arg_parser = ArgumentParser()
arg_parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
arg_parser.add_argument("--dataset", type=str, default="c4")
arg_parser.add_argument("--extra_args", type=str, default="")
args = arg_parser.parse_args()

# Split the extra_args string into a list
extra_args = args.extra_args.split()

# Load the dataset
# dataset = load_dataset(args.dataset, *extra_args, cache_dir=args.cache_dir)

# dataset = load_dataset("ccaligned_multilingual", language_code="fr_XX", type="sentences", cache_dir=args.cache_dir)
dataset = load_dataset(args.dataset, *extra_args, cache_dir=args.cache_dir)