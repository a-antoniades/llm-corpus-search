# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"

acess_token = "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"

import os
from argparse import ArgumentParser
# os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

arg_parser = ArgumentParser()
arg_parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
arg_parser.add_argument("--model", type=str, default="c4")
args = arg_parser.parse_args()

# load model
tokenizer = AutoTokenizer.from_pretrained(args.model, 
                                          use_auth_token=acess_token,
                                          cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(args.model, 
                                             use_auth_token=acess_token,
                                             cache_dir=CACHE_DIR)