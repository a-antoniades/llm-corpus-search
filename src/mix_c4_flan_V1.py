# %%
import os
import argparse
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import set_seed
# from promptsource import templates
import json
from collections import defaultdict
from utils import total_tokens

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

set_seed(420)


def filter_and_check(ds, tasks):
    ds1 = ds.filter(lambda example: any(task in example['task_name'] for task in tasks))

    print(f"Found {len(ds1)} rows for tasks {tasks}")

    return ds1

def concatenate_columns(dataset, column1, column2, new_column_name):
    def concat_example(example):
        example[new_column_name] = example[column1] + " " + example[column2]
        return example

    return dataset.map(concat_example)

def sample_dataset(dataset, num_samples):
    indices_to_remove = np.random.choice(len(dataset), len(dataset) - num_samples, replace=False)
    mask = np.ones(len(dataset), dtype=bool)
    mask[indices_to_remove] = False
    indices_to_keep = np.arange(len(dataset))[mask]
    return dataset.select(indices_to_keep)


class FLAN_V1_DATASET:

    def __init__(self):
        self.ds = load_dataset("conceptofmind/flan2021_submix_original", split="train")
        self.tasks = {
            "NLI": ["anli", "rte", "cb", "snli", "mnli", "wnli", "qnli"],
            "QA": ["arc", "nq", "triviaqa"],
            "Summarization": ["aelsc", "multi-news", "samsum", "ag_news",
                              "newsroom", "wiki_lingua", "cnn_dailymail",
                              "opinion_abstracts_idebate", "opinion_abstracts_rotten_tomatoes",
                              "gigaword", "xsum"],
            "Commonsense": ["copa", "hellaswag", "piqa", "story_cloze",
                            "cosmos_qa", "record"],
            
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a specific task.')
    parser.add_argument('--task', type=str, required=True, help='The task to process (e.g. "NLI").')
    args = parser.parse_args()

    TASK = args.task

    # Validate the task
    if TASK not in FLAN_V1_DATASET().tasks:
        raise ValueError(f"Task {TASK} is not recognized. Please choose from {list(FLAN_V1_DATASET().tasks.keys())}.")

    # %%
    task_ds = filter_and_check(FLAN_V1_DATASET().ds, FLAN_V1_DATASET().tasks[TASK])
    task_ds = concatenate_columns(task_ds, 'inputs', 'targets', 'text')
    task_tokens = total_tokens(task_ds, 'text')

    # %%
    ds_c4 = load_dataset("c4", "en", split="train", cache_dir=CACHE_DIR)
    ds_c4_small = sample_dataset(ds_c4, len(ds_c4) // 10)

    # %%
    n_samples_keep = len(ds_c4_small) - len(task_ds)
    ds_c4_sampled = sample_dataset(ds_c4_small, n_samples_keep)
    ds_c4_mixed = concatenate_datasets([ds_c4_sampled, task_ds])

    # %%
    base_path = os.path.join(CACHE_DIR, "flan_v1")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    ds_path = os.path.join(base_path, f"c4_mixed_{TASK}")

    ds_c4_mixed.save_to_disk(ds_path)
    # save num_tokens
    with open(os.path.join(base_path, f"task_tokens.txt"), "w") as f:
        f.write(str(task_tokens))

    print(f"Saved {TASK} mixed dataset to {os.path.join(base_path, f'c4_mixed_{TASK}')}")
    print(f"len dataset: {len(ds_c4_mixed)}")