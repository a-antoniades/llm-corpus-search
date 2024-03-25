# %%
import os
import argparse
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import set_seed
# from promptsource import templates
import json
from collections import defaultdict
from utils import total_tokens, limit_total_tokens
from itertools import chain

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
    return dataset.select(indices_to_keep), indices_to_keep


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
            "Sentiment": ["imdb", "sst2", "sentiment140", "yelp"],
            "Paraphrase": ["qqp", "mrpc", "paws", "stsb"],
            "Reading Comp.": ["bool_q", "openbookqa", "drop",
                              "squad", "multirc"],
            "Reading Comp. w/ Commonsense": ["cosmos_qa", "record"],
            "Coreference": ["definite_pronoun_resolution", "winogrande",
                            "wsc"],
            "Misc": ["coqa", "trec", "quac", "cola",
                     "wic", "fix_punct"],
            "Math": ["math_dataset"]
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a specific task.')
    parser.add_argument('--task', type=str, help='The task to process (e.g. "NLI").')
    parser.add_argument('--corpus_only', action='store_true', help='Whether to only use corpus dataset.')

    args = parser.parse_args()

    # TASK = args.task


    # # Validate the task
    # if TASK not in FLAN_V1_DATASET().tasks:
    #     raise ValueError(f"Task {TASK} is not recognized. Please choose from {list(FLAN_V1_DATASET().tasks.keys())}.")

    corpus = "EleutherAI/the_pile_deduplicated"
    # %%
    if args.corpus_only:
        corpus_path = f"/share/edc/home/antonis/datasets/huggingface/{corpus}/ds_{corpus}_small"
        if not os.path.exists(corpus_path):
            ds_corpus = load_dataset(corpus, split="train", cache_dir=CACHE_DIR, verification_mode='no_checks')
            print(f"doing it")
            ds_corpus_small, corpus_indexes = limit_total_tokens(ds_corpus, 50e09, pretokenized=False)
            ds_corpus_small.save_to_disk(corpus_path)
            np.save(os.path.join(corpus_path, "corpus_indexes.npy"), corpus_indexes)
            print(f"Saved {corpus} small dataset to {corpus_path}")
        exit()
    else:
        ds_corpus_small = load_from_disk(f"/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_{corpus}_small")

    # %%
    # TASK = ["QA", "NLI", "Summarization", "Commonsense"]
    if args.task == "all":
        TASK = ["QA", "NLI", "Summarization", "Commonsense"]
    else:
        TASK = [args.task]
    task_ds = filter_and_check(FLAN_V1_DATASET().ds, list(chain.from_iterable(FLAN_V1_DATASET().tasks[task] for task in TASK)))
    task_ds = concatenate_columns(task_ds, 'inputs', 'targets', 'text')
    task_tokens = total_tokens(task_ds, 'text')

    # %%
    n_samples_keep = len(ds_corpus_small) - len(task_ds)
    ds_corpus_sampled, corpus_indexes = sample_dataset(ds_corpus_small, n_samples_keep)
    ds_corpus_mixed = concatenate_datasets([ds_corpus_sampled, task_ds])

    # %%
    base_path = os.path.join(CACHE_DIR, "flan_v1")
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    ds_path = os.path.join(base_path, f"{corpus}_mixed_{TASK}")

    ds_corpus_mixed.save_to_disk(ds_path)
    # save num_tokens
    with open(os.path.join(ds_path, f"task_tokens.txt"), "w") as f:
        f.write(str(task_tokens))

    print(f"Saved {TASK} mixed dataset to {os.path.join(ds_path, f'{corpus}_mixed_{TASK}')}")
    print(f"len dataset: {len(ds_corpus_mixed)}")