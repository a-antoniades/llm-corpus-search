# %%
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
# from promptsource import templates
import json
import pickle
import collections
from itertools import combinations
import random
from tqdm import tqdm
import argparse

# from incidental-supervision.src.utils import concatenate_columns, count_gpt2_tokens

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
import os
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
DEVICE_ID = 3

import mauve

# # Get a list of all supported datasets
# datasets = templates.get_dataset_names()
# print(datasets)


# %%
max_len = 1000
n_groups = 10

mauve_scaling_factor = 2
## pre-sampling data

from datasets import DatasetDict

# Define the file paths
task_samples_path = f"/share/edc/home/antonis/datasets/huggingface/flan_v1/task_ds_sampled_{max_len}_{n_groups}.pkl"
dataset_dict_path = f"/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small_sampled_{max_len}_{n_groups}"

# Check if the task_samples file exists
if not os.path.exists(task_samples_path):
    ds_task = load_from_disk('/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds')
    task_samples_dict = collections.defaultdict(dict)
    # Pre-sample the ds_task['text']
    for task in tqdm(ds_task.keys(), desc="Tasks"):
        for n in tqdm(range(n_groups), desc="Groups"):
            task_samples_dict[task][str(n)] = random.sample(ds_task[task]['text'], max_len)

    # save 
    with open(task_samples_path, 'wb') as f:
        pickle.dump(dict(task_samples_dict), f)
else:
    # load
    with open(task_samples_path, 'rb') as f:
        task_samples_dict = collections.defaultdict(dict, pickle.load(f))

tasks = ['NLI', 'QA', 'Summarization', 'Commonsense']
# tasks = list(task_samples_dict.keys())

# Check if the dataset_dict file exists
if not os.path.exists(dataset_dict_path):
    # Pre-sample the ds['text']
    ds = load_from_disk("/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small")
    dataset_dict = {str(n): ds.select(indices=random.sample(range(len(ds)), max_len)) for n in tqdm(range(n_groups), desc="Pre-sampling")}

    # Convert to DatasetDict
    dataset_dict = DatasetDict(dataset_dict)

    # Save the dataset_dict
    dataset_dict.save_to_disk(dataset_dict_path)
else:
    dataset_dict = load_from_disk(dataset_dict_path)



# # Create the parser
parser = argparse.ArgumentParser(description='Compute Mauve results')
parser.add_argument('--mode', type=str, choices=['results', 'results_tasks', 'results_corpus', 'all'], required=True, help='Mode to run the script in')

# Parse the arguments
args = parser.parse_args()

# args = argparse.Namespace()
# args.mode = 'results_tasks'

print(f"--- RUNNING IN MODE: {args.mode} ---")

# %%
# Compute results or results_tasks based on the mode
if args.mode == 'results' or args.mode == 'all':
    print("Computing results")
    if os.path.exists('mauve_results.pkl'):
        with open('mauve_results.pkl', 'rb') as f:
            results = collections.defaultdict(dict, pickle.load(f))
    else:
        results = collections.defaultdict(dict)
    for task in tqdm(tasks, desc=f"Tasks"):
        for n in tqdm(range(n_groups), desc=f"{task}_Groups"):
            if int(n) in results[task]:
                continue
            text_1 = dataset_dict[str(n)]['text']
            text_2 = task_samples_dict[task][str(n)]
            task_samples_dict[task][n] = text_2
            print(f"Computing Mauve for {task} {n}")
            out =  mauve.compute_mauve(p_text=text_1, q_text=text_2, 
                            device_id=DEVICE_ID, max_text_length=512,
                            batch_size=1, verbose=True, 
                            featurize_model_name='gpt2-large',
                            mauve_scaling_factor=mauve_scaling_factor)
            results[task][n] = out.__dict__
            with open('mauve_results.pkl', 'wb') as f:
                pickle.dump(results, f)

    # save 
    with open('mauve_results.pkl', 'wb') as f:
        pickle.dump(results, f)

if args.mode == 'results_tasks' or args.mode == 'all':
    print("Computing results_tasks")
    if os.path.exists('mauve_results_tasks.pkl'):
        with open('mauve_results_tasks.pkl', 'rb') as f:
            results_tasks = collections.defaultdict(lambda: collections.defaultdict(dict), pickle.load(f))
    else:
        results_tasks = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Get all pairs of tasks
    # tasks = list(task_samples_dict.keys())
    task_pairs = list(combinations(tasks, 2))

    for task1, task2 in tqdm(task_pairs, desc="Task pairs"):
        for n in tqdm(range(n_groups), desc=f"{task1}_{task2}_Groups"):
            if int(n) in results_tasks[task1][task2]:
                print(f"Skipping {task1} vs {task2}: {n}")
                continue
            text_1 = task_samples_dict[task1][str(n)]
            text_2 = task_samples_dict[task2][str(n)]
            print(f"Computing Mauve for {task1} vs {task2}: {n}")
            out = mauve.compute_mauve(p_text=text_1, q_text=text_2, 
                            device_id=DEVICE_ID, max_text_length=512,
                            batch_size=1, verbose=True,
                            featurize_model_name='gpt2-large',
                            mauve_scaling_factor=mauve_scaling_factor)
            results_tasks[task1][task2][n] = out.__dict__
            # save 
            print(f"Saving {task1} vs {task2}: {n}")
            with open('mauve_results_tasks.pkl', 'wb') as f:
                pickle.dump(dict(results_tasks), f)

    # save 
    with open('mauve_results_tasks.pkl', 'wb') as f:
        pickle.dump(dict(results_tasks), f)

if args.mode == 'results_corpus' or args.mode == 'all':
    print("Computing results_corpus")
    if os.path.exists('mauve_results_corpus.pkl'):
        with open('mauve_results_corpus.pkl', 'rb') as f:
            results_tasks = collections.defaultdict(lambda: collections.defaultdict(dict), pickle.load(f))
    else:
        results_tasks = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Get all pairs of tasks
    tasks = list(dataset_dict.keys())
    task_pairs = list(combinations(dataset_dict, 2))

    print(dataset_dict.keys())

    for task1, task2 in tqdm(task_pairs, desc="Task pairs"):
        text_1 = dataset_dict[task1]['text']
        text_2 = dataset_dict[task2]['text']
        print(f"Computing Mauve for {task1} vs {task2}")
        out = mauve.compute_mauve(p_text=text_1, q_text=text_2, 
                        device_id=DEVICE_ID, max_text_length=512,
                        batch_size=1, verbose=True,
                        featurize_model_name='gpt2-large',
                        mauve_scaling_factor=mauve_scaling_factor)
        results_tasks[task1][task2] = out.__dict__
        # save 
        print(f"Saving {task1} vs {task2}")
        with open('mauve_results_corpus.pkl', 'wb') as f:
            pickle.dump(dict(results_tasks), f)

    # save
    with open('mauve_results_corpus.pkl', 'wb') as f:
        pickle.dump(dict(results_tasks), f)