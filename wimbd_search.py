# %%
import sys
sys.path.append('/share/edc/home/antonis/LLM-Incidental-Supervision/wimbd')
import os
import pickle
from functools import partial
import time
from tqdm import tqdm
import collections
import numpy as np
from distutils.util import strtobool
 
from elasticsearch import Elasticsearch
from wimbd.es import get_documents_containing_phrases
from src.utils import running_jupyter

from transformers import set_seed
import datasets
set_seed(420)

import argparse
from src.wimbd_ import WimbdTasks, DataConfigs, _load_dataset, clean_text

from wimbd_preprocess import parse_args as preprocess_parse_args, main as preprocess_main
from wimbd_process import parse_args as process_parse_args, main as process_main


import logging
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.ERROR)

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"

# docs_v1.5_2023-11-02


def get_nested_value(dct, key_path):
    print(f"dct: {dct}, key_path: {key_path}")
    if isinstance(key_path, str):
        return dct[key_path]
    elif isinstance(key_path, list):
        for key in key_path:
            print(f"key: {key}, dct: {dct}")
            dct = dct[key]
    else:
        raise ValueError(f"key_path must be of type str or list, not {type(key_path)}")
    return dct

def run_preprocess(args, method, save_path):
    preprocess_args = preprocess_parse_args(args=[])
    preprocess_args.n_grams = args.n_grams
    preprocess_args.base_dir = save_path
    preprocess_args.task = args.dataset
    preprocess_args.method = method
    return preprocess_main(preprocess_args)

def run_process(args, method, preprocessed_data, save_path):
    process_args = process_parse_args(args=[])
    process_args.base_dir = save_path
    process_args.dataset = args.dataset
    process_args.method = method
    process_args.n_grams = args.n_grams

    if method == "all":
        preprocessed_file_path = preprocessed_data["dfs_all"]["path"]
    elif method == "common":
        preprocessed_file_path = preprocessed_data["dfs_common_filtered"]["path"]
    else:
        print(f"Invalid method: {method}")
        return

    if os.path.exists(preprocessed_file_path):
        process_args.filename = os.path.basename(preprocessed_file_path)
    else:
        print(f"Preprocessed file not found: {preprocessed_file_path}")
        return

    process_main(process_args)

def parse_args():
    if running_jupyter():
        # declare namespace
        args = argparse.Namespace()
        # add arguments
        args.method = "common"
        args.n_grams = 2
        args.language_pair = ("cs", "en")  # default language pair
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, default=None, help="Name of the experiment")
        parser.add_argument("--type", type=str, default="wimbd", help="Type of the experiment")
        parser.add_argument("--corpus", type=str, default="re_pile", help="Dataset to use")
        parser.add_argument("--method", type=str, default=None, help="Method to use for the query")
        parser.add_argument("--get_docs", type=lambda x: bool(strtobool(x)), default=False, help="Explicitly enable or disable getting the documents")
        parser.add_argument("--n_grams", type=int, default=5, help="Number of n-grams to use")
        parser.add_argument("--language_pair", type=str, nargs=2, default=("en", "en"), help="Language pair to use")
        parser.add_argument("--dataset", type=str, default="mmlu", help="Dataset to use")
        parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Tasks to use")
        parser.add_argument("--debug", action="store_true", help="Print debug information")
        parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use")
        parser.add_argument("--no_split_text2", type=lambda x: bool(strtobool(x)), default=False, help="Explicitly enable or disable splitting of text2")
        parser.add_argument("--filter_keywords", type=lambda x: bool(strtobool(x)), default=False, help="Explicitly enable or disable filtering of keywords")
        parser.add_argument("--filter_stopwords", type=lambda x: bool(strtobool(x)), default=None, help="Explicitly enable or disable filtering of stopwords")
        parser.add_argument("--filter_punc", type=lambda x: bool(strtobool(x)), default=True)
        parser.add_argument("--align_pairs", type=lambda x: bool(strtobool(x)), default=True)
        parser.add_argument("--replace_keywords", type=lambda x: bool(strtobool(x)), default=False, help="Explicitly enable or disable filtering of kewywords")
        parser.add_argument("--only_alpha", type=lambda x: bool(strtobool(x)), default=False, help="Explicitly enable or disable filtering of non-alphabetic characters")
        parser.add_argument("--delimeter", type=str, default=" ", help="Delimeter to use for splitting text")
        parser.add_argument("--cont_from", type=str, default=None, help="Continue from a specific task")
        args = parser.parse_args()
    return args

def main(args):

    # SETUP PARAMS ACCORDING TO NGRAMS
    if args.filter_stopwords is None:
        if args.n_grams <= 3:
            args.filter_stopwords = True
        else:
            args.filter_stopwords = False

    base = f"./results/n-grams/{args.dataset}/{args.corpus}"
    if args.name is None:
        base_path = os.path.join(base, f"./exp_3/test-set")
    else:
        base_path = os.path.join(base, args.name)
    if args.tasks is not None:
        base_path = os.path.join(base_path, *args.tasks)
    settings = f"fkey{args.filter_keywords}_rkey{args.replace_keywords}_fstop{args.filter_stopwords}_onlyalpha{args.only_alpha}"
    save_path = f"{base_path}/n_samples_{str(args.n_samples)}_{settings}/{str(args.n_grams)}/{args.method}"
    if not os.path.exists(save_path) and not args.debug:
        os.makedirs(save_path)

    if args.type == 'wimbd':
        from wimbd.es import count_documents_containing_phrases
        if args.corpus == 'pile':
            index = 're_pile'
            cloud_id = "m-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl"
            api_key = "RlZBbHpZc0J1MEw4LVVWVk9SaTE6bXJlSUM2QnlSQmFHemhwVElVUnZyQQ=="
        elif args.corpus == 'dolma':
            index = "docs_v1.5_2023-11-02"
            cloud_id = "dolma-v15:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ1MjQyM2ZiNjk0NGE0YzdkOGQ5N2Y3NDM2MmMzODY3ZSQxMDNiM2ZkYTUwYzk0MTNmYmUwODA1ZDMyNjQ5YTliNQ=="
            api_key = "QTJiajFJMEIxR1JtTm13YUZBVGc6dEpudXhEd19SRzJUOVZNYUpDdlItdw=="
        else:
            raise ValueError(f"Method {args.corpus} not recognized")
        es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            retry_on_timeout=True,
            http_compress=True)
    elif args.type == 'infini':
        from src.infini_search import count_documents_containing_phrases
        if args.corpus == 'dolma':
            index = "v4_dolma-v1_6_llama"
            es = None
        elif args.corpus == 'pile':
            index = "v4_piletrain_llama"
            es = None
    elif args.type == 'rust':
        from src.rust_search import count_documents_containing_phrases
        if args.corpus == 'tuluv2':
            index = 'tuluv2'
            es = None
    else:
        raise ValueError(f"Method {args.type} not recognized")

    wt = WimbdTasks(align_pairs=args.align_pairs, 
                    language_pair=args.language_pair)
    print(f"type: {args.type}, index: {index}, corpus: {args.corpus}")

    print(f"tasks: {args.tasks}")
    print(f"dataset: {args.dataset}")
    args.language_pair = tuple(args.language_pair)
    args.tasks = args.tasks
    print(f"language_pair: {args.language_pair}")
    ds = _load_dataset(
        args.dataset, 
        tasks=args.tasks,
        languages=[args.language_pair],

    )
    print(f"dataset keys: {ds.keys()}")

    total_tasks = len(ds)
    completed_tasks = 0
    if not isinstance(ds, dict or not isinstance(ds, datasets.DatasetDict)):
        ds = {args.language_pair: ds}

    for task_name, task_ds in tqdm(ds.items(), desc="Processing tasks", total=total_tasks):
        
        # continue from a specific task
        if args.cont_from is not None:
            if task_name != args.cont_from:
                print(f"Skipping task {task_name}")
                continue
            else:
                args.cont_from = None

        completed_tasks += 1
        if not args.debug:
            filename = os.path.join(save_path, f"{task_name}.pkl")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # if os.path.exists(filename):
            #     raise ValueError(f"File {filename} already exists")
        n_gram_freqs = collections.defaultdict(lambda: {'value': 0, 
                                                        'example': None,
                                                        'task': task_name})

        # check if split
        if 'split' in DataConfigs.data_keys[args.dataset].keys():
            ds_split = DataConfigs.data_keys[args.dataset]['split']
            print(f"Splitting task {task_name}, using split {ds_split}")
            task_ds = task_ds[ds_split]

        if args.n_samples is not None:
            if len(task_ds) < args.n_samples:
                print(f"Skipping task {task_name} because it has less than {args.n_samples} samples")
                continue
            # indexes =  np.random.choice(len(task_ds), args.n_samples, replace=False)
            indexes = np.arange(args.n_samples)
            task_ds_full = task_ds.select(indexes)
            # save indexes
            if not args.debug:
                with open(os.path.join(save_path, f"indexes_{task_name}.npy"), 'wb') as f:
                    np.save(f, indexes)
        else:
            task_ds_full = task_ds
        
        for idx, example in tqdm(enumerate(task_ds_full), desc=f"Processing examples for task {task_name}"):
            print(f"Task {task_name}, ({completed_tasks}/{total_tasks}")
            print(example)
            start_time = time.time()

            dataset_keys = DataConfigs.data_keys[args.dataset]
            answer_key = dataset_keys.get('answer', None)

            if args.dataset in ["wmt", "europarl"]:
                text_1_key = ['translation', args.language_pair[0]]
                text_2_key = ['translation', args.language_pair[1]]
            elif args.dataset in ["wmt09_gens"]:
                text_1_key = ['translation', args.language_pair[0]]
                text_2_key = dataset_keys['text_2']
            else:
                text_1_key, text_2_key = dataset_keys['text_1'], dataset_keys['text_2']

            if answer_key is not None:
                answer = get_nested_value(example, answer_key)
                print(f"Answer: {answer}")
            
            print(f"example: {example}")
            text_1 = get_nested_value(example, text_1_key)
            text_2 = get_nested_value(example, text_2_key)

            # clean text
            clean = partial(clean_text, 
                            dataset=args.dataset, 
                            filter=args.filter_keywords, 
                            replace=args.replace_keywords)
            
            text_1_ori, text_2_ori = text_1, text_2
            text_1, text_2 = clean(text_1), clean(text_2, answer_idx=answer)
            print(f"Text 1: {text_1_ori} -> {text_1}")
            print(f"Text 2: {text_2_ori} -> {text_2}")
            
            example_counts = 0
            n_coverage = 0
            coverage = []
            p_coverage = []
            if args.method == "common":

                ngram_combinations = wt.get_combinations(text_1, text_2, 
                                                        args.language_pair, args.n_grams,
                                                        no_split_text2=args.no_split_text2,
                                                        filter_stopwords=args.filter_stopwords,
                                                        filter_punc=args.filter_punc,
                                                        only_alpha=args.only_alpha,
                                                        align_pairs=args.align_pairs)
                
                if args.debug:
                    print(f"Question: {text_1}, Answer: {text_2}")
                    print(f"ngram_combinations: {ngram_combinations}")
                for n_gram in ngram_combinations:
                    success = False
                    while not success:
                        try:
                            counts = count_documents_containing_phrases(index, n_gram,
                                                                        es=es, all_phrases=True)
                            if args.get_docs:
                                docs = get_documents_containing_phrases(index, n_gram,
                                                                        es=es, all_phrases=True,
                                                                        return_all_hits=True)
                            success = True  # If the function call was successful, exit the loop
                        except Exception as e:
                            print(f"An error occurred: {e}. Retrying...")
                            time.sleep(1)
                    example_counts += counts
                    
                    if args.debug:
                        print(f"Processing ngram combination {n_gram}")
                        print(f"counts: {counts}")
                        if args.get_docs:
                                print(f"docs: {[doc['_id'] for doc in docs]}")

                    n_gram_freqs[n_gram]['value'] += counts
                    n_gram_freqs[n_gram]['pair'] = n_gram
                    n_gram_freqs[n_gram]['example'] = example
                    n_gram_freqs[n_gram]['example_clean'] = text_1
                    
                    if args.get_docs:
                        doc_ids = [doc['_id'] for doc in docs]
                        source_ds = [doc['_source']['meta']['pile_set_name'] for doc in docs]
                        n_gram_freqs[n_gram]['docs'] = doc_ids
                        n_gram_freqs[n_gram]['set_name'] = source_ds

                    if counts > 0:
                        n_coverage += 1
                    
                    if args.debug:
                        print(f"ngrams: {n_gram}")
                        input("Press Enter to continue...")
            
            elif args.method == "all":
                text_1 = str(text_1) + args.delimeter + str(text_2)     # concatenate text_1 and text_2
                ngram_combinations = wt.process_text(text_1, wt.get_language_name(args.language_pair),
                                                     n_gram=args.n_grams,
                                                     filter_stopwords=args.filter_stopwords,
                                                     filter_punc=args.filter_punc,
                                                     only_alpha=args.only_alpha)
                print(f"ngram_combinations: {ngram_combinations}")
                for n_gram in ngram_combinations:
                    success = False
                    while not success:
                        try:
                            counts = count_documents_containing_phrases(index, (n_gram),
                                                                        es=es, all_phrases=True)
                            if args.get_docs:
                                docs = get_documents_containing_phrases(index, (n_gram),
                                    es=es, all_phrases=True,
                                    return_all_hits=True)
                            success = True  # If the function call was successful, exit the loop
                        except Exception as e:
                            print(f"An error occurred: {e}. Retrying...")
                            time.sleep(1)
                    example_counts += counts
                    
                    if args.debug:
                        print(f"example: {example}")
                        print(f"ngram: {n_gram}")
                        print(f"counts: {counts}")
                        if args.get_docs:
                            print(f"docs: {[doc['_id'] for doc in docs]}")
                            
                    n_gram_freqs[n_gram]['value'] += counts
                    n_gram_freqs[n_gram]['example'] = example
                    n_gram_freqs[n_gram]['example_clean'] = text_1
                    n_gram_freqs[n_gram]['sequence'] = n_gram
                    
                    if args.get_docs:
                        doc_ids = [doc['_id'] for doc in docs]
                        source_ds = [doc['_source']['meta']['pile_set_name'] for doc in docs]
                        n_gram_freqs[n_gram]['docs'] = doc_ids
                        n_gram_freqs[n_gram]['set_name'] = source_ds

                    if counts > 0:
                        n_coverage += 1
                    
                    if args.debug:
                        print(f"ngrams: {n_gram}")
                        input("Press Enter to continue...")
            
            # record wether any ngrams where found for this example
            example_dict = {
            'task': task_name,
            'example': example,
            'coverage': 1 if example_counts > 0 else 0
            }

            coverage.append(example_dict)

            # calculate p_coverage (proportion of ngrams found)
            p_coverage.append(n_coverage / len(ngram_combinations) if len(ngram_combinations) > 0 else 0)
            instances_left = len(task_ds_full) - idx
            remaining_time = instances_left * (time.time() - start_time)
            # Convert time to days/hours/minutes
            days, remainder = divmod(remaining_time, 24 * 3600)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)
            print(f"completed instance {idx}, remaining: {int(days)}d {int(hours)}h {int(minutes)}m")
            
            # save as .pkl
            if not args.debug:
                with open(filename, 'wb') as f:
                    pickle.dump(dict(n_gram_freqs), f)
                with open(os.path.join(save_path, f"task-coverage.pkl"), 'wb') as f:
                    pickle.dump(coverage, f)
                with open(os.path.join(save_path, f"task-p-coverage.pkl"), 'wb') as f:
                    pickle.dump(p_coverage, f)
        
    # if everything runs smoothly, save "completed flag"
    if not args.debug:
        with open(os.path.join(save_path, "completed.txt"), 'w') as f:  # Open in text mode
            f.write("completed")

    if not args.debug:
        with open(os.path.join(save_path, "completed.txt"), 'w') as f:  # Open in text mode
            f.write("completed")

    # # Run wimbd_preprocess.py for the "all" method
    # if args.method in [None, "all"]:
    #     preprocessed_data_all = run_preprocess(args, "all", save_path)
    #     run_process(args, "all", preprocessed_data_all, save_path)

    # # Run wimbd_preprocess.py for the "common" method
    # if args.method in [None, "common"]:
    #     preprocessed_data_common = run_preprocess(args, "common", save_path)
    #     run_process(args, "common", preprocessed_data_common, save_path)
    


if __name__ == "__main__":
    args = parse_args()
    # device which method to use to extract ngrams
    if args.method == None:
        for method in ["common", "all"]:
            args.method = method
            main(args)
    else:
        if args.dataset == 'wmt09_gens':
            tasks = args.tasks
            for task in tasks:
                args.tasks = [task]
                print(f"Processing task {task}")
                main(args)
        else:
            main(args)


"""

# Example usage for MMLU
CUDA_VISIBLE_DEVICES="" python wimbd_search.py \
                        --type rust \
                        --corpus tuluv2 \
                        --n_grams 2 \
                        --dataset mmlu \
                        --filter_stopwords true \
                        --replace_keywords false \
                        --only_alpha false \
                        --name local/all \
                        --method common

CUDA_VISIBLE_DEVICES=1 python wimbd_search.py \
                        --type infini \
                        --corpus pile \
                        --n_grams 2 \
                        --dataset mmlu \
                        --replace_keywords false \
                        --filter_stopwords true \
                        --only_alpha false \
                        --name exp5 \
                        --align_pairs true \
                        --tasks marketing \
                                management \
                                high_school_world_history \
                                high_school_european_history \
                                miscellaneous \
                        --method common
                        
# Example usage for translation
CUDA_VISIBLE_DEVICES="" python wimbd_search.py \
                        --type infini \
                        --corpus pile \
                        --n_grams 1 \
                        --dataset europarl \
                        --language_pair en es \
                        --filter_stopwords true \
                        --replace_keywords false \
                        --only_alpha false \
                        --get_docs false \
                        --debug


# Example usage for translation GENS
CUDA_VISIBLE_DEVICES="" python wimbd_search.py \
                        --type infini \
                        --corpus pile \
                        --n_grams 2 \
                        --method common \
                        --dataset wmt09_gens \
                        --tasks pythia-160m \
                        --language_pair en es \
                        --filter_stopwords false \
                        --replace_keywords false \
                        --only_alpha false \
                        --debug \
                        --get_docs false 


# Example usage for sciq
CUDA_VISIBLE_DEVICES="" python wimbd_search.py \
                        --type infini \
                        --corpus pile \
                        --n_grams 3 \
                        --dataset triviaqa \
                        --filter_stopwords false \
                        --replace_keywords false \
                        --only_alpha false \
                        --n_samples 500 \
                        --name debug


"""