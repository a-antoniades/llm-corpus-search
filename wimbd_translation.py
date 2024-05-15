# %%
import sys
sys.path.append('/share/edc/home/antonis/LLM-Incidental-Supervision/wimbd')
import os
import pickle
from itertools import product
from tqdm import tqdm
import collections

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
 
from elasticsearch import Elasticsearch
from wimbd.es import count_documents_containing_phrases
from src.utils import running_jupyter

import datasets
from datasets import load_dataset, load_from_disk

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from stop_words import get_stop_words # for czech and polish

from laserembeddings import Laser
import numpy as np

import argparse
from src.wimbd_ import WimbdTasks as wt

import logging
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.ERROR)



dataset_mapping = {
    "wmt09": [("cs", "en"), ("de", "en"), ("fr", "en"), ("es", "en"), ("it", "en"),
              ("hu", "en")],
    "wmt14": [("en", "fr")], # ("fr", "en") 
    "wmt16": [("ro", "en"), ("fi", "en")], # ("de", "en"), 
    "wmt19": [("lt", "en")],
    "wmt20": [("ja", "en"), ("ru", "en"), ("zh", "en"), ("pl", "en")], # ("cs", "en")
}

language_mapping = {"fr": "french", 
                    "en": "english", 
                    "de": "german", 
                    "ru": "russian", 
                    "zh": "chinese",
                    "ja": "japanese",
                    "ro": "romanian",
                    "cs": "czech",
                    "pl": "polish",
                    "es": "spanish",
                    "it": "italian",
                    "lt": "lithuanian",
                    "hu": "hungarian"}

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
        parser.add_argument("--method", type=str, default=None, help="Method to use for the query")
        parser.add_argument("--n_grams", type=int, default=2, help="Number of n-grams to use")
        parser.add_argument("--language_pair", type=str, nargs=2, default=("fr", "en"), help="Language pair to use")
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # %%
    cloud_id = "m-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl"
    api_key = "RlZBbHpZc0J1MEw4LVVWVk9SaTE6bXJlSUM2QnlSQmFHemhwVElVUnZyQQ=="

    es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            retry_on_timeout=True,
            http_compress=True)

    # %%

    dataset_name = wt.get_dataset_name_from_language_pair(args.language_pair, dataset_mapping)
    ds = wt.load_ds(dataset_name, language_pair=args.language_pair)
    print(f"-- Dataset name: {dataset_name}, language pair: {args.language_pair} --")
    print(f"Dataset: {ds}")


    # %%
    text_1 = 'Spectacular Wingsuit Jump Over Bogota'
    text_2 = 'Spectaculaire saut en "wingsuit" au-dessus de Bogota'
    print(wt.generate_ngrams(text_1, 3))  # For bigrams
    print(wt.generate_ngrams(text_2, 3))  # For trigrams

    text_1_gram = wt.generate_ngrams(text_1, 3)
    text_2_gram = wt.generate_ngrams(text_2, 3)

    text_combinations = list(product(text_1_gram, text_2_gram))


    # %%

    base_path = f"./results/n-grams/exp_full/{args.n_grams}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    n_examples = 0
    # Wrap your iterable with tqdm() for a progress bar
    if args.method == None:
        do_all = True
    else:
        do_all = False

    if args.method == "common" or do_all:
        save_path = f"{base_path}/common"
        filename = f"{args.language_pair[0]}-{args.language_pair[1]}-{args.n_grams}-grams.pkl"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # if os.path.exists(save_path):
        #     print(f"File {save_path} already exists. Skipping...")
        # else:
        n_gram_freqs = collections.defaultdict(lambda: {'value': 0, 'example': None})
        for example in tqdm(ds['translation']):
            example_lang1 = example[args.language_pair[0]]
            example_lang2 = example[args.language_pair[1]]
            text_combinations = wt.get_combinations(example_lang1, example_lang2, language_pair=args.language_pair, n_gram=args.n_grams)
            # align language pairs to only keep the ones that are translations of each other
            text_combinations = wt.align_lang_pairs(text_combinations, args.language_pair[0], args.language_pair[1])
            print(f"text_combinations: {text_combinations}")
            example_counts = 0
            for text_combination in text_combinations:
                counts = count_documents_containing_phrases("re_pile", text_combination, 
                                                    es=es, all_phrases=True)
                example_counts += counts
                n_gram_freqs[text_combination]['value'] += counts
                # n_gram_freqs[text_combination]['example'] = example
            print(f"no. text_combinations: {len(text_combinations)}")
            print(f"Example counts: {example_counts}")
            n_examples += 1
            # save as .pkl, specify language and n_gram amount)
            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(dict(n_gram_freqs), f)
                
    if args.method == "all" or do_all:
        print(f"/// Running all method ///")
        save_path = f"{base_path}/all/"
        filename = f"{args.language_pair[0]}-{args.language_pair[1]}-{args.n_grams}-grams-all.pkl"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        n_gram_freqs = collections.defaultdict(lambda: {'value': 0, 'language': None})
        for example in tqdm(ds['translation']):
            example_lang1 = example[args.language_pair[0]]
            example_lang2 = example[args.language_pair[1]]
            full_lang_1_name = language_mapping[args.language_pair[0]]
            full_lang_2_name = language_mapping[args.language_pair[1]]
            text_1_gram = wt.process_text(example_lang1, full_lang_1_name, n_gram=args.n_grams)
            text_2_gram = wt.process_text(example_lang2, full_lang_2_name, n_gram=args.n_grams)
            unique_n_grams = {args.language_pair[0]: text_1_gram, args.language_pair[1]: text_2_gram}
            example_counts = 0
            for lang in args.language_pair:
                for n_gram in unique_n_grams[lang]:
                    counts = count_documents_containing_phrases("re_pile", [n_gram], 
                                                        es=es, all_phrases=False)
                    example_counts += counts
                    n_gram_freqs[n_gram]['value'] += counts
                    n_gram_freqs[n_gram]['lang'] = lang

            print(f"no. text_combinations: {len(text_combinations)}")
            print(f"Example counts: {example_counts}")
            n_examples += 1
            # save as .pkl, specify language and n_gram amount
            with open(os.path.join(save_path, filename), 'wb') as f:
                pickle.dump(dict(n_gram_freqs), f)

