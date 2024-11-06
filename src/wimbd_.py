import os
import collections
from collections import defaultdict
from tqdm import tqdm
import json
import glob
import pickle
from itertools import product
import ast
from IPython import display

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import plotly.graph_objs as go
import seaborn as sns

import numpy as np
import pandas as pd

from laserembeddings import Laser
from langdetect import detect, LangDetectException
import nltk
from nltk import bigrams, FreqDist, word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
# import mauve

from scipy.stats import entropy
from scipy.spatial.distance import cosine

from datasets import (load_dataset, load_from_disk, 
                      get_dataset_config_names, DatasetDict)
import pycountry
import nagisa # for japanese
import jieba # for chinese
from stop_words import get_stop_words # for czech and polish
import string
import re
import spacy
nlp = spacy.load("en_core_web_sm")
import langid
from tqdm import tqdm
tqdm.pandas()


from transformers import AutoTokenizer, AutoModel

from src.utils import (remove_string, normalize_string, softmax,
                       remove_nested_lists, extract_dict_entry_to_column)

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
WEIGHTS_DIR = "/share/edc/home/antonis/weights/huggingface"

os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_WEIGHTS_PATH"] = WEIGHTS_DIR

MODEL_NAMES = [
    "OLMo-7B", "pythia-12b", "pythia-31m", "pythia-1.4b",
    "pythia-410m", "pythia-70m", "pythia-14m", "pythia-2.8b",
    "pythia-6.9b", "pythia-160m"
]

PYTHIA_MODEL_NAMES = [
    'pythia-14m', 'pythia-31m', 'pythia-70m',
    'pythia-160m', 'pythia-410m', 'pythia-1.4b',
    'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'
]

# Define country-specific colors
country_colors = {
    'fr-en': '#0055A4',   # France - Blue
    'es-en': '#FF9900',   # Spain - Orange
    'it-en': '#008C45',   # Italy - Green
    'de-en': '#FFCC00',   # Germany - Yellow
    'cs-en': '#D7141A',   # Czech Republic - Red
    'hu-en': '#8B4513'    # Hungary - SaddleBrown
}

MARKERS = ['o', 's', 'D', 'v', '^', 'p', '*', 'h', 'x', '+']

LANG_MARKERS = {
    'cs': 'o',   # Czech Republic - Red
    'fr': 's',   # France - Blue
    'es': 'D',   # Spain - Orange
    'de': 'v',   # Germany - Yellow
    'hu': '^',   # Hungary - SaddleBrown
    'it': 'p'    # Italy - Green
}

# Define the Colors class
class Colors:
    PYTHIA_MODEL_NAMES = [
        'pythia-14m', 'pythia-31m', 'pythia-70m',
        'pythia-160m', 'pythia-410m', 'pythia-1.4b',
        'pythia-2.8b', 'pythia-6.9b', 'pythia-12b'
    ]

    model_colormap = plt.cm.get_cmap('coolwarm', len(PYTHIA_MODEL_NAMES))
    # model_color_mapping = {model: model_colormap(1 - i / len(PYTHIA_MODEL_NAMES)) for i, model in enumerate(reversed(PYTHIA_MODEL_NAMES))}

dataset_mapping = {
    "wmt09": [("cs", "en"), ("de", "en"), ("fr", "en"), ("es", "en"), ("it", "en"), ("hu", "en"),
              ("en", "cs"), ("en", "de"), ("en", "fr"), ("en", "es"), ("en", "it"), ("en", "hu")],
    "wmt14": [("en", "fr")], # ("fr", "en") 
    "wmt16": [("ro", "en"), ("fi", "en")], # ("de", "en"), 
    "wmt19": [("lt", "en")],
    "wmt20": [("ja", "en"), ("ru", "en"), ("zh", "en"), ("pl", "en")], # ("cs", "en")
}

reverse_dataset_mapping = {
    ("cs", "en"): "wmt09", ("de", "en"): "wmt09", ("fr", "en"): "wmt09",
    ("es", "en"): "wmt09", ("it", "en"): "wmt09", ("hu", "en"): "wmt09",
    ("en", "fr"): "wmt14", # ("fr", "en"): "wmt14",
    ("ro", "en"): "wmt16", ("fi", "en"): "wmt16", # ("de", "en"): "wmt16",
    ("lt", "en"): "wmt19",
    ("ja", "en"): "wmt20", ("ru", "en"): "wmt20", ("zh", "en"): "wmt20", ("pl", "en"): "wmt20", # ("cs", "en"): "wmt20"
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


def load_ds_translation(languages=None, **kwargs):
    ds_dict = {}
    if languages is None:
        languages = dataset_mapping['wmt09']
    for language_pair in languages:
        if isinstance(language_pair, str):
            language_pair = language_pair.split("-")
        default_dataset_names = ["wmt14", "wmt16"]
        print(f"Loading dataset for language pair {language_pair}")
        dataset_name = 'wmt09'  # reverse_dataset_mapping[language_pair]
        if dataset_name in default_dataset_names:
            ds = load_dataset(dataset_name, language_pair=language_pair)['test']
        else:
            ds = load_from_disk(f'/share/edc/home/antonis/datasets/huggingface/{dataset_name}-{language_pair[0]}-{language_pair[1]}')
        ds_dict[f"{language_pair[0]}-{language_pair[1]}"] = ds
    return ds_dict

def load_trivia_qa(**kwargs):
    return {'triviaqa':
                load_dataset("trivia_qa", "rc.nocontext", cache_dir=CACHE_DIR)
            }

def load_europarl(languages=None, **kwargs):
    ds_to_load = {
    'Helsinki-NLP/europarl': ['cs-en', 'en-fr', 'de-en', 
                              'en-hu', 'en-es', 'en-it']
    }
    dataset = {}
    languages = [f"{task[0]}-{task[1]}" for task in languages]
    for ds, ds_tasks in ds_to_load.items():
        for task in ds_tasks:
            if languages is None or task in languages:
                dataset[task] = load_dataset(ds, task, cache_dir=CACHE_DIR)['train']
    return dataset

def load_mmlu(tasks=None, **kwargs):
    configs = get_dataset_config_names("hendrycks_test")
    ds_full = {}
    for config in configs:
        if tasks is None or config in tasks:
            ds_full[config] = load_dataset("hendrycks_test", config, cache_dir=CACHE_DIR)
    return ds_full

def load_bigbench(**kwargs):
    # load bigbench tasks
    bigbench_tasks = []
    with open("./configs/data/bigbench_tasks.txt", "r") as f:
        for line in f:
            bigbench_tasks.append(line.strip())
    ds_full = {}
    for task in bigbench_tasks:
        ds_full[task] = load_dataset("bigbench", task, cache_dir=CACHE_DIR)
    return ds_full

def load_wmt09_gens(tasks=None, languages=None, **kwargs):
    dataset_name = "wmt09_gens"
    dataset_dict = {}
    language_pairs = [f"{task[0]}-{task[1]}" for task in languages]
    for model_name in MODEL_NAMES:
        if tasks is None or model_name in tasks:
            # Define the path for the current model's dataset
            model_dataset_path = os.path.join(CACHE_DIR, f'{dataset_name}_{model_name}')
            
            # Load the dataset from disk
            dataset_dict[model_name] = load_from_disk(model_dataset_path)

            if languages:
                dataset_dict[model_name] = {k: v for k, v in dataset_dict[model_name].items() if k in language_pairs}
            
            print(f"Loaded dataset for model {model_name} from {model_dataset_path}")

    # Combine all the loaded datasets into a DatasetDict
    combined_dataset_dict = DatasetDict(dataset_dict)

    if tasks and len(tasks) == 1:
        combined_dataset_dict = combined_dataset_dict[tasks[0]]

    return combined_dataset_dict


def _load_dataset(dataset_name, **kwargs):
    """
    Load the dataset based on the given dataset name.

    :param dataset_name: Name of the dataset to load.
    :param kwargs: Additional keyword arguments for loading datasets.
    :return: Loaded dataset.
    """
    print(f"Loading dataset {dataset_name}")
    if dataset_name == "mmlu":
        # load MMLU
        return load_mmlu(**kwargs) if not kwargs.get('debug', False) else load_mmlu(['elementary_mathematics'])
    elif dataset_name == "bigbench":
        return load_bigbench(**kwargs)
    elif dataset_name == "arithmetic":
        return load_arithmetic(**kwargs)
    elif dataset_name in ["wmt", "translation"]:
        # Assuming wt.lang_datasets_exp1 is available in the scope where this function is called
        return load_ds_translation(**kwargs)
    elif dataset_name == "wmt09_gens":
        return load_wmt09_gens(**kwargs)
    elif dataset_name == "europarl":
        return load_europarl(**kwargs)
    elif dataset_name == "triviaqa":
        return load_trivia_qa(**kwargs)
    elif dataset_name == "sciq":
        return {"sciq": load_from_disk(f"{CACHE_DIR}/sciq_converted")}
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")

def filter_strings(text, str_remove=None, replacements=None):
    """
    Remove all instances contained in list str_remove from text and
    replace words according to the replacements dictionary.
    """
    if str_remove:
        for s in str_remove:
            text = text.replace(s, "")
    if replacements:
        for old, new in replacements.items():
            text = text.replace(old, new)
    text = text.lstrip()  # Remove leading spaces
    return text

def clean_text(string, dataset, filter, replace,
               answer_idx=None):
    string_configs = {
        "arithmetic": {
            "filter_words": ['Question:', 'Answer:', 'What is', '?', '\n'],
            "replace_words": {
                "plus": "+"
            }
        }
    }
    if answer_idx is not None:
        string = string[answer_idx]

    if not isinstance(string, str):
        if isinstance(string, list) or isinstance(string, tuple):
            string = ' '.join(string)

    if filter:
        filter_words = string_configs[dataset]["filter_words"]
        for s in filter_words:
            string = string.replace(s, "")
    
    if replace:
        replace_words = string_configs[dataset]["replace_words"]
        for old, new in replace_words.items():
            string = string.replace(old, new)
    
    # filter any leading spaces
    string = string.lstrip()

    return string

def load_arithmetic(tasks=None, filtered=False, replacement=False, **kwargs):
    """
    todo: augment the data with alternate 
    operator symbols e.g: plus/+ minus/- times/* divide/÷ etc
    """
    filter_words = ['Question:', 'Answer:', 'What is',
                    '?', '\n']

    replace_words = {
        "plus": "+"
    }

    # load arithmetic tasks
    configs = [
        'arithmetic_2da', 'arithmetic_2ds', 
        'arithmetic_2dm', 'arithmetic_1dc',
        'arithmetic_3da', 'arithmetic_3ds',
        'arithmetic_4da', 'arithmetic_4ds', 
        'arithmetic_5da', 'arithmetic_5ds',
    ] 
    ds_arithmetic = {}
    for config in configs:
        if tasks is None or config in tasks:
            ds = load_dataset("EleutherAI/arithmetic", 
                                name=config, 
                                cache_dir=CACHE_DIR)['validation']
        if filtered:
            # filter all columns to not contain filter_words
            ds = ds.map(lambda x: {k: filter_strings(v, str_remove=filter_words) for k, v in x.items()})
        if replacement:
            ds = ds.map(lambda x: {k: filter_strings(v, replacements=replace_words) for k, v in x.items()})

        ds_arithmetic[config] = ds
    
    return ds_arithmetic

def load_lang_ds(lang_pairs=None, **kwargs):
    lang_ds = {}
    wb = WimbdTasks()
    if lang_pairs is None:
        lang_pairs = wb.lang_datasets_exp1
    for lang_pair in lang_pairs:
        lang_pair_ls = lang_pair.split("-")
        ds_name = wb.get_dataset_name_from_language_pair(lang_pair_ls)
        ds = load_ds(ds_name, language_pair=lang_pair_ls)
        lang_ds[lang_pair] = ds['translation']
    return lang_ds

# def load_results(base_results_path, models, task, shot, subtasks, replace_str='wmt09-'):
#     """
#     Load results from a structured directory into a dictionary.

#     :param base_results_path: The base directory where results are stored.
#     :param models: A list of model names.
#     :param task: The task name.
#     :param shot: The shot type (e.g., 'zero-shot').
#     :param tasks: A list of tasks to include in the results.
#     :return: A dictionary with the loaded results.
#     """
#     results_dict = defaultdict(dict)

#     # Iterate over each model and dataset, loading the results.json file
#     for model in models:
#         results_path = os.path.join(base_results_path, model, task)
#         for subtask in subtasks:
#             print(f"subtask: {subtask}")
#             results_file = glob.glob(os.path.join(results_path, subtask, shot, '**/results.json'), recursive=True)
#             # print(f"results_file: {results_file}")
#             assert len(results_file) == 1, f"Found {len(results_file)} results files for model {model}, \
#                                              task {task}, shot {shot}, subtask {subtask} \
#                                              {results_file}"
#             results = json.load(open(results_file[0]))['results']
#             if replace_str:
#                 subtask_ = subtask.replace(replace_str, '')
#             else:
#                 subtask_ = subtask
#             results_dict[model][subtask_] = results[subtask]
    
#     return results_dict


def load_results(base_results_path, models, task, shot, subtasks, replace_str='wmt09-', calc_reverse=False):
    """
    Load results from a structured directory into a dictionary.

    :param base_results_path: The base directory where results are stored.
    :param models: A list of model names.
    :param task: The task name.
    :param shot: The shot type (e.g., 'zero-shot').
    :param subtasks: A list of tasks to include in the results.
    :param replace_str: A string to replace in the subtask name.
    :param calc_reverse: A boolean to determine if reverse tasks should be included.
    :return: A dictionary with the loaded results.
    """
    results_dict = defaultdict(dict)

    # Helper function to load results from a specified path
    def load_results_from_path(path):
        results_file = glob.glob(os.path.join(path, '**/results.json'), recursive=True)
        assert len(results_file) == 1, f"Found {len(results_file)} results files at {path} {results_file}"
        return json.load(open(results_file[0]))['results']

    # Helper function to generate reverse task name
    def generate_reverse_subtask(subtask):
        parts = subtask.split('-')
        if len(parts) == 3 and parts[0] == replace_str.rstrip('-'):
            return f"{parts[0]}-{parts[2]}-{parts[1]}"
        return subtask

    # Iterate over each model and dataset, loading the results.json file
    for model in models:
        results_path = os.path.join(base_results_path, model, task)
        for subtask in subtasks:
            subtask_results_path = os.path.join(results_path, subtask, shot)
            original_results = load_results_from_path(subtask_results_path)[subtask]

            if replace_str:
                subtask_ = subtask.replace(replace_str, '')
            else:
                subtask_ = subtask

            if calc_reverse:
                reverse_subtask = generate_reverse_subtask(subtask)
                reverse_results_path = os.path.join(results_path, reverse_subtask, shot)
                reverse_results = load_results_from_path(reverse_results_path)[reverse_subtask]

                # Calculate the average score between the original and reverse results
                avg_results = {k: (v + reverse_results[k]) / 2 for k, v in original_results.items()}
                results_dict[model][subtask_] = avg_results
            else:
                results_dict[model][subtask_] = original_results

    return results_dict


def drop_rows_without_words(df, language):
    def check_words(row):
        example_text = row['example']['translation'][language]
        sequence_words = row['sequence'].split()
        for word in sequence_words:
            if word.lower() not in example_text.lower():
                return False
        return True

    tqdm.pandas(desc=f"Filtering rows for language: {language}")
    mask = df.progress_apply(check_words, axis=1)
    return df[mask]

def filter_stop_words_(text, lang):
    # Add punctuation to the list of stopwords
    # stop_words = set(string.punctuation)
    stop_words = set()

    if lang == 'japanese':
        words = nagisa.tagging(text)
        stop_words.update([word for word, pos in zip(words.words, words.postags) if pos in ["助詞", "助動詞", "記号"]])
        filtered_text = [w for w in words.words if w not in stop_words]
    elif lang in ['czech', 'polish']:
        stop_words.update(get_stop_words(lang))
        filtered_text = [w for w in text.split() if not w in stop_words]
    elif lang == 'chinese':
        stop_words.update(set(stopwords.words('chinese')))
        word_tokens = jieba.cut(text, cut_all=False)
        filtered_text = [w for w in word_tokens if not w in stop_words]
    else:
        stop_words.update(set(stopwords.words(lang)))
        word_tokens = word_tokenize(text)
        filtered_text = [w for w in word_tokens if not w in stop_words]

    return filtered_text


def filter_percentile(df, percentile, column='value'):
    # filter outliers
    if percentile > 0:
        upper_quantile = df[column].quantile(percentile)
        df = df[df[column] < upper_quantile]
    return df.reset_index(drop=True).sort_values(by=column, ascending=False)

def drop_rows_with_stopwords(df, language):
    def check_stopwords(row):
        sequence_words = row[lang_col_name].split()
        filtered_words = filter_stop_words_(row[lang_col_name], language)
        return len(filtered_words) == len(sequence_words)

    # convert language to full language name
    lang_col_name = language
    language = language_mapping[language]
    tqdm.pandas(desc=f"Filtering stopwords for language: {language}")
    mask = df.progress_apply(check_stopwords, axis=1)
    return df[mask]

class BasePaths:
    base_ngram_paths = {
        "mmlu": {
            # "base_path": "./results/n-grams/mmlu/pile/exp4_nofilter/test-set/exp_full_None",
            # "pile": {"base_path": "./results/n-grams/mmlu/pile/exp4_filter/test-set/exp_full_None"},
            "pile": {"base_path": "./results/n-grams/mmlu/exp3/test-set/exp_full_None"},
            "dolma": {"base_path": "./results/n-grams/mmlu/dolma/exp4_infini/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse"},
            "tasks": [
                        'abstract_algebra',
                        'anatomy',
                        'astronomy',
                        'business_ethics',
                        'clinical_knowledge',
                        'college_biology',
                        'college_chemistry',
                        'college_computer_science',
                        'college_mathematics',
                        'college_medicine',
                        'college_physics',
                        'computer_security',
                        'conceptual_physics',
                        'econometrics',
                        'electrical_engineering',
                        'elementary_mathematics',
                        'formal_logic',
                        'global_facts',
                        'high_school_biology',
                        'high_school_chemistry',
                        'high_school_computer_science',
                        'high_school_european_history',
                        'high_school_geography',
                        'high_school_government_and_politics',
                        'high_school_macroeconomics',
                        'high_school_mathematics',
                        'high_school_microeconomics',
                        'high_school_physics',
                        'high_school_psychology',
                        'high_school_statistics',
                        'high_school_us_history',
                        'high_school_world_history',
                        'human_aging',
                        'human_sexuality',
                        'international_law',
                        'jurisprudence',
                        'logical_fallacies',
                        'machine_learning',
                        'management',
                        'marketing',
                        'medical_genetics',
                        'miscellaneous',
                        'moral_disputes',
                        'moral_scenarios',
                        'nutrition',
                        'philosophy',
                        'prehistory',
                        'professional_accounting',
                        'professional_law',
                        'professional_medicine',
                        'professional_psychology',
                        'public_relations',
                        'security_studies',
                        'sociology',
                        'us_foreign_policy',
                        'virology',
                        'world_religions'
                    ],
        },
        "arithmetic": {
            "base_path": "./results/n-grams/arithmetic/pile/exp1/unchanged_operator/n_samples_None_fkeyTrue_rkeyFalse_fstopFalse_onlyalphaFalse",
            "tasks": ['wmt09-cs-en', 'wmt09-de-en', 'wmt09-fr-en', 'wmt09-es-en', 'wmt09-it-en', 'wmt09-hu-en',
                      'wmt09-en-cs', 'wmt09-en-de', 'wmt09-en-fr', 'wmt09-en-es', 'wmt09-en-it', 'wmt09-en-hu']
        },
        "triviaqa": {
            "base_path": {
                5 : "./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse",
                3 : "./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse"
            },
        },
        "sciq": {
            "base_path": " ", 
        },
        "translation": {
            "tasks" : ['cs-en', 'de-en', 'fr-en', 'es-en', 'it-en', 'hu-en']
        }
    }

    base_results_paths = {
        "mmlu": {
        "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_4/inference/EleutherAI": [
            'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
            'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
            'pythia-70m', 'pythia-31m', 'pythia-14m'
        ],
        "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/OLMO": [
            'OLMo-7b'
        ]
        },
        "arithmetic": {
            "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_5/inference/EleutherAI": [
                'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
                'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
                'pythia-70m', 'pythia-31m', 'pythia-14m'
            ],
        }
        # "triviaqa": {
        #     ""
        # }
    }


class DataConfigs:
    data_keys = {
        'mmlu': {
            'text_1': 'question',
            'text_2': 'choices',
            'split': 'test',
            'answer': 'answer',
        },
        'bigbench': {
            'text_1': 'inputs',
            'text_2': 'targets',
            'split': 'default'
        },
        'arithmetic': {
            'text_1': 'context',
            'text_2': 'completion',
            'split': 'test',
        },
        'wmt': {
            'text_1': 'translation',
            'text_2': 'translation'
        },
        'wmt09_gens': {
            'text_1': ['translation', 'src'],
            'text_2': ['translation', 'gen']
        },
        'europarl': {
            'text_1': 'translation',
            'text_2': 'translation'
        },
        'triviaqa': {
            'text_1': 'question',
            'text_2': ['answer', 'value'],
            'split': 'validation'
        },
        'sciq': {
            'text_1': 'question',
            'text_2': 'choices',
            'split': 'train',
        }
    }

    task_configs = {
        'mmlu': {'ommit': 
                 [
                    'high_school_computer_science', 'elementary_mathematics',
                    'college_computer_science', 'high_school_mathematics', 
                    'public_relations', 'miscellaneous', 'nutrition',
                    'machine_learning', 'college_mathematics'
                 ]
        },
    }

    mmlu_tasks = {
            'math': [
                'elementary_mathematics',   # no filtering
                'abstract_algebra', # no filtering
                'econometrics', # not enough data
                'high_school_statistics', # good, use no filtering 
                'college_mathematics',  # use no filtering
                'high_school_mathematics',  # use no filtering
                'college_physics',  # use no filtering
                'high_school_physics',  # potentially ok
                'college_chemistry' # potentially ok

            ],
            'coding': [
                'college_computer_science', # use no filtering
                'high_school_computer_science', # use no filtering
                'electrical_engineering', # potentially ok
                'machine_learning', # potentially ok
                'computer_security' # quite ok
            ],
            'logic': [
                'logical_fallacies' # use no filtering
            ],
            'professional': [
                    'professional_psychology', # use no filtering
                    'professional_law', # use no filtering,
                    'nutrition',
                    'public_relations'

            ],
            'other': [
                'business_ethics',
                'management'
            ],
            'psychology': [
                'high_school_psychology'
            ],
            'top_diff_sft_olmo': [
                'marketing',
                'management',
                'high_school_world_history',
                'high_school_european_history',
                'miscellaneous'      
            ]
        }

import torch

def align_e5_pairs_df(df, col_1, col_2):
    """
    Create a new column called alignment_score with the e5 scores
    """
    from transformers import AutoTokenizer, AutoModel

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
    model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
    model.eval()

    print(f"Using device: {device}")
    
    def get_max_batch_size(text_1_batch, text_2_batch):
        def can_allocate_memory(batch_size):
            try:
                input_text = ['query: ' + query for query in text_1_batch[:batch_size]] + \
                            ['passage: ' + passage for passage in text_2_batch[:batch_size]]
                batch = tokenizer(input_text, max_length=512, padding=True,
                                truncation=True, return_tensors='pt').to(device)
                _ = batch.input_ids.size()  # Trigger CUDA memory allocation
                return True
            except RuntimeError:
                return False

        low, high = 1, len(text_1_batch)
        max_batch_size = 1024

        while low <= high:
            mid = (low + high) // 2
            if can_allocate_memory(mid):
                max_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
                torch.cuda.empty_cache()  # Clear cache after a failed allocation

        return max_batch_size

    alignment_scores = []

    for i in tqdm(range(0, len(df)), desc="Aligning pairs"):
        print(f"Processing row {i}")
        text_1_batch = df[col_1].iloc[i:].tolist()
        text_2_batch = df[col_2].iloc[i:].tolist()
        
        # max_batch_size = get_max_batch_size(text_1_batch, text_2_batch)
        max_batch_size = 32
        
        for j in tqdm(range(0, len(text_1_batch), max_batch_size), desc="Processing batches"):
            batch_text_1 = text_1_batch[j:j+max_batch_size]
            batch_text_2 = text_2_batch[j:j+max_batch_size]
            
            scores = calculate_qa_score(batch_text_1, batch_text_2, model=model, tokenizer=tokenizer)
            alignment_scores.extend(scores)
            
        if len(alignment_scores) >= len(df):
            break

    df['alignment_score'] = alignment_scores[:len(df)]
    return df

def calculate_qa_score(text_1, text_2, model, tokenizer, threshold=0.7):
    """
    Align pairs using e5 embeddings
    text_1, text_2: lists of texts to be aligned
    """
    import torch
    import torch.nn.functional as F

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(text_1, list):
        text_1 = [text_1]
    if not isinstance(text_2, list):
        text_2 = [text_2]

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    with torch.no_grad():
        querries = ['query: ' + query for query in text_1]
        passages = ['passage: ' + passage for passage in text_2]
        input_text = querries + passages
        batch = tokenizer(input_text, max_length=512, padding=True,
                          truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        num_pairs = min(len(querries), len(passages))
        scores = (embeddings[:num_pairs].cpu() @ embeddings[num_pairs:].cpu().T)
        corresponding_scores = scores.diag().detach().numpy() * 100  # Extract the diagonal and scale by 100

    return corresponding_scores

def align_ngram_combinations(ngram_combinations, model, tokenizer, threshold=None):
    """
    Calculate the QA score for all pairs of n-grams.

    ngram_combinations: list of tuples, where each tuple contains two n-grams
    model: the pre-trained model to be used for embeddings
    tokenizer: the tokenizer corresponding to the pre-trained model
    threshold: optional score threshold to filter out low-score pairs

    Returns:
        A list of tuples with the n-grams and their QA score, optionally filtered by the threshold.
    """
    
    aligned_pairs = []
    for pair in ngram_combinations:
        ngram_1, ngram_2 = pair
        score = calculate_qa_score(ngram_1, ngram_2, model, tokenizer)[0]
        if threshold is None or score >= threshold:
            aligned_pairs.append((ngram_1, ngram_2))

    print(f"kept {len(aligned_pairs)} out of {len(ngram_combinations)} pairs")
    
    return aligned_pairs


class WimbdTasks:
    def __init__(self, **kwargs):
        self.translation_dataset_mapping = {
            "wmt09": [("cs", "en"), ("de", "en"), 
                      ("fr", "en"), ("es", "en"), 
                      ("it", "en"), ("hu", "en")],
            "wmt14": [("en", "fr")],  # ("fr", "en")
            "wmt16": [("ro", "en"), ("fi", "en")],  # ("de", "en"),
            "wmt19": [("lt", "en")],
            "wmt20": [("ja", "en"), ("ru", "en"), ("zh", "en"), ("pl", "en")],  # ("cs", "en")
        }

        self.language_mapping = {"fr": "french",
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
        
        self.lang_datasets_exp1 = ['hu-en', 
                                   'es-en', 
                                   'it-en', 
                                   'fr-en', 
                                   'cs-en', 
                                   'de-en']


        if kwargs.get('align_pairs', True):
            lang_pair = kwargs.get('language_pair', ('en', 'fr'))
            if lang_pair[0] == lang_pair[1]:
                self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
                device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
                self.model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
                self.model.eval()
        self.Laser = Laser()

    def load_ds_translation(self, language_pair):
        dataset_name = self.get_dataset_name_from_language_pair(language_pair)
        default_dataset_names = ["wmt14", "wmt16"]
        if dataset_name in default_dataset_names:
            ds = load_dataset(dataset_name, language_pair=language_pair)['test']
        else:
            ds = load_from_disk(f'/share/edc/home/antonis/datasets/huggingface/{dataset_name}-{language_pair[0]}-{language_pair[1]}')
        return ds

    def get_dataset_name_from_language_pair(self, language_pair):
        if not isinstance(language_pair, tuple):
            language_pair = tuple(language_pair)
        for dataset, language_pairs in self.translation_dataset_mapping.items():
            if language_pair in language_pairs:
                return dataset
        raise ValueError(f"No dataset found for language pair {language_pair}")

    def get_language_name(self, bigram):
        try:
            lang_code = detect(bigram)
            language = pycountry.languages.get(alpha_2=lang_code)
            return language.name
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def filter_stop_words(self, text):
        words = nagisa.tagging(text)
        stopwords = [word for word, pos in zip(words.words, words.postags) if pos in ["助詞", "助動詞", "記号"]]
        return ' '.join(stopwords)

    def process_text(self, text, lang, n_gram, 
                     filter_stopwords=False, only_alpha=False,
                     filter_punc=False):
        if isinstance(text, (int, float)):
            text = str(text)
        if isinstance(text, list):
            text = [str(t) for t in text]
            text = ' '.join(text)
        
        # text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.strip() # remove leading spaces

        if filter_punc:
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        # Filter out stop words
        if filter_stopwords:
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            filtered_text = filter_stop_words_(text, lang)
            print(f"filtered_text: {filtered_text}")
        else:
            filtered_text = text.split(' ')
        
        text_gram = ngrams(filtered_text, n_gram)
        text_gram = [' '.join(ngram) for ngram in text_gram]
        if len(text_gram) == 0:
            text_gram = [' '.join(filtered_text)]

        if only_alpha:
            # Filter out n-grams that contain non-alphabetic characters or are too short
            text_gram = [ngram for ngram in text_gram if all(word.isalpha() and len(word) > 0 for word in ngram.split())]
        
        # filter out n-grams whose size is not equal to n_gram
        text_gram = [ngram for ngram in text_gram if len(ngram) != n_gram]

        return list(text_gram)

    def generate_ngrams(self, text, n):
        tokens = nltk.word_tokenize(text)
        ngrams = list(nltk.ngrams(tokens, n))
        return [' '.join(ngram) for ngram in ngrams]

    def laser_similarity(self, sentence1, lang1, sentence2, lang2):
        # print(f"Calculating laser similarity for {sentence1} and {sentence2}")
        embeddings1 = self.Laser.embed_sentences(sentence1, lang=lang1).squeeze()
        embeddings2 = self.Laser.embed_sentences(sentence2, lang=lang2).squeeze()

        # Cosine similarity
        similarity = 1 - cosine(embeddings1, embeddings2)
        return similarity

    def get_combinations(self, text_1, text_2, language_pair, n_gram=2,
                         no_split_text2=False, filter_stopwords=False, 
                         only_alpha=False, filter_punc=True, align_pairs=False):
        
        full_lang_1_name = self.get_language_name(language_pair[0])
        full_lang_2_name = self.get_language_name(language_pair[1])
        
        text_1_gram = self.process_text(text_1, full_lang_1_name, n_gram,
                                        filter_stopwords=filter_stopwords, 
                                        filter_punc=filter_punc,
                                        only_alpha=only_alpha)
        if no_split_text2:
            text_2_gram = text_2
        else:
            text_2_gram = self.process_text(text_2, full_lang_2_name, n_gram,
                                            filter_stopwords=filter_stopwords, 
                                            only_alpha=only_alpha)

        # If there are duplicate n-grams, return an empty list
        if len(set(text_1_gram) & set(text_2_gram)) > 0:
            return []
        
        text_combinations = list(product(text_1_gram, text_2_gram))
        
        # Sort each combination and eliminate duplicates
        text_combinations = list(set(combination for combination in text_combinations))

        # filter out combinations that contain the same n-gram
        text_combinations = [combination for combination in text_combinations if combination[0] != combination[1]]
        
        # align lang pairs 
        if language_pair[0] != language_pair[1]:
            if align_pairs:
                if n_gram <= 2:
                    threshold = 0.6
                else:
                    threshold = 0.9
                text_combinations = self.align_lang_pairs(text_combinations, 
                                                        language_pair[0], 
                                                        language_pair[1],
                                                        threshold=threshold)
        else:
            if align_pairs:
                print(f"aligning pairs!!")
                text_combinations = align_ngram_combinations(
                    text_combinations, self.model, self.tokenizer, threshold=0.85
                )

                
        text_combinations = tuple(text_combinations)
        return text_combinations

    def get_language_name(self, lang_code):
        try:
            language = pycountry.languages.get(alpha_2=lang_code)
            return language.name.lower()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def align_lang_pairs(self, text_combinations, 
                         lang_pair_1, lang_pair_2,
                         threshold=0.7):
        # filter out combinations where laser similarity is below 0.8
        text_combinations = [combination for combination in text_combinations if self.laser_similarity(combination[0], lang_pair_1, combination[1], lang_pair_2) > threshold]
        return text_combinations

    """
    use the below two functions to filter 
    lang_dfs by language in wimbd_analysis_translation.ipynb
    for analysis
    """

    def apply_laser_similarity(self, row, lang_pair, threshold):
        similarity = self.Laser.laser_similarity(row[lang_pair[0]], lang_pair[0], row[lang_pair[1]], lang_pair[1])
        return similarity > threshold

    def filter_lang_dfs_language(self, lang_dfs, threshold, wt):
        for lang_pair, df in tqdm(lang_dfs.items()):
            len_before = len(df)
            lang_pair = tuple(lang_pair.split("-"))
            df = df[df.apply(self.apply_laser_similarity, 
                        axis=1, 
                        laser_instance=wt, 
                        lang_pair=lang_pair, 
                        threshold=threshold)]
            len_after = len(df)
            len_diff = len_before - len_after
            len_p = len_diff / len_before
            print(f"lang_pair: {lang_pair}, len_before: {len_before}, \
                    len_after: {len_after}, len_diff: {len_diff} \
                    p: {len_p}")
        return lang_dfs

    def filter_df_by_language(self, df, threshold):
        def filter_row(row, laser_instance, lang_pair, threshold):
            lang_pair = tuple(row['task'].split('-'))
            example = row['index']
            similarity = laser_instance.laser_similarity(example[lang_pair[0]], lang_pair[0], example[lang_pair[1]], lang_pair[1])
            return similarity > threshold

        filtered_df = df[df.apply(filter_row, axis=1, lang_pair=('de', 'en'), threshold=threshold)]

        return filtered_df

class WimbdAnalysis:
    def __init__(self, base_path, dataset_list, n_gram, filter_chars):
        self.task_str = "_".join(dataset_list)
        self.n_gram = n_gram
        self.filter_chars = filter_chars
        self.base_path = base_path
        # self.filename = os.path.join(self.base_path, str(n_gram))
        self.base_path_commom = os.path.join(self.base_path, "common")
        self.base_path_all = os.path.join(self.base_path, "all")
        self.plot_path = os.path.join(self.base_path, "plots")
        self.wt = WimbdTasks()

    def check_language(self, text, lang):
        language, _ = langid.classify(text)
        return lang == self.get_language_name(language)
    
    def check_language_df(self, df, lang):
        df_filtered = df[df[lang].apply(lambda x: self.check_language(x, lang))]
        return df_filtered

    def calc_coverage(self, df):
        zero = df[df['value'] == 0]
        non_zero = df[df['value'] > 0]
        return len(non_zero) / (len(zero) + len(non_zero))

    def sum_values(self, data):
        return sum(item['value'] for item in data.values())
   
    def filter_column_by_language(self, data, language, column=None):
        """
        this function is to be used when both languages are in the same
        column (for example in the _all case) when we only want to keep
        the target language
        """
        column = 'lang' if column is None else column
        # return {k: v for k, v in data.items() if v[column] == language}
        return data[data[column] == language]

    def filter_chars_by_language(self, df, lang, filter_chars=False):
        regex_patterns = {
            "ja": r'[一-龯ぁ-んァ-ンａ-ｚＡ-Ｚ々〆〤]',
            "ru": r'[а-яА-ЯёЁ]',
            "zh": r'[一-龯]',
            "fr": r'[àâéèêëîïôùûüÿçœæÀÂÉÈÊËÎÏÔÙÛÜŸÇŒÆ]',
            "ro": r'[ăâîșțĂÂÎȘȚ]',
            "de": r'[äöüÄÖÜß]',
            "cs": r'[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]',
            # Add more patterns for other languages here
        }

        if filter_chars:
            lang_regex = regex_patterns.get(lang)
            if lang_regex:
                df = df[df[lang].str.contains(lang_regex, regex=True, na=False)]
        
        return df
        
    def extract_language_pair(self, filename):
        # Match both 'de-en' and "('de', 'en').pkl" formats
        match = re.search(r"([a-z]{2})-([a-z]{2})|\('([a-z]{2})', '([a-z]{2})'\)\.pkl", filename)
        if match:
            # Check which pattern was matched and extract accordingly
            if match.group(1) and match.group(2):
                # This is for 'de-en' format
                return match.group(1), match.group(2)
            elif match.group(3) and match.group(4):
                # This is for "('de', 'en').pkl" format
                return match.group(3), match.group(4)
        return None  # If no match, return None
    
    def filter_rows_with_names_and_places(self, df, text_column):
        """
        Filters out rows in a pandas DataFrame that contain names of people or places.

        Args:
        - df (pd.DataFrame): The input DataFrame.
        - text_column (str): The name of the column containing text to check for named entities.

        Returns:
        - pd.DataFrame: A DataFrame with rows containing names of people or places removed.
        """

        # Define a function to check if a text contains person or place names
        def contains_person_or_place(text):
            try:
                doc = nlp(text)
                return any(ent.label_ in ["PERSON", "GPE", "LOC"] for ent in doc.ents)
            except Exception as e:
                print(f"An error occurred while processing text: {text}. Error: {e}")
                return False  # If there's an error, we'll assume no names or places are present.

        # Apply the check function to each row and filter the DataFrame
        initial_row_count = len(df)
        
        tqdm.pandas(desc="Filtering rows")
        filtered_df = df[~df[text_column].progress_apply(lambda x: contains_person_or_place(x) if x is not None else False)]
        
        final_row_count = len(filtered_df)
        rows_deleted = initial_row_count - final_row_count
        
        print(f"Initial length: {initial_row_count}, Final length: {final_row_count}, Rows deleted: {rows_deleted}")
        
        return filtered_df
    
    def align_lang_pairs_df(self, df, lang_1, lang_2, 
                            col_1=None, col_2=None):
        """
        Add a new column to the DataFrame with the similarity score between two languages.

        Args:
        - df (pd.DataFrame): The input DataFrame.
        - lang_1 (str): The name of the first language column.
        - lang_2 (str): The name of the second language column.

        Returns:
        - pd.DataFrame: The DataFrame with the new similarity score column.
        """
        col_1_name = lang_1 if col_1 is None else col_1
        col_2_name = lang_2 if col_2 is None else col_2

        def calculate_laser_similarity(row):
            return self.wt.laser_similarity(row[col_1_name], lang_1, row[col_2_name], lang_2)

        # Apply the similarity function to each row to create a new column with the scores
        print(f"Calculating multilingual alignment scores for {lang_1} and {lang_2}")
        df['alignment_score'] = df.progress_apply(calculate_laser_similarity, axis=1)

        return df

    def prepare_ngram_df(self, n_gram_freqs_path, is_all=False, 
                        filter_chars=False, detect_lang=False,
                        percentile=0.95, n_gram=None, filter_entities=False,
                        align_langs=0, filter_stopwords=False, remove_english=True,
                        remove_non_english=False, **kwargs):

        def find_english_column(df, languages):
            if 'index' in df.columns:
                return 'index'
            elif languages[0] == 'en':
                return languages[0]
            else:
                return languages[1]

        with open(n_gram_freqs_path, "rb") as f:
            n_gram_freqs = pickle.load(f)
        
        n_gram_freqs = dict(sorted(n_gram_freqs.items(), key=lambda item: item[1]['value'], reverse=True))
        languages = self.extract_language_pair(n_gram_freqs_path.split("/")[-1])

        df = pd.DataFrame(n_gram_freqs).T
        if kwargs.get("debug"):
            df = df.iloc[:1000]
        df['coverage'] = self.calc_coverage(df)
        
        non_english_lang = languages[0] if languages[0] != 'en' else languages[1]
        en_lang_col_name = find_english_column(df, languages)
        other_lang_col_name = languages[0] if languages[0] != en_lang_col_name else languages[1]
        
        if n_gram:
            # Assuming the n-grams are in the first level of the MultiIndex
            # and that they are separated by spaces (e.g., 'word1 word2')
            df = df[df.index.get_level_values(0).map(lambda x: len(x.split()) == n_gram)]
        
        # df = df[df['value'] > 0]

        if is_all:
            df = df.reset_index().rename(columns={"index": non_english_lang})
            print(df.head(1))
            if remove_english:
                if 'language' in df.columns:
                    df = df.drop(columns=['language'])
                if 'lang' in df.columns:
                    df = self.filter_column_by_language(df, other_lang_col_name, 'lang')
                else:
                    df = drop_rows_without_words(df, non_english_lang)
            if remove_non_english:
                df = drop_rows_without_words(df, en_lang_col_name)
        else:
            df = df.reset_index().rename(columns={"level_0": languages[0], "level_1": languages[1]})

        # if filter_chars:
        #     target_column = other_lang_col_name if other_lang_col_name in df.columns else "index"
        #     df = self.filter_column_by_language(df, target_language, target_column)
        
        if filter_stopwords:
            print(f"columns: {df.columns}")
            print(f"other_lang_col_name: {other_lang_col_name}")
            if other_lang_col_name in df.columns:
                df = drop_rows_with_stopwords(df, other_lang_col_name)
            if en_lang_col_name in df.columns:
                df = drop_rows_with_stopwords(df, en_lang_col_name)

        if detect_lang:
            print(f"detecting langs!")
            target_column = non_english_lang if non_english_lang in df.columns else "index"
            df = filter_by_language(df, non_english_lang, target_column)
        
        if filter_entities:
            print(f"filtering entitites!")
            # print(f"df: {df}")
            print(f"df.columns: {df.columns}")
            chosen_column = en_lang_col_name if en_lang_col_name in df.columns else other_lang_col_name
            if en_lang_col_name in df.columns:
                df = self.filter_rows_with_names_and_places(df, en_lang_col_name)
            if other_lang_col_name in df.columns:
                df = self.filter_rows_with_names_and_places(df, other_lang_col_name)
        
        if align_langs > 0:
            df = self.align_lang_pairs_df(df, languages[0], languages[1])
        
        df['task'] = f"{languages[0]}-{languages[1]}"

        if len(df) == 0:
            return n_gram_freqs, df

        if df.columns[0] == languages[0]:
            # rename to 'lang_1'
            df = df.rename(columns={languages[0]: 'lang_1', languages[1]: 'lang_2'})
            # combine lang_1 and lang_2
            # df['example'] = df.apply(lambda row: [row['lang_1'], row['lang_2']], axis=1)
        
        # filter outliers
        if percentile > 0:
            upper_quantile = df['value'].quantile(percentile)
            df = df[df['value'] < upper_quantile]

        return n_gram_freqs, df
    
    def prepare_ngram_df_(self, n_gram_freqs_path, is_all=False, 
                          filter_chars=False, detect_lang=False, 
                          percentile=0, **kwargs):
     
        df = pd.read_pickle(n_gram_freqs_path)
        if isinstance(df, list):
            df = df[0]
        
        df = pd.DataFrame(df).T.sort_values("value", ascending=False)
        df['coverage'] = self.calc_coverage(df)
        # df = df[df['value'] > 0]
        # df_0 = df[df['value'] > 0]
        df = df.reset_index().rename(columns={"level_0": "Q", "level_1": "A"})
        print(f"df cols: {df.columns}")
        df = extract_dict_entry_to_column(df, 'example', 'question', 'question')

        if percentile > 0:
            df = filter_percentile(df, percentile)
        
        # if kwargs.get("align_pairs"):
        print(f"aligning pairs")
        df = align_e5_pairs_df(df, 'Q', 'question')
    
        return df

    def load_lang_paths(self, files, lang_pairs):
        # Prepare the patterns for matching both 'cs-en' and "('cs', 'en').pkl" formats
        print(f"lang_pairs: {lang_pairs}")
        patterns = ['-' .join(lang) for lang in lang_pairs] + \
                   [f"({repr(lang[0])}, {repr(lang[1])}).pkl" for lang in lang_pairs]
        
        # Filter files that match any of the patterns
        files = [file for file in files if any(pattern in file for pattern in patterns)]
        return files

    def get_lang_dfs(self, base_path, datasets, **kwargs):
        kwarg_string = '_'.join(f"{key}{value}" for key, value in kwargs.items())
        filename = f"lang_dfs_{kwarg_string}.pkl"
        save_pth = os.path.join(base_path, filename)
        task_name = os.path.join(base_path, f"{file}.csv")
        
        def get_file_name(lang_pair):
            file = [file for file in files if lang_pair[0] in file and lang_pair[1] in file]
            print(f"file: {file}, {files}")
            assert len(file) == 1, f"file: {file} not found, or found multiple times"
            return file[0]
        
        print(f"datasets: {datasets}")
        lang_pairs = [lang.split('-')[-2:] for lang in datasets]
        files = [file for file in os.listdir(base_path) if file.endswith(".pkl")]
        print(f"lang_pairs: {lang_pairs}")
        files = self.load_lang_paths(files, lang_pairs)
        lang_dfs = {}
        for lang_pair in lang_pairs:
            file = get_file_name(lang_pair)
            # check if file exists
            if os.path.exists(task_name):
                # load file
                df_clean = pd.read_csv(task_name)
                # check if 'alignment_score' column exists
                if 'alignment_score' in df_clean.columns:
                    print(f"---- file {file} exists, skipping ----")
                else:
                    print(f"---- file {file} exists, but alignment_score column is missing ----")
                    n_gram_freqs, df_clean = self.prepare_ngram_df(os.path.join(base_path, file), **kwargs)
            languages = f"{lang_pair[0]}-{lang_pair[1]}"
            lang_dfs[languages] = df_clean
            df_clean.to_csv(task_name, index=False)

        # save lang_df
        with open(save_pth, "wb") as f:
            pickle.dump(lang_dfs, f)
        
        print(f"saved file: {save_pth}")
        return lang_dfs, save_pth
    
    def get_task_dfs(self, base_path, datasets, **kwargs):
        # save task_dfs
        kwarg_string = '_'.join(f"{key}{value}" for key, value in kwargs.items())
        filename = f"task_df_{kwarg_string}.pkl"
        save_pth = os.path.join(base_path, filename)
        
        if not isinstance(datasets, list):
            datasets = [datasets]

        files = [file for file in os.listdir(base_path) if file.endswith(".pkl")]
        files = [file for file in files if any(dataset in file for dataset in datasets)]
        task_dfs = {}

        for file in files:
            print(f"processing {file}")
            file_path = os.path.join(base_path, f"{file}.csv")
            # check if file exists
            if os.path.exists(file_path):
                # Load file
                df_clean = pd.read_csv(file_path)

                # Check if 'alignment_score' column exists
                if 'alignment_score' in df_clean.columns:
                    print(f"---- File {file} exists, skipping ----")
                else:
                    print(f"---- File {file} exists, but alignment_score column is missing ----")
                    df_clean = self.prepare_ngram_df_(os.path.join(base_path, file), **kwargs)
                    df_clean.to_csv(file_path, index=False)
            else:
                print(f"---- File {file} does not exist ----")
                df_clean = self.prepare_ngram_df_(os.path.join(base_path, file), **kwargs)
                df_clean.to_csv(file_path, index=False)
            print(f"df_clean {df_clean.columns}")
            print(os.path.join(base_path, f'{file}.csv'))
            task = remove_string(file.split("/")[-1].split(".")[0])
            task_dfs[task] = df_clean
            # export
            df_clean.to_csv(os.path.join(base_path, f"{file}.csv"), index=False)
        
        with open(save_pth, "wb") as f:
            pickle.dump(task_dfs, f)
        
        print(f"saved file: {save_pth}")

        return task_dfs, save_pth

    def load_all(self, ext="all.pkl.csv"):
        files = [file for file in os.listdir(self.base_path_all) if file.endswith(ext)]
        n_gram_freqs_all = {}
        for file in files:
            print(f"Loading {file}")
            with open(os.path.join(self.base_path_all, file), "rb") as f:
                lang_pair = file.split(".")[0].split("-")
                lang_pair = "-".join(lang_pair[:2])
                n_gram_freqs_all[lang_pair] = pd.read_csv(f)
        return n_gram_freqs_all

    def plot_n_samples(self, df_dict, color_map):
        # Prepare data for plotting
        langs = []
        total_samples = []
        for lang, df in df_dict.items():
            langs.append(lang)
            total_samples.append(df['value'].sum())

        # Create a DataFrame for easier plotting
        plot_df = pd.DataFrame({
            'Language': langs,
            'Total Samples': total_samples
        })

        # Sort the DataFrame by total samples
        plot_df = plot_df.sort_values('Total Samples', ascending=False)

        # Create a color list corresponding to languages
        colors = [color_map[lang] for lang in plot_df['Language']]

        # Create the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(plot_df['Language'], plot_df['Total Samples'], color=colors)
        plt.xlabel('Language')
        plt.ylabel('Total Samples')
        plt.title('Total Samples for Each Language')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        plt.savefig(f"{self.plot_path}/total_samples.png")
        plt.show()


    def plot_distribution(self, df, lang, ax=None, color=None): 
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(df['value'], bins=50, log=True, color=color)
        ax.set_title(f'{lang}', fontsize=20)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Count')
        plt.savefig(f"{self.plot_path}/{lang}_hist.png")
        if ax is None:
            plt.show()

    def plot_cumulative_distribution(self, df, lang, ax=None, color=None):
        if ax is None:
            fig, ax = plt.subplots()
        values_list_sorted = np.sort(df['value'])
        cumulative_frequencies = np.cumsum(values_list_sorted)
        cdf = cumulative_frequencies / float(cumulative_frequencies[-1])
        ax.scatter(values_list_sorted, cdf, color=color)
        ax.plot(values_list_sorted, cdf, color=color)
        ax.set_title(f'{lang}', fontsize=20)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Cumulative Distribution')
        plt.savefig(f"{self.plot_path}/{lang}.png")
        if ax is None:
            plt.show()

    def analyze_and_plot_distributions(self, lang_dfs):
        # Create a color palette
        colors = sns.color_palette('hls', len(lang_dfs))

        # Create a figure for the plots with subplots for each language
        fig, axs = plt.subplots(2, len(lang_dfs), figsize=(5.5*len(lang_dfs), 10))  # 2 rows for distribution and cumulative distribution

        # Plot the distributions and cumulative distributions
        for i, (lang, df) in enumerate(lang_dfs.items()):
            self.plot_distribution(df, lang, axs[0, i], color=colors[i])  # Row 0 for distribution
            self.plot_cumulative_distribution(df, lang, axs[1, i], color=colors[i])  # Row 1 for cumulative distribution

        # Set the figure name and save the figure
        fig_name = os.path.join(self.plot_path, "combined_distributions")
        plt.savefig(fig_name + ".png")
        plt.savefig(fig_name + ".pdf")
        print(f"Saved figure to {fig_name}.png")

        # Adjust the layout and show the plots
        plt.tight_layout()
        plt.show()


    def plot_bleu_scores(self, bleu_scores, models, color_mapping):
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = []

        for dataset in bleu_scores:
            scores = [bleu_scores[dataset][model] for model in models]
            bar = ax.bar(models, scores, color=color_mapping[dataset], alpha=0.5)
            bars.append(bar)

        ax.set_xlabel('Models')
        ax.set_ylabel('BLEU Score')
        ax.set_title('BLEU Scores for Datasets')
        # ax.legend(bars, bleu_scores.keys(), title="Datasets")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_scores(self, model_scores, dataset_scores,
                    color_mapping, 
                    model_color_mapping,
                    x_key='n_samples', name=None,
                    log_axis=False,
                    fit_polynomial=None):
        plt.figure(figsize=(6, 4.5))
        texts = []
        linestyle = "-"
        fontsize = 14

        plt.rcParams.update({'font.size': fontsize})

        for model in model_scores:
            # Sort the data by x-values
            sorted_indices = np.argsort(model_scores[model][x_key])
            x_values = np.array(model_scores[model][x_key])[sorted_indices]
            y_values = np.array(model_scores[model]['score'])[sorted_indices]

            if fit_polynomial:
                z = np.polyfit(x_values, y_values, fit_polynomial)
                p = np.poly1d(z)
                plt.plot(x_values, p(x_values), 
                        color=model_color_mapping[model], 
                        label=model.replace('pythia-', ''))
            else:
                plt.plot(x_values, y_values, 
                        color=model_color_mapping[model], 
                        label=model.replace('pythia-', ''),
                        linestyle=linestyle)  # Increase line width for better visibility
            # texts.append(plt.text(x_values[-1], y_values[-1], model, ha='left', va='bottom', fontsize=fontsize))
        
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
        plt.xlabel('# N-grams (n)', fontsize=fontsize)
        plt.ylabel('BLEU Score', fontsize=fontsize)

        # set min x value to 0
        # plt.xlim(left=0)

        # Custom legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # for handle in handles:
        #     handle.set_linewidth(3)  # Make lines thicker
        #     handle.set_markersize(8)  # Make markers larger


        for dataset in dataset_scores:
            foreign_lang = dataset.split("-")[0]
            plt.scatter(dataset_scores[dataset][x_key], dataset_scores[dataset]['score'], 
                        color=color_mapping[dataset], 
                        # label=dataset, 
                        s=50, 
                        marker=LANG_MARKERS.get(foreign_lang, 'o'),
                        zorder=10)  # Default to 'o' if not found

        # plt.legend(handles, labels, bbox_to_anchor=(1.2, 1), loc='upper right', fontsize=fontsize)

        # if name is not None:
        #     plt.title(name, fontsize=fontsize)
        if log_axis:
            plt.xscale('log')
        
        plt.tight_layout()
        fig_name = os.path.join(self.plot_path, f"lang_scores_{name.replace(' ', '_')}")
        plt.savefig(fig_name + ".png")
        plt.savefig(fig_name + ".pdf")
        print(f"Saved figure to {fig_name}.png")

    # def plot_scores(self, model_scores, dataset_scores,
    #                 color_mapping, 
    #                 model_color_mapping,
    #                 x_key='n_samples', name=None,
    #                 log_axis=False,
    #                 fit_polynomial=None,
    #                 ax=None, y_limits=None,
    #                 show_legend=True, show_xlabel=True,
    #                 show_ylabel=True):
    #     if ax is None:
    #         plt.figure(figsize=(9.5, 7.5))
    #         texts = []
    #     else:
    #         texts = []

    #     for model in model_scores:
    #         # Sort the data by x-values
    #         sorted_indices = np.argsort(model_scores[model][x_key])
    #         x_values = np.array(model_scores[model][x_key])[sorted_indices]
    #         y_values = np.array(model_scores[model]['score'])[sorted_indices]

    #         if fit_polynomial:
    #             z = np.polyfit(x_values, y_values, fit_polynomial)
    #             p = np.poly1d(z)
    #             if ax is None:
    #                 plt.plot(x_values, p(x_values), color=model_color_mapping[model], label=model)
    #             else:
    #                 ax.plot(x_values, p(x_values), color=model_color_mapping[model], label=model)
    #         else:
    #             if ax is None:
    #                 plt.plot(x_values, y_values, 
    #                          color=model_color_mapping[model], label=model,
    #                          linestyle='--', linewidth=2)
    #             else:
    #                 ax.plot(x_values, y_values, 
    #                         color=model_color_mapping[model], label=model,
    #                         linestyle='--')
    #         if ax is None:
    #             texts.append(plt.text(x_values[-1], y_values[-1], model, ha='left', va='bottom', fontsize=fontsize))
    #         else:
    #             pass
    #             # texts.append(ax.text(x_values[-1], y_values[-1], model, ha='left', va='bottom', fontsize=12))

    #     markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h', 'x', '+']
    #     for i, dataset in enumerate(dataset_scores):
    #         if ax is None:
    #             plt.scatter(dataset_scores[dataset][x_key], dataset_scores[dataset]['score'], 
    #                     color=color_mapping[dataset], label=dataset, s=125, marker=markers[i % len(markers)])
    #         else:
    #             ax.scatter(dataset_scores[dataset][x_key], dataset_scores[dataset]['score'], 
    #                 color=color_mapping[dataset], label=dataset, s=100, linestyle='--', marker=markers[i % len(markers)])

    #     if ax is None:
    #         adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    #         plt.xlabel(x_key, fontsize=15)
    #         plt.ylabel('BLEU Score', fontsize=15)
    #         plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    #         if name is not None:
    #             plt.title(name)
    #         if log_axis:
    #             plt.xscale('log')
    #         fig_name = os.path.join(self.plot_path, f"lang_scores_{name.replace(' ', '_')}")
    #         plt.savefig(fig_name + ".png")
    #         plt.savefig(fig_name + ".pdf")
    #         print(f"Saved figure to {fig_name}.png")
    #     else:
    #         if show_xlabel:
    #             ax.set_xlabel(x_key, fontsize=15)
    #         if show_ylabel:
    #             ax.set_ylabel('BLEU Score', fontsize=15)
    #         if show_legend:
    #             ax.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
    #         if name is not None:
    #             ax.set_title(name)
    #         if log_axis:
    #             ax.set_xscale('log')
        
    #     if y_limits is not None:
    #         if ax is None:
    #             plt.ylim(y_limits)
    #         else:
    #             ax.set_ylim(y_limits)
    
    def plot_average_scores(self, dataset_scores, model_scores, color_mapping, 
                       model_list=None, x_key='n_samples',
                       log_axis=False, title=None, annotate=False, show_legend=True):
        avg_dataset_scores = collections.defaultdict(lambda: collections.defaultdict(list))
        avg_scores = {'score': [], x_key: []} 

        for dataset in dataset_scores:
            if model_list is None:
                avg_dataset_scores[dataset]['score'] = np.mean(dataset_scores[dataset]['score'])
                avg_dataset_scores[dataset][x_key] = np.mean(dataset_scores[dataset][x_key])
            else:
                model_indices = [i for i, model in enumerate(model_scores.keys()) if model in model_list]
                avg_dataset_scores[dataset]['score'] = np.mean([dataset_scores[dataset]['score'][i] for i in model_indices])
                avg_dataset_scores[dataset][x_key] = np.mean([dataset_scores[dataset][x_key][i] for i in model_indices])

            plt.scatter(avg_dataset_scores[dataset][x_key], avg_dataset_scores[dataset]['score'], 
                        color=color_mapping[dataset], label=dataset)
            if annotate:
                plt.annotate(dataset, (avg_dataset_scores[dataset][x_key], avg_dataset_scores[dataset]['score']))
            avg_scores['score'].append(avg_dataset_scores[dataset]['score'])
            avg_scores[x_key].append(avg_dataset_scores[dataset][x_key])

        # sort by n_samples
        avg_scores['score'] = [x for _, x in sorted(zip(avg_scores[x_key], avg_scores['score']))]
        avg_scores[x_key] = sorted(avg_scores[x_key])
        
        plt.plot(avg_scores[x_key], avg_scores['score'], color='red', label='large models')
        plt.xlabel(x_key)
        plt.ylabel('score')
        if title is not None:
            plt.title(title)
        if log_axis:
            plt.xscale('log')

        if title is not None:    
            title_path = title.replace(' ', '_').replace(',', '_')
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
            plt.savefig(os.path.join(self.plot_path, f"avg_scores_{title_path}.png"))
        
        plt.show()

        return avg_scores
    
    def plot_average_scores_plotly(self, dataset_scores, model_scores, color_mapping, 
                                model_list=None, x_key='n_samples',
                                log_axis=False, title=None):
        avg_dataset_scores = collections.defaultdict(lambda: collections.defaultdict(list))
        avg_scores = {'score': [], x_key: []} 

        # Create Plotly figure
        fig = go.Figure()

        for dataset in dataset_scores:
            if model_list is None:
                avg_dataset_scores[dataset]['score'] = np.mean(dataset_scores[dataset]['score'])
                avg_dataset_scores[dataset][x_key] = np.mean(dataset_scores[dataset][x_key])
            else:
                model_indices = [i for i, model in enumerate(model_scores.keys()) if model in model_list]
                avg_dataset_scores[dataset]['score'] = np.mean([dataset_scores[dataset]['score'][i] for i in model_indices])
                avg_dataset_scores[dataset][x_key] = np.mean([dataset_scores[dataset][x_key][i] for i in model_indices])

            # Append data for plotting
            avg_scores['score'].append(avg_dataset_scores[dataset]['score'])
            avg_scores[x_key].append(avg_dataset_scores[dataset][x_key])
            
            # Add trace for each dataset to the figure
            fig.add_trace(go.Scatter(
                x=[avg_dataset_scores[dataset][x_key]],
                y=[avg_dataset_scores[dataset]['score']],
                text=[dataset],  # Will show this text on hover
                mode='markers',
                marker=dict(color=color_mapping[dataset]),
                name=dataset
            ))

        # sort by x_key (e.g., 'n_samples')
        sorted_indices = np.argsort(avg_scores[x_key])
        sorted_scores = np.array(avg_scores['score'])[sorted_indices]
        sorted_x_values = np.array(avg_scores[x_key])[sorted_indices]

        # Add sorted average line to the figure
        fig.add_trace(go.Scatter(
            x=sorted_x_values,
            y=sorted_scores,
            mode='lines',
            marker=dict(color='red'),
            name='Average'
        ))

        # Set log axis if specified
        if log_axis:
            fig.update_xaxes(type='log')

        # Update layout with title and axis labels
        fig.update_layout(
            title=title,
            xaxis_title=x_key,
            yaxis_title='Score',
            hovermode='closest'
        )

        # disable legend
        fig.update_layout(showlegend=False)

        # Show figure
        fig.show()

        return avg_scores

    def plot_model_size_vs_scores(self, results_dict, models, 
                                  model_param_map, color_mapping):
        plt.figure(figsize=(6, 4.5))

        for dataset in results_dict:
            print(f"dataset: {dataset}")
            non_en_lang = dataset.split("-")[0]
            scores = [results_dict[dataset][model] for model in models]
            plt.scatter([model_param_map[model] for model in models], 
                        scores, color=color_mapping[dataset], label=dataset, 
                        marker=LANG_MARKERS.get(non_en_lang, 'o'),
                        s=50)
            plt.plot([model_param_map[model] for model in models], scores, 
                     color=color_mapping[dataset])
        
        fontisize = 14

        plt.axvline(x=1e09, color='black', linestyle='--')
        plt.text(1e09, 30, '1b', fontsize=fontisize)
        plt.axvline(x=410e06, color='black', linestyle='--')
        plt.text(410e06, 30, '410M', fontsize=fontisize)

        # plt.title("Model Size vs. Dataset Scores")
        plt.xlabel("Number of Parameters", fontsize=fontisize)
        plt.ylabel("BLEU Score", fontsize=fontisize)
        # set xvalues and yvalues
        plt.xticks(fontsize=fontisize)
        plt.yticks(fontsize=fontisize)
        plt.xscale('log')
        plt.legend(fontsize=fontisize-3)
        plt.tight_layout()
        fig_name = os.path.join(self.plot_path, "model_size_vs_scores")
        plt.savefig(fig_name + ".png")
        plt.savefig(fig_name + ".pdf")
        print(f"Saved figure to {fig_name}.png")

    def build_lang_ds(self, lang_pairs):
        language_pair_tuples = [tuple(lang_pair.split("-")) for lang_pair in lang_pairs]
        ds = load_ds_translation(language_pair=language_pair_tuples)
        # extract translation column
        lang_ds = {}
        for lang_pair, data in zip(lang_pairs, ds.values()):
            lang_ds[lang_pair] = data['translation']
        return lang_ds

    def calculate_bigram_frequencies(self, dataset):
        # Initialize a dictionary to hold FreqDist objects for each language
        language_freqs = {}
        nlp_multi = spacy.load('xx_ent_wiki_sm')

        # Iterate over each entry in the dataset
        for entry in dataset:
            for lang_code, text in entry.items():
                # Tokenize text
                lang = pycountry.languages.get(alpha_2=lang_code).name.lower()
                # Lazily initialize a FreqDist object for this language
                if lang_code not in language_freqs:
                    language_freqs[lang_code] = FreqDist()
                
                text = self.wt.process_text(text, lang, n_gram=2)
                text = " ".join(text)
                if lang == 'hungarian':
                    doc = nlp_multi(text.lower())
                    tokens = [token.text for token in doc]
                else:
                    tokens = word_tokenize(text.lower(), language=lang)

                # Optionally, you can remove stopwords and punctuation from the analysis
                # Note: stopwords are language-specific
                tokens = self.wt.filter_stop_words_(' '.join(tokens), lang)

                # Create bigrams and update frequency distribution for this language
                bigrams_in_text = bigrams(tokens)
                language_freqs[lang_code].update(bigrams_in_text)

        # Return the dictionary of FreqDist objects
        return language_freqs
    
    def prepare_scores(self, bleu_scores, lang_dfs, models, **kwargs):
        model_scores = defaultdict(lambda: defaultdict(list))
        dataset_scores = defaultdict(lambda: defaultdict(list))

        for dataset in lang_dfs:
            for model in models:
                lang_df = lang_dfs[dataset]
                lang_df = lang_df[lang_df['task'] == dataset]
                n_samples = lang_df['value'].sum()
                coverage = lang_df['coverage'].mean() if 'coverage' in lang_df else 0
                print(f"dataset: {dataset}, model: {model}, n_samples: {n_samples}, coverage: {coverage}")
                print(f"bleu scores: {bleu_scores.keys()}")
                score = bleu_scores[dataset][model]
                model_scores[model]['score'].append(score)
                model_scores[model]['n_samples'].append(n_samples)
                model_scores[model]['coverage'].append(coverage)
                dataset_scores[dataset]['score'].append(score)
                dataset_scores[dataset]['n_samples'].append(n_samples)
                dataset_scores[dataset]['coverage'].append(coverage)

                for kwarg, value in kwargs.items():
                    value = value[dataset]
                    model_scores[model][kwarg].append(value)
                    dataset_scores[dataset][kwarg].append(value)

        for model in model_scores:
            model_score = model_scores[model]
            # Create a sorted index based on n_samples
            sorted_index = sorted(range(len(model_score['n_samples'])), key=lambda k: model_score['n_samples'][k])
            model_scores[model]['score'] = [model_score['score'][i] for i in sorted_index]
            model_scores[model]['n_samples'] = [model_score['n_samples'][i] for i in sorted_index]
            model_scores[model]['coverage'] = [model_score['coverage'][i] for i in sorted_index]

            for kwarg, value in kwargs.items():
                model_scores[model][kwarg] = [model_score[kwarg][i] for i in sorted_index]

        return model_scores, dataset_scores


    def df_to_bigram_freqs(self, df):
        # Initialize a dictionary to hold FreqDist objects for each language
        language_freqs = collections.defaultdict(FreqDist)

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            # Get the frequencies
            freq = row['value']

            # Iterate over all columns except for 'value'
            for col in df.columns:
                if col != 'value' and isinstance(row[col], str):
                    # Get the bigrams
                    bigram = tuple(row[col].split())

                    # Update the frequency distribution for this language
                    language_freqs[col][bigram] += freq

        # Return the dictionary of FreqDist objects
        return language_freqs


    def calculate_mutual_information(self, lang_dfs, lang_ds, smoothing=1e-5):
        """
        Calculate mutual information between language distributions.
        
        Args:
            lang_dfs: Dictionary of DataFrames containing language pair data
            lang_ds: Dictionary of language datasets
            smoothing: Small constant for smoothing probabilities
        """
        # Extract frequency values from DataFrames
        lang_frequencies = {}
        for lang, df in lang_dfs.items():
            lang_frequencies[lang] = df['value'].values
        
        # Calculate pairwise mutual information
        lang_mutual_info = {}
        for lang1 in lang_frequencies:
            lang_mutual_info[lang1] = {}
            for lang2 in lang_frequencies:
                if lang1 != lang2:
                    # Get frequency distributions
                    dist1 = lang_frequencies[lang1]
                    dist2 = lang_frequencies[lang2]
                    
                    # Ensure arrays are of same length by padding shorter one
                    max_len = max(len(dist1), len(dist2))
                    dist1_padded = np.pad(dist1, (0, max_len - len(dist1)), 'constant', constant_values=smoothing)
                    dist2_padded = np.pad(dist2, (0, max_len - len(dist2)), 'constant', constant_values=smoothing)
                    
                    # Apply smoothing
                    dist1_smoothed = dist1_padded + smoothing
                    dist2_smoothed = dist2_padded + smoothing
                    
                    # Convert to probability distributions
                    prob_dist1 = dist1_smoothed / np.sum(dist1_smoothed)
                    prob_dist2 = dist2_smoothed / np.sum(dist2_smoothed)
                    
                    # Calculate KL divergence
                    mutual_info = entropy(prob_dist1, prob_dist2)
                    lang_mutual_info[lang1][lang2] = mutual_info
        
        # Calculate merged mutual information
        merged_mutual_info = {}
        for lang in lang_frequencies:
            if lang in lang_mutual_info:
                merged_mutual_info[lang] = np.mean(list(lang_mutual_info[lang].values()))
        
        return lang_mutual_info, merged_mutual_info


    def merge_distributions(self, distribution_dict):
        # Initialize a new FreqDist object to hold the merged distribution
        merged_freqs = FreqDist()

        # Iterate over each language in the distribution dictionary
        for lang, freqs in distribution_dict.items():
            # Add the frequencies of the current language to the merged distribution
            merged_freqs += freqs

        # Return the merged distribution
        return merged_freqs


    def align_dists(self, freqs1, freqs2):
        """
        function to align the two freq dicts to have the same keys
        e.g if one has bigrams that are not in the other, fill with 0
        """
        keys1 = set(freqs1.keys())
        keys2 = set(freqs2.keys())
        keys = keys1.union(keys2)
        for key in keys:
            if key not in freqs1:
                freqs1[key] = 0
            if key not in freqs2:
                freqs2[key] = 0
        # align keys according to keys variable above
        freqs1 = {k: freqs1[k] for k in keys}
        freqs2 = {k: freqs2[k] for k in keys}
        return freqs1, freqs2


    def calculate_mutual_info(self, lang_dfs, lang_ds):
        lang_mutual_info = {}
        merged_mutual_info = {}

        for lang_pair in lang_dfs.keys():
            lang_key = lang_pair.split("-")[0]
            # if 'hu' in lang_pair:
                # continue
            lang_df = lang_dfs[lang_pair]
            lang_pair_ds = lang_ds[lang_pair]
            gram_freqs_data = self.calculate_bigram_frequencies(lang_pair_ds)
            gram_freqs_corpus = self.df_to_bigram_freqs(lang_df)
            for key in gram_freqs_data.keys():
                gram_freqs_data[key], gram_freqs_corpus[key] = self.align_dists(gram_freqs_data[key], gram_freqs_corpus[key])
            mutual_info = self.calculate_mutual_information(gram_freqs_corpus[lang_key], gram_freqs_data[lang_key])
            lang_mutual_info[lang_pair] = mutual_info
            merged_grams_corpus = self.merge_distributions(gram_freqs_corpus)
            merged_grams_data = self.merge_distributions(gram_freqs_data)
            merged_grams_data, merged_grams_corpus = self.align_dists(merged_grams_data, merged_grams_corpus)
            mutual_info = self.calculate_mutual_information(merged_grams_corpus, merged_grams_data)
            merged_mutual_info[lang_pair] = mutual_info

        return lang_mutual_info, merged_mutual_info

    def calculate_mauve_score(self, ds_1, ds_2):
        out = mauve.compute_mauve(p_text=ds_1, q_text=ds_2, 
                                device_id=0, max_text_length=512,
                                batch_size=32, verbose=True, 
                                featurize_model_name='gpt2-large')
        return out['mauve_score']


def post_filter(df):
    # Extract the 'gold' label from the 'example' column if it exists
    if 'example' in df.columns:
        df['gold'] = df['example'].apply(lambda x: x.get('answer', None))
        if 'answer' in df.iloc[0]['example']:
                df['answer'] = df['example'].apply(lambda x: x['choices'][x['answer']])

    # # Drop rows where 'A' does not match the 'gold' label
    # df = df[df['A'] == df['gold']]

    # Calculate predicted labels and accuracy
    if 'probs' in df.columns:
        df['predicted'] = df['probs'].apply(lambda x: np.argmax(x))
    elif 'probs_nll' in df.columns:
        df['predicted'] = df['probs_nll'].apply(lambda x: np.argmax(x))

    if 'result' in df.columns:
        df['accuracy'] = df['result']
    else:
        df['accuracy'] = (df['predicted'] == df['gold']).astype(int)

    # Calculate softmax probabilities if not already present
    if 'probs_softmax' not in df.columns and 'probs' in df.columns:
        df['probs_softmax'] = df['probs'].apply(lambda x: softmax(x))
        df['probs_softmax_gold'] = df.apply(lambda row: row['probs_softmax'][row['gold']], axis=1)

    # Merge task accuracy and model task accuracy
    task_accuracy = df.groupby('task')['accuracy'].mean().reset_index(name='ds_score')
    model_task_accuracy = df.groupby(['model', 'task'])['accuracy'].mean().reset_index(name='model_ds_score')
    df = df.merge(task_accuracy, on='task', how='left')
    df = df.merge(model_task_accuracy, on=['model', 'task'], how='left')

    # Convert 'example' dictionaries to strings to make them hashable
    df['example_str'] = df['example'].apply(lambda x: str(x))
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop any rows where 'A' is just a number
    df = df[~df['A'].str.isnumeric()]

    return df


def filter_by_language(df, lang=None, column=None):
    column = lang if column is None else column
    # Define a mapping from ISO 639-1 language codes to langdetect language codes
    lang_map = {
        'ja': 'ja',
        'ru': 'ru',
        'zh': 'zh-cn',
        'fr': 'fr',
        'ro': 'ro',
        'de': 'de',
        'cs': 'cs',
        'pl': 'pl',
        'hu': 'hu',
        'it': 'it',
        'es': 'es',
    }

    # Filter the DataFrame
    def detect_lang(row):
        # Get the langdetect language code
        detect_lang = lang if lang is not None else row['task'].split('-')[0]
        langdetect_code = lang_map.get(detect_lang, detect_lang)  # Default to the same code if not found in map
        text_to_detect = row[column] if column is not None else row[detect_lang]  # Adjust this line based on your DataFrame structure

        try:
            return detect(text_to_detect) == langdetect_code
        except LangDetectException:
            return False

    # Initialize progress bar
    pbar = tqdm(total=len(df), desc=f"Detecting {lang} language")

    # Apply the detect_lang function to each row in the DataFrame with progress bar update
    def apply_and_update_progress(row):
        result = detect_lang(row)
        pbar.update(1)
        return result

    df_filtered = df[df.apply(apply_and_update_progress, axis=1)]

    # Close progress bar after processing is complete
    pbar.close()

    print(f"ori: {len(df)}, \
            filtered: {len(df_filtered)}, \
            diff: {len(df) - len(df_filtered)}")

    return df_filtered


def display_language_pairs(lang_dfs_dict, language_pairs=None, rows=500):
    pd.options.display.min_rows = rows
    language_pairs = lang_dfs_dict.keys() if language_pairs is None else language_pairs

    # Iterate over the language pairs and display the top rows for each
    for lang_pair in language_pairs:
        print(f"{lang_pair.replace('-', ' - ')}")
        display(lang_dfs_dict[lang_pair].head(rows))
        print("\n")  # Add a newline for better readability between displays


# def align_pairs(df):

def align_pairs_e5(text_1, text_2, threshold=0.7, gpu=None):
    """
    Align pairs using e5 embeddings
    text_1, text_2: lists of texts to be aligned
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() \
                            and gpu is not None else "cpu")

    if not isinstance(text_1, list):
        text_1 = [text_1]
    if not isinstance(text_2, list):
        text_2 = [text_2]

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    with torch.no_grad():

        querries = ['query: ' + query for query in text_1]
        passages = ['passage: ' + passage for passage in text_2]

        input_text = querries + passages

        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device)
        model.eval()

        batch = tokenizer(input_text, max_length=512, padding=True, 
                          truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        num_pairs = min(len(querries), len(passages))

        scores = (embeddings[:num_pairs] @ embeddings[num_pairs:].T)
        corresponding_scores = scores.diag().cpu().detach().numpy() * 100  # Extract the diagonal and scale by 100

    return corresponding_scores


class ProcessTriviaQA:

    @staticmethod
    def check_exact_match(row):
        if pd.isna(row['result']) or pd.isna(row['answer']):
            return False
        aliases = row['answer'].get('aliases', [])
        normalized_result = normalize_string(row['result'])
        return int(any(normalize_string(alias) == normalized_result for alias in aliases))

    @staticmethod
    def process_model_examples(df_dict, p=0.9999):
        # df_dict = {model: data.iloc[:10] for model, data in df_dict.items()}
        df_dict = {model: filter_percentile(data, p) for model, data in df_dict.items()}
        df_dict = {model: remove_nested_lists(data) for model, data in df_dict.items()}
        df_dict = {model: ProcessTriviaQA.create_exact_match_column(data) for model, data in df_dict.items()}
        df_0 = df_dict[list(df_dict.keys())[0]]
        # df_aligned = ProcessTriviaQA.calculate_alignment_scores(df_0, gpu=3)
        # # df_dict = {model: data.iloc[keep_indices] for model, data in df_dict.items()}
        # df_dict = {model: data.assign(align_score=df_aligned['align_score']) for model, data in df_dict.items()}
        return df_dict

    @staticmethod
    def create_exact_match_column(examples):
        examples['em'] = examples.progress_apply(ProcessTriviaQA.check_exact_match, axis=1)
        return examples
    
    @staticmethod
    def calculate_alignment_scores(df, batch_size=256, gpu=None,
                                   col_1='Q', col_2='question'):
        """
        Calculate the alignment scores between 'Q' and 'Question' columns and add them as a new column 'align_score'.
        Drop rows where the alignment score falls below the threshold.
        
        :param df: pandas DataFrame containing the data
        :param threshold: minimum alignment score required:param batch_size: number of pairs to process in each batch (default: 32)
        :return: tuple containing the modified DataFrame with 'align_score' column and the indices of the kept rows
        """
        from tqdm import tqdm  # Import tqdm for progress bar functionality

        # Get the total number of rows
        num_rows = len(df)

        # Initialize an array to store the alignment scores
        alignment_scores = np.zeros(num_rows)

        # Initialize a progress bar
        pbar = tqdm(total=num_rows, desc='Calculating alignment scores')

        # Process the rows in batches
        for i in range(0, num_rows, batch_size):
            # Update the progress bar with the number of rows processed
            pbar.update(min(batch_size, num_rows - i))

            # Get the batch of 'Q' and 'Question' columns
            batch_text_1 = df[col_1][i:i+batch_size].tolist()
            batch_text_2 = df[col_2][i:i+batch_size].tolist()

            # Calculate the alignment scores for the batch
            batch_scores = align_pairs_e5(batch_text_1, batch_text_2, gpu=gpu)

            # Store the batch scores in the alignment_scores array
            alignment_scores[i:i+batch_size] = batch_scores

        # Close the progress bar
        pbar.close()

        # Add the alignment scores as a new column 'align_score' to the DataFrame
        df['align_score'] = alignment_scores

        # Create a boolean mask indicating which rows to keep
        # print(f"Threshold is: {threshold}")
        # keep_rows = df['align_score'] >= threshold
        # kept_indices = df.index[keep_rows].tolist()
        # df_filtered = df.iloc[kept_indices]

        # print the number of rows kept and dropped and percentage
        # num_kept = len
        # num_dropped = num_rows - num_kept
        # print(f"Kept {num_kept} rows and dropped {num_dropped} rows ({num_dropped / num_rows:.2%})")

        return df
    
    @staticmethod
    def get_random_samples_by_interval(df, sample_size, column='interval'):
        # Remove rows with missing column values
        df = df.dropna(subset=[column])

        # Convert column to string type
        df[column] = df[column].astype(str)

        # Create an empty list to store the random samples
        samples_list = []

        # Get the unique log intervals
        intervals = df[column].unique()

        # Iterate over each interval
        for interval in intervals:
            # Get the rows corresponding to the current interval
            interval_df = df[df[column] == interval]

            # Check if the interval has enough samples
            if len(interval_df) >= sample_size:
                # Select random samples from the interval
                samples = interval_df.sample(n=sample_size)
                samples_list.append(samples)
            else:
                print(f"Interval {interval} does not have enough samples. Skipping.")

        # Concatenate the random samples into a single DataFrame
        samples_df = pd.concat(samples_list, ignore_index=True)

        return samples_df
    
    @staticmethod
    def plot_variable_by_interval(df, x_column, y_column, 
                                  log_axis=False, save_pth=None, 
                                  ax=None, color=None, label=None,
                                  title=None, x_label=None,
                                  plot_scatter=False, plot_std=False,
                                  xlimits=None, ylimits=None):
        
        # Check if the columns exist in the DataFrame
        if x_column not in df.columns or y_column not in df.columns:
            raise ValueError(f"Columns {x_column} and/or {y_column} not found in DataFrame")
        
        # set all fonts to 14
        plt.rcParams.update({'font.size': 14})

        # Remove rows with missing values in the specified columns
        df = df.dropna(subset=[x_column, y_column])

        # Convert the x_column to string type if not already
        df[x_column] = df[x_column].astype(str)
        
        # color_map = Colors.model_colormap
        # model_2_color = {model: color_map(i - 1 - i / len(PYTHIA_MODEL_NAMES)) for i, model in reversed(enumerate(PYTHIA_MODEL_NAMES))}

        # Extract the numeric value for the x_column if it's an interval
        if any(char in df[x_column].iloc[0] for char in '[]()'):
            df['interval_start'] = df[x_column].apply(lambda x: float(x.split(',')[0].strip('[](-∞ ')))
            x_values = 'interval_start'
        else:
            x_values = x_column

        # Group by the x_column and calculate the mean for the y_column
        avg_by_x = df.groupby(x_column)[y_column].mean()

        # Create a new DataFrame with x_values and the average of y_column
        avg_df = pd.DataFrame({x_values: df.groupby(x_column)[x_values].first(),
                               'avg_y': avg_by_x})

        # Sort the DataFrame based on x_values
        avg_df = avg_df.sort_values(x_values)

        print(f"avg_df = {avg_df}")
        print(f"avg_y: {avg_df['avg_y']}")

        # Create a new figure and axis if ax is not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))

        # Create a line plot on the provided or new axis
        ax.plot(avg_df[x_values], avg_df['avg_y'], marker='o', color=color, label=label)
        ax.set_xlabel(x_column)
        ax.set_ylabel(f'Average {y_column}')
        # ax.set_title(f'Average {y_column} by {x_column}')

        # Optionally plot standard deviation
        if plot_std:
            # Group by the x_column and calculate the standard deviation for the y_column
            std_by_x = df.groupby(x_column)[y_column].std()

            # Assuming avg_df is already sorted by x_values
            # Plot the standard deviation as a shaded area around the mean line
            ax.fill_between(avg_df[x_values], 
                            avg_df['avg_y'] - std_by_x.values, 
                            avg_df['avg_y'] + std_by_x.values, 
                            color=color, alpha=0.1)

        # optionally plot the individual points
        if plot_scatter:
            ax.scatter(df[x_column], df[y_column], color=color, alpha=0.25)
        
        if xlimits is not None:
            ax.set_xlim(xlimits)
        
        if ylimits is not None:
            ax.set_ylim(ylimits)

        if log_axis:
            ax.set_xscale('log')
        
        # if title is not None:
        #     ax.set_title(title)
        
        if x_label is None:
            ax.set_xlabel('N-grams (n)')
        else:
            ax.set_xlabel(x_label)
        ax.set_ylabel('Accuracy (p)')

        # # Make the legend flat and position it at the bottom
        ncol = len(ax.lines) // 6
        # ax.legend(loc='upper left', 
        #           ncol=ncol if ncol >= 1 else 1)
        bbox_to_anchor=(0.5, -0.15),

        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0.15, 1, 1])  # Adjust the bottom space for the legend

        if save_pth is not None:
            os.makedirs(os.path.dirname(save_pth), exist_ok=True)
            plt.savefig(save_pth, bbox_inches='tight')  # Save the figure with the legend
            print(f"Saved figure in {save_pth}")

        # Return the axis object
        return ax
    


# %%
# from rapidfuzz import process, fuzz

# # Create a new column for the id
# example_dfs_all['id'] = None
# instances = pd.DataFrame(model_instance_results['pythia-12b'])

# for instance in tqdm(instances):
#     # Skip if this instance has already been matched
#     # if example_dfs_all.loc[example_dfs_all['query'] == instance['query'], 'id'].notna().any():
#         # continue

#     # Get the best match in example_dfs_all['query']
#     best_match, score, _ = process.extractOne(instance['query'], example_dfs_all['query'], scorer=fuzz.token_sort_ratio)
    
#     # If the score is above a certain threshold, assign the id
#     example_dfs_all.loc[example_dfs_all['query'] == best_match, 'id'] = instance['id']

# # Save the DataFrame to a CSV file in BASE_PATH
# example_dfs_all.to_csv(os.path.join(BASE_PATH, 'example_dfs_all_.csv'), index=False)

# # instances = model_instance_results['pythia-12b']
# # for row, example in enumerate(tqdm(example_dfs_all['query'])):
# #     for instance in instances:
# #         if example == instance['query']:
# #             print(f"Found match for {example}")
# #             example_dfs_all.loc[row, 'id'] = instance['id']
# #             instances.remove(instance)
# #             break

# %%
# import pandas as pd
# import os
# from tqdm import tqdm

# def normalize_text(text):
#     # Convert to lowercase, strip whitespace, remove punctuation, etc.
#     return text.lower().strip()

# def match_and_update_ids(example_dfs, model_instance_results, filename, key='query'):
#     # Convert model_instance_results to a DataFrame
#     instances = pd.DataFrame(model_instance_results)

#     # Normalize the 'query' column in both DataFrames
#     example_dfs[key] = example_dfs[key].apply(normalize_text)
#     instances[key] = instances[key].apply(normalize_text)

#     print(f"example_dfs: {example_dfs.head(1)}")
#     print(f"instances: {instances.head(1)}")
#     # Merge the example_dfs with instances on 'query' column
#     merged_df = example_dfs.merge(instances[['id', key]], on=key, how='left')

#     # Save the merged DataFrame to a CSV file if it doesn't exist
#     if not os.path.exists(filename):
#         merged_df.to_csv(filename, index=False)
    
#     return merged_df

# # Assuming BASE_PATH is defined
# # Process all tasks
# example_dfs_all_filename = os.path.join(BASE_PATH, 'example_dfs_all_exact.csv')
# example_dfs_all = match_and_update_ids(example_dfs_all, model_instance_results_all['pythia-12b'], example_dfs_all_filename)

# # Process common tasks
# example_dfs_common_filename = os.path.join(BASE_PATH, 'example_dfs_common_exact.csv')
# example_dfs_common = match_and_update_ids(example_dfs_common, model_instance_results_common['pythia-12b'], example_dfs_common_filename)
# # Process all tasks
# example_dfs_all_filename = os.path.join(BASE_PATH, 'example_dfs_all_exact.csv')
# example_dfs_all = match_and_update_ids(example_dfs_all, model_instance_results_all['pythia-12b'], example_dfs_all_filename)

# # Process common tasks
# example_dfs_common_filename = os.path.join(BASE_PATH, 'example_dfs_common_exact.csv')
# example_dfs_common = match_and_update_ids(example_dfs_common, model_instance_results_common['pythia-12b'], example_dfs_common_filename)
# # Assuming BASE_PATH is defined
# # Process all tasks
# example_dfs_all_filename = os.path.join(BASE_PATH, 'example_dfs_all_exact.csv')
# example_dfs_all = match_and_update_ids(example_dfs_all, model_instance_results_all['pythia-12b'], example_dfs_all_filename)

# # Process common tasks
# example_dfs_common_filename = os.path.join(BASE_PATH, 'example_dfs_common_exact.csv')
# example_dfs_common = match_and_update_ids(example_dfs_common, model_instance_results_common['pythia-12b'], example_dfs_common_filename)