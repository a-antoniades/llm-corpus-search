# %%
import os
import collections
import json
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)
pd.options.display.max_rows = 100

import matplotlib.pyplot as plt
import seaborn as sns

# %%
import langid
from src.wimbd_ import WimbdAnalysis, WimbdTasks, display_language_pairs

import argparse

bigram = "bonjour"  # Replace with your bigram
language, _ = langid.classify(bigram)
print("Detected language:", language)

# %%
# pth = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue/2/all/('cs', 'en').pkl"
# with open(pth, "rb") as f:
#     data = pickle.load(f)

# %%
# lang params
# BASE_DIR = f"./results/n-grams/exp_full/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_grams", type=int, default=1)
    parser.add_argument("--base_dir", type=str, default="./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue")
    # parser.add_argument("--base_path", type=str, default=None)
    # parser.add_argument("--base_path_common", type=str, default=None)
    # parser.add_argument("--base_path_all", type=str, default=None)
    # parser.add_argument("--filter_chars", action="store_true")
    # parser.add_argument("--detect_lang", action="store_true")
    # parser.add_argument("--percentile", type=float, default=0.1)

    return parser.parse_args()

def main(args):
    N_GRAMS = args.n_grams
    BASE_DIR = args.base_dir
    BASE_PATH = os.path.join(BASE_DIR, str(N_GRAMS))
    BASE_PATH_COMMON = os.path.join(BASE_PATH, "common")
    BASE_PATH_ALL = os.path.join(BASE_PATH, "all")
    FILTER_CHARS = False
    DETECT_LANG = False
    PERCENTILE = 0.1

    # model params
    base_results_path = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_3/inference/EleutherAI"
    models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b', 'pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m',]


    # %%
    n_gram_freqs_path = "./results/n-grams/exp_full/2/common/pl-en-2-grams.pkl"
    n_gram_freqs = pickle.load(open(n_gram_freqs_path, "rb"))
    n_gram_freqs = dict(sorted(n_gram_freqs.items(), key=lambda item: item[1]['value'], reverse=True))
    # # export as text
    # # with open("n_gram_freqs.txt", "w") as f:
    # #     for k, v in n_gr
    # # 3am_freqs.items():
    # #         f.write(f"{k}: {v}\n")
    # LANGUAGES = ['ru-en', 'fr-en', 'ro-en', 'de-en', 'pl-en', 'cs-en']
    # LANGUAGES = ['ru-en', 'ro-en', 'de-en', 'pl-en', 'cs-en', 'fr-en', 'ja-en', 'zh-en']

    # wmt09
    LANGUAGES = ['wmt09-cs-en', 'wmt09-de-en', 'wmt09-fr-en', 'wmt09-es-en', 'wmt09-it-en', 'wmt09-hu-en',
                'wmt09-en-cs', 'wmt09-en-de', 'wmt09-en-fr', 'wmt09-en-es', 'wmt09-en-it', 'wmt09-en-hu',]


    wa = WimbdAnalysis(BASE_PATH, LANGUAGES, N_GRAMS, FILTER_CHARS)
    wt = WimbdTasks()


    LANGUAGES_STR = "_".join(LANGUAGES)
    PLOT_PATH = f"./results/n-grams/exp_full/{N_GRAMS}/plots/{LANGUAGES_STR}_filter_{FILTER_CHARS}"
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

    # %%
    # lang_dfs_all = wa.get_lang_dfs(BASE_PATH_ALL, LANGUAGES, filter_chars=False,
    #                               percentile=0.95)
    # lang_dfs_common = wa.get_lang_dfs(BASE_PATH_COMMON, LANGUAGES, filter_chars=FILTER_CHARS,
    #                                     percentile=0.999, n_gram=N_GRAMS)
    # lang_dfs_common_0 = wa.get_lang_dfs(BASE_PATH_COMMON, LANGUAGES, filter_chars=FILTER_CHARS,
    #                                     percentile=0.0)
    # lang_dfs_all_filtered = wa.get_lang_dfs(BASE_PATH_COMMON, LANGUAGES, filter_chars=True,
    #                                         percentile=0.99, detect_lang=False, filter_entities=False,
    #                                         align_langs=True)
    lang_dfs_all_filtered = wa.get_lang_dfs(BASE_PATH_ALL, LANGUAGES, is_all=True, filter_chars=False,
                                            percentile=0.95, detect_lang=False, filter_entities=True)
    lang_dfs_common_filtered = wa.get_lang_dfs(BASE_PATH_COMMON, LANGUAGES, filter_chars=False,
                                               percentile=0.995, detect_lang=False, filter_entities=True,
                                               align_langs=0.8)

if __name__ == "__main__":
    args = parse_args()
    for n_grams in [1, 2, 3, 4]:
        args.n_grams = n_grams
        main(args)