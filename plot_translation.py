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

# %%
# lang params
N_GRAMS = 1
BASE_DIR = f"./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue"
LANG_DF_PTH = "./results/n-grams/exp_full/2/examples_dfs_0-shot_common_models.pkl"
# BASE_DIR = f"./results/n-grams/exp_full/"

BASE_PATH = os.path.join(BASE_DIR, str(N_GRAMS))
BASE_PATH_COMMON = os.path.join(BASE_PATH, "common")
BASE_PATH_ALL = os.path.join(BASE_PATH, "all")
FILTER_CHARS = False
DETECT_LANG = False
PERCENTILE = 0.1

# model params
base_results_path = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_3/inference/EleutherAI"
models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b', 'pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m',]

# wmt09
LANGUAGES = ['wmt09-cs-en', 'wmt09-de-en', 'wmt09-fr-en', 'wmt09-es-en', 'wmt09-it-en', 'wmt09-hu-en']
TASKS = [task.replace('wmt09-', '') for task in LANGUAGES]


wa = WimbdAnalysis(BASE_PATH, LANGUAGES, N_GRAMS, FILTER_CHARS)
wt = WimbdTasks()


LANGUAGES_STR = "_".join(LANGUAGES)
PLOT_PATH = f"./results/n-grams/exp_full/{N_GRAMS}/plots/{LANGUAGES_STR}_filter_{FILTER_CHARS}"
if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

# %%
# lang_dfs_all_filtered = wa.get_lang_dfs(BASE_PATH_ALL, LANGUAGES, filter_chars=True,
#                                         percentile=0.95, detect_lang=False, filter_entities=True)
lang_dfs = pickle.load(open(LANG_DF_PTH, 'rb'))

# get total samples per language pair
lang_dfs_total_samples = {lang: df['value'].sum() for lang, df in lang_dfs.items()}
colors = sns.color_palette('hls', len(TASKS))
color_mapping = {lang: color for lang, color in zip(TASKS, colors)}
# %%
"""
This block is used to display
the language pairs that are available
"""
# for lang, df in lang_dfs_all.items():
#     print(f"{lang}: {len(df)}")

# # display top 100 of each language
# for lang, df in lang_dfs.items():
#     print(f"-------- lang: {lang} --------")
#     df = df.sort_values(by=['value'], ascending=False)
#     print(f"// top 100")
#     display(df.iloc[:100])
#     midpoint = np.median(df['value'])
#     print(f"// mid 100")
#     display(df[df['value'] <= midpoint].iloc[:100])


def get_model_results_from_df(df, metric='bleu'):
    """"
    returns a dict of structure {task: {model:{scores, nsamples}}
    """
    model2score = {}
    dataset2score = {}
    for model, model_results in df.items():
        for task in sorted(model_results['task'].unique()):
            if model not in model2score:
                model2score[model] = {'score': [], 'n_samples': []}
            if task not in dataset2score:
                dataset2score[task] = {'score': [], 'n_samples': []}
            task_df = model_results[model_results['task'] == task]
            task_score = task_df.iloc[0]['bleu']
            task_n_samples = task_df['value'].sum()
            model2score[model]['score'].append(task_score)
            model2score[model]['n_samples'].append(task_n_samples)
            dataset2score[task]['score'].append(task_score)
            dataset2score[task]['n_samples'].append(task_n_samples)
    return model2score, dataset2score

def get_task_ngrams_from_df(df):
    examples = df['pythia-12b'] # the ngram vals are same for all models
    tasks = examples['task'].unique()
    task_ngrams = {}
    for task in tasks:
        task_ngrams[task] = {'score': [], 'n_samples': []}
        task_df = examples[examples['task'] == task]
        score = task_df.iloc[0]['bleu']
        # n_samples = 
        task_ngrams[task] = task_df
    return task_ngrams


model2score, dataset2score = get_model_results_from_df(lang_dfs)
task_ngrams = get_task_ngrams_from_df(lang_dfs)

# model colormap
model_colormap = plt.cm.get_cmap('coolwarm', len(models))
model_color_mapping = {model: model_colormap(1 - i / len(models)) for i, model in enumerate(models)}

# plot
wa.plot_scores(model2score, dataset2score, color_mapping,
               model_color_mapping, name="XY_pairs")


# %%
