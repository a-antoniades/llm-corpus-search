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
from src.wimbd_ import (WimbdAnalysis, WimbdTasks,
                        display_language_pairs)
from src.wimbd_ import BasePaths as PATHS

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
    parser.add_argument("--n_grams", type=int, nargs='+', default=None)
    parser.add_argument("--base_dir", type=str, nargs='+', default=None)
    parser.add_argument("--filename", nargs='+', default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--sub_tasks", nargs='+', default=None)
    parser.add_argument("--percentile", default=0)
    parser.add_argument("--method", default=None)
    parser.add_argument("--filter_entities", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--detect_lang", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--filter_stopwords", type=lambda x: (str(x).lower() == 'true'), default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--multi_task", action="store_true")
    # parser.add_argument("--base_path", type=str, default=None)
    # parser.add_argument("--base_path_common", type=str, default=None)
    # parser.add_argument("--base_path_all", type=str, default=None)
    # parser.add_argument("--filter_chars", action="store_true")
    # parser.add_argument("--detect_lang", action="store_true")
    # parser.add_argument("--percentile", type=float, default=0.1)

    return parser.parse_args()

def main(args):
    TASK_CONF = PATHS.base_ngram_paths[args.task] if args.task in PATHS.base_ngram_paths.keys() else None
    N_GRAMS = args.n_grams
    BASE_DIR = args.base_dir
    BASE_PATH = os.path.join(BASE_DIR, str(N_GRAMS))
    BASE_PATH_COMMON = os.path.join(BASE_PATH, "common")
    BASE_PATH_ALL = os.path.join(BASE_PATH, "all")
    FILTER_CHARS = False
    DETECT_LANG = False
    if args.filter_stopwords is None:
        FILTER_STOPWORDS = True if N_GRAMS < 3 else False
    else:
        FILTER_STOPWORDS = args.filter_stopwords

    # model params
    models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
              'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
              'pythia-70m', 'pythia-31m', 'pythia-14m',]
    # wmt09
    if args.sub_tasks is not None:
        TASKS = args.sub_tasks
    elif TASK_CONF is not None:
        TASKS = TASK_CONF['tasks'] if 'tasks' in TASK_CONF.keys() else args.task
    else:
        TASKS = args.task
    print(f"TASKS: {TASKS}")

    wa = WimbdAnalysis(BASE_PATH, TASKS, N_GRAMS, FILTER_CHARS)
    wt = WimbdTasks()

    TASKS_STR = "_".join(TASKS)
    if len(TASKS_STR) > 25:
        TASKS_STR = f"{args.task}"
    
    PLOT_PATH = f"./results/n-grams/exp_full/{N_GRAMS}/plots/{TASKS_STR}_filter_{FILTER_CHARS}"
    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)

    # %%
    ## Filter ALL
    # lang_dfs_all_filtered, lang_dfs_all_pth = wa.get_lang_dfs(BASE_PATH_ALL, TASKS, is_all=True, 
    #                                         filter_chars=False, percentile=0, detect_lang=True, 
    #                                         filter_entities=True, filter_stopwords=True,
    #                                         remove_english=True, remove_non_english=False)
    # lang_dfs_common_filtered, lang_dfs_common_pth = wa.get_lang_dfs(BASE_PATH_COMMON, TASKS, filter_chars=False,
    #                                            percentile=0, detect_lang=True, filter_entities=True,
    #                                            filter_stopwords=True, align_langs=0)
    
    if args.task == "translation":
        print(f"taskssss: {TASKS}")
        if args.method in ['all', None]:
            dfs_all_filtered, dfs_all_pth = wa.get_lang_dfs(
                BASE_PATH_ALL, TASKS, is_all=True, filter_chars=False, percentile=0,
                detect_lang=args.detect_lang, filter_entities=args.filter_entities, filter_stopwords=FILTER_STOPWORDS,
                remove_english=True, remove_non_english=False, debug=args.debug
            )
        else:
            dfs_all_filtered, dfs_all_pth = None, None

        if args.method in ['common', None]:
            dfs_common_filtered, dfs_common_pth = wa.get_lang_dfs(
                BASE_PATH_COMMON, TASKS, filter_chars=False, percentile=0,
                detect_lang=args.detect_lang, filter_entities=args.filter_entities, filter_stopwords=FILTER_STOPWORDS,
                align_langs=0.8, debug=args.debug
            )
        else:
            dfs_common_filtered, dfs_common_pth = None, None

    else:
        if args.method in ['all', None]:
            dfs_all_filtered, dfs_all_pth = wa.get_task_dfs(
                BASE_PATH_ALL, TASKS, filter_chars=True, percentile=args.percentile
            )
        else:
            dfs_all_filtered, dfs_all_pth = None, None

        if args.method in ['common', None]:
            print("loading common!")
            dfs_common_filtered, dfs_common_pth = wa.get_task_dfs(
                BASE_PATH_COMMON, TASKS, filter_chars=True, percentile=args.percentile,
                align_pairs=True
            )
        else:
            dfs_common_filtered, dfs_common_pth = None, None
            
    return {
        "dfs_all": {
            'data': dfs_all_filtered,
            'path': dfs_all_pth
        },
        "dfs_common_filtered": {
            'data': dfs_common_filtered,
            'path': dfs_common_pth
        }
    }


if __name__ == "__main__":
    
    args = parse_args()
    if args.multi_task:
        base_dirs = args.base_dir
        methods = [args.method] if args.method is not None else ['all', 'common']
        models = ['pythia-12b', 'pythia-2.8b', 'pythia-410m']
        base_dirs = ["./results/n-grams/wmt09_gens/pile/exp4_"] if args.base_dir is None else args.base_dir
        filenames = ["n_samples_100_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue",
                     "n_samples_100_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue"] if args.filename is None else args.filename
        
        n_grams = args.n_grams
        
        for base_dir in base_dirs:
            
            for model in models:
                for filename in filenames:
                    args.base_dir = os.path.join(base_dir, model, filename)

                    for method in methods:
                        args.method = method

                        for n_gram in n_grams:
                            print(f"method: {method}, base_dir: {base_dir}, n_gram: {n_gram}, filename: {filename}")
                            args.n_grams = n_gram
                            try:
                                main(args)
                            except Exception as e:
                                print(f"Error: {args.base_dir}, {args.n_grams}")
                                print(str(e))

    else:
        base_dirs = args.base_dir
        methods = [args.method] if args.method is not None else ['all', 'common']
        
        if args.n_grams is None:
            n_grams = range(1, 5)
        else:
            n_grams = args.n_grams

        for method in methods:
            args.method = method

            for base_dir in base_dirs:
                args.base_dir = base_dir

                for n_gram in n_grams:
                    print(f"method: {method}, base_dir: {base_dir}, n_gram: {n_gram}")
                    args.n_grams = n_gram
                    main(args)


"""

CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py --n_grams 3 5 \
    --base_dir "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue"
    
CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py --n_grams 2 \
    --base_dir "./results/n-grams/wmt/pile/exp4/n _samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue"

CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py \
    --base_dir "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue"
    
CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py --n_grams 2 \
    --base_dir "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue"

CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py \
                        --task "translation" \
                        --base_dir ./results/n-grams/exp_full \
                        --method common \
                        --n_grams 4

europarl fstop filtered: /results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue

europarl, fstop true
CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py \
                        --base_dir "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue"


trivia_qa
CUDA_VISIBLE_DEVICES=1 python wimbd_preprocess.py \
                        --task "triviaqa" \
                        --base_dir "./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse" \
                        --n_grams 5

CUDA_VISIBLE_DEVICES=1 python wimbd_preprocess.py \
                        --task "triviaqa" \
                        --base_dir "./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse" \
                        --n_grams 3

trivia_qa -> xinyi
CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py \
                        --base_dir ./results/n-grams/trivia_qa/pile/exp_3/test-set/n_samples_10000_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse \
                        --task "triviaqa" \
                        --n_grams 5

                        --base_dir "./results/n-grams/trivia_qa/pile/exp_3/test-set/n_samples_10000_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse" \
sciq
CUDA_VISIBLE_DEVICES="" python wimbd_preprocess.py \
                        --task "sciq" \
                        --base_dir "./results/n-grams/trivia_qa/pile/exp_3/test-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse" \
                        --n_grams 5

- mmlu
-- pile
CUDA_VISIBLE_DEVICES=0 python wimbd_preprocess.py \
                        --task "mmlu" \
                        --base_dir "./results/n-grams/mmlu/exp3/test-set/exp_full_None" \
                        --method common \
                        --sub_tasks marketing management \
                                  high_school_world_history \
                                  high_school_european_history \
                                  miscellaneous \
                        --n_grams 5

CUDA_VISIBLE_DEVICES=6 python wimbd_preprocess.py \
                        --task "mmlu" \
                        --base_dir "./results/n-grams/mmlu/pile/exp4_filter/test-set/exp_full_None" \
                        --method common \
                        --n_grams 5 6 7

--base_dir "./results/n-grams/mmlu/exp3/test-set/exp_full_None"
                        
wmt09gens
-- 2 grams
CUDA_VISIBLE_DEVICES=7 python wimbd_preprocess.py \
                        --task "translation" \
                        --sub_tasks en-es \
                        --base_dir ./results/n-grams/wmt09_gens/pile/exp_3/test-set/pythia-12b/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse \
                                   ./results/n-grams/wmt09_gens/pile/exp_3/test-set/pythia-410m/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse \
                        --method common \
                        --filter_entities False \
                        --detect_lang False \
                        --n_grams 2
-- 4 grams
CUDA_VISIBLE_DEVICES=7 python wimbd_preprocess.py \
                        --task "translation" \
                        --sub_tasks en-es \
                        --base_dir ./results/n-grams/wmt09_gens/pile/exp4/pythia-2.8b \
                                   ./results/n-grams/wmt09_gens/pile/exp4/pythia-12b \
                                   ./results/n-grams/wmt09_gens/pile/exp4/pythia-410m \
                        --method common \
                        --filter_entities False \
                        --detect_lang False \
                        --n_grams 2 4 

-- models
CUDA_VISIBLE_DEVICES=6 python wimbd_preprocess.py --task translation \
                           --sub_tasks en-es \
                           --method common \
                           --n_grams 1 2 3 \
                           --detect_lang False \
                           --filter_entities False \
                           --filter_stopwords False \
                           --base_dir ./results/n-grams/wmt09_gens/pile/exp4__ \
                           --filename n_samples_100_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue \
                           --multi_task

--- pythia 410m 4 grams filter
python wimbd_preprocess.py --task translation \
                           --sub_tasks en-es \
                           --base_dir "./results/n-grams/wmt09_gens/pile/exp4/pythia-410m/n_samples_100_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue" \
                           --method common \
                           --n_grams 4 \
                           --detect_lang False \
                           --filter_entities False \
                           --filter_stopwords False

"""
# %%

