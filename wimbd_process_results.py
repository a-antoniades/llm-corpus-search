# %%
import os
import glob
import re
import collections
from collections import defaultdict
from tqdm import tqdm
import json
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from adjustText import adjust_text

from langdetect import detect, LangDetectException
import langid
from src.wimbd_ import WimbdAnalysis, _load_dataset
from src.utils import remove_string


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='WIMBD Analysis')
    parser.add_argument('--dataset', type=str, default="mmlu", help='Dataset to use')
    parser.add_argument('--ngrams', type=int, default=5, help='N-grams')
    parser.add_argument('--shots', type=int, default=0, help='Number of shots')
    parser.add_argument('--base_dir', type=str, help='Base directory')
    parser.add_argument('--filename', type=str, default=None, help='Filename of preprocessed task_dfs if it exists')
    parser.add_argument('--method', type=str, default=None, help='Method to use')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()


# Define the softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()

def nll_to_prob(nll):
    return np.exp(nll)


def get_metric(results):
    metrics_to_check = [
        'bleu',
        'acc'
    ]
    for metric in metrics_to_check:
        if metric in results:
            return results[metric]


def fetch_paths_with_criteria(search_dir, *criteria):
    """
    Fetch file paths that contain all the given criteria.

    Args:
        search_dir (str): The base directory to search in.
        file_pattern (str): The file pattern to match (e.g., "results.json").
        *criteria (str): Variable length argument list of criteria to match in the file paths.

    Returns:
        list: A list of file paths that match all the given criteria.
    """
    # Use glob to find all files that match the search pattern
    matching_paths = [path for path in search_dir if all(criterion in path for criterion in criteria)]
    return matching_paths


def process_model_results(base_results_paths, TASK, TASKS, SHOT):
    results_dict = collections.defaultdict(dict)
    instance_results_dict = collections.defaultdict(dict)

    # Iterate over each model and dataset, loading the results.json file
    for task in TASKS:
        for base_results_path, models_ in base_results_paths.items():
            result_paths = glob.glob(os.path.join(base_results_path, "**", "results.json"), recursive=True)
            instance_results_paths = glob.glob(os.path.join(base_results_path, "**", "doc_results.json"), recursive=True)
            for model in models_:
                # get file that contains model, task and shot
                result_path = fetch_paths_with_criteria(result_paths, "results.json", model, TASK, SHOT, task)[0]
                instance_results_path = fetch_paths_with_criteria(instance_results_paths, model, TASK, SHOT, task)[0]
                results = json.load(open(result_path, 'r'))['results']
                doc_results = json.load(open(instance_results_path, 'r'))
                for task in results.keys():
                    # remove hendrycksTest-
                    task_str = remove_string(task)
                    print(f"Processing {model} {task_str}")
                    results_dict[task_str][model] = get_metric(results[task])
                    if os.path.exists(instance_results_path):
                        instance_results_dict[model][task_str] = doc_results[task]                     
    return results_dict, instance_results_dict


def process_and_save_results(example_dfs, model_instance_results, base_path, suffix):
    
    examples_dfs_models = {}
    for model in model_instance_results.keys():
        instances = pd.DataFrame(model_instance_results[model])
        print(f"instances: {instances.head()}")
        print(f"example_dfs: {example_dfs.head()}")

        def process_int_bool_col(col, extract_single=False, make_list=False):

            if extract_single:
                # extract elements from list if just 1
                col = col.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
            if make_list:
                # make into a list (so it can be indexed)
                col = col.apply(lambda x: [x])
            
            # convert to int if bool
            if col.dtype == bool:
                col = col.astype(int)
            # check if it's a list
            elif col.dtype == object:
                pass
            elif col.dtype != float:
                col = col.astype(float)
            return col

        # handle bool/int result/lls columns (according to task type)
        if 'lls' in instances.columns:
            instances['lls'] = process_int_bool_col(instances['lls'], make_list=True)
        if 'result' in instances.columns:
            instances['result'] = process_int_bool_col(instances['result'], extract_single=True)
        
        # if gold col doesn't exist, it means there is only 1 ll (e.g generative tasks)
        if 'gold' not in example_dfs.columns:
            example_dfs['gold'] = 0
            if 'answer' in example_dfs.iloc[0].keys():
                example_dfs['gold'] = example_dfs['example'].apply(lambda x: x['answer'])

        # convert nll to probs
        if 'lls' in instances.columns:
            instances['probs'] = instances['lls'].apply(nll_to_prob)
            instances['probs_softmax'] = instances['probs'].apply(softmax) # get normalized probs

            # merge on ids
            examples_dfs_model = example_dfs.merge(instances, on='id', how='inner', suffixes=('', '_drop'))
            examples_dfs_model.drop('gold_drop', axis=1, inplace=True) # drop duplicate gold col

            print(f"examples_dfs_model: {examples_dfs_model.head()}")
            # Assuming 'gold' column exists and contains the index of the correct label
            examples_dfs_model['probs_gold'] = examples_dfs_model.apply(
                lambda row: row['probs'][row['gold']], axis=1
            )
            examples_dfs_model['probs_softmax_gold'] = examples_dfs_model.apply(
                lambda row: row['probs_softmax'][row['gold']], axis=1
            )
        
        if 'bleu' in instances.columns:
            # merge on ids
            examples_dfs_model = example_dfs.merge(instances, on='id', how='inner', suffixes=('', '_drop'))
        
        # add the model results to the model dict
        examples_dfs_models[model] = examples_dfs_model

    # Save model_instance_results to pickle
    model_instance_results_pth = os.path.join(base_path, f'model_instance_results_{suffix}.pkl')
    with open(model_instance_results_pth, 'wb') as f:
        pickle.dump(model_instance_results, f)
    print(f"Saved model_instance_results to {model_instance_results_pth}")

    # Save examples_dfs_models to pickle
    example_dfs_models_pth = os.path.join(base_path, f'examples_dfs_{suffix}_models.pkl')
    with open(example_dfs_models_pth, 'wb') as f:
        pickle.dump(examples_dfs_models, f)
    print(f"Saved examples_dfs_models to {example_dfs_models_pth}")

    return examples_dfs_models


def process_instance_results(instance_results, task_dfs, label=''):
    model_instance_results = collections.defaultdict(list)
    for model in instance_results.keys():
        for i, task in enumerate(instance_results[model].keys()):
            print(f"Processing {model} {task}, columns: {instance_results[model][task][0].keys()}")
            model_task_results = instance_results[model][task].copy()
            # Add task label to each id, with an optional common label
            for row in model_task_results:
                row['id'] = f"{str(row['id'])}_{i}"
            model_instance_results[model].extend(model_task_results)
    return model_instance_results


def merge_and_process_dfs(task_dfs):
    example_dfs = pd.concat(task_dfs.values())
    
    # Extract 'query' from the 'example' column, handling both 'question' and 'context'
    def extract_query(example):
        if 'question' in example:
            return example['question']
        elif 'context' in example:
            return example['context']
        else:
            return example

    example_dfs['query'] = example_dfs['example'].apply(extract_query).astype(str)
    
    # Perform the aggregation to get the sum of 'value' and other statistics if needed
    aggregated_data = example_dfs.groupby(['query', 'task'])['value'].agg(['sum', 'count']).reset_index()
    
    # Merge the aggregated data back to the original DataFrame
    example_dfs = example_dfs.merge(aggregated_data, on=['query', 'task'], how='left')
    
    # Sort the DataFrame based on the 'sum' of 'value'
    example_dfs = example_dfs.sort_values(by='sum', ascending=False)
    
    return example_dfs


def find_matching_rows(df, src_phrase, ref_phrase):
    # Split the phrases into individual words
    src_words = src_phrase.split()
    ref_words = ref_phrase.split()

    # Use all() to check if all words are in the 'src' and 'ref' columns
    match = df[
        df['src'].apply(lambda x: all(re.search(re.escape(word), x, re.IGNORECASE) for word in src_words)) &
        df['ref'].apply(lambda x: all(re.search(re.escape(word), x, re.IGNORECASE) for word in ref_words))
    ]
    return match


def find_matching_id(row, instances_df):
    # Check if the 'query' is a list and has two elements to match against 'src' and 'ref'
    if 'translation' in row['example']:
        translation = row['example']['translation']
        langs = list(translation.keys())
        # print(f"translation: {translation[langs[1]]}")
        # first entry is src, second is ref
        match = instances_df[
            instances_df['ref'].str.contains(translation['en'], na=False, regex=False)
        ]
    elif isinstance(row['example'], list) and len(row['example']) == 2:
        match = find_matching_rows(instances_df, row['example'][0], row['example'][1])
    else:
        match = instances_df[instances_df['query'].str.contains(row['query'], na=False, regex=False)]
    if not match.empty:
        return match.iloc[0]['id']
    else:
        print(f"No match found for {row['query']}")
        return None


def add_ids_and_save(example_dfs, instances, filename):
    # Add a new column 'id' to example_dfs by applying the find_matching_id function with a progress bar
    print(f"example_dfs: {example_dfs.head()['query']}")
    print(f"instances: {instances.head()['query']}")
    example_dfs['id'] = example_dfs.progress_apply(find_matching_id, axis=1, instances_df=instances)

    # Save the DataFrame to CSV
    example_dfs.to_csv(filename, index=False)
    print(f"Saved example_dfs to {filename}")

    return example_dfs


dataset_to_eval_task = {
    'mmlu': "MMLU/hendrycks*",
    'arithmetic': "ARITHMETIC/arithmetic*",
    'translation': "TRANSLATION/wmt09"
}


def main(args):
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
    LANGUAGES = ['cs-en', 'hu-en', 'de-en', 'it-en', 'fr-en', 'es-en']
    ds = _load_dataset(args.dataset, tasks=LANGUAGES if not args.debug else ["miscellaneous", "international_law"])

    # %%
    # lang params
    N_GRAMS = args.ngrams
    BASE_DIR = args.base_dir
    BASE_PATH = os.path.join(BASE_DIR, f"{N_GRAMS}")
    BASE_PATH_DF = os.path.join(BASE_PATH, args.method)
    FILTER_CHARS = False
    DETECT_LANG = False
    LOG_AXIS = True
    TASK = dataset_to_eval_task[args.dataset]
    METHOD = args.method

    # model params
    base_results_config = {
        "mmlu": {
            'paths': {
                "0-shot": {
                    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_4/inference/EleutherAI": [
                        'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
                        'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
                        'pythia-70m', 'pythia-31m', 'pythia-14m'
                    ],
                    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/OLMO": [
                        'OLMo-7b'
                    ]
                },
                "5-shot": {
                    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_5/inference/EleutherAI" : [
                        'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
                        'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
                        'pythia-70m', 'pythia-31m', 'pythia-14m'
                    ],
                }
            },
        },
        "arithmetic": {
            "paths" : {
                "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_5/inference/EleutherAI": [
                    'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
                    'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
                    'pythia-70m', 'pythia-31m', 'pythia-14m'
                ]
            },
            "shot": "16-shot"
        },
        "translation": {
            "paths": {
                "0-shot": {
                    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_5/inference/EleutherAI/" : [
                            'pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
                            'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
                            'pythia-70m', 'pythia-31m', 'pythia-14m'
                    ],
                },
            },
        }
                
    }

    result_shots = f"{args.shots}-shot"
    base_config = base_results_config[args.dataset]
    base_results_paths = base_config['paths'][result_shots]
    TASKS_OMMIT = base_config['tasks_ommit'] if 'tasks_ommit' in base_config else []
    print(f"base_results_paths: {base_results_paths}")

    models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 
              'pythia-1.4b', 'pythia-410m', 'pythia-160m', 
              'pythia-70m', 'pythia-31m', 'pythia-14m']
            #   'OLMo-7b']

    # %%
    TASKS = [remove_string(t) for t in list(ds.keys()) if t not in TASKS_OMMIT]
    print(f"TASKS: {TASKS}")
    wa = WimbdAnalysis(BASE_PATH, TASKS, N_GRAMS, FILTER_CHARS)

    # %%
    if args.filename is not None:
        with open(os.path.join(BASE_PATH_DF, args.filename), "rb") as file:
            task_dfs = pickle.load(file)
    else:
        task_dfs = wa.get_task_dfs(BASE_PATH_DF, TASKS)

    print(f"task dfs: {task_dfs}")
    # %%
    # get model performance results on tasks
    results_dict, instance_results_dict = process_model_results(base_results_paths, TASK, TASKS, result_shots)
    print(f"results_dict: {results_dict}")

    # %%
    model_scores, dataset_scores = wa.prepare_scores(results_dict, task_dfs, models,)
                                                    # coverage_=task_cov_common,
                                                    # cov_mean=task_cov_mean)

    single_model = ["pythia-12b"]
    single_model_score, single_model_dataset_score = wa.prepare_scores(results_dict, task_dfs, single_model)
                                                                    #    coverage_=task_cov_common)

    # %%
    # Assuming your dictionary is named `dataset_scores`
    sorted_data = sorted(dataset_scores.items(), key=lambda x: max(x[1]['n_samples']), reverse=True)
    top_5_datasets = {k: v['n_samples'] for k, v in sorted_data[:5]}
    print(f"Top 5 datasets: {top_5_datasets}")


    model_param_map = {'pythia-12b': 12e09,
                        'pythia-6.9b': 6.9e09,
                        'pythia-2.8b': 2.8e09,
                        'pythia-1.4b': 1.4e09,
                        'pythia-410m': 410e06,
                        'pythia-160m': 160e06,
                        'pythia-70m': 70e06,
                        'pythia-31m': 31e06,
                        'pythia-14m': 14e06,}

    model_scores, dataset_scores = wa.prepare_scores(results_dict, task_dfs, models)
                                                    # coverage_=task_cov_all,
                                                    # cov_mean=task_cov_mean)

    # Process all tasks
    model_instance_results = process_instance_results(instance_results_dict.copy(), task_dfs.copy(), label=f'_{METHOD}')

    if args.debug:
        max_len = 10
        for task in task_dfs.keys():
            task_dfs[task] = task_dfs[task].head(max_len)

    example_dfs = merge_and_process_dfs(task_dfs)

    tqdm.pandas()

    # Assuming model_instance_results is a list of dictionaries
    instances_df = process_instances(model_instance_results[list(model_instance_results.keys())[0]])

    # Assuming BASE_PATH is defined
    example_dfs_filename = os.path.join(BASE_PATH, f'example_dfs_{METHOD}_exact.csv')

    print(f"example dfs all: {example_dfs}")

    # Process all tasks
    example_dfs = add_ids_and_save(example_dfs, instances_df, example_dfs_filename)
    # count how many examples are not None or Nan
    examples_matched = example_dfs['id'].notna().sum()
    examples_umatched = example_dfs['id'].isna().sum()

    ## Assuming BASE_PATH is defined
    example_dfs_models = process_and_save_results(example_dfs, model_instance_results, 
                                                  BASE_PATH, suffix=f"{result_shots}_{METHOD}")

    print(f"Examples matched: {examples_matched}")
    print(f"Examples unmatched: {examples_umatched}")

def process_instances(model_instance_results):
    model_instance_results = pd.DataFrame(model_instance_results)
    if 'context' in model_instance_results.columns:
        model_instance_results = model_instance_results.rename(columns={'context': 'query'})
    if all(col in model_instance_results.columns for col in ['src', 'ref']):
        model_instance_results['query'] = model_instance_results.apply(lambda row: [row['src'], row['ref']], axis=1)
    return model_instance_results


if __name__ == "__main__":
    args = parse_args()
    # n_grams = args.ngrams
    n_grams = [2, 3, 4]
    for n_gram in n_grams:
        args.ngrams = n_gram
        main(args)

    # if args.method is None:
    #     methods = ["all", "common"]
    #     for method in methods:
    #         args.method = method
    #         main(args)
    # else:
    #     main(args)



"""


CUDA_VISIBLE_DEVICES="" python wimbd_process_results.py \
                          --base_dir "./results/n-grams/exp_full" \
                          --filename "lang_dfs_filter_charsFalse_percentile0.99_detect_langFalse_filter_entitiesTrue_align_langs0.8.pkl" \
                          --dataset "translation" \
                          --method "common" \
                          --ngrams 1

                          --base_dir "./results/n-grams/exp_full" \
                          --filename "lang_dfs_filter_charsFalse_lower_percentile0.005.pkl" \
                          
                          --base_dir "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue" \
                          --filename "lang_dfs_filter_charsFalse_lower_percentile0.005.pkl" \


CUDA_VISIBLE_DEVICES="" python wimbd_process_results.py \
                          --base_dir "./results/n-grams/mmlu/pile/exp4_filter/test-set/exp_full_None" \
                          --dataset "mmlu" \
                          --method "common" \
                          --ngrams 5


                          --base_dir "./results/n-grams/mmlu/pile/exp4_nofilter/test-set/exp_full_None" \

                          
"""


    # %%
    # coverage_path_common = os.path.join(BASE_PATH_COMMON, "task-coverage.pkl")
    # coverage_path_all = os.path.join(BASE_PATH_ALL, "task-coverage.pkl")

    # with open(coverage_path_common, "rb") as f:
    #     coverage_common = pickle.load(f)

    # with open(coverage_path_all, "rb") as f:
    #     coverage_all = pickle.load(f)

    # coverage_common = pd.DataFrame(coverage_common)
    # coverage_common = coverage_common[coverage_common['task'].isin(TASKS)]

    # coverage_all = pd.DataFrame(coverage_all)
    # coverage_all = coverage_all[coverage_all['task'].isin(TASKS)]

    # task_cov_common = coverage_common.groupby('task')['coverage'].mean().to_dict()
    # task_cov_all = coverage_all.groupby('task')['coverage'].mean().to_dict()

    # # now get avg task coverage
    # N_GRAM_LIST = N_GRAMS if isinstance(N_GRAMS, list) else [N_GRAMS]
    # task_coverage = defaultdict(list)
    # for n_grams in N_GRAM_LIST:
    #     coverage_path_common = os.path.join(BASE_PATH_COMMON, "task-coverage.pkl")
    #     coverage_path_all = os.path.join(BASE_PATH_ALL, "task-coverage.pkl")

    #     with open(coverage_path_common, "rb") as f:
    #         coverage_common = pickle.load(f)

    #     with open(coverage_path_all, "rb") as f:
    #         coverage_all = pickle.load(f)

    #     coverage_common = pd.DataFrame(coverage_common)
    #     coverage_common = coverage_common[coverage_common['task'].isin(TASKS)]

    #     coverage_all = pd.DataFrame(coverage_all)
    #     coverage_all = coverage_all[coverage_all['task'].isin(TASKS)]

    #     task_cov_common = coverage_common.groupby('task')['coverage'].mean().to_dict()
    #     task_cov_all = coverage_all.groupby('task')['coverage'].mean().to_dict()

    #     print(f"task_cov_common: {task_cov_common.keys()}")
    #     for task in TASKS:
    #         task_coverage[task].append(task_cov_common[task])

    # task_coverage = pd.DataFrame(task_coverage)
    # task_coverage = task_coverage.melt().rename(columns={"variable": "task", "value": "coverage"})
    # task_cov_mean = task_coverage.groupby('task')['coverage'].mean().to_dict()
