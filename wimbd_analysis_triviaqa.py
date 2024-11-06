import os
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from src.wimbd_ import BasePaths as PATHS
from src.wimbd_ import DataConfigs as CONFIG
from src.wimbd_ import post_filter, filter_percentile, load_results
from src.wimbd_ import ProcessTriviaQA as ptqa
from src.utils import remove_nested_lists
from wimbd_process_results import softmax
from datetime import datetime
from src.analysis import AnalyzeNgrams as an

pd.set_option('display.max_columns', 10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for the data")
    parser.add_argument("--ngrams", type=int, nargs="+", default=[1, 2, 3], help="List of n-gram sizes to process (default: [1, 2, 3])")
    args = parser.parse_args()
    return args


def main(args):
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")

    all_models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b', 'pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m']
    large_models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b']
    small_models = ['pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m']

    DATASET = "examples_dfs_4-shot_common_models.pkl"
    TASKS = "triviaqa"
    OMMIT_TASKS = False
    TASKS_OMMIT = ["formal_logic"]
    POST_FILTER = True

    METHOD = "0-shot_common"
    SAVE_DIR = f"./figures/ngrams/{TASKS}"

    for N_GRAMS in args.ngrams:
        print(f"Processing {N_GRAMS}-grams...")

        BASE_PATH = os.path.join(args.base_dir, f"{N_GRAMS}")
        BASE_PATH_COMMON = os.path.join(BASE_PATH, "common")
        BASE_PATH_ALL = os.path.join(BASE_PATH, "all")

        RESULTS_PATH = os.path.join(args.base_dir, str(N_GRAMS), DATASET)
        print(f"BASE PATH: {BASE_PATH}")

        # Filter outliers
        pickle_file = pd.read_pickle(RESULTS_PATH)
        examples_models = {model: filter_percentile(data, 0.9999) for model, data in pickle_file.items()}
        example_model = examples_models['pythia-12b']
        examples_model = remove_nested_lists(example_model)

        examples_model.head(0)
        examples_model = ptqa.create_exact_match_column(example_model)

        coverage_path = os.path.join(BASE_PATH_COMMON, "task-coverage.pkl")
        task_cov = an.calculate_average_task_coverage(BASE_PATH_ALL, [TASKS], [N_GRAMS])

        # Assuming 'value' is the column with values to bin and we want 5 bins between 1 and 10000
        examples_model = an.create_bins(examples_model, 'value', 1, 10000, 4)
        example_samples_model = ptqa.get_random_samples_by_interval(examples_model, 100)

        print(f"Plotting for {N_GRAMS}-grams...")
        filename = DATASET.replace('.pkl', '')
        save_pth = os.path.join(SAVE_DIR, N_GRAMS)
        ptqa.plot_avg_em_by_log_interval(example_samples_model, 
                                        log_axis=True,
                                        save_pth=os.path.join(save_pth, f"{filename}_log_samples.png"))
        ptqa.plot_avg_em_by_log_interval(examples_model, 
                                         log_axis=True,
                                         save_pth=os.path.join(save_pth, f"{filename}_log.png"))

        pd.set_option('display.max_columns', 20)
        examples_model['Q']

if __name__ == "__main__":
    args = parse_args()
    main(args)