import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PATHS = {
    "triviaqa": {
        "3": f"./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse/",
        "5": f"./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse/" 
    },
    "sciq": {
        
    }
}


class AnalyzeNgrams:

    @staticmethod
    def load_coverage_data(base_path):
        coverage_path = os.path.join(base_path, "task-coverage.pkl")
        with open(coverage_path, "rb") as f:
            coverage = pickle.load(f)
        return pd.DataFrame(coverage)

    @staticmethod
    def filter_coverage_data(coverage, tasks):
        return coverage[coverage['task'].isin(tasks)]

    @staticmethod
    def calculate_task_coverage(coverage):
        return coverage.groupby('task')['coverage'].mean().to_dict()

    @staticmethod
    def calculate_average_task_coverage(base_path, tasks, n_grams_list):
        task_coverage = defaultdict(list)
        
        for n_grams in n_grams_list:
            coverage = AnalyzeNgrams.load_coverage_data(base_path)
            coverage = AnalyzeNgrams.filter_coverage_data(coverage, tasks)
            task_cov = AnalyzeNgrams.calculate_task_coverage(coverage)

            for task in tasks:
                task_coverage[task].append(task_cov.get(task, 0))

        task_coverage = pd.DataFrame(task_coverage)
        task_coverage = task_coverage.melt().rename(columns={"variable": "task", "value": "coverage"})
        task_cov_mean = task_coverage.groupby('task')['coverage'].mean().to_dict()

        return task_coverage, task_cov_mean
    
    @staticmethod
    def create_bins(df, value_column, bin_start, bin_end, num_bins, is_log=True):
        """
        Create a new column in the dataframe with logarithmic or linear intervals based on the specified value column.
        Zero values are assigned a special interval label, and values between 0 and 1 are handled separately.
        
        :param df: pandas DataFrame containing the data
        :param value_column: the name of the column with values to bin
        :param bin_start: the start of the bins (inclusive)
        :param bin_end: the end of the bins (exclusive)
        :param num_bins: number of bins to create between bin_start and bin_end
        :param is_log: boolean indicating whether to use logarithmic (True) or linear (False) intervals, default is True
        :return: DataFrame with a new column 'interval' added
        """
        df_copy = df.copy()

        if is_log:
            # Adjust bin_start to be slightly above 0 to handle the 0-1 interval
            adjusted_bin_start = max(bin_start, 1e-10)
            
            # Create logarithmically spaced bins between adjusted_bin_start and bin_end
            bins = np.logspace(np.log10(adjusted_bin_start), np.log10(bin_end), num_bins + 1)
            
            # Map bin index to interval string representation
            interval_labels = [f"[{bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(num_bins)]
            zero_to_one_label = "[0.00, 1.00)"
            above_max_label = f"[{bin_end}, ∞)"
            
            # Apply the labels to the 'interval' column, including the special label for zeros and 0-1 interval
            df_copy['interval'] = df_copy[value_column].apply(
                lambda x: zero_to_one_label if x > 0 and x < 1 else (
                    zero_to_one_label if x == 0 else (
                        above_max_label if x >= bin_end else interval_labels[np.digitize(x, bins, right=False)-1]
                    )
                )
            )
        else:
            # Create linearly spaced bins between bin_start and bin_end
            bins = np.linspace(bin_start, bin_end, num_bins + 1)
            
            # Map bin index to interval string representation
            interval_labels = [f"[{bins[i]:.4f}, {bins[i+1]:.4f})" for i in range(num_bins)]
            below_min_label = f"(0, {bin_start})"
            above_max_label = f"[{bin_end}, ∞)"
            
            # Apply the labels to the 'interval' column, including the special labels for values outside the range
            df_copy['interval'] = df_copy[value_column].apply(
                lambda x: below_min_label if x < bin_start else (
                    above_max_label if x >= bin_end else interval_labels[np.digitize(x, bins, right=False)-1]
                )
            )

        # print number of samples per interval
        print(df_copy['interval'].value_counts())
        
        return df_copy
    
    @staticmethod
    def calculate_qa_coverage(df, metric='em'):
        coverage_results = []

        questions = df['question'].unique()

        for question in tqdm(questions, desc="Processing questions"):
            df_q = df[df['question'] == question]
            result = df_q[metric].iloc[0]
            query_q = ' '.join(list(df_q['Q']))
            query_a = ' '.join(list(df_q['A']))
            answers = df_q['answer'].iloc[0]['aliases']

            # Calculate the proportion of the question which is covered by
            # the query_q
            q_coverage = len(set(query_q.split()).intersection(set(question.split()))) / len(set(question.split()))

            # Calculate answer coverage
            a_coverage = int(any(query_a_term in answers for query_a_term in query_a.split()))

            # Store the results as a dictionary
            coverage_results.append({
                'question': question,
                'q_coverage': q_coverage,
                'a_coverage': a_coverage,
                'em': result
            })

        # Convert the list of dictionaries to a DataFrame
        coverage_df = pd.DataFrame(coverage_results)
        coverage_df['total_coverage'] = (coverage_df['q_coverage'] + coverage_df['a_coverage']) / 2
        
        return coverage_df

        