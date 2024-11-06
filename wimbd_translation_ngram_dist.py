# %%
import os
import collections
import json
import pickle
from tqdm import tqdm
from natsort import natsorted

import argparse
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
# pd.set_option('display.max_columns', 500)
# pd.options.display.max_rows = 500
# pd.options.display.max_columns = 20
# pd.set_option('display.max_columns', 200)
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.min_rows', 100)
# pd.set_option('display.expand_frame_repr', True)

from scipy.stats import linregress, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import jensenshannon

from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns

from src.wimbd_ import _load_dataset, WimbdTasks, filter_percentile
from src.utils import running_jupyter
wt = WimbdTasks()

# %%
percentile = 0.99
TASK = 'en-fr'
LANG_COLS = 'fr'
LANG_COL_COMMON = 'lang_2'
LANG_COL_ALL = 'fr'
LANG_COL = 'fr'
VALUE_COL = 'value' # 'value_diff'
model = 'EleutherAI/pythia-12b'
tokenizer = AutoTokenizer.from_pretrained(model)

# %%
import glob

def load_logits(logits_pth):
    logit_dict = {}
    all_pths = sorted(glob.glob(os.path.join(logits_pth, "**/*.pt"), recursive=True))
    for pth in tqdm(all_pths):
        logits = torch.load(pth)
        id_ = int(pth.split("_")[-1].split(".")[0])
        logit_dict[id_] = logits
        # print(f"id: {id_}, logits shape: {logits.shape}")
    return logit_dict

def load_tokens(tokens_path):
    token_dict = {}
    all_paths = sorted(glob.glob(os.path.join(tokens_path, "**/*.json"), recursive=True))
    
    for path in tqdm(all_paths):
        with open(path, 'r') as f:
            logits = json.load(f)
        
        id_ = int(path.split("_")[-1].split(".")[0])
        token_dict[id_] = logits
        # print(f"id: {id_}, number of logits: {len(logits)}")
    
    return token_dict

def subtract_values(df1, df2, col1, col2, value_col):
    # Merge the two DataFrames based on the specified columns
    merged_df = pd.merge(df1, df2, left_on=col1, right_on=col2, suffixes=('_1', '_2'), how='left')
    
    # Subtract the values from df2 from df1 where the specified columns match
    merged_df[value_col + '_diff'] = merged_df[value_col + '_1'] - merged_df[value_col + '_2']
    
    # Calculate the average difference for the values that have matches
    avg_diff = merged_df[value_col + '_diff'].mean()
    
    # Fill the missing values in the '_diff' column with the average difference
    merged_df[value_col + '_diff'].fillna(avg_diff, inplace=True)
    
    # For the values that have no matches, subtract the average difference from the original value
    merged_df.loc[merged_df[value_col + '_2'].isna(), value_col + '_diff'] = merged_df[value_col + '_1'] - avg_diff
    
    return merged_df

def calc_ngram_dist(df, lang_col, value_col):
    ngram_counts = df.groupby(lang_col)[value_col].sum()
    n_gram_dist = (ngram_counts / ngram_counts.sum()).astype(float)
    n_gram_dist = n_gram_dist[n_gram_dist > 0]
    n_gram_dist = np.log(n_gram_dist)
    return n_gram_dist

def calc_ngram_dist_factorized(df_1, df_2, lang_col_1, lang_col_2):
    df_1['dist'] = df_1['value'] / df_1['value'].sum()
    df_1 = df_1[df_1['dist'] > 0]

    df_2['dist'] = df_2['value'] / df_2['value'].sum()
    df_2 = df_2[df_2['dist'] > 0]

    # Replace zero values with a small positive value (e.g., 1e-8)
    df_1['dist'].replace(0, 1e-8, inplace=True)
    df_2['dist'].replace(0, 1e-8, inplace=True)

    # Ensure no NaN or inf values before taking log
    df_1['dist'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_2['dist'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df_1.dropna(subset=['dist'], inplace=True)
    df_2.dropna(subset=['dist'], inplace=True)

    df_1['log_dist'] = np.log(df_1['dist'])
    df_2['log_dist'] = np.log(df_2['dist'])

    df_1 = subtract_values(df_1, df_2, lang_col_1, lang_col_2, 'log_dist')

    return df_1

def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def find_ngrams_in_dist(ngrams, n_gram_log_dist, col_name, lang_col):
    ngram_probs = []
    lang_col = lang_col if lang_col in n_gram_log_dist.columns else 'lang_2'
    for ngram in ngrams:
        # print(ngram)
        if ngram in list(n_gram_log_dist[lang_col]):
            # print("found")
            prob = n_gram_log_dist[n_gram_log_dist[lang_col] == ngram][col_name].values[0]
            ngram_probs.append((ngram, prob))
    return ngram_probs

def pick_nonoverlapping_ngrams(ngram_probs):
    ngram_probs.sort(key=lambda x: x[1], reverse=True)
    chosen_ngrams = []
    for ngram, prob in ngram_probs:
        if not any(ngram in chosen_ngram for chosen_ngram in chosen_ngrams):
            chosen_ngrams.append(ngram)
    return chosen_ngrams

def match_ngrams_to_probs(chosen_ngrams, tokens, probs, tokenizer):
    ngram_probs = []
    for ngram in chosen_ngrams:
        ngram_tokens = tokenizer.encode(ngram)
        # token_indices = [tokens.index(token) for token in ngram_tokens]
        token_indices = []
        # print(tokens)
        for token in ngram_tokens:
            if token in tokens:
                # print(f"token found!")
                token_indices.append(tokens.index(token))
            else:
                # print(f"Token {token} not found in tokens")
                continue
        # assert len(token_indices) == len(ngram_tokens), f"Token indices: {token_indices}, ngram tokens: {ngram_tokens}"
        # assert token_indices <= len(tokens), f"Token indices: {token_indices}, tokens: {tokens}"
        ngram_logit = probs[token_indices].mean()
        ngram_probs.append((ngram, ngram_logit.item()))
    # if len(ngram_probs) == 0:
        # print(f"No ngrams found in tokens")
    return ngram_probs

def process_doc_results(doc_results, model_tokens, model_probs, 
                        tokenizer, n_gram_log_dist, value_col, n_iters=None,
                        save_pth=None):
    if n_iters is None:
        n_iters = len(doc_results)
    
    for i in tqdm(range(n_iters)):
        result = doc_results[i]
        generation = result['result'][0]
        tokens = model_tokens[result['id']][0]
        probs = model_probs[result['id']][0]
        
        ngrams = generate_ngrams(generation, n=2)
        ngram_ds_probs = find_ngrams_in_dist(ngrams, n_gram_log_dist, 
                                             value_col, LANG_COL)
        chosen_ngrams = pick_nonoverlapping_ngrams(ngram_ds_probs)
        ngram_lm_probs = match_ngrams_to_probs(chosen_ngrams, tokens, probs, tokenizer)
        
        # save results
        doc_results[i]['ngram_ds_probs'] = ngram_ds_probs
        doc_results[i]['ngram_lm_probs'] = ngram_lm_probs
        doc_results[i]['chosen_ngrams'] = chosen_ngrams
    
    if save_pth is not None:
        os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        with open(save_pth, 'w') as f:
            json.dump(save_pth, f)
    
    print(f"doc results: {doc_results}")
    return doc_results

def create_ngram_dataframe(doc_results_new):
    data = []

    for row in doc_results_new:
        chosen_ngrams = row['chosen_ngrams']
        ngram_ds_probs = row['ngram_ds_probs']
        ngram_lm_probs = row['ngram_lm_probs'] if 'ngram_lm_probs' in row else row['ngram_lm_logits']
        
        for ngram, ds_prob, lm_logit in zip(chosen_ngrams, ngram_ds_probs, ngram_lm_probs):
            data.append({
                'ngrams': ngram,
                'ds_probs': ds_prob[1],
                'lm_probs': lm_logit[1]
            })

    df = pd.DataFrame(data)

    # if there's more than one
    # entry for an ngram, take the mean
    df = df.groupby('ngrams').mean().reset_index()
    
    df['lm_probs'] = np.log(df['lm_probs'])
    return df

def calculate_similarity(lm_probs, ds_probs, method='kl_divergence'):
    """
    Calculate the similarity between two probability distributions.
    'lm_probs' and 'ds_probs' should be lists of probabilities for each ngram.
    """
    # Normalize the probabilities
    lm_probs = np.array(lm_probs)
    ds_probs = np.array(ds_probs)
    lm_probs_normalized = lm_probs / lm_probs.sum()
    ds_probs_normalized = ds_probs / ds_probs.sum()
    
    # Choose the method to calculate similarity
    if method == 'kl_divergence':
        # Calculate the Jensen-Shannon divergence, which is the symmetrized and smoothed version of KL divergence
        similarity = jensenshannon(lm_probs_normalized, ds_probs_normalized)
    elif method == 'correlation':
        # Calculate the Pearson correlation coefficient
        correlation, _ = pearsonr(lm_probs_normalized, ds_probs_normalized)
        similarity = correlation
    elif method == 'mse':
        similarity = mean_squared_error(lm_probs_normalized, ds_probs_normalized)
    elif method == 'cross_entropy':
        similarity = -np.sum(ds_probs_normalized * np.log(lm_probs_normalized))
    elif method == 'xinyi':
        similarity = -np.sum(np.log(lm_probs_normalized))
    else:
        raise ValueError("Unsupported method for similarity calculation.")
    
    return similarity

def create_prob_seq_df(doc_results_new):
    data = []
    
    for row in doc_results_new:
        chosen_ngrams = row['chosen_ngrams']
        ngram_ds_probs = dict(row['ngram_ds_probs'])
        ngram_lm_probs = dict(row['ngram_lm_probs'] if 'ngram_lm_probs' in row else row['ngram_lm_logits'])
        len_ngrams = len(ngram_ds_probs)
        len_probs = len(ngram_lm_probs) 

        ds_probs = [ngram_ds_probs[ngram] for ngram in chosen_ngrams]
        lm_probs = [ngram_lm_probs[ngram] for ngram in chosen_ngrams]

        lm_probs_kldiv = calculate_similarity(np.log(lm_probs), ds_probs, method='kl_divergence')
        # lm_probs_corr = calculate_similarity(lm_probs, ds_probs, method='correlation')
        if len_ngrams and len_probs > 0:
            lm_probs_mse = calculate_similarity(lm_probs, ds_probs, method='mse')
            lm_probs_cross_entropy = calculate_similarity(lm_probs, ds_probs, method='cross_entropy')
            lm_probs_xinyi = calculate_similarity(lm_probs, ds_probs, method='xinyi')
        
        data.append({
            'example': row['result'][0],
            'id': row['id'],
            'ngrams': chosen_ngrams,
            'ds_probs': ds_probs,
            'ds_probs_sum': np.sum(ds_probs),
            'ds_probs_mean': np.mean(ds_probs) if ds_probs else None,
            'lm_probs': lm_probs,
            'lm_probs_log': np.log(lm_probs) if lm_probs else None,
            'lm_probs_sum': np.sum(np.log(lm_probs)) if lm_probs else None,
            'lm_probs_mean': np.mean(np.log(lm_probs)) if lm_probs else None,
            'len_ngrams': len_ngrams,
            'len_probs': len_probs,
            'lm_probs_kldiv': lm_probs_kldiv,
            'lm_probs_mse': lm_probs_mse if len_ngrams and len_probs > 0 else None,
            'lm_probs_cross_entropy': lm_probs_cross_entropy if len_ngrams and len_probs > 0 else None,
            'lm_probs_xinyi': lm_probs_xinyi if len_ngrams and len_probs > 0 else None
        })
    
    return pd.DataFrame(data).dropna()

def fit_and_evaluate_linear_regression(df, column_1, column_2,
                                       test_size=0.2, random_state=42,
                                       save_pth=None, title='Linear Regression'):
    
    # Ensure 'len_ngrams' is in the DataFrame and is not the target or a feature
    if 'len_ngrams' not in df:
        raise ValueError("The DataFrame does not contain the 'len_ngrams' column.")

    # len_min, len_max = -3, 0
    # df = df[(df[column_2] >= len_min) & (df[column_2] <= len_max)]

    # Split the data into training and testing sets
    X = df[[column_1]]  # Independent variable
    y = df[column_2]    # Dependent variable
    len_ngrams = df['len_ngrams']  # This column is just for plotting purposes, not for training
    X_train, X_test, y_train, y_test, len_ngrams_train, len_ngrams_test = train_test_split(
        X, y, len_ngrams, test_size=test_size, random_state=random_state
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    # Fit the linear regression model on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)
    coeff = model.coef_

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R^2):", r2)

    # Add the predicted values to the testing set DataFrame
    df_test = X_test.copy()
    df_test[column_2] = y_test
    df_test['predicted'] = y_pred
    df_test['len_ngrams'] = len_ngrams_test  # Add the len_ngrams for plotting

    # Normalize len_ngrams for color mapping
    len_ngrams_normalized = (len_ngrams_test - len_ngrams_test.min()) / (len_ngrams_test.max() - len_ngrams_test.min())

    # Create a colormap
    cmap = cm.get_cmap('viridis')

    stats = {
        'mse': mse,
        'r2': r2
    }

    save_title = title.replace(' ', '_')
    if save_pth is not None:
        os.makedirs(save_pth, exist_ok=True)
        with open(os.path.join(save_pth, f'{save_title}.json'), 'w') as f:
            print(os.path.join(save_pth, f'{save_title}.json'))
            json.dump(stats, f)
        print(f"Stats saved to {save_pth}")

    # Calculate the mean and standard deviation of y
    y_mean = np.mean(y_test)
    y_std = np.std(y_test)

    # Filter the points within 2 standard deviations
    mask = np.abs(y_test - y_mean) <= 2 * y_std
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]
    y_pred_filtered = y_pred[mask]
    len_ngrams_normalized_filtered = len_ngrams_normalized[mask]

    plt.figure(figsize=(6, 5.5))  # Set the figure size to be square

    # Use the normalized len_ngrams for the color mapping
    scatter = plt.scatter(X_test_filtered, y_test_filtered, 
                          c=cmap(len_ngrams_normalized_filtered), label='Actual',
                          alpha=0.85, edgecolors='k', linewidth=0.5)
    plt.plot(X_test_filtered, y_pred_filtered, color='red', linewidth=2, label='Predicted')

    plt.xlabel(column_1)
    plt.ylabel(column_2)
    plt.title(title)

    # Add the R-squared value to the figure
    plt.text(0.75, 0.1, f'R^2 = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.75, 0.2, f'm = {coeff[0]:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.75, 0.3, f'σ = {y_std:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    plt.legend(loc='upper left')
    plt.colorbar(scatter, label='Length of n-grams')

    if save_pth is not None:
        plt.savefig(os.path.join(save_pth, f'{save_title}.png'))
    plt.show()

    return model, df_test, stats

def plot_len_ngrams_vs_kldiv(df, y_axis='lm_probs_kldiv', save_pth=None,
                             name="ngrams vs kldiv"):
    # Ensure the necessary columns are in the DataFrame
    if 'len_ngrams' not in df or 'lm_probs_kldiv' not in df:
        raise ValueError("DataFrame must contain 'len_ngrams' and 'lm_probs_kldiv' columns.")

    plt.figure(figsize=(6, 6))
    
    # Fit a linear regression model
    model = LinearRegression()
    X = df[['len_ngrams']]
    y = df[y_axis]
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Calculate R-squared value
    r_squared = r2_score(y, predictions)
    
    # Plot the data points
    sns.scatterplot(data=df, x='len_ngrams', y=y_axis, alpha=0.6, edgecolor=None)
    
    # Plot the regression line
    sns.lineplot(x=df['len_ngrams'], y=predictions, color='red', label=f'Linear Regression\nR-squared = {r_squared:.2f}')
    
    plt.title(f'Length of n-grams vs. {y_axis}')
    plt.xlabel('Length of n-grams')
    plt.ylabel(f'{y_axis}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_pth is not None:
        plotname = f"len_ngrams_vs_{y_axis}_{name}.png"
        plt.savefig(os.path.join(save_pth, plotname))
    
    plt.show()

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
        parser.add_argument("--model_results_base_path")
        parser.add_argument("--model", default=None)
        parser.add_argument("--save_pth", default="./figures/translation/dist")
        parser.add_argument("--examples_pth", default=None)
        parser.add_argument("--lang_pair", default="en-fr")
        parser.add_argument("--exp_name")
        parser.add_argument("--debug", action="store_true", default=False)
        args = parser.parse_args()
    return args

def main(args):
    SAVE_PTH = os.path.join(args.save_pth, args.model, args.exp_name)
    NAME = f'log({args.lang_pair}), p=0.975'  # - log(p_en")'
    os.makedirs(SAVE_PTH, exist_ok=True)
    
    # model_results_pth =  "./models/pythia/debug"
    model_results_pth = os.path.join(args.model_results_base_path, args.model, f"TRANSLATION/wmt09-{TASK}/0-shot")
    model_logits_pth = os.path.join(model_results_pth, "logits")
    model_tokens_pth = os.path.join(model_results_pth, "tokens")
    model_logits = load_logits(model_logits_pth)
    model_tokens = load_tokens(model_tokens_pth)
    doc_results = json.load(open(os.path.join(model_results_pth, "doc_results.json"), 'r'))
    doc_results = doc_results[list(doc_results.keys())[0]]

    # %%
    # examples_pth = "./results/n-grams/exp_full/2/examples_dfs_0-shot_common_models.pkl"
    # Common
    # examples_pth = "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue/2/common/lang_dfs_filter_charsFalse_percentile0.999_n_gram2.pkl"
    # All
    # examples_pth = "./results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue/1/all/lang_dfs_filter_charsFalse_percentile0.95.pkl"
    # europarl
    # All
    if args.examples_pth is None:
        examples_pth_all = "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/all/lang_dfs_is_allTrue_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue.pkl"
        # examples_pth_all = "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/all/lang_dfs_is_allTrue_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_filter_stopwordsTrue.pkl"
        # examples_pth_all = "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/all/lang_dfs_is_allTrue_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_filter_stopwordsFalse_remove_englishFalse_remove_non_englishTrue.pkl"
        examples_all = pickle.load(open(examples_pth_all, "rb"))[args.lang_pair]
        examples_all = filter_percentile(examples_all, 0.95)
        # common
        examples_pth_common = "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/common/lang_dfs_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_align_langs0.pkl"
        # examples_pth_common = "./results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/all/lang_dfs_is_allTrue_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_filter_stopwordsTrue.pkl"
        # examples_pth_common = "./incidental-supervision/results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/2/common/lang_dfs_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_filter_stopwordsTrue_align_langs0.pkl"
        examples_common = pickle.load(open(examples_pth_common, "rb"))[args.lang_pair]
        examples_common = filter_percentile(examples_common, 0.975)
        examples_diff = subtract_values(examples_all, examples_common, LANG_COL_ALL, LANG_COL_COMMON, 'value')
    else:
        examples_pth_common = args.examples_pth
        examples_common = pickle.load(open(examples_pth_common, "rb"))[args.lang_pair]
        examples_common = filter_percentile(examples_common, 0.975)

    if args.debug:
        # keep only 1/20th of the data
        examples_common = examples_common.iloc[:len(examples_common) // 20]
        SAVE_PTH = os.path.join(SAVE_PTH, "debug")
        os.makedirs(SAVE_PTH, exist_ok=True)

    # %%
    # test_logits(model_logits, doc_results, tokenizer, model_tokens)

    # %%
    df = examples_common
    # calculate ngram distribution
    ngram_counts = df.groupby(LANG_COL_COMMON)['value'].sum()
    n_gram_dist = (ngram_counts / ngram_counts.sum()).astype(float)
    n_gram_dist = n_gram_dist[n_gram_dist > 0]
    n_gram_log_dist = np.log(n_gram_dist).reset_index()

    # calculate ngram distribution
    # n_gram_dist_all = calc_ngram_dist(examples_all, LANG_COL_ALL, VALUE_COL)
    n_gram_dist_common = calc_ngram_dist(examples_common, LANG_COL_COMMON, VALUE_COL)
    # n_gram_dist = calc_ngram_dist(df, LANG_COL_COMMON, VALUE_COL)

    # subtract ngram log dist all from common
    # n_gram_dist_all_ = n_gram_dist_all.reset_index(name='log_prob')
    n_gram_dist_common_ = n_gram_dist_common.reset_index(name='log_prob')
    log_prob_col = 'log_prob'
    # n_gram_dist_diff = subtract_values(n_gram_dist_common_, n_gram_dist_all_, LANG_COL_COMMON, LANG_COL_ALL, log_prob_col)
    # n_gram_dist_diff.dropna(subset=[log_prob_col + '_diff'], inplace=True)
    # n_gram_dist_diff.reset_index(inplace=True)

    VALUE_COL_DIST= log_prob_col + '_diff'


    n_gram_log_dist = n_gram_log_dist.reset_index().rename(columns={'index': 'lang_2'})

    # %%
    examples_common.head(2)

    # %%
    # examples_common[examples_common['value'] > 0]
    # df_common_factorized = calc_ngram_dist_factorized(examples_common, examples_all, LANG_COL_COMMON, LANG_COL_ALL)
    # VALUE_COL_DIST = 'log_dist_1'

    result_1 = doc_results[0]
    logit_1 = model_logits[result_1['id']]
    tokens_1 = model_tokens[result_1['id']][0]

    print(result_1)
    print(logit_1.shape)
    print(len(tokens_1))

    """
    1. break down the generation into all possible ngrams
    2. search for each possible ngram in the n_gram_log_dist
    3. pick the ones with the highest probability. make sure
    no ngrams overlap in the chosen set
    4. match the chosen ngrams, to the tokens and their associated
    logits
    """
    # %%
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_probs = model_logits

    doc_pth = os.path.join(SAVE_PTH, f"doc_results_ngrams.json")
    # if os.path.exists(doc_pth):
    #     doc_results_new = json.load(open(doc_pth, 'r'))
    # else:
    doc_results_new = process_doc_results(doc_results, model_tokens, model_probs, 
                                        tokenizer, n_gram_dist_common_, 'log_prob',
                                        n_iters=None, save_pth=doc_pth)

    df = create_ngram_dataframe(doc_results_new)
    df_examples = create_prob_seq_df(doc_results_new)
    # %%
    total_set = set()  # Use a set instead of a tuple to store unique ngrams
    for row in doc_results_new:
        chosen_ngrams = row['chosen_ngrams']
        for chosen in chosen_ngrams:
            total_set.add(chosen)  # Add each chosen ngram to the set
    print(f"total_set: {len(total_set)}")

    column_1 = 'ds_probs'
    column_2 = 'lm_probs'
    df = create_ngram_dataframe(doc_results_new)

    # Assuming 'df' is your DataFrame containing the data
    examples_pth_common_dir = os.path.dirname(examples_pth_common)
    save_plot_pth = os.path.join(SAVE_PTH, 'plots')
    model_examples, df_examples_test, stats = fit_and_evaluate_linear_regression(df_examples, 
                                                                                'ds_probs_sum', 
                                                                                'lm_probs_sum',
                                                                                save_pth=save_plot_pth,
                                                                                title=f'Dataset Content (ds) vs. Language Model (lm) gens - {NAME}')
    
    plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_kldiv', 
                             save_plot_pth, NAME)
    if 'lm_probs_mse' in df_examples.columns:
        plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_mse', 
                                 save_plot_pth, NAME)
    if 'lm_probs_cross_entropy' in df_examples.columns:
        plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_cross_entropy', 
                                 save_plot_pth, NAME)
    if 'lm_probs_xinyi' in df_examples.columns:
        plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_xinyi', 
                                 save_plot_pth, NAME)



if __name__ == "__main__":
    args = parse_args()
    models = ['pythia-12b-deduped', 'pythia-6.9b-deduped', 
              'pythia-2.8b-deduped', 'pythia-1.4b-deduped', 
              'pythia-410m-deduped', 'pythia-160m-deduped', 
              'pythia-70m-deduped', 'pythia-31m-deduped', 
              'pythia-14m-deduped']
    
    # 'EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 'EleutherAI/pythia-2.8b'
    models = [ 'EleutherAI/pythia-1.4b', 
                'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                'EleutherAI/pythia-70m', 'EleutherAI/pythia-31m', 
                'EleutherAI/pythia-14m']

    if args.model is None:
        for model in models:
            args.model = model
            print(f"Running model: {model}")
            main(args)
    else:
        main(args)

"""

CUDA_VISIBLE_DEVICES="" python wimbd_translation_ngram_dist.py \
                        --model_results_base_path "./models/experiment_6_logits_max_5/inference/" \
                        --save_pth "./figures/translation/dist" \
                        --exp_name "wmt09-en-fr-0-shot-(en-fr)"
                        


- wmt09gens --
-- 410m
CUDA_VISIBLE_DEVICES="" python wimbd_translation_ngram_dist.py \
                        --model_results_base_path ./models/experiment_6_logits_max_5/inference/ \
                        --save_pth ./figures/translation/dist/wmt09gens/pythia-12b-gens \
                        --lang_pair en-es \
                        --examples_pth ./results/n-grams/wmt09_gens/pile/exp_3/test-set/pythia-12b/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse/2/common/lang_dfs_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesFalse_filter_stopwordsTrue_align_langs0.8_debugFalse.pkl \
                        --model EleutherAI/pythia-12b \
                        --exp_name "wmt09-en-fr-0-shot-(en-fr)"
                        

"""

# # randomly permute x axis
# df_random_x = df_examples_test.copy()
# df_random_x['ds_probs_sum'] = np.random.permutation(df_random_x['ds_probs_sum'].values)
# model_random_x, df_random_x_test, stats_random_x = fit_and_evaluate_linear_regression(df_random_x, 
#                                                                                       'ds_probs_sum', 
#                                                                                       'lm_probs_sum',
#                                                                                       save_pth=model_results_pth,
#  


# %%
# len_ngrams_unique = df_examples['len_ngrams']
# for len_ngram in len_ngrams_unique:
#     df_len_ngram = df_examples[df_examples['len_ngrams'] == len_ngram]
#     model_examples, df_examples_test, stats = fit_and_evaluate_linear_regression(df_len_ngram, 
#                                                                                  'ds_probs_sum', 
#                                                                                  'lm_probs_sum',
#                                                                                  save_pth=model_results_pth,
#                                                                                  title=f'Dataset Content (ds) vs. Language Model (lm) gens - {NAME} - {len_ngram} ngrams')

# %%