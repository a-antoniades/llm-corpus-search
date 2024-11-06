# %%
import os, re
import json
import pickle
import string
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import torch
import glob

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import jensenshannon

from matplotlib import cm
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

from src.wimbd_ import filter_stop_words_

language_map = {'fr': 'french', 'cs': 'czech', 'de': 'german', 'es': 'spanish',
                'it': 'italian', 'hu': 'hungarian'}

mmlu_tasks = {'prehistory', 'business_ethics', 'econometrics', 'college_medicine', 
            'professional_law', 'philosophy', 'abstract_algebra', 'moral_disputes', 
            'college_chemistry', 'medical_genetics', 'high_school_government_and_politics', 
            'human_aging', 'us_foreign_policy', 'high_school_macroeconomics', 
            'logical_fallacies', 'moral_scenarios', 'college_mathematics', 
            'international_law', 'computer_security', 'sociology', 'professional_psychology', 
            'marketing', 'human_sexuality', 'high_school_chemistry', 'professional_accounting', 
            'college_computer_science', 'anatomy', 'high_school_us_history', 
            'college_biology', 'public_relations', 'high_school_computer_science', 
            'high_school_mathematics', 'college_physics', 'professional_medicine', 
            'high_school_microeconomics', 'clinical_knowledge', 'elementary_mathematics', 
            'machine_learning', 'security_studies', 'nutrition', 'world_religions', 
            'high_school_psychology', 'high_school_geography', 'management', 
            'global_facts', 'high_school_world_history', 'electrical_engineering', 
            'high_school_european_history', 'jurisprudence', 'high_school_physics', 
            'conceptual_physics', 'high_school_statistics', 'virology', 
            'high_school_biology', 'astronomy', 'miscellaneous'}

def load_logits(logits_pth):
    print("loading logits...")
    logit_dict = {}
    all_pths = sorted(glob.glob(os.path.join(logits_pth, "**/*.pt"), recursive=True))
    for pth in tqdm(all_pths):
        logits = torch.load(pth)
        id_ = int(pth.split("_")[-1].split(".")[0])
        logit_dict[id_] = logits
        # print(f"id: {id_}, logits shape: {logits.shape}")
    return logit_dict

def load_tokens(tokens_path):
    print("loading tokens...")
    token_dict = {}
    all_paths = sorted(glob.glob(os.path.join(tokens_path, "**/*.json"), recursive=True))
    
    for path in tqdm(all_paths):
        with open(path, 'r') as f:
            logits = json.load(f)
        
        id_ = int(path.split("_")[-1].split(".")[0])
        token_dict[id_] = logits
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

def calc_ngram_dist(df, lang_col, value_col='value'):
    ngram_counts = df.groupby(lang_col)[value_col].sum()
    n_gram_dist = (ngram_counts / ngram_counts.sum()).astype(float)
    n_gram_dist = n_gram_dist[n_gram_dist > 0]
    n_gram_dist = np.log(n_gram_dist)
    n_gram_dist = n_gram_dist.reset_index(name='log_prob')
    n_gram_dist.rename(columns={lang_col: 'ngram'}, inplace=True)
    print(n_gram_dist)
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

def clean(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def generate_ngrams(text, n):
    all_ngrams = []
    text_vars = [text.split()]
    text = clean(text)
    text_vars.append(text.split())
    if args.task[-2:] in language_map:
        filtered_text = filter_stop_words_(text, language_map[args.task[-2:]])
    else:
        filtered_text = filter_stop_words_(text, 'english')
    text_vars.append(filtered_text)
    for words in text_vars:
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            all_ngrams.append(ngram)
    return list(set(all_ngrams))

def find_ngrams_in_dist(ngrams, n_gram_log_dist, col_name, lang_col):
    ngram_probs = []
    lang_col = lang_col if lang_col in n_gram_log_dist.columns else 'lang_2'
    for ngram in ngrams:
        if ngram in list(n_gram_log_dist[lang_col]):
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
    token_texts = [tokenizer.decode(i) for i in tokens]
    for ngram in chosen_ngrams:
        ngram_tokens = tokenizer.encode(' ' + ngram + ' ' +clean(ngram))
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
        for i, text in enumerate(token_texts):
            if ngram.find(text) > -1 or ngram.find(text.lower()) > -1:
                token_indices.append(i)
        token_indices = list(set(token_indices))
        print(f"ngram: {ngram}, sentence: {token_texts}")
        print(f"ngram tokens: {ngram_tokens}, Token indices: {token_indices}")
        if len(token_indices) > 0:
            ngram_logit = (probs[token_indices].log().sum() * (len(ngram_tokens)/len(token_indices))).item()
        else:
            ngram_logit = 0.0
        ngram_probs.append((ngram, ngram_logit))
    # if len(ngram_probs) == 0:
        # print(f"No ngrams found in tokens")
    return ngram_probs

def process_doc_results(doc_results, model_tokens, model_probs, tokenizer, 
                        n_gram_log_dist, value_col, n_iters=None, save_pth=None):
    print("matching ngram searching results with model logits...")
    if n_iters is None:
        n_iters = len(doc_results)
        
    for i in tqdm(range(n_iters)):
        doc_results[i]['ngram_ds_probs'] = {}
        doc_results[i]['ngram_lm_probs'] = {}
        doc_results[i]['chosen_ngrams'] = {}
        doc_results[i]['len_ngrams'] = {}
        result = doc_results[i]
        
        if model_tokens is None:
            tokens = probs = []
        else:
            generation = result['result'][0]
            tokens = model_tokens[result['id']][0]
            probs = model_probs[result['id']][0]
        
        if len(tokens) == len(probs):
            for n in args.ngrams:
                if model_tokens is None:
                    ngrams = [result['ngram']]
                    try: 
                        len_ngrams = result["len_ngram"]
                    except:
                        print("no len_ngram in results file")
                        continue
                    ngram_ds_probs = [(result['ngram'], result["probs"])]
                    chosen_ngrams = [p[0] for p in ngram_ds_probs]
                    ngram_lm_probs = [(result['ngram'], -result['loss'] * len_ngrams)]
                else:
                    ngrams = generate_ngrams(generation, n=n)
                    ngram_ds_probs = find_ngrams_in_dist(ngrams, n_gram_log_dist[n], value_col, 'ngram')
                    print(ngram_ds_probs)
                    # chosen_ngrams = pick_nonoverlapping_ngrams(ngram_ds_probs)
                    chosen_ngrams = list(set([p[0] for p in ngram_ds_probs]))
                    ngram_lm_probs = match_ngrams_to_probs(chosen_ngrams, tokens, probs, tokenizer)
                    chosen_ngrams = list(set([p[0] for p in ngram_lm_probs]))
                    len_ngrams = len(chosen_ngrams)
                
                if len_ngrams > 0:
                    # save results
                    doc_results[i]['ngram_ds_probs'][n] = ngram_ds_probs
                    doc_results[i]['chosen_ngrams'][n] = chosen_ngrams
                    doc_results[i]['ngram_lm_probs'][n] = ngram_lm_probs
                    doc_results[i]['len_ngrams'][n] = len_ngrams
                    print(chosen_ngrams)
                    print(ngram_ds_probs)
                    print(ngram_lm_probs)
                    print(len_ngrams)
    
    if save_pth is not None:
        os.makedirs(os.path.dirname(save_pth), exist_ok=True)
        with open(save_pth, 'w') as f:
            json.dump(doc_results, f)
            
    return doc_results

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
    else:
        raise ValueError("Unsupported method for similarity calculation.")
    
    return similarity

def create_prob_seq_df(doc_results_new, ds_temp=1, lm_temp=0.1):
    data = []
    count = 0
    
    for row in doc_results_new:
        all_len_ngrams = []
        all_ds_log_probs = []
        all_ds_log_probs_sum = []
        all_lm_log_probs = []
        all_lm_log_probs_sum = []
        all_lm_probs_kldiv = []
        all_lm_probs_mse = []
        all_lm_probs_cross_entropy = []
        # print(row)
        
        for n in args.ngrams:
            try:
                chosen_ngrams = row['chosen_ngrams'][n]
            except:
                n = str(n)
                try:
                    chosen_ngrams = row['chosen_ngrams'][n]
                except:
                    print("cannot read results")
                    continue
                
            if len(chosen_ngrams) > 0:
                ngram_ds_log_probs = dict(row['ngram_ds_probs'][n])
                ds_log_probs = np.array([ngram_ds_log_probs[ngram] for ngram in chosen_ngrams])
                ds_log_probs_sum = np.sum(ds_log_probs)
                ds_probs = np.exp(ds_log_probs/ds_temp)
                ngram_lm_log_probs = dict(row['ngram_lm_probs'][n])
                lm_log_probs = np.array([ngram_lm_log_probs[ngram] for ngram in chosen_ngrams])
                lm_log_probs_sum = np.sum(lm_log_probs)
                lm_probs = np.exp(lm_log_probs/lm_temp)
                
                all_len_ngrams.append(len(chosen_ngrams))
                all_ds_log_probs.append(ds_log_probs)
                all_ds_log_probs_sum.append(ds_log_probs_sum)
                all_lm_log_probs.append(lm_log_probs)
                all_lm_log_probs_sum.append(lm_log_probs_sum)
                
                if args.task[:3] == '-en':
                    lm_probs_kldiv = calculate_similarity(lm_probs, ds_probs, method='kl_divergence')
                    # lm_probs_corr = calculate_similarity(lm_probs, ds_probs, method='correlation')
                    lm_probs_mse = calculate_similarity(lm_probs, ds_probs, method='mse')
                    lm_probs_cross_entropy = calculate_similarity(lm_probs, ds_probs, method='cross_entropy')
                    # lm_probs_xinyi = calculate_similarity(lm_probs, ds_probs, method='xinyi')
                    all_lm_probs_kldiv.append(lm_probs_kldiv)
                    all_lm_probs_mse.append(lm_probs_mse)
                    all_lm_probs_cross_entropy.append(lm_probs_cross_entropy)
                else:
                    all_lm_probs_kldiv.append(0)
                    all_lm_probs_mse.append(0)
                    all_lm_probs_cross_entropy.append(0)
            else:
                # all_len_ngrams.append(0.0)
                # all_ds_probs.append(0.0)
                break
        
        if len(all_len_ngrams) == len(args.ngrams):
            
            if args.task[:3] == 'en-':
                example = row['result'][0]
                i = row['id']
            else:
                example = row['text']
                i = count
            
            data.append({
                'example': example,
                'id': i,
                'ngrams': chosen_ngrams,
                'ds_log_probs_sum': all_ds_log_probs_sum,
                'ds_log_probs': all_ds_log_probs,
                'lm_log_probs': all_lm_log_probs,
                'lm_log_probs_sum': np.sum(all_lm_log_probs_sum),
                'len_ngrams': [row['len_ngrams'][n] for n in args.ngrams],
                'lm_probs_kldiv': all_lm_probs_kldiv,
                'lm_probs_mse': all_lm_probs_mse,
                'lm_probs_cross_entropy': all_lm_probs_cross_entropy,
                # 'lm_probs_xinyi': lm_probs_xinyi if len_ngrams and len_probs > 0 else None
            })
            
            count += 1
            
    return pd.DataFrame(data).dropna()


def fit_and_evaluate_linear_regression(df, x_column, y_column,
                                    test_size=0.2, random_state=42,
                                    save_pth=None, title='Linear Regression'):
    
    # Ensure 'len_ngrams' is in the DataFrame and is not the target or a feature
    if 'len_ngrams' not in df:
        raise ValueError("The DataFrame does not contain the 'len_ngrams' column.")

    # len_min, len_max = -3, 0
    # df = df[(df[column_2] >= len_min) & (df[column_2] <= len_max)]

    # Split the data into training and testing sets
    X = np.array(df[x_column].to_list()) # Independent variable
    y = np.array(df[y_column].to_list()) # Dependent variable
    len_ngrams = np.array(df['len_ngrams'].to_list())  # This column is just for plotting purposes, not for training
    print(X)
    print(y)
    print(len_ngrams)
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

    # Normalize len_ngrams for color mapping
    len_ngrams_normalized = (len_ngrams_test - len_ngrams_test.min(0)) / (len_ngrams_test.max(0) - len_ngrams_test.min(0))

    # Create a colormap
    cmap = cm.get_cmap('viridis')

    stats = {
        'mse': mse,
        'r2': r2,
        'coeff': list(coeff)
    }
    
    print(stats)

    save_title = title.replace(' ', '_')
    if save_pth is not None:
        os.makedirs(save_pth, exist_ok=True)
        with open(os.path.join(save_pth, f'{save_title}.json'), 'w') as f:
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
    
    for i in range(len(args.ngrams)):
        x_test_i = [x[i] for x in X_test_filtered]
        
        plt.figure(figsize=(7, 5.5))  # Set the figure size to be square

        # # Use the normalized len_ngrams for the color mapping
        # scatter = plt.scatter(x_test_i, y_test_filtered, 
        #                     c=cmap(len_ngrams_normalized_filtered[:, i]), label='Actual',
        #                     alpha=0.85, edgecolors='k', linewidth=0.5)
        scatter = plt.scatter(x_test_i, y_test_filtered, label='Actual',
                    alpha=0.85, edgecolors='k', linewidth=0.5)
        plt.plot(x_test_i, y_pred_filtered, color='black', 
                    linewidth=2, label='Predicted', linestyle='--')

        plt.xlabel('log data ngram prob')
        plt.ylabel('log LM ngram prob')
        plt.title('Data distribution v.s. LM distribution')

        # Add the R-squared value to the figure
        plt.text(0.75, 0.1, f'R2 = {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.75, 0.2, f'm = {coeff[0]:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.75, 0.3, f'σ = {y_std:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        plt.legend(loc='upper left')
        # plt.colorbar(scatter, ticks=[], label='Num of n-grams per example')

        if save_pth is not None:
            plt.savefig(os.path.join(save_pth, f'data_vs_lm_dist_{args.task}.pdf'))
            plt.savefig(os.path.join(save_pth, f'data_vs_lm_dist_{args.task}.png'))
        plt.close()

    return model, stats

def plot_len_ngrams_vs_kldiv(df, y_axis='lm_probs_kldiv', save_pth=None):
    # Ensure the necessary columns are in the DataFrame
    if 'len_ngrams' not in df or 'lm_probs_kldiv' not in df:
        raise ValueError("DataFrame must contain 'len_ngrams' and 'lm_probs_kldiv' columns.")
    
    X = np.array(df['len_ngrams'].to_list()) # Independent variable
    y = np.array(df[y_axis].to_list()) # Dependent variable
    stats = {}
    
    for i, n in enumerate(args.ngrams):
        plt.figure(figsize=(6, 6))
        # Fit a linear regression model
        model = LinearRegression()
        X_n = np.expand_dims(X[:, i], axis=-1).astype(np.float64)
        y_n = y[:, i]
        model.fit(X_n, y_n)
        
        # Make predictions
        predictions = model.predict(X_n)
        
        # Calculate R-squared value
        r2 = r2_score(y_n, predictions)
        mse = mean_squared_error(y_n, predictions)
        coeff = model.coef_
        
        stats[n] = {
            'mse': mse,
            'r2': r2,
            'coeff': list(coeff),
            'mean': np.mean(y_n)
        }
        print(stats)
        
        # Plot the data points
        plt.scatter(X[:, i], y_n, alpha=0.6, edgecolor=None)
        
        # Plot the regression line
        plt.plot(X[:, i], predictions, color='black', 
                    linewidth=2, label=f'Linear Regression\nR2 = {r2:.2f}', linestyle='--')
        
        plt.title(f'Length of {n}-grams vs. {y_axis}')
        plt.xlabel(f'Length of {n}-grams')
        plt.ylabel(f'{y_axis}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_pth is not None:
            plotname = f"len_{n}-grams_vs_{y_axis}.png"
            plt.savefig(os.path.join(save_pth, plotname))
        
        plt.close()
    
    save_title =f'ngrams_len_vs_{y_axis}'
    if save_pth is not None:
        os.makedirs(save_pth, exist_ok=True)
        with open(os.path.join(save_pth, f'{save_title}.json'), 'w') as f:
            json.dump(stats, f)
        print(f"Stats saved to {save_pth}")

def main(args):
    ngrams = set(args.ngrams)
    
    n_gram_log_dist = {}
    for ngram in ngrams:
        # /share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/results/n-grams/wmt/pile/exp4/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaTrue/
        if args.all:
            examples_dir = f"data/translation/{ngram}/all"
            lan_col = args.task[-2:]
        else:
            if args.task[:3] == 'en-':
                # examples_dir = f"data/translation/{ngram}/common"
                examples_dir = f"/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/results/n-grams/europarl/pile/exp4/n_samples_20000_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaTrue/{ngram}/common"
                task = args.task[-2:] + '-en'
                
                if ngram in [1, 2, 7]:
                    # lan_col = 'lang_1'
                    # examples_pth = f'{examples_dir}/lang_dfs_filter_charsFalse_percentile0_detect_langTrue_filter_entitiesTrue_filter_stopwordsTrue_align_langs0.pkl' 
                    lan_col = 'lang_2'
                    examples_pth = f'{examples_dir}/lang_dfs_filter_charsFalse_percentile0_detect_langFalse_filter_entitiesTrue_filter_stopwordsTrue_align_langs0.pkl'
                    df = pickle.load(open(examples_pth, "rb"))[task]
                elif ngram in [3, 4, 5, 13]:
                    examples_pth = f'{examples_dir}/{args.task}.pkl'
                    if os.path.exists(examples_pth):
                        lan_col = 'lang_2'
                        d = pickle.load(open(examples_pth, "rb"))
                        df = pd.DataFrame(d).T.reset_index(names=['lang_1', 'lang_2'])
                    else:
                        examples_pth = f'{examples_dir}/{task}.pkl'
                        if os.path.exists(examples_pth):
                            lan_col = 'lang_1'
                            try:
                                d = pickle.load(open(examples_pth, "rb"))
                                df = pd.DataFrame(d).T.reset_index(names=['lang_1', 'lang_2'])
                            except:
                                print(f'No ngrams files for {args.task}, ngram={ngram}')
                                exit(1)
                            
            elif args.task == 'trivia_qa':
                examples_dir = f"data/trivia_qa/{ngram}/"
                lan_col = 'A'
                examples_pth = f'{examples_dir}/common/task_df_filter_charsTrue_percentile0.pkl'
                df = pickle.load(open(examples_pth, "rb"))['triviaqa']
                
            elif args.task == 'mmlu':
                examples_dir = f"data/mmlu/{ngram}/"
                lan_col = 'A'
                examples_pth = f'{examples_dir}/examples_dfs_common_models.pkl'
                df = pickle.load(open(examples_pth, "rb"))
    
            else:
                examples_dir = f"data/mmlu/{ngram}/common"
                lan_col = 'A'
                examples_pth = f'{examples_dir}/{args.task}.pkl'
                d = pickle.load(open(examples_pth, "rb"))
                df = pd.DataFrame(d).T.reset_index(names=['Q', 'A'])
                            
        # df = filter_percentile(df, 0.975)

        if args.debug:
            # keep only 1/20th of the data
            df = df.iloc[:len(df) // 20]

        if args.task[:3] == 'en-':
            # calculate ngram distribution
            n_gram_log_dist[ngram] = calc_ngram_dist(df, lan_col)

    """
    1. break down the generation into all possible ngrams
    2. search for each possible ngram in the n_gram_log_dist
    3. pick the ones with the highest probability. make sure
    no ngrams overlap in the chosen set
    4. match the chosen ngrams, to the tokens and their associated
    logits
    """
    for model in args.models:
        
        SAVE_PTH = os.path.join(args.save_pth, args.task, '-'.join([str(n) for n in ngrams]), model)
        os.makedirs(SAVE_PTH, exist_ok=True)
        
        if args.debug:
            SAVE_PTH = os.path.join(SAVE_PTH, "debug")
            os.makedirs(SAVE_PTH, exist_ok=True)
        
        # model_results_pth =  "./models/pythia/debug"
        if args.task[:3] == 'en-':
            model_results_pth = f"/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_6_logits_max_5/inference/{model}/TRANSLATION/wmt09-{args.task}/{args.shots}-shot"
            model_logits_pth = os.path.join(model_results_pth, "logits")
            model_tokens_pth = os.path.join(model_results_pth, "tokens")
            model_logits = load_logits(model_logits_pth)
            model_tokens = load_tokens(model_tokens_pth)
            doc_results = json.load(open(os.path.join(model_results_pth, "doc_results.json"), 'r'))
            doc_results = doc_results[f'wmt09-{args.task}']
        else:
            model_tokens = None
            model_logits = None
            doc_results = []
            if args.task == 'mmlu':
                for task in mmlu_tasks:
                    model_results_pth = f"out/{task}/{model}/loss.jsonl"
                    try:
                        with open(model_results_pth) as rf:
                            for l in rf:
                                doc_results.append(json.loads(l.strip()))
                    except:
                        print("no file: ", model_results_pth)
                        continue
            else:
                model_results_pth = f"out/{args.task}/{model}/loss.jsonl"
                try:
                    with open(model_results_pth) as rf:
                        for l in rf:
                            doc_results.append(json.loads(l.strip()))
                except:
                    print("no file: ", model_results_pth)
                    return
            
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        if args.processed_data is not None and os.path.exists(args.processed_data):
            doc_results_new = json.load(open(args.processed_data))
        # elif os.path.exists(f'{SAVE_PTH}/processed_data.json'):
        #     doc_results_new = json.load(open(f'{SAVE_PTH}/processed_data.json'))
        else:
            doc_results_new = process_doc_results(doc_results, model_tokens, model_logits, 
                                            tokenizer, n_gram_log_dist, 'log_prob',
                                            n_iters=None, save_pth=f'{SAVE_PTH}/processed_data.json')

        df_examples = create_prob_seq_df(doc_results_new, lm_temp=args.temperature)
        total_set = {n: set() for n in args.ngrams}  # Use a set instead of a tuple to store unique ngrams
        for row in doc_results_new:
            for n in args.ngrams:
                try:
                    chosen_ngrams = row['chosen_ngrams'][n]
                except:
                    try:
                        chosen_ngrams = row['chosen_ngrams'][str(n)]
                    except:
                        continue
                for chosen in chosen_ngrams:
                    total_set[n].add(chosen)  # Add each chosen ngram to the set
        print(f"total_set: {len(total_set)}")

        save_plot_pth = os.path.join(SAVE_PTH, 'plots')
        model_examples, stats = fit_and_evaluate_linear_regression(df_examples, 
                                'ds_log_probs_sum', 
                                'lm_log_probs_sum',
                                save_pth=save_plot_pth,
                                title=f'Dataset Content (ds) vs. Language Model (lm) gens')
        
        plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_kldiv', save_plot_pth)
        if 'lm_probs_mse' in df_examples.columns:
            plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_mse', save_plot_pth)
        if 'lm_probs_cross_entropy' in df_examples.columns:
            plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_cross_entropy', save_plot_pth)
        # if 'lm_probs_xinyi' in df_examples.columns:
        #     plot_len_ngrams_vs_kldiv(df_examples, 'lm_probs_xinyi', save_plot_pth)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--models", nargs="+", type=str, default=None)
    parser.add_argument("--save_pth", type=str, default="./figures/translation/dist")
    parser.add_argument("--task", type=str, default='mmlu')
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument('--ngrams', nargs="+", type=int, default=2, help='value of n', )
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--processed_data", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    
    if args.models is None:
        args.models = ['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 
                        'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
                        'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                        'EleutherAI/pythia-70m', 'EleutherAI/pythia-31m', 
                        'EleutherAI/pythia-14m']

    if args.task == 'mmlu':
        args.models = ['allenai/OLMo-7B', 'allenai/OLMo-7B-SFT']

    main(args)


"""

python plot_ngram_dist_xinyi.py \
                    --task en-fr \
                    --ngrams 2 


"""


