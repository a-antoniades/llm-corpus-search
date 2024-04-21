import os
import glob
import collections
import json
import pickle
from tqdm import tqdm
from natsort import natsorted

from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from src.wimbd_ import _load_dataset, WimbdTasks, filter_percentile


class NgramDist:

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

    def test_logits(model_logits, doc_results, tokenizer, model_tokens):
        for idx, row in enumerate(doc_results):
            id_ = row['id']
            logits = model_logits[id_]
            gen_text = row['result'][0]
            
            # tokens = tokenizer(gen_text, return_tensors='pt', add_special_tokens=False)
            tokens = model_tokens[id_][0] if id_ in model_tokens else tokenizer.encode(gen_text)
            decoded_text = tokenizer.decode(tokens) # [:len(gen_text)])
            # Assuming 'tokens' is a list of token IDs:
            decoded_tokens = [tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in tokens]
            decoded_text_with_delimiter = ' | '.join(decoded_tokens)
            
            # Align logits with tokens, including special tokens
            # This assumes that logits are for each token in the sequence, including special tokens.
            aligned_logits = logits[:, :len(tokens)]
            
            # print(f"id: {id_}, gen: {gen_text}")
            print(f"logits: {logits.shape}, tokens: {len(tokens)}, aligned_logits: {aligned_logits.shape}")
            print(f"ori_text: {gen_text} -> {len(gen_text)}") 
            print(f"dec_text: {decoded_text} -> {len(decoded_text)}")
            print(f"dec_tokens: {decoded_text_with_delimiter}")
            
            if idx > 1:
                break



    


