# %%
import os
import re
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.wimbd_ import BasePaths as PATHS
from src.wimbd_ import DataConfigs as CONFIG
from src.wimbd_ import post_filter
from src.utils import softmax
from wimbd_process import find_best_match


from datetime import datetime
pd.set_option('display.max_columns', 10)

# Generate a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")

# %%
# Function to clean the text
def clean_text(text):
    pattern = r'(.*?D\..*?)(?:\nAnswer:|$)'
    # Find all matches and keep only up to "D."
    cleaned_text = re.sub(pattern, r'\1', text, flags=re.DOTALL)
    return cleaned_text.strip()


# %%
METHOD = "common"
BASE_PATH = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/results/n-grams/mmlu/pile/exp4_filter/test-set/exp_full_None/5"
dfs_all_models_pth = os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_models.pkl")
df_all_models = pickle.load(open(dfs_all_models_pth, "rb"))

# %%
models = list(df_all_models.keys())
df_all_model = df_all_models[models[0]]

# %%
pd.set_option('display.max_columns', 20)

df_all_model.iloc[0]['example']

# %%
def create_new_df_model(json_file, df_model):
    """
    Update the DataFrame df_model with new model results from a JSON file.
    
    Args:
        df_model (DataFrame): The DataFrame to be updated.
        json_file (str): The path to the JSON file containing the new model results.
    
    Returns:
        DataFrame: The updated DataFrame with the new model results.
    """
    # Load the JSON file containing the new model results
    with open(json_file) as f:
        new_model_results = json.load(f)


    # Flatten the new model results and prepare for DataFrame conversion
    flat_results = []
    for task, results in tqdm(new_model_results.items(), desc="Processing tasks"):
        for result in results:
            query = clean_text(result["query"])
            flat_results.append({
                "query": query,
                "lls": result["result"],
                "gold": result["gold"],
                "probs": np.exp(result["result"]),  # Assuming "result" contains the nlls
                "probs_gold": np.exp(result["result"][result["gold"]])
            })

    # Create a DataFrame from the flattened new model results
    df_new_model = pd.DataFrame(flat_results)

    print(f"df_new_model_cols: {df_new_model.columns}")
    print(f"df_model_cols: {df_model.columns}")

    # Convert 'query' columns to strings and strip whitespace
    df_model['query'] = df_model['query'].astype(str).str.strip()
    df_new_model['query'] = df_new_model['query'].astype(str).str.strip()

    # Define a function to apply the matching process row-wise
    def match_row(row, instances_df, instances_key):
        # Use the find_best_match function to find a match for the 'query'
        match_df = find_best_match(row['query'], instances_df, instances_key)
        if not match_df.empty:
            # If a match is found, return the ID from the match
            return match_df.iloc[0]['id']
        else:
            # If no match is found, return None or some indicator
            return None

    # Apply the match_row function to each row in df_model
    df_new_model['id'] = df_new_model.progress_apply(lambda x: match_row(x, df_model, 'query'), axis=1)

    # Select only the necessary columns from df_new_model for the update
    df_new_model_relevant = df_new_model[['id', 'lls', 'probs', 'probs_gold']].copy()

    # drop this cols from df_model
    df_model.drop(columns=['lls', 'probs', 'probs_gold'], inplace=True)

    # Merge df_model with the relevant columns from df_new_model based on the 'id'
    df_updated_model = pd.merge(df_model, df_new_model_relevant, left_on='id', right_on='id', how='left')

    # Drop the 'id' column and any columns from df_new_model that are not needed
    # df_updated_model.drop(columns=['id', 'id'], inplace=True)
    print(f"columns: {df_updated_model.columns}")
    # Check for rows that only exist in one of the DataFrames
    unmatched_rows = df_updated_model[df_updated_model['lls'].isna()]
    num_unmatched_rows = len(unmatched_rows)

    # Report unmatched rows
    if num_unmatched_rows > 0:
        print(f"Number of unmatched rows: {num_unmatched_rows}")
    else:
        print("All rows matched successfully.")

    return df_updated_model


new_model_results_pth = "./models/experiment_5/inference/allenai/open-instruct-pythia-6.9b-tulu/MMLU/hendrycks*/0-shot/doc_results.json"
df_model = create_new_df_model(new_model_results_pth, df_all_model)

# save the results
new_model_name = "open-instruct-pythia-6.9b-tulu"
df_all_models[new_model_name] = df_model

dfs_all_models_pth_new = os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_models_sft_6_9pythia.pkl")
with open(dfs_all_models_pth, "wb") as f:
    pickle.dump(df_all_models, f)

# %%



