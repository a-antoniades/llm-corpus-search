import os
import json
import collections
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import DatasetDict
from transformers import set_seed

import numpy as np
set_seed(42)


def concatenate_columns(example, new_col_name='concatenated_text'):
    """
    Concatenate multiple columns in a dataset example into a single text,
    including the column name before each content. Also removes certain characters.
    
    :param example: A single example from a dataset.
    :param new_col_name: Name of the new column to store the concatenated text.
    :return: Updated example with concatenated text in a new column.
    """
    old_cols = list(example.keys())
    skip_cols = ['id', 'concept_set_idx', 'eid', 'subtree_was_extended']
    concatenated_text = " ".join(f"{col}: {example[col]}" for col in old_cols if col not in skip_cols)
    
    # Remove certain characters from concatenated_text
    characters_to_remove = ["{", "}"]
    for char in characters_to_remove:
        concatenated_text = concatenated_text.replace(char, "")
    
    # Store concatenated text in a new column
    example[new_col_name] = concatenated_text
    
    # Remove the old columns
    for col in old_cols:
        example.pop(col)
    
    return example

def sample_and_concatenate_split(dataset, split, proportion, new_col_name='text'):
    """
    Sample and concatenate columns of a given split of a dataset.
    
    :param dataset: The dataset object.
    :param split: The split name, e.g., 'train' or 'validation'.
    :param proportion: Proportion of the dataset to sample.
    :param new_col_name: Name of the new column to store the concatenated text.
    :return: The sampled and modified dataset split.
    """
    if split not in dataset:
        return None
    
    # Sample the dataset split
    if proportion >= 1.0:
        sampled_dataset = dataset[split]
    else:
        num_examples = int(len(dataset[split]) * proportion)
        indices = np.random.choice(range(len(dataset[split])), num_examples, replace=False)
        sampled_dataset = dataset[split].select(indices)
    
    # Concatenate columns if there are more than one
    if len(sampled_dataset.column_names) > 1:
        print(f"Concatenating columns of split '{split}'...")
        sampled_dataset = sampled_dataset.map(lambda example: concatenate_columns(example, new_col_name=new_col_name))
    
    return sampled_dataset


def load_and_sample_dataset(dataset_config, cache_dir):
    """
    Load and sample the dataset based on the given configuration.
    
    :param dataset_config: Dictionary with dataset configuration.
    :param cache_dir: Directory where the datasets are cached.
    :return: A tuple containing the sampled train and validation datasets.
    """
    dataset_name = dataset_config['dataset_name']
    dataset_config_name = dataset_config['dataset_config_name']
    proportion = dataset_config['p']
    use_auth_token = dataset_config.get('use_auth_token', False)
    # streaming = dataset_config.get('streaming', False)

    # Load dataset
    dataset = load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
    )

    # Sample training dataset
    if 'train_split' in dataset_config:
        train_dataset = sample_and_concatenate_split(dataset, dataset_config['train_split'], proportion)
    else:
        train_dataset = None

    # Sample validation dataset
    if 'validation_split' in dataset_config:
        validation_dataset = sample_and_concatenate_split(dataset, dataset_config['validation_split'], proportion)
    else:
        validation_dataset = None

    return train_dataset, validation_dataset

def merge_datasets(*dataset_configs, cache_dir="./cache"):
    """
    Merge datasets based on the given configurations.
    
    :param dataset_configs: List of dictionaries with dataset configurations.
    :param cache_dir: Directory where the datasets are cached.
    :return: A dictionary containing the concatenated train and validation datasets.
    """
    # Dictionary to hold the datasets
    datasets_by_type = {
        'train': {'QA': [], 'text': []}, 
        'validation': []
    }
    
    # Load and split datasets
    for config in dataset_configs:
        print(f"Processing dataset {config['dataset_name']}...")
        train_dataset, validation_dataset = load_and_sample_dataset(config, cache_dir)
        ds_type = config['dataset_type']

        # Add the datasets to the respective lists
        if train_dataset is not None:
            datasets_by_type['train'][ds_type].append(train_dataset)
        if validation_dataset is not None:
            datasets_by_type['validation'].append(validation_dataset)
    

    # Randomly sample from 'text' datasets
    qa_train_datasets = datasets_by_type['train']['QA']
    text_train_datasets = datasets_by_type['train']['text']

    # Get the total number of examples for each ds_type
    n_qa_train_examples = sum(len(ds) for ds in qa_train_datasets)
    n_text_train_examples = sum(len(ds) for ds in text_train_datasets)
    assert n_text_train_examples > n_qa_train_examples, f"Number of text examples ({n_text_train_examples}) \
                                                        is less than the number of QA examples ({n_qa_train_examples})."
    
    sampled_text_train_datasets = []
    for n, ds in enumerate(text_train_datasets):
        n_examples_remove = int((len(ds) / n_text_train_examples) * n_qa_train_examples)
        print(f"Removing {n_examples_remove} examples from dataset {n}...")
        if n_examples_remove <= 0:
            sampled_text_train_datasets.append(ds)
        else:
            assert n_examples_remove < len(ds), f"Number of examples to remove ({n_examples_remove}) \
                                                is greater than the number of examples in the dataset ({len(ds)})."
            # Randomly select indices to remove
            indices_to_remove = np.random.choice(len(ds), n_examples_remove, replace=False)
            print("done")
            # Create a boolean array representing whether each index should be kept
            mask = np.ones(len(ds), dtype=bool)
            mask[indices_to_remove] = False
            print("done2")
            # Get the indices to keep
            indices_to_keep = np.arange(len(ds))[mask]
            print("done3")
            sampled_text_train_datasets.append(ds.select(indices_to_keep))
            print("done4")

    # Concatenate datasets
    concatenated_train = concatenate_datasets(datasets_by_type['train']['QA'] + sampled_text_train_datasets)
    concatenated_validation = concatenate_datasets(datasets_by_type['validation'])
    print("done5")
    # Convert to HuggingFace datasets
    raw_datasets = { 
        'train': concatenated_train, 
        'validation': concatenated_validation
    }
    print("done6")

    return raw_datasets


if __name__ == "__main__":
    CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
    DATASET_DIR = os.path.join(CACHE_DIR, "merged_datasets")
    P_QA = 0
    
    dataset_configs = [
        {
            'dataset_type': 'QA',
            'dataset_name': 'common_gen',
            'dataset_config_name': 'common_gen',
            'p': P_QA,
            'train_split': 'train',
        },
        {
            'dataset_type': 'QA',
            'dataset_name': 'e2e_nlg',
            'dataset_config_name': 'e2e_nlg',
            'p': P_QA,
            'train_split': 'train',
        },
        {
            'dataset_type': 'QA',
            'dataset_name': 'dart',
            'dataset_config_name': 'dart',
            'p': 1,
            'validation_split': 'validation',
        },
        {
            'dataset_type': 'QA',
            'dataset_name': 'web_nlg',
            'dataset_config_name': 'release_v3.0_en',
            'p': 1,
            'validation_split': 'test',
        },
        {
            'dataset_type': 'text',
            'dataset_name': 'wikitext',
            'dataset_config_name': 'wikitext-2-v1',
            'p': 1,
            'train_split': 'train',
        },
        {
            'dataset_type': 'text',
            'dataset_name': 'bookcorpus',
            'dataset_config_name': None,
            'p': 1,
            'train_split': 'train',
        }
    ]
    merged_datasets = merge_datasets(*dataset_configs, cache_dir=CACHE_DIR)

    # export newly created dataset
    ds_folder = os.path.join(DATASET_DIR, f"dataset_{P_QA}")
    print(f"Saving dataset to {ds_folder}...")
    if not os.path.exists(ds_folder):
        os.makedirs(ds_folder)
    n_folders = len(os.listdir(DATASET_DIR))
    ds_folder = os.path.join(DATASET_DIR, f"dataset_{P_QA}")
    for split in merged_datasets.keys():
        merged_datasets[split].save_to_disk(os.path.join(ds_folder, f"dataset_{split}.arrow"))
    # save config
    with open(os.path.join(ds_folder, "config.json"), 'w') as f:
        json.dump(dataset_configs, f)
    # Print example dataset information
    print(merged_datasets)