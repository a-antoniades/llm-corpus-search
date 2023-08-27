import os
from typing import Optional
import json
import collections
from datasets import load_dataset, load_from_disk, Dataset
from datasets import concatenate_datasets
from datasets import DatasetDict
from transformers import set_seed

from promptsource.templates import DatasetTemplates, TemplateCollection

import random
import numpy as np
set_seed(42)

NEW_COL_NAME = 'text'


def concatenate_cols(example, columns: Optional[dict] = None, new_col_name=NEW_COL_NAME):
    """
    Concatenate multiple columns in a dataset example into a single text,
    including the column name before each content. Also removes certain characters.
    
    :param example: A single example from a dataset.
    :param new_col_name: Name of the new column to store the concatenated text.
    :return: Updated example with concatenated text in a new column.
    """
    def find_value(d, key):
        """
        Recursively search for key
        in possibly nested dictionaries.
        """
        if key in d:
            return d[key]
        for k, v in d.items():
            if isinstance(v, dict):
                item = find_value(v, key)
                if item is not None:
                    return item
        return None
    # old_cols = list(example.keys())
    # skip_cols = ['id', 'concept_set_idx', 'eid', 'subtree_was_extended']
    if columns is None:
        concatenated_text = example['text']
    else:
        concatenated_text = " ".join(f"{new_col} {str(find_value(example, old_col))}" for old_col, new_col in columns.items())
    #     concatenated_text = " ".join(
    #     f"{new_col}: {' '.join(str(item) for item in find_value(example, old_col))}"
    #     if isinstance(find_value(example, old_col), list)
    #     else f"{new_col}: {str(find_value(example, old_col))}"
    #     for old_col, new_col in columns.items()
    # )
    
    # Remove certain characters from concatenated_text
    characters_to_remove = ["{", "}"]
    for char in characters_to_remove:
        concatenated_text = concatenated_text.replace(char, "")
    
    # Store concatenated text in a new column
    new_example = {new_col_name: concatenated_text}
    
    # print(f"new_example: {new_example}")
    
    return new_example
    
def sample_split(dataset, proportion, max_examples=None):
    """
    Sample and concatenate columns of a given split of a dataset.
    
    :param dataset: The dataset object.
    :param split: The split name, e.g., 'train' or 'validation'.
    :param proportion: Proportion of the dataset to sample.
    :param new_col_name: Name of the new column to store the concatenated text.
    :return: The sampled and modified dataset split.
    """
    
    # Sample the dataset split
    if proportion >= 1.0:
        sampled_dataset = dataset
    else:
        num_examples = int(len(dataset) * proportion)
        indices = np.random.choice(range(len(dataset)), num_examples, replace=False)
        sampled_dataset = dataset.select(indices)

    if max_examples is not None:
        print(f"Sampling {max_examples} examples from the dataset.")
        indexes = np.random.choice(range(len(sampled_dataset)), max_examples, replace=False)
        sampled_dataset = sampled_dataset.select(indexes)
    
    return sampled_dataset

def remove_redundant_columns(dataset, columns_to_keep=['text']):
    if isinstance(columns_to_keep, str):
        columns_to_keep = [columns_to_keep]
    columns_not_keep = set(dataset.column_names) - set(columns_to_keep)
    dataset = dataset.remove_columns(columns_not_keep)
    return dataset

def concatenate_columns(dataset, columns, new_col_name=NEW_COL_NAME):
    # Concatenate columns if there are more than one
    if len(dataset.column_names) > 1:
        print(f"dataset_cols: {dataset.column_names}")
        dataset = dataset.map(lambda example: concatenate_cols(example, columns, new_col_name=new_col_name))

    return remove_redundant_columns(dataset, new_col_name)

def to_promptsource(dataset, templates, is_validation=False, new_col_name=NEW_COL_NAME):
    print(f"Converting to promptsouce with templates: {templates}")
    
    def apply_template(example, template, is_validation=False, new_col_name=NEW_COL_NAME):
        if isinstance(template, list):
            template = random.choice(template)
        if is_validation:
            prompt, answer = template.apply(example)
            return {
                new_col_name: prompt,
                'answer_choices': template.answer_choices,
                'answer': answer
                }
        else:
            return {
                new_col_name: template.apply(example),
                'answer_choices': template.answer_choices
                }
    
    prompted_dataset = dataset.map(lambda example: apply_template(example, templates,
                                                                  is_validation, new_col_name=new_col_name))

    # if column is list of lists, make into list of strings
    if isinstance(prompted_dataset[new_col_name][0], list) and not is_validation:
        prompted_dataset = prompted_dataset.map(lambda example: {new_col_name: ' '.join(example[new_col_name])})
        prompted_dataset = remove_redundant_columns(prompted_dataset, new_col_name)
    elif is_validation:
        prompted_dataset = remove_redundant_columns(prompted_dataset, [new_col_name, 'answer_choices', 'answer'])

    return prompted_dataset
    

def map_labels(dataset, label_mapping, column_mapping):
    """
    Map labels in a dataset to new labels.
    
    :param dataset: The dataset object.
    :param label_mapping: A dictionary mapping old labels to new labels.
    :return: The dataset with mapped labels.
    """
    print(f"Mapping labels: {label_mapping}")
    text_column = list()
    label_column = 'label'
    # new_label_column = column_mapping[label_column]
    label_mapping = label_mapping[label_column]

    def map_label(example):
        # try:
        example[label_column] = label_mapping[example[label_column]]
        # except:
        #     # print(f"Label {example[label_column]} not found in mapping, skipping")
        #     # skip example if label not found in mapping
        #     example[label_column] = None
        return example

    # map labels and replace the original column
    # convert column to string first, 
    from datasets import ClassLabel, Features, Value
    new_dataset = dataset.features.copy()    
    new_dataset[label_column] = Value('string')
    # round to 1 decimal place
    dataset = dataset.cast(new_dataset)
    dataset = dataset.map(map_label)

    dataset = concatenate_columns(dataset, columns=column_mapping, new_col_name=NEW_COL_NAME)

    # remove the original label column
    dataset = remove_redundant_columns(dataset, columns_to_keep=[NEW_COL_NAME])

    return dataset

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
    train_split = dataset_config.get('train_split', None)
    validation_split = dataset_config.get('validation_split', None)
    label_mapping = dataset_config.get('mapping', None)
    # streaming = dataset_config.get('streaming', False)

    columns = dataset_config.get('columns', None)

    def process_dataset(dataset, dataset_name, dataset_config_name, 
                        is_promptsource, proportion, column_mapping, label_mapping, dataset_config):

        is_validation = dataset_config.get('validation_split', False) != False

        if label_mapping is not None:
            mapping_labels = set(label_mapping['label'].keys())
            print(f"mapping_labels: {mapping_labels}")
            print(f"dataset_labels: {set(dataset['label'])}")
            dataset = dataset.filter(lambda example: str(example['label']) in mapping_labels)
            assert len(dataset) > 0, f"Dataset {dataset_name} has no examples after filtering"
            
        processed_dataset = sample_split(dataset, proportion, 
                                         dataset_config.get('max_examples', None))
        
        # assert label_mapping is None or is_promptsource == False, "If dataset is a promptsource, label_mapping must not be specified"

        # verbalize numeric labels
        if label_mapping is not None and not is_promptsource:
            if MAP_COLS is False:
                column_mapping = {k: f"{k}:" for k in column_mapping.keys()}
            processed_dataset = map_labels(processed_dataset, label_mapping, column_mapping)

        # convert text to promptsource
        if is_promptsource:
            templates = [template for id, template in DatasetTemplates(dataset_name, dataset_config_name).templates.items()]
            print(f"Dataset {dataset_name}, has templates, applying promptsouce")
            processed_dataset = to_promptsource(processed_dataset, templates, is_validation)

        elif column_mapping is not None:
            if isinstance(column_mapping, dict):
                columns = list(column_mapping.keys())
            processed_dataset = concatenate_columns(processed_dataset, column_mapping)
        else:
            processed_dataset = remove_redundant_columns(processed_dataset)

        return processed_dataset

    # Check if dataset is a promptsource
    is_promptsource = False
    if 'promptsource' in dataset_config:
        if dataset_config['promptsource'] == True:
            is_promptsource = True

    # Sample training dataset
    train_split = dataset_config.get('train_split', None)
    print(train_split)
    if train_split is not None and proportion > 0:
        # load train dataset
        if dataset_config.get('path', None) is not None:
            print(F"Loading dataset from path: {dataset_config['path']}")
            train_dataset = load_from_disk(dataset_config['path'])
        else:
            train_dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
                split=train_split,
            )
        print(f"ds_name: {dataset_name}, ds_config_name: {dataset_config_name}, train_split: {train_split}")
        train_dataset = process_dataset(train_dataset, 
                                        dataset_name, dataset_config_name, 
                                        is_promptsource, proportion, columns,
                                        label_mapping, dataset_config)
    else:
        train_dataset = None

    # Sample validation dataset
    val_split = dataset_config.get('validation_split', None)
    if val_split is not None and proportion > 0:
        # load train dataset
        validation_dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            split=validation_split,
        )
        validation_dataset = process_dataset(validation_dataset,
                                            dataset_name, dataset_config_name, 
                                            is_promptsource, proportion, columns,
                                            label_mapping, dataset_config)
    else:
        validation_dataset = None

    

    return train_dataset, validation_dataset


def pack_examples(dataset, max_n):
    # Create a list to store the packed examples
    packed_text = []
    
    # Number of rows in the dataset
    num_rows = len(dataset)
    
    index = 0
    
    # Extract all text into a single array for faster access
    all_text = [example['text'] for example in dataset]
    # Loop through the dataset
    while index < num_rows:
        # Randomly select a value of n from uniform distribution
        n = np.random.randint(1, max_n + 1)
        # Select n examples starting from current index
        selected_text = all_text[index: index + n]
        # Concatenate the text of the selected examples (and add newline)
        concatenated_text = ' \n '.join(selected_text)
        # Append the concatenated text to packed_text list
        packed_text.append(concatenated_text)
        # Update the index for next iteration
        index += n
    
    # Convert the packed_text list to a Dataset
    packed_dataset = Dataset.from_dict({'text': packed_text})
    
    # Return the packed dataset
    return packed_dataset


def merge_datasets(dataset_configs, pack_qa: Optional[int] = None, cache_dir: Optional[str] = None):
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
        print(f"config: {config}")
        print(f"Processing dataset {config['dataset_name']}...")
        train_dataset, validation_dataset = load_and_sample_dataset(config, cache_dir)
        if train_dataset is not None:
            print(f"train_dataset: {train_dataset[0:2]}")
        if validation_dataset:
            print(f"validation_dataset: {validation_dataset[0:2]}")
        print(f"train_dataset: {train_dataset}")
        print(f"validation_dataset: {validation_dataset}")
        ds_type = config['dataset_type']

        # Add the datasets to the respective lists
        if train_dataset is not None:
            datasets_by_type['train'][ds_type].append(train_dataset)
        if validation_dataset is not None:
            datasets_by_type['validation'].append(validation_dataset)
        print(f"train_dataset: {train_dataset}")
        print(f"validation_dataset: {validation_dataset}")
        # assert train_dataset is not None or validation_dataset is not None, f"Neither train nor validation dataset is available for {config['dataset_name']}."

    # Randomly sample from 'text' datasets
    qa_train_datasets = datasets_by_type['train']['QA']
    text_train_datasets = datasets_by_type['train']['text']

    # Get the total number of examples for each ds_type
    n_qa_train_examples = sum(len(ds) for ds in qa_train_datasets)
    n_text_train_examples = sum(len(ds) for ds in text_train_datasets)
    # assert n_text_train_examples > n_qa_train_examples, f"Number of text examples ({n_text_train_examples}) \
    #                                                     is less than the number of QA examples ({n_qa_train_examples})."
    
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
            # Create a boolean array representing whether each index should be kept
            mask = np.ones(len(ds), dtype=bool)
            mask[indices_to_remove] = False
            # Get the indices to keep
            indices_to_keep = np.arange(len(ds))[mask]
            sampled_text_train_datasets.append(ds.select(indices_to_keep))
    
    # random pack QA dataset examples
    if pack_qa is not None:
        qa_train_datasets = datasets_by_type['train']['QA']
        packed_qa_train_datasets = []
        for ds in qa_train_datasets:
            packed_ds = pack_examples(ds, pack_qa)
            packed_qa_train_datasets.append(packed_ds)
        datasets_by_type['train']['QA'] = packed_qa_train_datasets

    # Concatenate datasets
    train_sets = datasets_by_type['train']['QA'] + sampled_text_train_datasets
    val_sets = datasets_by_type['validation']
    concatenated_train = concatenate_datasets(train_sets) if len(train_sets) > 0 else None
    concatenated_validation = concatenate_datasets(val_sets) if len(val_sets) > 0 else None
    
    # Convert to HuggingFace datasets
    raw_datasets = { 
        'train': concatenated_train, 
        'validation': concatenated_validation
    }

    return raw_datasets


if __name__ == "__main__":
    CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
    DATASET_DIR = os.path.join(CACHE_DIR, "merged_datasets")
    DATASET_TYPE = "NLI_eval_2"
    P_QA = 1
    P = 1
    QA_PACKING = 1
    PROMPTSOURCE = True
    MAP_COLS = False
    SKIP_SPLIT = None
    
    # Define dataset configurations
    from dataset_configs import DatasetConfig
    config = DatasetConfig(P_QA=P_QA, P=P, PROMPTSOURCE=PROMPTSOURCE)
    dataset_config = config.dataset_configs[DATASET_TYPE]

    merged_datasets = merge_datasets(dataset_config, QA_PACKING, CACHE_DIR)
    print(f"datasets: {merged_datasets}")

    # export newly created dataset
    ds_folder = os.path.join(DATASET_DIR, DATASET_TYPE, 
                             f"P_{P}_PQA_{str(QA_PACKING)}_psource_{PROMPTSOURCE}_prompt_{MAP_COLS}", f"dataset_{P_QA}")
    print(f"Saving dataset to {ds_folder}...")
    if not os.path.exists(ds_folder):
        os.makedirs(ds_folder)
    n_folders = len(os.listdir(DATASET_DIR))
    for split in merged_datasets.keys():
        if split == SKIP_SPLIT or merged_datasets[split] is None:
            print(f"skipping {SKIP_SPLIT}")
            continue
        else:
            merged_datasets[split].save_to_disk(os.path.join(ds_folder, f"dataset_{split}.arrow"))
    # save config
    with open(os.path.join(ds_folder, "config.json"), 'w') as f:
        json.dump(dataset_config, f)
    # Print example dataset information
    print(merged_datasets)