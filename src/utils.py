import os
import numpy as np
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm
from transformers import GPT2Tokenizer

def running_jupyter():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # 'ZMQInteractiveShell' is the class name for the Jupyter Notebook shell
            return True
        else:
            # Probably in IPython or other interactive shell
            return False
    except (NameError, ImportError):
        # Probably in a standard Python shell
        pass
    return False

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max(x) for numerical stability
    return e_x / e_x.sum()


def nll_to_prob(nll):
    return np.exp(nll)


def remove_string(input_value):
    strings_to_remove = [
        "hendrycks*",
        "hendrycksTest-",
        "wmt09-"
    ]
    if isinstance(input_value, str):
        for string_to_remove in strings_to_remove:
            input_value = input_value.replace(string_to_remove, "")
        return input_value
    elif isinstance(input_value, tuple):
        return tuple(remove_string(item) for item in input_value)
    else:
        raise ValueError("input_value must be a string or a tuple of strings")


def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of max_len.
        assert total_length != 0, f"total_length is 0, {len(concatenated_examples[list(examples.keys())[0]])}"
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        return result

def count_tokens_in_example(example):
    # Note that the function now operates on a single example, not a batch
    # Also, it returns a dictionary
    return {"num_tokens": len(example['input_ids'])}

def count_tokens(dataset):
    # Create a new dataset with an additional column that contains the number of tokens in each example
    with_lengths = dataset.map(count_tokens_in_example, num_proc=multiprocessing.cpu_count())

    # Now, sum up the lengths. Note that this operation is not parallelized, 
    # so it may be slow if the dataset is very large.
    total_tokens = sum(tqdm(with_lengths['num_tokens'], desc="Counting tokens"))

    return total_tokens

def count_gpt2_tokens(dataset, text_column):
    """
    Tokenize a Hugging Face Dataset using GPT-2 tokenizer and count the total number of tokens.

    Parameters:
    dataset (datasets.Dataset): Hugging Face Dataset to tokenize.
    text_column (str): Name of the column in the dataset that contains the text to tokenize.

    Returns:
    int: Total number of tokens.
    """

    # Load pre-trained GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define a function to tokenize each example and return the number of tokens
    def count_tokens(example):
        tokens = tokenizer.encode(example[text_column], truncation=True)
        return {"num_tokens": len(tokens)}

    # Map the count_tokens function to the dataset
    dataset = dataset.map(count_tokens, remove_columns=dataset.column_names)

    # Sum the num_tokens column to get the total number of tokens
    num_tokens = sum(dataset['num_tokens'])

    return num_tokens

def limit_total_tokens(dataset, max_tokens, pretokenized=False):
    """
    Limit the total number of tokens in a Hugging Face Dataset.

    Parameters:
    dataset (datasets.Dataset): Hugging Face Tokenized Dataset to limit.
    max_tokens (int): Maximum total number of tokens.
    tokenize (transformers.PreTrainedTokenizer): The tokenizer used for tokenizing the dataset.

    Returns:
    datasets.Dataset: Limited Hugging Face Tokenized Dataset.
    """
    if not pretokenized:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    total_tokens = 0
    indices_to_keep = []

    # iterate with a progress bar
    for i, example in enumerate(tqdm(dataset)):
        if pretokenized:
            num_tokens = len(example["input_ids"])
        else:
            num_tokens = len(tokenizer.tokenize(example["text"]))
        total_tokens += num_tokens

        if total_tokens > max_tokens:
            break

        indices_to_keep.append(i)

    return dataset.select(indices_to_keep), indices_to_keep


def filter_path(path, keywords):
    parts = path.split(os.sep)
    filtered_parts = []
    for part in parts:
        if any(keyword in part for keyword in keywords):
            filtered_parts.append(part)
        if len(filtered_parts) == len(keywords):
            break
    return os.sep.join(filtered_parts)


def concatenate_columns(dataset, column1, column2, new_column_name):
    def concat_example(example):
        example[new_column_name] = example[column1] + " " + example[column2]
        return example

    return dataset.map(concat_example)

def total_tokens(dataset, text_field):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    total_count = 0
    
    for example in tqdm(dataset, desc="Counting tokens"):
        text = example[text_field]
        tokens = tokenizer.tokenize(text)
        total_count += len(tokens)

    # Print the total count in scientific notation
    print(f"Total number of tokens: {total_count:.2e}")

    return f"{total_count:.2e}"

# def set_grad_accum


# def split_train_params():
#     average_score_ = collections.defaultdict(lambda: collections.defaultdict(dict))
#     average_ranking_ = collections.defaultdict(lambda: collections.defaultdict(dict))

#     for key in average_score.keys():
#         model_split = key.split(' ')
#         if len(model_split) == 2:
#             training_set, param_size = model_split
#         elif len(model_split) == 3:
#             pre_training_set, training_set, param_size = model_split
#             training_set = pre_training_set + ' ' + training_set
#         average_score_[training_set][param_size] = average_score[key]
#         average_ranking_[training_set][param_size] = average_rankings[key]

# def get_averages()

#     average_score_ = collections.defaultdict(lambda: collections.defaultdict(dict))
#     average_ranking_ = collections.defaultdict(lambda: collections.defaultdict(dict))

#     for key in average_score.keys():
#         model_split = key.split(' ')
#         if len(model_split) == 2:
#             training_set, param_size = model_split
#         elif len(model_split) == 3:
#             pre_training_set, training_set, param_size = model_split
#             training_set = pre_training_set + ' ' + training_set
#         average_score_[training_set][param_size] = average_score[key]
#         average_ranking_[training_set][param_size] = average_rankings[key]


# import os
# import shutil

# # Define the root directory where the models are located
# root_dir = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1"

# # Iterate over all subdirectories
# for dirpath, dirnames, filenames in os.walk(root_dir):
#     # Check if 'results1.json' exists in the directory
#     if 'results1.json' in filenames:
#         # Construct the full file paths
#         src = os.path.join(dirpath, 'results1.json')
#         dst = os.path.join(dirpath, 'results.json')
#         # Copy 'results1.json' to 'results.json'
#         shutil.copy(src, dst)
#         print(f"Moved {src} to {dst}")