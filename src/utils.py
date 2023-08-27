import os
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset
import multiprocessing
from tqdm import tqdm

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


def limit_total_tokens(tokenized_dataset, max_tokens):
    """
    Limit the total number of tokens in a Hugging Face Tokenized Dataset.

    Parameters:
    tokenized_dataset (datasets.Dataset): Hugging Face Tokenized Dataset to limit.
    max_tokens (int): Maximum total number of tokens.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenizing the dataset.

    Returns:
    datasets.Dataset: Limited Hugging Face Tokenized Dataset.
    """
    
    total_tokens = 0
    indices_to_keep = []

    # iterate with a progress bar
    for i, example in enumerate(tqdm(tokenized_dataset)):
        num_tokens = len(example["input_ids"])
        total_tokens += num_tokens

        if total_tokens > max_tokens:
            break

        indices_to_keep.append(i)

    return tokenized_dataset.select(indices_to_keep)


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