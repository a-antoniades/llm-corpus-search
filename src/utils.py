from transformers import AutoTokenizer
from datasets import load_dataset

def count_tokens(dataset, tokenizer):
    """Count the total number of tokens in the dataset
    Args:
        dataset (datasets.Dataset): Dataset to count the tokens of
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
    """
    total_tokens = 0
    # Loop through the dataset
    for example in dataset:
        # Tokenize the text and increment the token count
        tokens = tokenizer.tokenize(example['text']) # assuming the text field is called 'text'
        total_tokens += len(tokens)
    return total_tokens