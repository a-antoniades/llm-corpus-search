import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling
from datasets import load_dataset

from torch.utils.data import IterableDataset
import logging
transformers.logging.set_verbosity_info()

# from data_generation.generator import Grapher

# show cuda devices
print(f"CUDA DEVSSSS: {torch.cuda.device_count()}")

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    random_initialize: Optional[bool] = field(default=False)
    model_name_or_path: Optional[str] = field(default="gpt2",
        metadata={"help": "gpt2: 124M; gpt2-medium: 355M; gpt2-large: 774M; gpt2-xl: 1.5B; RWKV/rwkv-4-169m-pile; sgugger/rwkv-430M-pile; sgugger/rwkv-7b-pile."},)
    entity_as_new_token: Optional[bool] = field(default=False)
    relation_as_new_token: Optional[bool] = field(default=False)
    model_type: Optional[str] = field(default="Transformer", 
                                      metadata={"choices": ["Transformer", "LSTM", "RWKV"]})


@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset: str = field(default=None, metadata={"help": "dataset name."})
    randomize_entity_name: Optional[bool] = field(default=False)
    weighted_r: int = field(default=None, metadata={"help": "double weight a relation."})
    use_inverse_r: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # device: Optional[str] = field(default="cuda:0")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    path_len: int = field(default=20,
        metadata={"help": "Maximum reasoning path length."},
    )
    mode: str = field(default="random_walk") # choices=['random_walk', 'proof']
    resume: Optional[bool] = field(default=False)
    gradient_checkpointing: Optional[bool] = field(default=False)
    # place_model_on_device: Optional[bool] = field(default=True)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    # new_tokens_list: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # num_new_tokens += tokenizer.add_tokens(new_tokens_list)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # output_embeddings = model.get_output_embeddings().weight.data
        
        ids = random.sample(list(range(len(input_embeddings))), k=num_new_tokens)
        # input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings[ids]
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.tie_weights()



def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# def create_RNN(num_layers=8, hidden_units=128, dense_units=128, input_shape=(1024,)):
#     model = Sequential()
#     for _ in range(num_layers):
#         model.add(LSTM(hidden_units,input_shape=input_shape))
#     model.add(Dense(units=dense_units,activation='softmax'))
#     model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary() 
#     return model



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.random_initialize:
        print("Random initializing...")
        if model_args.model_type in ["Transformer", "RWKV"]:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
            model = transformers.AutoModelForCausalLM.from_config(config)
        # elif model_args.model_type == "LSTM":
        #     model = create_RNN()
        else:
            print(f"model type {model_args.model_type} unimplemented.")
            exit(1)
    else:
        print("Using pre-trained model weights...")
        if model_args.model_type in ["Transformer", "RWKV"]:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        # elif model_args.model_type == "LSTM":
        #     model = load_model(model_args.model_name_or_path)
        else:
            print(f"pretrained model type {model_args.model_type} unimplemented.")
            exit(1)
            

    ## DATASET, TOKENIZER, COLLATOR ##
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # Add special tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # since we added new tokens, we need to resize the embedding
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # Load dataset from HuggingFace datasets library
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=training_args.model_max_length)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # drop all examples that have a length of 0
    tokenized_dataset = tokenized_dataset.filter(lambda example: len(example["input_ids"]) > 0)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize our Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args,
                      data_collator=data_collator,
                      train_dataset=tokenized_dataset["train"], eval_dataset=None)
    
    # Training
    if training_args.resume:
        trainer.train(model_args.model_name_or_path)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()