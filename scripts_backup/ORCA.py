# %%
# model_name_or_path = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_NLI/EleutherAI/pythia-160M-deduped_ckpt_False/checkpoint-70000"
# dataset_path = '/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'

import os
import collections
import random
import pickle
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed
)
import transformers
from transformers.testing_utils import CaptureLogger
import datasets
from datasets import DatasetDict
from datasets import load_from_disk

from functorch import make_functional_with_buffers, vmap, grad
import torch
import torch.nn as nn
import numpy as np

set_seed(420)
text_column_name = "text"
os.environ["HF_DATASETS_CACHE"] = "/share/edc/home/antonis/datasets/huggingface"

max_len = 1000
n_groups = 10
max_iter = 100
model_name_or_path = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_Commonsense/EleutherAI/pythia-1.4B-deduped_ckpt_False/checkpoint-70000"
task_samples_path = f"/share/edc/home/antonis/datasets/huggingface/flan_v1/task_ds_sampled_{max_len}_{n_groups}.pkl"
dataset_dict_path = f"/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small_sampled_{max_len}_{n_groups}"
device = "cpu"

with open(task_samples_path, 'rb') as f:
    ds_task = collections.defaultdict(dict, pickle.load(f))

ds = load_from_disk(dataset_dict_path)

def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

def load_model(model_name_or_path, device=device):
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, config=config)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return config, tokenizer, model

# raw_dataset = load_from_disk(dataset_path)

# %%
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

def compute_loss_stateless_model(params, buffers, batch, targets):
    outputs = fmodel(params, buffers, batch)
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
    return loss

_, tokenizer, model = load_model(model_name_or_path)

# Convert your model to a functional model
fmodel, params, buffers = make_functional_with_buffers(model)

loss_fn = nn.CrossEntropyLoss()

# Create a new function that computes the gradient with respect to the params
ft_compute_grad = grad(compute_loss_stateless_model)

# Use vmap to get the function to compute the gradient over an entire batch of samples and targets
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

# Initialize a variable to store the sum of squared gradients
sum_squared_grads = 0
num_samples = 0

# Number of samples and iterations
num_samples_per_iter = 1000
num_iters = 10

import json

def compute_grads(inputs):
    inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs['input_ids'].to(device).unsqueeze(0)

    # Shift the input_ids one token to the right to create the labels
    labels = inputs.clone()
    labels[:, :-1] = inputs[:, 1:]

    # Compute per-sample-gradients
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, inputs, labels)

    # zero out the gradients
    model.zero_grad()
    torch.cuda.empty_cache()

    return ft_per_sample_grads

def compute_grad_mag(grads):
    total_gradient_mag = 0
    num_grad_tensors = 0
    for grad_tuple in grads:
        for grad_tensor in grad_tuple:
            total_gradient_mag += torch.norm(grad_tensor).item()
            num_grad_tensors += 1
    normalized_gradient_mag = total_gradient_mag / num_grad_tensors
    return normalized_gradient_mag


results = collections.defaultdict(dict)
batch_size = 1  # Define your batch size
for n in tqdm(ds.keys(), desc="Processing DS datasets"):            
    grad_mag = []
    gradients = []
    for sample in tqdm(ds[n]):
        # Compute the gradients
        grads = compute_grads(sample['text'])
        gradients.append(grads)
        # compute total magnitude
        # gradient_mag = compute_grad_mag(grads)
        # grad_mag.append(gradient_mag)
        if len(gradients) > max_iter:
            break
    results[n] = gradients
# save as pkl
with open('grads_c4.pkl', 'wb') as f:
    pickle.dump(results, f)


task_grads = collections.defaultdict(lambda: collections.defaultdict(dict))
for task_cluster in tqdm(ds_task.keys(), desc="Processing task datasets"):
    grad_mag = []
    gradients = []
    for n in tqdm(ds_task[task_cluster].keys(), desc="Processing task groups"):
        for sample in tqdm(ds_task[task_cluster][n]):
            # Compute the gradients
            grads = compute_grads(sample)
            gradients.append(grads)
            # gradient_mag = compute_grad_mag(grads)
            # grad_mag.append(gradient_mag)
            # gradients.append(grads)
            if len(gradients) > max_iter:
                break
        task_grads[task_cluster][n] = gradients
# save as pkl
with open('grads_tasks.pkl', 'wb') as f:
    pickle.dump(task_grads, f)
        
    