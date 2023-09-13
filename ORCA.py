# %%
model_name_or_path = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_NLI/EleutherAI/pythia-160M-deduped_ckpt_False/checkpoint-70000"
dataset_path = '/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
device = "cpu"

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed
)
from transformers.testing_utils import CaptureLogger
import transformers
from transformers.testing_utils import CaptureLogger
import datasets
from datasets import DatasetDict
from datasets import load_from_disk
import os
from functorch import make_functional_with_buffers, vmap, grad
import torch.nn as nn

set_seed(420)
text_column_name = "text"
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
    return config, tokenizer, model

_, tokenizer, model = load_model(model_name_or_path)

raw_dataset = load_from_disk(dataset_path)

# %%
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
tokenized_datasets = {}
for x in raw_dataset:
    if isinstance(raw_dataset, datasets.dataset_dict.DatasetDict):
        pth = os.path.join(dataset_path, x)
        assert os.path.exists(pth), f"Dataset {x} not found at {pth}"
        tokenized_datasets[x] = load_from_disk(pth)
    else:
        tokenized_datasets = load_from_disk(dataset_path)

def compute_loss_stateless_model(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    outputs = fmodel(params, buffers, batch)
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
    return loss

# %%
# inputs = tokenized_datasets['NLI'][0]['inputs']
# inputs = tokenizer(inputs, return_tensors='pt')

# # Shift the input_ids one token to the right to create the labels
# labels = inputs['input_ids'].clone()
# labels[:-1] = inputs['input_ids'][1:]

# # Calculate the loss
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss

# %%

# Convert your model to a functional model
fmodel, params, buffers = make_functional_with_buffers(model)

loss_fn = nn.CrossEntropyLoss()

# Create a new function that computes the gradient with respect to the params
ft_compute_grad = grad(compute_loss_stateless_model)

# Use vmap to get the function to compute the gradient over an entire batch of samples and targets
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

max_iters = 2
# Assuming tokenized_datasets is your dataset
for dataset_name, dataset in tokenized_datasets.items():
    for i in range(len(dataset)):
        inputs = dataset[i]['inputs']
        inputs = tokenizer(inputs, return_tensors='pt')

        # Shift the input_ids one token to the right to create the labels
        labels = inputs['input_ids'].clone()
        labels[:-1] = inputs['input_ids'][1:]

        # Extract the input_ids tensor from the BatchEncoding object
        input_ids = inputs['input_ids']

        # Compute per-sample-gradients
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, input_ids, labels)
        ft_per_sample_grads_target = ft_compute_sample_grad(params, buffers, input_ids, labels)
        
        print(f"len inputs: {len(input_ids)}")
        print(f"len grads: {len(ft_per_sample_grads)}")
# %%



