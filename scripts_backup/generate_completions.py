# %%
import os
import pickle
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, GPTNeoXTokenizerFast
from generate import load_model
from datasets import load_from_disk
import collections
import torch
from tqdm import tqdm

# %%
ds_path = "/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds"
ds  = load_from_disk(ds_path)

# %%
device = "cuda:0"
model_name_or_path = "EleutherAI/gpt-neox-20b" # "EleutherAI/pythia-160M-deduped"
model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path).half().to(device).eval()

# %%
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name_or_path)

# %%
"""
dict levels:

task -> prompt / target / generated -> n_generations -> generation

"""

def default_dict():
    return collections.defaultdict(dict)

def default_dict_of_dicts():
    return collections.defaultdict(default_dict)

gen_dict = collections.defaultdict(default_dict_of_dicts)

save_path = "./results/generations/gpt-neo-x"

for task in tqdm(ds.keys(), desc="Tasks"):
    # sample 10 rows
    ds_task = ds[task].select(range(10))
    for idx, row in tqdm(enumerate(ds_task), desc="Rows"):
        prompt = row["inputs"]
        targets = row["targets"]
        # concatenate
        input_ = prompt + " " + targets
        # split into two
        input_1, input_2 = input_[:len(input_)//2], input_[len(input_)//2:]
        gen_dict[task]["prompt"] = input_1
        gen_dict[task]["continuation"] = input_2
        # tokenize
        input_1_ = tokenizer(input_1, return_tensors="pt").input_ids.to(device)
        input_2_ = tokenizer(input_2, return_tensors="pt").input_ids.to(device)
        
        n_samples = 10
        # generate 10 samples
        tqdm.write(f"--- True continuation: {input_2} ---")
        for i in range(n_samples):
            gen = model.generate(input_1_, max_new_tokens=int(len(input_2)*1.2), 
                                 do_sample=True, top_p=0.95, 
                                 num_return_sequences=1)
            gen_dec = tokenizer.decode(gen[0])
            gen_dict[task][idx]["generated"][i] = gen_dec
            tqdm.write(f"{i}: {gen_dec}")
            # clear server
            torch.cuda.empty_cache()
            # save dict as pkl
            with open(os.path.join(save_path, "gpt-neo-x.pkl"), "wb") as f:
                pickle.dump(gen_dict, f)


