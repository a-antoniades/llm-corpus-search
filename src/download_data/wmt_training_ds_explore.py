# %%
from datasets import load_dataset, load_from_disk, concatenate_datasets, get_dataset_config_names
# from promptsource import templates
import pickle
import pandas as pd

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"

# %%
ds_to_load = ['cs-en', 'en-fr', 'de-en']
ds_to_load = {
    'Helsinki-NLP/europarl': ['cs-en', 'en-fr', 'de-en', 
                              'en-hu', 'en-es', 'en-it']
    # 'wmt20': ['pl-en', 'ro-en', ]
}
dataset = {}
for ds, tasks in ds_to_load.items():
    for task in tasks:
        dataset[task] = load_dataset(ds, task, cache_dir=CACHE_DIR)


print(dataset.keys())