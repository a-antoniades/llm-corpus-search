# %%
import os
import json
import glob
import re
from src.wimbd_ import _load_dataset

HF_HOME = "/share/edc/home/antonis/datasets/huggingface"
save_path = os.makedirs(os.path.join(HF_HOME, 'wmt09_gens'), exist_ok=True)

base_pth = "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/experiment_6_logits_max_4/inference/allenai/OLMo-7B/TRANSLATION"
doc_res_files = glob.glob(os.path.join(base_pth, "**/doc_results.json"), recursive=True)

# %%
pattern = re.compile(r'wmt\d{2}-(\w{2}-\w{2})')
language_pairs = [pattern.search(path).group(1) for path in doc_res_files if pattern.search(path)]
doc = json.load(open(doc_res_files[0]))

def find_lang(path):
    pattern = re.compile(r'wmt\d{2}-(\w{2}-\w{2})')
    return pattern.search(path).group(1)



# %%
doc['wmt09-en-hu'][0]

# %%
doc_res_files[0].split('/')[-3]

# %%
wmt = _load_dataset('wmt')

# %%
wmt['cs-en']['translation']

from tqdm import tqdm
from datasets import DatasetDict
from datasets import Dataset

def insert_gens_into_wmt(wmt, doc_res_files):
    for path in tqdm(doc_res_files, desc="Processing files"):
        with open(path, 'r') as f:
            doc_res = json.load(f)
        task = path.split('/')[-3]
        lang = find_lang(path)
        lang1, lang2 = lang.split('-')
        total_rows = len(wmt[lang]['translation'])
        matched_rows = 0

        for doc in tqdm(doc_res[task], desc=f"Inserting generations for {task}", leave=False):
            doc_res_task_dict = wmt[lang].to_dict()
            src = doc['src']
            ref = doc['ref']
            gen = doc['result'][0]
            for idx, row in enumerate(doc_res_task_dict['translation']):
                doc_res_task_dict['translation'][idx]['gen'] = gen
                matched_rows += 1
        wmt[lang] = Dataset.from_dict(doc_res_task_dict)
        dataset_dict = {lang: Dataset.from_dict({'translation': value['translation']}) for lang, value in wmt.items()}
        dataset_dict = DatasetDict(dataset_dict)
        # Now you can save it to disk
        dataset_dict.save_to_disk(os.path.join(HF_HOME, 'wmt09_gens'))
        print(f"Matched {matched_rows}/{total_rows} rows for {task}")
    return wmt


wmt = insert_gens_into_wmt(wmt, doc_res_files)

dataset_dict = {lang: Dataset.from_dict({'translation': value['translation']}) for lang, value in wmt.items()}
dataset_dict = DatasetDict(dataset_dict)

# Now you can save it to disk
dataset_dict.save_to_disk(os.path.join(HF_HOME, 'wmt09_gens'))

# %%



