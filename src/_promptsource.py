"""
Code from Alon Albalak, 
FLAD - Few-shot Learning with Auxiliary Data
https://github.com/alon-albalak/FLAD

"""


from datasets import load_dataset, load_from_disk, concatenate_datasets
from promptsource import templates

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
import os
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


import os
import json
import logging
from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates, TemplateCollection
# set logging level to INFO
logger = logging.getLogger(__name__)
logger.setLevel(20)

class TOMixture:
    # ("glue","mrpc"), # Paraphrase identification
    # ("glue","qqp"),
    # ("paws","labeled_final"),
    # ("kilt_tasks", "hotpotqa"), # Closed-book QA
    # ("wiki_qa", None),
    # ("adversarial_qa", "dbidaf"), # Extractive QA
    # ("adversarial_qa","dbert"),
    # ("adversarial_qa","droberta"),
    # ("duorc","SelfRC"),
    # ("duorc","ParaphraseRC"),
    # ("ropes",None),
    # ("quoref",None),
    # ("cos_e","v1.11"), # Multiple-choice QA
    # ("cosmos_qa",None),
    # ("dream",None),
    # ("qasc",None),
    # ("quail",None),
    # ("quarel",None),
    # ("quartz",None),
    # ("sciq",None),
    # ("social_i_qa",None),
    # ("wiki_hop","original"),
    # ("wiqa",None),
    # ("amazon_polarity",None), # Sentiment
    # ("app_reviews",None),
    ("sst","default"), # Senitment Classification")
    ("imdb",None),
    # ("rotten_tomatoes",None),
    # ("yelp_review_full",None),
    # ("common_gen",None), # Structure-to-text
    # ("wiki_bio",None),
    # ("cnn_dailymail","3.0.0"), # Summarization
    # ("gigaword",None),
    # ("multi_news",None),
    # ("samsum",None),
    # ("xsum",None),
    # ("ag_news",None), # Topic Classification
    # ("dbpedia_14",None),
    # ("trec",None)

def get_dataset_name(name: str, subset: str):
    if subset is not None:
        canonized_name = f"{name}/{subset}"
    else:
        canonized_name = name
    return canonized_name

def get_T0MixtureDatasets(split, max_samples=None, return_as_dict=True):
    """
    T0MixtureDatasets creates a separate dataset for each dataset in the mixture
    """
    datasets = {} if return_as_dict else []
    for name, subset in TOMixture:
        dataset = load_dataset(name, subset, split=split, cache_dir=CACHE_DIR)
        if max_samples:
            dataset = Dataset.from_dict(dataset[:max_samples])
        templates = [template for id, template in DatasetTemplates(name, subset).templates.items()]
        dataset.templates = templates
        dataset.name = get_dataset_name(name, subset)

        if return_as_dict:
            datasets[get_dataset_name(name, subset)] = dataset
        else:
            datasets.append(dataset)


        logger.info(f"Loaded dataset {name}/{subset} with {len(templates)} templates")
        assert(len(templates) > 0), "No templates"
    return datasets