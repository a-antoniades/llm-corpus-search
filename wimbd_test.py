import sys
sys.path.append('/share/edc/home/antonis/LLM-Incidental-Supervision/wimbd')
from wimbd.es import count_documents_containing_phrases, get_documents_containing_phrases
from elasticsearch import Elasticsearch

CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
# cloud_id = "m-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl"
# api_key = "RlZBbHpZc0J1MEw4LVVWVk9SaTE6bXJlSUM2QnlSQmFHemhwVElVUnZyQQ=="

corpus = 'pile'

if corpus == 'pile':
    index = 're_pile'
    cloud_id = "m-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl"
    api_key = "RlZBbHpZc0J1MEw4LVVWVk9SaTE6bXJlSUM2QnlSQmFHemhwVElVUnZyQQ=="
elif corpus == 'dolma':
    index = "docs_v1.5_2023-11-02"
    cloud_id = "dolma-v15:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ1MjQyM2ZiNjk0NGE0YzdkOGQ5N2Y3NDM2MmMzODY3ZSQxMDNiM2ZkYTUwYzk0MTNmYmUwODA1ZDMyNjQ5YTliNQ=="
    api_key = "QTJiajFJMEIxR1JtTm13YUZBVGc6dEpudXhEd19SRzJUOVZNYUpDdlItdw=="

n_gram = ("i love animals",
          "halo")

corpus_to_index = {
    'pile': 're_pile',
    'dolma': 'docs_v1.5_2023-11-02'
}

es = Elasticsearch(
    cloud_id=cloud_id,
    api_key=api_key,
    retry_on_timeout=True,
    http_compress=True)

counts = count_documents_containing_phrases(corpus_to_index[corpus], n_gram,
                                            es=es, all_phrases=True)

docs = get_documents_containing_phrases(corpus_to_index[corpus], n_gram,
                                       es=es, all_phrases=True,
                                       return_all_hits=True)

print(f"counts: {counts}")
print(f"docs: {[doc['_source']['meta']['pile_set_name'] for doc in docs]}")
print(f"docs_ids: {[doc['_id'] for doc in docs]}")
# print(next(iter(docs['_source']['meta'])))