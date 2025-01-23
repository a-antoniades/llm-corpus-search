from typing import Dict, Optional, Sequence
import json, copy, math, os, random
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
import sys, re
import hf_olmo
sys.path.append('../wimbd')
from wimbd.es import get_documents_containing_phrases

language_map = {'fr': 'French', 'cs': 'Czech', 'de': 'German', 'es': 'Spanish',
                'it': 'Italian', 'hu': 'Hungarian'}

mmlu_tasks = {'prehistory', 'business_ethics', 'econometrics', 'college_medicine', 
            'professional_law', 'philosophy', 'abstract_algebra', 'moral_disputes', 
            'college_chemistry', 'medical_genetics', 'high_school_government_and_politics', 
            'human_aging', 'us_foreign_policy', 'high_school_macroeconomics', 
            'logical_fallacies', 'moral_scenarios', 'college_mathematics', 
            'international_law', 'computer_security', 'sociology', #professional_psychology', 
            'marketing', 'human_sexuality', 'high_school_chemistry', 'professional_accounting', 
            'college_computer_science', 'anatomy', 'high_school_us_history', 
            'college_biology', 'public_relations', 'high_school_computer_science', 
            'high_school_mathematics', 'college_physics', 'professional_medicine', 
            'high_school_microeconomics', 'clinical_knowledge', 'elementary_mathematics', 
            'machine_learning', 'security_studies', 'nutrition', 'world_religions', 
            'high_school_psychology', 'high_school_geography', 'management', 
            'global_facts', 'high_school_world_history', 'electrical_engineering', 
            'high_school_european_history', 'jurisprudence', 'high_school_physics', 
            'conceptual_physics', 'high_school_statistics', 'virology', 
            'high_school_biology', 'astronomy', 'miscellaneous'}

IGNORE_INDEX = -100
MAX_CHARS = 3000
MIN_CHARS = 500

def set_seed(seed):
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)

def generate_ngrams(text, n):
    all_ngrams = []
    words = clean_text(text).split()
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        all_ngrams.append(ngram)
    return list(set(all_ngrams))
    
def random_ngram(text, n):
    all_ngrams = generate_ngrams(text, n)
    if len(all_ngrams) > 0:
        return random.choice(all_ngrams)
    else:
        return None

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            padding="longest",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]       
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )

def prepare_task_data(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    device='cuda'
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) 
                                            for strings in (examples, sources)]
    eos = torch.tensor([tokenizer.eos_token_id])
    input_ids = [torch.cat((ids, eos)) for ids in examples_tokenized["input_ids"]]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids.to(device), labels=labels.to(device))

def prepare_ngram_data(
    docs: Sequence[str],
    ngram: str,
    tokenizer: PreTrainedTokenizer,
    max_length=1024,
    device='cuda'
) -> Dict:
    """Preprocess the data by tokenizing."""
    docs_tokenized = tokenizer(docs, padding="longest",
                            return_tensors="pt", truncation=False,)
    ngram_ids = tokenizer(' ' + ngram, return_tensors="pt").input_ids[0]
    input_ids = docs_tokenized.input_ids
    attention_mask = docs_tokenized.attention_mask
    num_ngram_tokens = len(ngram_ids)
    ngram_mask = torch.zeros_like(attention_mask)
    doc_len = torch.zeros(input_ids.size(0), dtype=torch.int32)
    for i in range(input_ids.size(1) - num_ngram_tokens + 1):
        m = False
        m = torch.all(input_ids[:, i: i + num_ngram_tokens] == 
                    ngram_ids[None, :], dim=1) | m
        ngram_mask[:, i: i + num_ngram_tokens] += m[:, None].int()
        doc_len[m] = i + num_ngram_tokens
    
    print('doc length used: ', doc_len)
    labels = torch.zeros_like(attention_mask) + IGNORE_INDEX
    labels[ngram_mask.bool()] = input_ids[ngram_mask.bool()]
    
    max_length = min(max_length, input_ids.shape[-1])
    input_mask = torch.zeros_like(attention_mask)
    for i, l in enumerate(doc_len):
        if l-max_length > 0:
            start = l-max_length
            end = l
        else:
            start = 0
            end = max_length
        input_mask[i, start: end] = 1
    input_mask = input_mask.bool()
    
    input_ids = input_ids[input_mask].reshape(-1, max_length).to(device)
    labels = labels[input_mask].reshape(-1, max_length).to(device)
    attention_mask = attention_mask[input_mask].reshape(-1, max_length).to(device)
    
    return dict(input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels)

def grad_exists(p):
    return p is not None and p.requires_grad == True
    
def compute_grad(loss, params):
    grads = []
    loss = loss.double()
    for i, p in enumerate(params):
        if i == len(params) - 1:
            grad = torch.autograd.grad(loss, p, retain_graph=False)
        else:
            grad = torch.autograd.grad(loss, p, retain_graph=True)
        # print(grad[0].sum())
        grads.append(grad[0].detach().cpu())
    del grad
    return grads
    
def compute_grad_sim(grads_1, grads_2, eps=1e-6):
    prod = 0
    for g1, g2 in zip(grads_1, grads_2):
        prod += torch.matmul(g1.view(1, -1).double(), g2.view(-1).double())
    prod = prod.item()
    print("prod: ", prod)
    cos_sim = prod #/max(norm, eps)
    
    return cos_sim

def accum_grad(out_list, grads):
    if len(out_list) == 0:
        out_list = grads
    else:
        out_list = [g1+g2 for g1, g2 in zip(out_list, grads)]
    return out_list

def compute_pretrain_grad(model, tokenizer, doc_list, ngram, batch_size=16, 
                        max_length=1024, log_file=None, task_grads=None):
    params = [p for p in model.parameters() if grad_exists(p)]
    grads_sum = []
    
    for i in range(math.ceil(len(doc_list)/batch_size)):
        doc_batch = doc_list[i*batch_size: min((i+1)*batch_size, len(doc_list))]
        tokenized_batch = prepare_ngram_data(doc_batch, ngram, tokenizer, max_length)
        loss = model(**tokenized_batch).loss
        print("    pretrain loss: ", loss.detach().cpu().item())
        grads = compute_grad(loss, params)
        if log_file is not None:
            assert task_grads is not None
            sim = compute_grad_sim(task_grads, grads)
            texts = tokenizer.batch_decode(tokenized_batch['input_ids'], 
                                        skip_special_tokens = True)
            # print(texts)
            for text in texts:
                text = text.strip()
                if len(text) > 0:
                    with open(log_file, 'a') as wf:
                        json.dump({'ngram': ngram, 'cos_sim': sim, 'text': text}, wf)
                        wf.write('\n')
        grads_sum = accum_grad(grads_sum, grads * len(doc_batch))
        del loss
        del grads
    return [g/len(doc_list) for g in grads_sum]

def shorten_docs(docs, ngram):
    shortend_docs = []
    for doc in docs:
        doc = clean_text(doc)
        ngram_pos = doc.find(ngram)
        end = ngram_pos + len(ngram) + 25 
        if end - MAX_CHARS >= 0:
            start = end - MAX_CHARS
        else:
            start = 0
            end = MAX_CHARS
        shortend_docs.append(doc[start: end])
        # print((start, end))
        # print(doc[end-len(ngram)-1: end])
    # print(shortend_docs)
    return shortend_docs

def clean_text(s):
    s = re.sub(r'[^\w\s]',' ',s)
    s = s.lower().strip()
    return ' '.join(s.split())

def read_df(data_path, dedup_ngram_key, example_key, qa_map, prompt_template, rand_ngram, n):
    data_df = pd.read_pickle(data_path)['pythia-12b']
    if args.task in ['trivia_qa', 'mmlu']:
        data_df = data_df[data_df["value"] >= args.n_docs]
    else:
        data_df = data_df[data_df['task'] == args.task][data_df["value"] >= args.n_docs]
    print(len(data_df))
    task_data = {}
    for text in set(data_df[example_key]):
        df = data_df[data_df[example_key] == text]
        ngram_pairs = df["pair"].to_list()
        if args.task[2:] == '-en':
            ngram_pairs = [(p[1], p[0]) for p in ngram_pairs]
        ans = qa_map(df, 'A')
        
        ngram_pairs = [p for p in ngram_pairs if ans.find(p[1]) > -1]
        
        if len(ngram_pairs) > 2:
            new_ngram_pairs = []
            ngram_1s = set(df[dedup_ngram_key])
            for p in ngram_pairs:
                if p[0] in ngram_1s:
                    new_ngram_pairs.append(p)
                    ngram_1s.remove(p[0])
            ngram_pairs = new_ngram_pairs
            
        input_text = clean_text(qa_map(df, 'Q'))
        output_text = clean_text(ans)
        input_text = prompt_template(input_text)
        if rand_ngram:
            if random_ngram(input_text + output_text, n) is None:
                continue
            ngram_pairs = [(random_ngram(input_text + output_text, n), p[1]) for p in ngram_pairs]
        task_data[input_text + output_text] = ngram_pairs
        
    print("num testing data: ", len(task_data))
    del data_df
    return task_data

def read_ngram_dict(data_path, prompt_template, task, rand_ngram, n):
    data_dict = pd.read_pickle(data_path)
    task_data = {}
    for pair in data_dict:
        count = data_dict[pair]['value']
        if count > 0:
            if task == 'mmlu':
                q = data_dict[pair]['example']['question']
                op = data_dict[pair]['example']['choices']
                a = data_dict[pair]['example']['answer']
                a_text = op[a]
            elif task == 'translation':
                q = data_dict[pair]['example']['translation']['en']
                op = None
                a = data_dict[pair]['example']['translation'][task[:2]]
                a_text = a
            elif task == 'gsm8k':
                q = data_dict[pair]['example']['question']
                op = None
                a = data_dict[pair]['example']['answer']
                a_text = a
                
            if clean_text(a_text).find(pair[-1]) > -1:
                text = prompt_template(q, op, a)
                if rand_ngram:
                    pair = (random_ngram(text, n), pair[1])
                if text in task_data:
                    task_data[text][pair] = count
                else:
                    task_data[text] = {pair: count}
    del data_dict
    return task_data

def load_data(task, subtask, ngram, random_baseline):
    data_path = f'data/task_gram_count/{task}/{subtask}/{ngram}.pkl'
    if task == 'translation':
        def prompt_template(q):
            return 'English text: ' + q + f'\n{language_map[subtask]} translation: '
        
    elif task == 'trivia_qa':
        def prompt_template(q):
            return 'Question: ' + q + '\nAnswer: '
        
    elif task == 'mmlu':
        def prompt_template(q):
            return 'Question: ' + q + ' '
    
    elif task == 'gsm8k':
        def prompt_template(q, op, a):
            return 'Question: ' + clean_text(q) + '\nAnswer: ' + a
    
    else:
        raise NotImplementedError
    
    task_data = read_ngram_dict(data_path, prompt_template, task, random_baseline, ngram)
    
    return task_data

def compute_scores(task_data, model, tokenizer, pretrain_doc_file, es, out_dir, args):
    if args.corpus == 'pile':
        index = 're_pile'
    elif args.corpus == 'dolma':
        index = 'docs_v1.5_2023-11-02'
    else:
        raise NotImplementedError
    if os.path.exists(pretrain_doc_file):
        try: 
            all_docs = json.load(open(pretrain_doc_file))
        except:
            try:
                with open(pretrain_doc_file) as rf:
                    text = rf.read()
                    all_docs = json.loads(rf'{text}')
            except:
                all_docs = {}
    else:
        all_docs = {}
        
    parallel_log_file = f'{out_dir}/parallel_log.jsonl'
    non_parallel_log_file = f'{out_dir}/non_parallel_log.jsonl'
    score_file = f'{out_dir}/scores.json'
    
    if os.path.exists(score_file):
        with open(score_file) as rf:
            scores = json.load(rf)
    else:
        scores = {'parallel': [], 'non_parallel': []}
    skip = len(scores['parallel'])
    num_score = 0
    for idx, example in enumerate(task_data):
        print(example)
        for ngram_1, ngram_2 in task_data[example]:
            num_score += 1
            if num_score < skip:
                continue
            ngram_1 = clean_text(ngram_1)
            ngram_2 = clean_text(ngram_2)
            print("    n-gram: ", (ngram_1, ngram_2))
            parallel_doc_ids = []
            
            task_grads = compute_pretrain_grad(model, tokenizer, 
                                    [example], ngram_2, 
                                    args.batch_size, args.max_length)
            
            if ngram_2 in all_docs and ngram_1 in all_docs[ngram_2]:
                parallel_doc_texts = all_docs[ngram_2][ngram_1]
            else:
                try:
                    parallel_docs = get_documents_containing_phrases(index, [ngram_1, ngram_2], 
                                                    all_phrases=True,
                                                    num_documents=args.n_docs, 
                                                    es=es)
                except:
                    continue
                parallel_doc_ids = []
                parallel_doc_texts = []
                for doc in iter(parallel_docs):
                    parallel_doc_ids.append(doc['_id'])
                    parallel_doc_texts.append(doc['_source']['text'])
                if ngram_2 in all_docs:
                    all_docs[ngram_2][ngram_1] = parallel_doc_texts
                else:
                    all_docs[ngram_2] = {ngram_1: parallel_doc_texts}
                json.dump(all_docs, open(pretrain_doc_file, 'w'))
                
            print("     num parallel docs: ", len(parallel_doc_texts))
            
            shortend_docs = shorten_docs(parallel_doc_texts, ngram_2)
            
            parallel_pretrain_grads = compute_pretrain_grad(model, tokenizer, 
                                    shortend_docs, ngram_2, 
                                    args.batch_size, args.max_length,
                                    parallel_log_file, task_grads)
            
            if not args.random_baseline:
                if ngram_2 in all_docs[ngram_2]:
                    non_parallel_doc_texts = all_docs[ngram_2][ngram_2]
                else:
                    try:
                        docs = get_documents_containing_phrases(index, ngram_2, 
                                                        num_documents=args.n_docs, 
                                                        es=es)
                    except:
                        continue
                    non_parallel_doc_texts = []
                    for doc in iter(docs):
                        if doc['_id'] not in parallel_doc_ids and ngram_1 not in doc['_source']['text']:
                            non_parallel_doc_texts.append(doc['_source']['text'])
                    
                    all_docs[ngram_2][ngram_2] = non_parallel_doc_texts
                    
                    json.dump(all_docs, open(pretrain_doc_file, 'w'))
                        
                print("     num non parallel docs: ", len(non_parallel_doc_texts))
                
                shortend_docs = shorten_docs(non_parallel_doc_texts, ngram_2)
                
                non_parallel_pretrain_grads = compute_pretrain_grad(model, tokenizer, 
                                        shortend_docs, ngram_2, 
                                        args.batch_size, args.max_length,
                                        non_parallel_log_file, task_grads)
            
            if len(parallel_pretrain_grads) > 0 and (args.random_baseline or len(non_parallel_pretrain_grads)>0) and len(task_grads) > 0:
                with torch.no_grad():
                    parallel_score = compute_grad_sim(task_grads, parallel_pretrain_grads)
                    if not args.random_baseline:
                        non_parallel_score = compute_grad_sim(task_grads, non_parallel_pretrain_grads)
                
                scores['parallel'].append(parallel_score)
                
                if args.random_baseline:
                    print(f"Example {idx}, ({ngram_1}, {ngram_2}): parallel score = {parallel_score}")
                    print(f"Average parallel score = {np.mean(scores['parallel'])}")
                else:
                    print(f"Example {idx}, ({ngram_1}, {ngram_2}): parallel score = {parallel_score}, non parallel score = {non_parallel_score}")
                    scores['non_parallel'].append(non_parallel_score)
                    print(f"Average parallel score = {np.mean(scores['parallel'])}, non parallel score = {np.mean(scores['non_parallel'])}")
                    
                json.dump(scores, open(score_file, 'w'))
                
                if len(scores) > args.max_examples:
                    print(f"reach maximum number of scores {args.max_examples}")
                    return scores
        
    return scores

def main(args):
    
    set_seed(args.seed)
    
    es = Elasticsearch(
        cloud_id=args.cloud_id,
        api_key=args.api_key,
        retry_on_timeout=True,
        http_compress=True,
        timeout=30, max_retries=10)
    
    task_data = load_data(args.task, args.subtask, args.ngrams, args.random_baseline)
    
    for model_name in args.models:
        for step in args.checkpoints:
            if step < 1e5:
                model = AutoModelForCausalLM.from_pretrained(model_name, 
                        revision = f'step{step}',
                        torch_dtype=torch.float16, device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, 
                        torch_dtype=torch.float16, device_map="auto")
                
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if args.random_baseline:
                out_dir = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}_random2/num_docs={args.n_docs}/{model_name}_prod/{step}'
                pretrain_doc_file = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}_random2/num_docs={args.n_docs}/retrived_docs_{args.corpus}.json'
            else:
                out_dir = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}/num_docs={args.n_docs}/{model_name}_prod/{step}'
                pretrain_doc_file = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}/num_docs={args.n_docs}/retrived_docs_{args.corpus}.json'
            os.makedirs(out_dir, exist_ok=True)

            scores = compute_scores(task_data, model, tokenizer, pretrain_doc_file, es, out_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find supportive examples for a task')
    parser.add_argument('--models', nargs="+", type=str, default=None,
                        help='Huggingface pretrained model', )
    parser.add_argument('--corpus', type=str, default='pile',
                        help='Pretraining corpus to search ngrams', )
    parser.add_argument('--checkpoints', nargs="+", type=int, default=None,
                        help='checkpoint steps')
    parser.add_argument('--task', type=str, default='fr-en',
                        help='Eval task', )
    parser.add_argument('--subtask', type=str, default='fr',
                        help='Eval subtask', )
    parser.add_argument('--out_dir', type=str, default='out/grad',
                        help='Output directory', )
    parser.add_argument('--reverse', type=bool, default=True,
                        help='reverse translation direction')
    parser.add_argument('--avg_all', type=bool, default=False,
                        help='average gradient over all testing examples')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='value of n', )
    parser.add_argument('--n_docs', type=int, default=50,
                        help='number of documents to retrieve per ngram', )
    parser.add_argument('--max_length', type=int, default=1024,
                        help='maximum sequence length', )
    parser.add_argument('--batch_size', type=int, default=1,
                        help='maximum num of sequences each time', )
    parser.add_argument('--max_examples', type=int, default=50,
                        help='maximum number of examples', )
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed', )
    parser.add_argument('--random_baseline', type=bool, default=False,
                        help='add random baseline', )
    parser.add_argument('--cloud_id', type=str, default=None,
                        help='WIMBD cloud id', )
    parser.add_argument('--api_key', type=str, default=None,
                        help='WIMBD api key', )
    args = parser.parse_args()
    
    checkpoints = reversed([int(i*2e4) for i in range(1, 5)])
    
    if args.models[0] == 'pythia':
        args.models = reversed(['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 
            'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
            'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
            'EleutherAI/pythia-70m'])
        args.corpus = 'pile'
    elif args.models[0] == 'olmo':
        args.models = ['OLMo-1B', 'OLMo-7B', 'OLMo-7B-instruct']
        args.corpus = 'dolma'
    else:
        raise NotImplementedError
        
    if args.checkpoints is None:
        args.checkpoints = checkpoints
    
    main(args)