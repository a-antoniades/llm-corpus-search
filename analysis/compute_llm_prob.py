from typing import Dict, Optional, Sequence
import json, copy, math, os
import torch
import argparse
import numpy as np
import pandas as pd
import sys, re
import hf_olmo
import requests

from torch.nn import CrossEntropyLoss
from transformers import GPTNeoXForCausalLM
from hf_olmo import OLMoForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

sys.path.append('../wimbd')
np.random.seed(42)

index = 're_pile'
cloud_id = "m-datasets:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1N2U5ODIwZDUxNTQ0YWViMjk0MmQwNzI1NjE0OTQ2JDhkN2M0OWMyZDEzMTRiNmM4NDNhNGEwN2U4NDE5NjRl"
api_key = "RlZBbHpZc0J1MEw4LVVWVk9SaTE6bXJlSUM2QnlSQmFHemhwVElVUnZyQQ=="

language_map = {'fr': 'French', 'cs': 'Czech', 'de': 'German', 'es': 'Spanish',
                'it': 'Italian', 'hu': 'Hungarian'}

mmlu_tasks = {'prehistory', 'business_ethics', 'econometrics', 'college_medicine', 
            'professional_law', 'philosophy', 'abstract_algebra', 'moral_disputes', 
            'college_chemistry', 'medical_genetics', 'high_school_government_and_politics', 
            'human_aging', 'us_foreign_policy', 'high_school_macroeconomics', 
            'logical_fallacies', 'moral_scenarios', 'college_mathematics', 
            'international_law', 'computer_security', 'sociology', 'professional_psychology', 
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

knowledge_tasks = ['prehistory', 'business_ethics', 'philosophy', 
                'moral_disputes', 'medical_genetics', 'high_school_government_and_politics', 
                'human_aging', 'us_foreign_policy', 'high_school_macroeconomics',
                'logical_fallacies', 'international_law', 'computer_security',
                'sociology', 'professional_psychology', 'marketing', 'human_sexuality',
                'anatomy', 'high_school_us_history', 'public_relations', 
                'high_school_microeconomics', 'clinical_knowledge', 'security_studies',
                'nutrition', 'world_religions', 'high_school_psychology', 
                'high_school_geography', 'management', 'global_facts', 
                'high_school_world_history', 'high_school_european_history',
                'jurisprudence', 'virology', 'astronomy', 'miscellaneous']
reasoning_tasks = ['econometrics', 'professional_law', 'abstract_algebra', 'college_medicine', 
                'college_chemistry', 'moral_scenarios', 'college_mathematics', 
                'high_school_chemistry', 'professional_accounting', 'college_computer_science',
                'college_biology', 'high_school_computer_science', 'high_school_mathematics', 
                'college_physics', 'professional_medicine', 'elementary_mathematics',
                'machine_learning', 'electrical_engineering', 'high_school_physics', 
                'conceptual_physics', 'high_school_statistics', 'high_school_biology']

IGNORE_INDEX = -100
MAX_CHARS = 3000
MIN_CHARS = 500

def pythia_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    loss_mode: Optional[str] = 'sum',
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
        `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
        only required when the model is used as a decoder in a Sequence to Sequence model.

        Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
        `past_key_values` input) to speed up sequential decoding.

        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
        `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
        ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
    >>> config.is_decoder = True
    >>> model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> prediction_logits = outputs.logits
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.gpt_neox(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    lm_logits = self.embed_out(hidden_states)

    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(lm_logits.device)
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        loss = loss.view([labels.size(0), labels.size(1)])
        if loss_mode == 'sum':
            loss = loss.sum(-1)
        elif loss_mode == 'average':
            loss = loss.sum(-1)/(loss!=0).sum(-1)
        else:
            print("no such loss mode: ", loss_mode)
            raise NotImplementedError

    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    
# monkey patching
GPTNeoXForCausalLM.forward = pythia_forward

def olmo_forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
        loss_mode: Optional[str] = 'sum',
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.config.embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view([labels.size(0), labels.size(1)-1])
            if loss_mode == 'sum':
                loss = loss.sum(-1)
            elif loss_mode == 'average':
                loss = loss.sum(-1)/(loss!=0).sum(-1)
            else:
                print("no such loss mode: ", loss_mode)
                raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )
        
# monkey patching
OLMoForCausalLM.forward = olmo_forward

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
    m = False
    for i in range(input_ids.size(1) - num_ngram_tokens + 1):
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

def compute_loss(model, tokenizer, doc, ngram, batch_size=16, 
                        max_length=1024, log_file=None,):    
    tokenized_batch = prepare_ngram_data([doc], ngram, tokenizer, max_length)
    loss = model(**tokenized_batch, loss_mode='sum').loss.detach().cpu()
    print("    pretrain loss: ", loss)
    texts = []
    if log_file is not None and not torch.isnan(loss):
        texts = tokenizer.batch_decode(tokenized_batch['input_ids'], 
                                    skip_special_tokens = True)
    return loss[0].item(), texts

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
    return shortend_docs

def clean_text(s):
    s = re.sub(r'[^\w\s]',' ',s)
    s = s.lower().strip()
    return ' '.join(s.split())

def get_inifi_gram_count(ngram):
    payload = {
        'index': 'v4_piletrain_llama',
        'query_type': 'count',
        'query': ngram,
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    print(result)
    return result['count']

def get_inifi_gram_prob(input_text, output_text):
    probs = []
    for word in output_text.split():
        input_text += ' ' + word
        payload = {
            'index': 'v4_piletrain_llama',
            'query_type': 'infgram_prob',
            'query': input_text,
        }
        result = requests.post('https://api.infini-gram.io/', json=payload).json()
        print(result)
        if 'prob' in result:
            probs.append(result['prob'])
    return probs

def read_ngram_dict(data_path, prompt_template, task):
    data_dict = pd.read_pickle(data_path)
    task_data = {}
    all_ngram_counts = {}
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
                if text in task_data:
                    task_data[text][pair] = count
                else:
                    task_data[text] = {pair: count}
                if pair not in all_ngram_counts:
                    all_ngram_counts[pair] = count
    del data_dict
    return task_data, all_ngram_counts

def load_data(ngram, task, subtask):
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
    
    task_data, all_ngram_pair_counts = read_ngram_dict(data_path, prompt_template, task)
    
    return task_data, all_ngram_pair_counts

def main(args):
    task_data_file = f'{args.out_dir}/{args.task}/{args.ngrams}/task_data_{args.corpus}.json'
    
    if os.path.exists(task_data_file):
        with open(task_data_file) as rf:
            r_task_data = json.load(rf)
        
        task_data = {}
        for text in r_task_data:
            task_data[text] = {}
            for ngram2 in r_task_data[text]:
                task_data[text][ngram2] = {'ngram2_stats': r_task_data[text][ngram2]['ngram2_stats']}
                for ngram_pair in r_task_data[text][ngram2]:
                    if ngram_pair == 'ngram2_stats':
                        continue
                    task_data[text][ngram2][tuple(ngram_pair.split('; '))] = r_task_data[text][ngram2][ngram_pair]
        del r_task_data
        
    else:
        os.makedirs(f'{args.out_dir}/{args.task}/{args.ngrams}', exist_ok=True)
        task_data, ngram_pair_counts = load_data(args.ngrams, args.task, args.subtask)
        print("wimbd data loaded")
        
        ngrams = set()
        for p in ngram_pair_counts:
            ngrams.add(p[0])
            ngrams.add(p[1])
        single_ngram_counts = {}
        for ngram in ngrams:
            single_ngram_counts[ngram] = get_inifi_gram_count(ngram)
        
        w_task_data = {}
        new_task_data = {}
        i = 0
        for text in task_data:
            new_task_data[text] = {}
            w_task_data[text] = {}
            ngram2topairs = {}
            for p in task_data[text]:
                if p[1] in ngram2topairs:
                    ngram2topairs[p[1]].append(p)
                else:
                    ngram2topairs[p[1]] = [p]
            for ngram2 in ngram2topairs:
                max_len = 500
                ngram_pos = text.find(ngram2)
                infini_gram_prob = get_inifi_gram_prob(text[max(ngram_pos-max_len, 0): ngram_pos], ngram2)
                try:
                    ngram2_count = single_ngram_counts[ngram2]
                except:
                    ngram2_count = get_inifi_gram_count(ngram2)
                stats = {'ngram2_count': ngram2_count,
                        'ngram2_infini_gram_probs': infini_gram_prob,}
                if ngram2_count == 0:
                    continue
                new_task_data[text][ngram2] = {'ngram2_stats': stats}
                w_task_data[text][ngram2] = {'ngram2_stats': stats}
                for ngram_pair in ngram2topairs[ngram2]:
                    try:
                        ngram1_count = single_ngram_counts[ngram_pair[0]]
                    except:
                        ngram1_count = get_inifi_gram_count(ngram_pair[0])
                    if ngram1_count == 0:
                        continue
                    ngram_pair_prob = task_data[text][ngram_pair]/ngram1_count
                    stats = {'ngram_pair_count': task_data[text][ngram_pair],
                            'ngram1_count': ngram1_count,
                            'ngram_pair_prob': ngram_pair_prob}
                    new_task_data[text][ngram2][ngram_pair] = stats
                    w_task_data[text][ngram2][ngram_pair[0] + '; ' + ngram_pair[1]] = stats
                
                if len(new_task_data[text][ngram2]) == 1:
                    w_task_data[text].pop(ngram2, None)
                    new_task_data[text].pop(ngram2, None)
                    
            if len(new_task_data[text]) == 0:
                w_task_data.pop(text, None)
                new_task_data.pop(text, None)
            else:
                i += 1
        
        with open(task_data_file, 'w') as wf:
            json.dump(w_task_data, wf, indent=4)
        del w_task_data
        
        task_data = new_task_data
        
    for model_name in args.models:
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        model = model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        out_dir = f'{args.out_dir}/{args.task}/{args.ngrams}/{model_name}'
        os.makedirs(out_dir, exist_ok=True)
        loss_file = f'{out_dir}/log_probs_new.jsonl'
        
        for idx, example in enumerate(task_data):
            print(example)
            for ngram in task_data[example]:
                ngram = clean_text(ngram)
                print("    n-gram: ", ngram)
                loss, texts = compute_loss(model, tokenizer, example, ngram, 
                            args.batch_size, args.max_length, loss_file)
                if len(texts) == 1: # loss is not NaN
                    text = texts[0].strip()
                    if len(text) > 0 and loss > 0:
                        ngram_pair_probs = []
                        for ngram_pair in task_data[example][ngram]:
                            if ngram_pair == 'ngram2_stats':
                                continue
                            ngram_pair_probs.append(task_data[example][ngram][ngram_pair]['ngram_pair_prob'])
                        
                        
                        infini_gram_prob = [p for p in task_data[example][ngram]['ngram2_stats']['ngram2_infini_gram_probs'] if p > 0]

                        log_infini_gram_prob = np.log(infini_gram_prob).sum()
                        print(log_infini_gram_prob)
                        
                        log_ngram_pair_prob = np.log(ngram_pair_probs).mean()
                        print(log_ngram_pair_prob)
                        
                        with open(loss_file, 'a') as wf:
                            json.dump({'ngram': ngram, 
                                '-log_ngram_pair_prob': -log_ngram_pair_prob, 
                                '-log_infini_gram_prob': -log_infini_gram_prob, 
                                '-lm_log_p': loss, 
                                'text': text}, wf)
                            wf.write('\n')
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find supportive examples for a task')
    parser.add_argument('--models', nargs="+", type=str, default=None,
                        help='Huggingface pretrained model', )
    parser.add_argument('--corpus', type=str, default='pile',
                        help='Pretraining corpus to search ngrams', )
    parser.add_argument('--task', type=str, default='translation',
                        help='Eval task', )
    parser.add_argument('--subtask', type=str, default='fr',
                        help='Eval subtask', )
    parser.add_argument('--out_dir', type=str, default='out/dist',
                        help='Output directory', )
    parser.add_argument('--reverse', type=bool, default=True,
                        help='reverse translation direction')
    parser.add_argument('--avg_all', type=bool, default=False,
                        help='average gradient over all testing examples')
    parser.add_argument('--ngrams', type=int, default=2,
                        help='value of n', )
    parser.add_argument('--max_length', type=int, default=1024,
                        help='maximum sequence length', )
    parser.add_argument('--batch_size', type=int, default=16,
                        help='maximum num of sequences each time', )
    args = parser.parse_args()
    
    if args.models[0] == 'pythia':
        args.models = reversed(['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 
            'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
            'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
            'EleutherAI/pythia-70m'])
        args.corpus = 'pile'
    elif args.models[0] == 'olmo':
        args.models = ['allenai/OLMo-1B', 'allenai/OLMo-7B', 'allenai/OLMo-7B-instruct']
        args.corpus = 'dolma'
    else:
        raise NotImplementedError
    print(args.models)
    main(args)