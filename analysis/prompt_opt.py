from openai import OpenAI
import re, json, copy, math, os, random, sys
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
sys.path.append('./wimbd')
from wimbd.es import count_documents_containing_phrases

task_dict = {'trivia_qa': {'prompt': "Provide a word or a short phrase as the answer to the given factual question.",
                            'examples': ['Question: Where in England was Dame Judi Dench born?\nAnswer: Park Grove',
                                         'Question: In which decade did Billboard magazine first publish and American hit chart?\nAnswer: 30s',
                                         'Question: Who won Super Bowl XX?\nAnswer: Chicago Bears',
                                         'Question: What did Clarice Cliff create?\nAnswer: Pots',
                                         "Question: Which James Bond film features a song by Louis Armstrong?\nAnswer: On Her Majesty's Secret Service"]},
             'gsm8k': {'prompt': "Solve the following math word problem step by step.",
                       'examples': ["Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"+\
                                   "Solution: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
                                   "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n" +\
                                    "Solution: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
                                    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n" +\
                                    "Solution: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."]}}

def generate_prompt(prompt: str, score: float, history: str, META_PROMPT, client):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": f"History: {history}\n\n" + \
                    f"Current Prompt: {prompt}\n" + \
                    f"Current score: {score}\n\n" + \
                    f"New prompt:"
            },
        ],
    )

    return completion.choices[0].message.content

def set_seed(seed):
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)

def get_all_ngrams(n, text):
    text = re.sub(r'[^\w\s]',' ',text)
    words = text.lower().strip().split()
    ngrams = []
    for i in range(len(words)-n):
        ngrams.append(' '.join(words[i: i+n]))
    return ngrams

def filter(prompt_ngrams, example_ngrams, threshold, model):
    prompt_embds = model.encode(prompt_ngrams)
    example_embds = model.encode(example_ngrams)
    sim = model.similarity(prompt_embds, example_embds).detach().cpu().numpy()
    all_pairs = []
    for ngram1 in prompt_ngrams:
        row = []
        for ngram2 in example_ngrams:
            row.append((ngram1, ngram2))
        all_pairs.append(row)
    all_pairs = np.array(all_pairs)
    return all_pairs[sim > threshold]

def search(ngram_pairs, es, index):
    score = 0
    num_ngrams = 0
    if type(ngram_pairs[0]) == str:
        for ngram in ngram_pairs:
            cp = count_documents_containing_phrases(
                index, ngram, es=es)
            score += cp
            num_ngrams += 1
            print(f"{ngram}: {cp}")
    else:
        for ngram1, ngram2 in ngram_pairs:
            cp = count_documents_containing_phrases(
                index, [ngram1, ngram2], all_phrases=True, es=es)
            # print(cp)
            # exit(0)
            score += cp
            num_ngrams += 1
            print(f"({ngram1}, {ngram2}): {cp}")
    
    return score/num_ngrams
            
            
def main(args):
    set_seed(args.seed)
    client = OpenAI()
    
    if args.corpus == 'pile':
        index = 're_pile'
    elif args.corpus == 'dolma':
        index = "docs_v1.5_2023-11-02"
    else:
        raise ValueError(f"Method {args.corpus} not recognized")
    
    es = Elasticsearch(
        cloud_id=args.cloud_id,
        api_key=args.api_key,
        retry_on_timeout=True,
        http_compress=True,
        request_timeout=30, max_retries=10)
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    out_file = f"{args.out_dir}/{args.task}"
    os.makedirs(out_file, exist_ok=True)
    if args.pair:
        out_file += "/pair"
    else:
        out_file += "/single"
    out_file += f"_n={args.n}_it={args.num_iter}_corpus={args.corpus}"
    if args.mem:
        out_file += '_mem.txt'
    else:
        out_file += '_gen.txt'
    
    if args.mem:
        META_PROMPT = """
**Task Description**:

You are tasked with optimizing a given prompt to guide an open-source language model (LM) in completing a specific task effectively. You will receive:

- The current prompt for the task.
- Its corresponding memorization score (Average frequency of task-related n-grams found in the LM's pretraining corpus).
- A history of previous prompt optimization iterations.

**Optimization Goals**:

Clearly describe the intended task with a general instruction that effectively guides the LM to perform the task.
Maximize the memorization score of the updated prompt. The memorization score reflects the distributional correlation between the prompt and the LM's pretraining corpus. A higher score encourages better alignment with the LM’s learned knowledge.

**Example task input-output pairs**:

""" + '\n\n'.join(task_dict[args.task]['examples']) +\
"""


**Output**:

Produce an updated prompt that balances clarity of task instruction with an improved memorization score."""

    else:
        META_PROMPT = """
**Task Description**:

You are tasked with optimizing a given prompt to guide an open-source language model (LM) in completing a specific task effectively. You will receive:

- The current prompt for the task.
- Its corresponding memorization score (Average frequency of task-related n-grams found in the LM's pretraining corpus).
- A few example input-output pairs illustrating the intended task.
- A history of previous prompt optimization iterations.

**Optimization Goals**:

Clearly describe the intended task with a general instruction that effectively guides the LM to perform the task.
Mimizing the memorization score of the updated prompt. The memorization score reflects the distributional correlation between the prompt and the LM's pretraining corpus. A lower score encourages the LM to generate more novel outputs.

**Example task input-output pairs**:

""" + '\n\n'.join(task_dict[args.task]['examples']) +\
"""


**Output**:

Produce an updated prompt that balances clarity of task instruction with an lower memorization score."""

    print(META_PROMPT)
        
    history = []
    prompt = task_dict[args.task]['prompt']
    wf = open(out_file, 'a')
    for i in range(args.num_iter):
        prompt_ngrams = get_all_ngrams(args.n, prompt)
        
        if args.pair:
            example_ngrams = []
            for example in task_dict[args.task]['examples']:
                example_ngrams += get_all_ngrams(args.n, example)
            ngram_pairs = filter(prompt_ngrams, example_ngrams, args.threshold, model)
            score = search(ngram_pairs, es, index)
        else:
            score = search(prompt_ngrams, es, index)
            
        print(f"Iteration {i}:\nPrompt: {prompt}\nScore: {score}.\n")
        wf.write(f"Iteration {i}:\nPrompt: {prompt}\nScore: {score}.\n")
        
        if len(history) == 0:
            history_text = 'No history yet.'
        else:
            history_text = []
            for j, (p, s) in enumerate(history):
                history_text.append(f"Iteration {j}:\nPrompt: {p}\nScore: {s}.\n")
            history_text = '\n'.join(history_text)
        
        new_prompt = generate_prompt(prompt, score, history_text, META_PROMPT, client)
        
        history.append((prompt, score))
        prompt = new_prompt
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optimize task prompt')
    parser.add_argument('--cloud_id', type=str, default=None,
                        help='WIMBD cloud id', )
    parser.add_argument('--api_key', type=str, default=None,
                        help='WIMBD api key', )
    parser.add_argument('--corpus', type=str, default='pile',
                        help='Pretraining corpus to search ngrams', )
    parser.add_argument('--task', type=str, default='trivia_qa',
                        help='Eval task', )
    parser.add_argument('--out_dir', type=str, default='out/prompt',
                        help='Output directory', )
    parser.add_argument('--n', type=int, default=3,
                        help='value of n', )
    parser.add_argument('--num_iter', type=int, default=100,
                        help='number of optimization iterations', )
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Minimum cosine similarity', )
    parser.add_argument('--mem', type=bool, default=False,
                        help='Encourage memorization or generalization', )
    parser.add_argument('--pair', type=bool, default=False,
                        help='Search for ngram pairs or single ngrams', )
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed', )
    args = parser.parse_args()
    
    main(args)
        
        