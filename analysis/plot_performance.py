import numpy as np
import pandas as pd
import re, os, argparse, json
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load
from sentence_transformers import SentenceTransformer

plt.rcParams.update({'font.size': 12})
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
lang_tasks = {'cs', 'de', 'es', 'fr', 'hu', 'it'}

def clean_text(s):
    s = re.sub(r'[^\w\s]',' ',s)
    s = s.lower().strip()
    return ' '.join(s.split())

def read_ngram_dict(data_path, task):
    data_dict = pd.read_pickle(data_path)
    task_data = {}
    all_ngram_counts = {}
    for pair in data_dict:
        count = data_dict[pair]['value']
        if task in ['mmlu', 'gsm8k']:
            text = data_dict[pair]['example']['question']
        elif task == 'translation':
            text = data_dict[pair]['example_clean']
        else:
            raise NotImplementedError
        
        if text in task_data:
            task_data[text][pair] = count
        else:
            task_data[text] = {pair: count}
        if pair not in all_ngram_counts:
            all_ngram_counts[pair] = count
    del data_dict
    return task_data, all_ngram_counts

def filter_cos_sim(ngram_counts, t, out_file):
    res = {}
    if os.path.exists(out_file):
        with open(out_file) as rf:
            sim = json.load(rf)
        for p in ngram_counts:
            if sim[p[0] + ', ' + p[1]]['cos_sim'] > t:
                res[p] = ngram_counts[p]
    else:
        embedding_model = SentenceTransformer(
            "intfloat/multilingual-e5-large",
            prompts={
                "classification": "Classify the following text: ",
                "retrieval": "Retrieve semantically similar text: ",
                "clustering": "Identify the topic or theme based on the text: ",
            },
        )
        sims = {}
        for ngram1, ngram2 in ngram_counts:
            embd1 = embedding_model.encode(ngram1)
            embd2 = embedding_model.encode(ngram2)
            sim = embedding_model.similarity(embd1, embd2).detach().cpu().numpy()[0][0]
            sim = float(sim)
            sims[ngram1 + ', ' + ngram2] = {'cos_sim': sim, 'count': ngram_counts[(ngram1, ngram2)]}
            print(f"({ngram1}, {ngram2}): {sim}")
            if sim > t:
                res[(ngram1, ngram2)] = ngram_counts[(ngram1, ngram2)]
            with open(out_file, 'w') as wf:
                json.dump(sims, wf, indent=4)
    return res

def main(args):
    df_pth = f'data/wimbd/{args.task}/{args.ngram}/model_perf.pkl'
    df = pd.read_pickle(df_pth)
    q_col = 'query_x'
    if args.task == 'mmlu':
        tasks = mmlu_tasks
    elif args.task == 'translation':
        tasks = lang_tasks
    elif args.task == 'trivia_qa':
        tasks = ['trivia_qa']
    elif args.task == 'gsm8k':
        tasks = ['gsm8k']
    else: 
        raise NotImplementedError
        
    task_data = {}
    all_ngram_pair_counts = {}
    for subtask in tasks:
        data_path = f'data/task_gram_count/{args.task}/{subtask}/{args.ngram}.pkl'
        sub_task_data, sub_ngram_pair_counts = read_ngram_dict(data_path, args.task)
        task_data = task_data | sub_task_data
        all_ngram_pair_counts = all_ngram_pair_counts | sub_ngram_pair_counts
    
    out_file = f'out/sim/{args.task}/ngrams={args.ngram}/cos_sim.json'
    os.makedirs(f'out/sim/{args.task}/ngrams={args.ngram}', exist_ok=True)
    if args.t > 0:
        all_ngram_pair_counts = filter_cos_sim(all_ngram_pair_counts, args.t, out_file)
        with open(f'out/sim/{args.task}/ngrams={args.ngram}/text_ngram.json', 'w') as wf:
            w_data = {text: [p for p in task_data[text]] for text in task_data}
            json.dump(w_data, wf, indent=4)
    
    num_zeros = len([p for p in all_ngram_pair_counts if all_ngram_pair_counts[p]==0])
    print("num no zero counts: ", len(all_ngram_pair_counts) - num_zeros)
    zero_rate = num_zeros/len(all_ngram_pair_counts)
    print("zero count rate: ", zero_rate)
    trigram_count = {k: sum([all_ngram_pair_counts[p] for p in task_data[k] if p in all_ngram_pair_counts]) for k in task_data}
    
    plt.figure(figsize=(7,6))
            
    for model in models:
        trigram_col = []
        for text in df[model][q_col]:
            try:
                trigram_col.append(trigram_count[text])
            except: 
                trigram_col.append(0)
        df[model]['trigram_count'] = trigram_col
        cur_df = df[model].groupby(q_col).first()
            
        nonzero_entry = cur_df[cur_df['trigram_count'] > args.min_count]
        test_examples = nonzero_entry[nonzero_entry['trigram_count'] < args.max_count]
        ngram_counts = test_examples['trigram_count']
        
        if args.task == 'mmlu':
            labels = np.array(test_examples['gold'].to_list())
            probs = test_examples['probs_softmax']
            prob_array = np.array([list(prob) for prob in probs.to_list()])
            pred_probs = []
            for prob, l in zip(prob_array, labels):
                pred_probs.append(prob[l])
            res = prob_array.argmax(-1) == labels
        elif args.task == 'translation':
            res = np.array(test_examples['bleu'].to_list())
        elif args.task == 'trivia_qa':
            labels = [a['aliases'] for a in test_examples['answer']]
            preds = test_examples['result'].to_list()
            res = []
            for l, p in zip(labels, preds):
                p = p.strip().lower()
                for _l in l:
                    _l = _l.strip().lower()
                    if _l == p:
                        res.append(1)
                    else:
                        res.append(0)
        elif args.task == 'gsm8k':
            if args.metric == 'acc':
                res = np.array(test_examples['acc'].to_list())
            elif args.metric == 'bertscore':
                res = bertscore.compute(predictions=test_examples['result'], references=test_examples['answer'], lang="en")['f1']
        else:
            raise NotImplementedError
        
        prob_bins = [[] for _ in range(args.bins)]
        bin_size = args.max_count//args.bins
        for c, p in zip(ngram_counts, res):
            for i in range(args.bins):
                if c < (i+1)*(bin_size):
                    prob_bins[i].append(p)
                    break
        acc = [np.mean(p) for p in prob_bins]
        
        x = np.arange(args.bins)
        plt.plot(x, acc, 'o-', color=color_mapping[model], label=model)
        plt.xticks(x, x*(bin_size))

    plt.xlabel('# N-grams')
    if args.task in ['mmlu', 'trivia_qa']:
        plt.ylabel('Accuracy')
    elif args.task == 'translation':
        plt.ylabel('BLEU')
    elif args.task == 'gsm8k':
        if args.metric == 'acc':
            plt.ylabel('Accuracy')
        elif args.metric == 'bertscore':
            plt.ylabel('BERTScore')
    plt.legend(loc="upper left")
    os.makedirs(f'out/perf/{args.task}/ngrams={args.ngram}', exist_ok=True)
    plt.savefig(f'out/perf/{args.task}/ngrams={args.ngram}/{args.task}_ngram_vs_perf_{args.shot}shot_{args.corpus}_t={args.t}_{args.metric}.pdf')
    plt.savefig(f'out/perf/{args.task}/ngrams={args.ngram}/{args.task}_ngram_vs_perf_{args.shot}shot_{args.corpus}_t={args.t}_{args.metric}.png')
    plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot acc v.s. counts')
    parser.add_argument('--task', type=str, default='translation')
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--corpus', type=str, default='pile',
                        help='Pretraining corpus to search ngrams.')
    parser.add_argument('--metric', type=str, default='acc',
                        help='Evaluation metric.')
    parser.add_argument('--max_count', type=int, default=1000)
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--t', type=float, default=0, 
                        help="cosine similarity threshold")
    args = parser.parse_args()
    
    if args.corpus == 'pile':
        models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b', 'pythia-410m', 
        'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m']
    elif args.corpus == 'dolma':
        models = ['OLMo-1B', 'OLMo-7B', 'OLMo-7B-instruct']
    else:
        raise NotImplementedError
    
    colors = sns.color_palette('coolwarm', len(models))
    color_mapping = {model: color for model, color in zip(models, colors)}
    
    main(args)