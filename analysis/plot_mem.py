import numpy as np
import os, json
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from compute_loss import clean_text

def bar_chart(x, y, corr_type='pearson', save_pth=None, x_name='ngram_pair', y_name='LM'):
    pearson_corr, pearson_p_value = pearsonr(x, y)
    print('Pearsons correlation: %.3f' % pearson_corr)
    spearman_corr, spearman_p_value = spearmanr(x, y)
    print('Spearmans correlation: %.3f' % spearman_corr)
    tau_corr, tau_p_value = kendalltau(x, y)
    
    stats = {
        'pearson': (pearson_corr, pearson_p_value),
        'spearman': (spearman_corr, spearman_p_value), 
        'kendall_tau': (tau_corr, tau_p_value),
        'kendall_tau_d': ((1 - tau_corr) / 2, None)
    }
    print(stats)
    
    title=f'{x_name}_vs_{y_name}'

    if save_pth is not None:
        os.makedirs(save_pth, exist_ok=True)
        with open(os.path.join(save_pth, f'{title}.json'), 'w') as f:
            json.dump(stats, f)
        print(f"Stats saved to {save_pth}")
    
    x_range = list(range(int(max(x)) + 2))
    y_avg = [[] for _ in x_range]
    
    for _x, _y in zip(x, y):
        for i, t in enumerate(x_range):
            if _x <= t:
                y_avg[i].append(_y)
                break
    y_avg = [np.mean(ys) for ys in y_avg]
    
    plt.bar(x_range, y_avg, align='edge')
    plt.xlabel(f'-log {x_name} prob')
    plt.ylabel(f'-log {y_name} prob')
    if save_pth is not None:
        plt.savefig(os.path.join(save_pth, f'{title}.pdf'))
        plt.savefig(os.path.join(save_pth, f'{title}.png'))
    plt.close()
    
    return stats[corr_type]


def main(args):
    pair_corrs = []
    pair_ps = []
    single_corrs = []
    single_ps = []
    for n in args.ngrams:
        print('ngram = ', n)
        
        if args.task == 'fr-en':
            task = 'translation'
        else:
            task = args.task
            
        if args.t > 0:
            sim_file = f'out/sim/{task}/ngrams={n}/cos_sim.json'
            if os.path.exists(sim_file):
                with open(sim_file) as rf:
                    sim_count = json.load(rf)
            else:
                print(sim_file + " doesn't exist.")
                exit(1)
            
            sims = {}
            for p in sim_count:
                ngram1, ngram2 = p.split(', ')
                k = clean_text(ngram2)
                if k in sims:
                    sims[k].append(sim_count[p]['cos_sim'])
                else:
                    sims[k] = [sim_count[p]['cos_sim']]
            
        pair_corr = []
        pair_p = []
        single_corr = []
        single_p = []
        for model in args.models:
            print(model)
            loss_file = f'{args.out_dir}/{args.task}/{n}/{model}/log_probs_new.jsonl'
            all_log_ngram_pair_probs = []
            all_log_infini_gram_probs = []
            all_lm_log_probs = []
            if os.path.exists(loss_file):
                with open(loss_file) as rf:
                    for l in rf:
                        d = json.loads(l)
                        if args.t > 0:
                            if d['ngram'] in sims:
                                sim = np.mean(sims[d['ngram']])
                            else:
                                sim = 0
                            print("cosine similarity: ", sim)
                        if args.t == 0 or sim >= args.t:
                            all_log_ngram_pair_probs.append(d['-log_ngram_pair_prob'])
                            all_log_infini_gram_probs.append(d['-log_infini_gram_prob'])
                            all_lm_log_probs.append(d['-lm_log_p'])
            else:
                print("log prob file does not exist")
                exit(1)
                
            all_log_ngram_pair_probs = np.array(all_log_ngram_pair_probs)
            all_log_infini_gram_probs = np.array(all_log_infini_gram_probs)
            all_lm_log_probs = np.array(all_lm_log_probs)
            
            corr, p = bar_chart(all_log_ngram_pair_probs, all_lm_log_probs, 
                corr_type=args.corr_type,
                save_pth=f'{args.out_dir}/{args.task}/{n}/{model}', 
                x_name='ngram_pair', y_name='LM')
            pair_corr.append(corr)
            pair_p.append(p)
            corr, p = bar_chart(all_log_infini_gram_probs, all_lm_log_probs, 
                corr_type=args.corr_type,
                save_pth=f'{args.out_dir}/{args.task}/{n}/{model}', 
                x_name='infini_gram', y_name='LM')
            single_corr.append(corr)
            single_p.append(p)

        pair_corrs.append(np.array(pair_corr))
        pair_ps.append(np.array(pair_p))
        single_corrs.append(np.array(single_corr))
        single_ps.append(np.array(single_p))
        
    line_style = ['-', '--']
    colors = ['g', 'b']
    
    markers = ['o', '*']
    x_name = ['-'.join(model.split('-')[1:]) for model in args.models]
    if len(args.models) > 3:
        if args.task == 'gsm8k':
            x = np.array([np.log(0.07), np.log(0.16), np.log(0.41), np.log(1.4), np.log(2.8), np.log(6.9), np.log(12)])
        elif  args.task[:4] == 'mmlu':
            x = np.array([np.log(0.014), np.log(0.031), np.log(0.07), np.log(0.16), np.log(0.41), np.log(1.4), np.log(2.8), np.log(6.9)])
        else:
            x = np.array([np.log(0.014), np.log(0.031), np.log(0.07), np.log(0.16), np.log(0.41), np.log(1.4), np.log(2.8), np.log(6.9), np.log(12)])
    else:
        x = np.arange(len(x_name))
    plt.rcParams.update({'font.size': 12})
    
    for k, n in enumerate(args.ngrams):
        if n == 3:
            plt.plot(x, pair_corrs[k], line_style[1] + colors[k], linewidth=1, label=f'ngram pair (n={n})')
            cluster_1 = []
            cluster_2  = []
            for i, p in enumerate(pair_ps[k]):
                if p is None or p < 0.05:
                    cluster_1.append(i)
                else:
                    cluster_2.append(i)
            plt.scatter(x[cluster_1], pair_corrs[k][cluster_1], marker=markers[0], color=colors[k])
            plt.scatter(x[cluster_2], pair_corrs[k][cluster_2], marker=markers[1], color='gray')
            
            plt.plot(x, single_corrs[k], line_style[0] + colors[k], linewidth=1, label=f'infini-gram (n={n})')
            cluster_1 = []
            cluster_2  = []
            for i, p in enumerate(single_ps[k]):
                if p is None or p < 0.05:
                    cluster_1.append(i)
                else:
                    cluster_2.append(i)
            plt.scatter(x[cluster_1], single_corrs[k][cluster_1], marker=markers[0], color=colors[k])
            plt.scatter(x[cluster_2], single_corrs[k][cluster_2], marker=markers[1], color='gray')
        else:
            plt.plot(x, pair_corrs[k], line_style[0] + colors[k], linewidth=1, label=f'ngram pair (n={n})')
            cluster_1 = []
            cluster_2  = []
            for i, p in enumerate(pair_ps[k]):
                if p is None or p < 0.05:
                    cluster_1.append(i)
                else:
                    cluster_2.append(i)
            plt.scatter(x[cluster_1], pair_corrs[k][cluster_1], marker=markers[0], color=colors[k])
            plt.scatter(x[cluster_2], pair_corrs[k][cluster_2], marker=markers[1], color='gray')
            
            plt.plot(x, single_corrs[k], line_style[1] + colors[k], linewidth=1, label=f'infini-gram (n={n})')
            cluster_1 = []
            cluster_2  = []
            for i, p in enumerate(single_ps[k]):
                if p is None or p < 0.05:
                    cluster_1.append(i)
                else:
                    cluster_2.append(i)
            plt.scatter(x[cluster_1], single_corrs[k][cluster_1], marker=markers[0], color=colors[k])
            plt.scatter(x[cluster_2], single_corrs[k][cluster_2], marker=markers[1], color='gray')
    
    plt.figure(figsize=(6.4, 6.4))
    plt.ylim(-0.05, 0.35)
    plt.xticks(x, x_name)
    plt.legend(loc="best")
    if args.models[0][0] == 'E':
        plt.xlabel(f'Pythia model size (log scale)')
    else:
        plt.xlabel(f'OLMo models')
        
    if args.corr_type == 'kendall_tau_d':
        plt.ylabel(f'Distributional generalization')
    else:
        plt.ylabel(f'Distributional memorization')
    
    plt.savefig(f'{args.out_dir}/{args.task}/{args.corr_type}_t={args.t}.pdf')
    plt.savefig(f'{args.out_dir}/{args.task}/{args.corr_type}_t={args.t}.png')
    plt.show()
    plt.close()
        
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find supportive examples for a task')
    parser.add_argument('--models', nargs="+", type=str, default=None,
                        help='Huggingface pretrained model', )
    parser.add_argument('--corpus', type=str, default='pile',
                        help='Pretraining corpus to search ngrams', )
    parser.add_argument('--task', type=str, default='fr-en',
                        help='Eval task', )
    parser.add_argument('--corr_type', type=str, default='spearman',
                        help='Correlation type', )
    parser.add_argument('--out_dir', type=str, default='out/dist',
                        help='Output directory', )
    parser.add_argument('--reverse', type=bool, default=True,
                        help='reverse translation direction')
    parser.add_argument('--avg_all', type=bool, default=False,
                        help='average gradient over all testing examples')
    parser.add_argument('--ngrams', nargs="+", type=int, default=[3, 5],
                        help='value of n', )
    parser.add_argument('--t', type=float, default=0, 
                        help="cosine similarity threshold")
    args = parser.parse_args()
    
    if args.models[0] == 'pythia':
        if args.task == 'gsm8k':
            args.models = list(reversed(['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 
                'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
                'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                'EleutherAI/pythia-70m']))
        elif args.task[:4] == 'mmlu':
            args.models = list(reversed(['EleutherAI/pythia-6.9b', 
                    'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
                    'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                    'EleutherAI/pythia-70m', 'EleutherAI/pythia-31m', 
                    'EleutherAI/pythia-14m']))
        else:
            args.models = list(reversed(['EleutherAI/pythia-12b', 'EleutherAI/pythia-6.9b', 
                    'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
                    'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                    'EleutherAI/pythia-70m', 'EleutherAI/pythia-31m', 
                    'EleutherAI/pythia-14m']))
        args.corpus = 'pile'
    elif args.models[0] == 'olmo':
        args.models = ['allenai/OLMo-1B', 'allenai/OLMo-7B', 'allenai/OLMo-7B-instruct']
        args.corpus = 'dolma'
        
    main(args)