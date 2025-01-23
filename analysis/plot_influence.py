import matplotlib.pyplot as plt
import numpy
import json, os, argparse

plt.rcParams.update({'font.size': 14})
languages = ['cs', 'fr', 'es', 'de', 'hu', 'it']
p_line_styles = ['r+-', 'bs-', 'g^-', 'yo-', 'cp-', 'm*-']
np_line_styles = ['r+--', 'bs--', 'g^--', 'yo--', 'cp--', 'm*--']
rand_line_styles = ['r+:', 'bs:', 'g^:', 'yo:', 'cp:', 'm*:']
xs = [numpy.log(0.014), numpy.log(0.031), numpy.log(0.07), numpy.log(0.16), numpy.log(0.41), numpy.log(1.4), numpy.log(2.8)]

def aggregate(args):
    x_names = []
    y_p = []
    y_np = []
    y_rand = []
    for model in args.models:
        if args.task == 'translation':
            ckpt_dirs = [f'{args.out_dir}/{lan}-en/ngrams={args.ngrams}/num_docs={args.n_docs}/{model}_prod'for lan in languages]
            if args.random_baseline:
                ckpt_dirs = [f'{args.out_dir}/{lan}-en/ngrams={args.ngrams}_random2/num_docs={args.n_docs}/{model}_prod'for lan in languages]
        else:
            ckpt_dirs = [f'{args.out_dir}/{args.task}/ngrams={args.ngrams}/num_docs={args.n_docs}/{model}_prod']
            if args.random_baseline:
                ckpt_dirs = [f'{args.out_dir}/{args.task}/ngrams={args.ngrams}_random2/num_docs={args.n_docs}/{model}_prod']
        if model.split('-')[-1] == 'deduped':
            args.checkpoints = ['']
        score_files = []
        for ckpt_dir in ckpt_dirs:
            for step in args.checkpoints:
                score_files.append(f'{ckpt_dir}/{step}/scores.json')
        
        cur_y_p = []
        cur_y_np =[]
        for score_file in score_files:
            if os.path.exists(score_file):
                scores = json.load(open(score_file))
                p_score = 0
                np_score = 0
                num = 0
                for p in scores["parallel"]:
                    if p == 0:
                        continue
                    p_score += p
                    num += 1
                if num > 0:
                    cur_y_p.append(p_score/num)
                
                num = 0
                for p in scores["non_parallel"]:
                    if p == 0:
                        continue
                    np_score += p
                    num += 1
                if num > 0:
                    cur_y_np.append(np_score/num)
            else:
                print(score_file, ' does not exist')
                continue
        cur_x = model.split('-')[1]
        print("model size: ", cur_x)
        print("parallel score across checkpoints: ", cur_y_p)
        print("non-parallel score across checkpoints: ", cur_y_np)
        x_names.append(cur_x)
        y_p.append(numpy.mean(cur_y_p))
        y_np.append(numpy.mean(cur_y_np))
    
    plt.figure(figsize=(7,6))
    x = list(range(len(x_names)))
    print(x)
    print(x_names)
    print(y_p)
    print(y_np)
    plt.plot(x, y_p, 'o-g', linewidth=2, label='docs w/ ngram pair')
    plt.plot(x, y_np, 's-b', linewidth=2, label='docs w/ single ngram')
    plt.xticks(x, x_names)
    plt.legend(loc="lower left")
    plt.xlabel(f'Pythia model size')
    plt.ylabel(f'Training influence tracing score')
    
    img_dir = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}/num_docs={args.n_docs}'
    if args.random_baseline:
        img_dir = f'{args.out_dir}/{args.task}/ngrams={args.ngrams}_random2/num_docs={args.n_docs}'
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(f'{img_dir}/score.pdf')
    plt.savefig(f'{img_dir}/score.png')
    plt.close()

def main(args):
    if args.task[2:] == '-en':
        
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(15, 7)
        plt.setp(axs, xticks=xs, xticklabels=model_sizes)
        fig.suptitle('Translation')
        idx = 0

        for lan, p_style, np_style in zip(languages, p_line_styles, np_line_styles):
            cur_x = []
            cur_y_p = []
            cur_y_np =[]
            for x, size in zip(xs, model_sizes):
                score_file = f'out/{lan}-en/ngrams=2/num_docs=50/EleutherAI/pythia-{size}-deduped/scores.json'
                if os.path.exists(score_file):
                    scores = json.load(open(score_file))
                    p_score = 0
                    np_score = 0
                    num = 0
                    for p, np in zip(scores["parallel"], scores["non_parallel"]):
                        if p == 0 or np == 0:
                            continue
                        p_score += p
                        np_score += np
                        num += 1
                    if num > 0:
                        cur_y_p.append(p_score/num)
                        cur_y_np.append(np_score/num)
                        cur_x.append(x)
            ax0 = idx // 3
            ax1 = idx % 3
            axs[ax0, ax1].plot(cur_x, cur_y_p, p_style, label=f'{lan}')
            axs[ax0, ax1].plot(cur_x, cur_y_np, np_style)
            axs[ax0, ax1].legend()
            idx += 1
        
        fig.supxlabel('Pythia model size (log scale)')
        fig.supylabel('Gradient similarity')
        fig.tight_layout()
        plt.title('Translation')
        fig.savefig('translation_grad_sim.pdf')
        
    elif args.task == 'trivia_qa':
        model_sizes = ['70m', '160m', '410m', '1.4b', '2.8b']
        xs = [numpy.log(0.07), numpy.log(0.16), numpy.log(0.41), numpy.log(1.4), numpy.log(2.8)]

        cur_x = []
        cur_y_p = []
        cur_y_np =[]
        for x, size in zip(xs, model_sizes):
            score_file = f'out/trivia_qa/ngrams=5/num_docs=50/EleutherAI/pythia-{size}-deduped/scores.json'
            if os.path.exists(score_file):
                scores = json.load(open(score_file))
                p_score = 0
                np_score = 0
                num = 0
                for p, np in zip(scores["parallel"], scores["non_parallel"]):
                    if p == 0 or np == 0:
                        continue
                    p_score += p
                    np_score += np
                    num += 1
                if num > 0:
                    cur_y_p.append(p_score/num)
                    cur_y_np.append(np_score/num)
                    cur_x.append(x)
        plt.plot(cur_x, cur_y_p, 'r+-')
        plt.plot(cur_x, cur_y_np, 'r+--')
        plt.xticks(xs, model_sizes)
        plt.xlabel('Pythia model size (log scale)')
        plt.ylabel('Gradient similarity')
        plt.title('Trivia QA')
        plt.savefig('trivia_qa_grad_sim.pdf')
        
    elif args.task == 'mmlu':
        models = ['OLMo-7B-SFT', 'OLMo-7B']
        line_styles = ['r+-', 'bs-', 'g^-', 'yo-', 'cp-', 'm*-']
        
        all_xs = []
        all_ps = []
        all_nps = []
        for model, style in zip(models, line_styles):
            cur_x = []
            cur_y_p = []
            cur_y_np =[]
            num_scores = []

            score_file = f'out/mmlu/ngrams=3/num_docs=1/allenai/{model}-prod/scores.json'
            if os.path.exists(score_file):
                scores = json.load(open(score_file))
                p_score = 0
                np_score = 0
                num = 0
                for p, np in zip(scores["parallel"], scores["non_parallel"]):
                    if p == 0 or np == 0 or numpy.isnan(p) or numpy.isnan(np):
                        continue
                    p_score += p
                    np_score += np
                    num += 1
                if num > 5:
                    cur_y_p.append(p_score/num)
                    cur_y_np.append(np_score/num)
                    num_scores.append(num)
            
            xs = list(range(len(cur_y_p)))
            sorted_ids = numpy.argsort(cur_y_p)
            
            print(numpy.array(num_scores)[sorted_ids])
            
            plt.plot(xs, numpy.array(cur_y_p)[sorted_ids], style, label=f'{model}')
            plt.plot(xs, numpy.array(cur_y_np)[sorted_ids], style + '-')
            plt.xticks(xs, list(numpy.array(cur_x)[sorted_ids]), rotation=45, ha='right')
            
            print(list(numpy.array(cur_x)[sorted_ids]))
            
            plt.legend(loc="lower left")
            plt.xlabel('MMLU tasks')
            plt.ylabel('Gradient similarity')
            plt.legend()
            plt.savefig(f'mmlu_grad_sim_{model}.png', bbox_inches='tight')
            plt.close()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot grad sim')
    parser.add_argument('--models', nargs="+", type=str, default=None,
                        help='Huggingface pretrained model', )
    parser.add_argument('--checkpoints', nargs="+", type=int, default=None,
                        help='checkpoint steps')
    parser.add_argument('--task', type=str, default='translation',
                        help='Eval task', )
    parser.add_argument('--out_dir', type=str, default='out/grad',
                        help='Output directory', )
    parser.add_argument('--ngrams', type=int, default=2,
                        help='value of n', )
    parser.add_argument('--n_docs', type=int, default=50,
                        help='number of documents to retrieve per ngram', )
    parser.add_argument('--random_baseline', type=bool, default=False,
                        help='add random baseline', )
    args = parser.parse_args()
    
    models = reversed(['EleutherAI/pythia-2.8b', 'EleutherAI/pythia-1.4b', 
                'EleutherAI/pythia-410m', 'EleutherAI/pythia-160m', 
                'EleutherAI/pythia-70m', 'EleutherAI/pythia-31m', 
                'EleutherAI/pythia-14m'])
    
    checkpoints = [int(i*2e4) for i in range(1, 6)]
    
    if args.models is None:
        args.models = models
        
    if args.checkpoints is None:
        args.checkpoints = checkpoints
    
    aggregate(args)
