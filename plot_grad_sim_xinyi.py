import matplotlib.pyplot as plt
import numpy
import json, os, argparse

plt.rcParams.update({'font.size': 15})


# Define country-specific colors
language_colors = {
    'fr-en': '#0055A4',   # France - Blue
    'es-en': '#FF9900',   # Spain - Orange
    'it-en': '#008C45',   # Italy - Green
    'de-en': '#FFCC00',   # Germany - Yellow
    'cs-en': '#D7141A',   # Czech Republic - Red
    'hu-en': '#8B4513'    # Hungary - SaddleBrown
}

def main(args):
    if args.task == 'translation':
        languages = ['cs', 'fr', 'es', 'de', 'hu', 'it']
        markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h', 'x', '+']
        p_line_styles = [f'{marker}-' for marker in markers]
        np_line_styles = ['r+--', 'bs--', 'g^--', 'yo--', 'cp--', 'm*--']
        np_line_styles = [f'{marker}--' for marker in markers]
        model_sizes = ['70m', '160m', '410m', '1.4b', '2.8b']
        xs = [numpy.log(0.07), numpy.log(0.16), numpy.log(0.41), numpy.log(1.4), numpy.log(2.8)]
        
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
                score_file = f'/share/edc/home/xinyi_wang/incidental-supervision/out/{lan}-en/ngrams=2/num_docs=50/EleutherAI/pythia-{size}-deduped/scores.json'
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
            color = language_colors[f'{lan}-en']
            axs[ax0, ax1].plot(cur_x, cur_y_p, p_style, label=f'{lan}',
                               color=color)
            axs[ax0, ax1].plot(cur_x, cur_y_np, np_style,
                               color=color)
            axs[ax0, ax1].legend()
            idx += 1
        
        fig.supxlabel('Pythia model size (log scale)')
        fig.supylabel('Gradient similarity')
        fig.tight_layout()
        plt.title('Translation')
        fig.savefig('translation_grad_sim.pdf')
        
    elif args.task == 'trivia_qa':
        model_sizes = ['70m', '160m', '410m', '1.4b', '2.8b']
        color = '#800080' 
        xs = [numpy.log(0.07), numpy.log(0.16), numpy.log(0.41), numpy.log(1.4), numpy.log(2.8)]

        cur_x = []
        cur_y_p = []
        cur_y_np =[]
        for x, size in zip(xs, model_sizes):
            score_file = f'/share/edc/home/xinyi_wang/incidental-supervision/out/trivia_qa/ngrams=5/num_docs=50/EleutherAI/pythia-{size}-deduped/scores.json'
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
        plt.figure(figsize=(7.5, 7))
        plt.plot(cur_x, cur_y_p, 'rx-', color=color,
                 markersize=10)
        plt.plot(cur_x, cur_y_np, 'rx--', color=color,
                 markersize=10)
        plt.xticks(xs, model_sizes)
        plt.xlabel('Pythia model size (log scale)')
        plt.ylabel('Gradient similarity')
        plt.title('Trivia QA')
        plt.savefig('trivia_qa_grad_sim.pdf')
        
    elif args.task == 'mmlu':
        models = ['OLMo-7B-SFT', 'OLMo-7B']
        line_styles = ['r+-', 'bs-', 'g^-', 'yo-', 'cp-', 'm*-']
        mmlu_tasks = ['business_ethics',  
            'philosophy', 'abstract_algebra', 'moral_disputes', 
            'college_chemistry', 'medical_genetics', 'high_school_government_and_politics', 
            'us_foreign_policy', 'high_school_macroeconomics', 
            'logical_fallacies', 'college_mathematics', 
            'international_law', 'computer_security', 'professional_psychology', 
            'marketing', 'human_sexuality', 'high_school_chemistry', 
            'college_computer_science', 'anatomy', 'high_school_us_history', 
            'college_biology', 'public_relations', 'high_school_computer_science', 
            'high_school_mathematics', 'college_physics', 'professional_medicine', 
            'high_school_microeconomics', 'clinical_knowledge', 'elementary_mathematics', 
            'machine_learning', 'security_studies', 'nutrition', 'world_religions', 
            'high_school_psychology', 'high_school_geography', 'management', 
            'global_facts', 'high_school_world_history', 'electrical_engineering', 
            'high_school_european_history', 'jurisprudence', 'high_school_physics', 
            'conceptual_physics', 'high_school_statistics', 'virology', 
            'high_school_biology', 'astronomy', 'miscellaneous']
        
        # mmlu_tasks = ['prehistory', 'business_ethics', 'econometrics', 'college_medicine', 
        #     'professional_law', 'philosophy', 'abstract_algebra', 'moral_disputes', 
        #     'college_chemistry', 'medical_genetics', 'high_school_government_and_politics', 
        #     'human_aging', 'us_foreign_policy', 'high_school_macroeconomics', 
        #     'logical_fallacies', 'moral_scenarios', 'college_mathematics', 
        #     'international_law', 'computer_security', 'sociology', 'professional_psychology', 
        #     'marketing', 'human_sexuality', 'high_school_chemistry', 'professional_accounting', 
        #     'college_computer_science', 'anatomy', 'high_school_us_history', 
        #     'college_biology', 'public_relations', 'high_school_computer_science', 
        #     'high_school_mathematics', 'college_physics', 'professional_medicine', 
        #     'high_school_microeconomics', 'clinical_knowledge', 'elementary_mathematics', 
        #     'machine_learning', 'security_studies', 'nutrition', 'world_religions', 
        #     'high_school_psychology', 'high_school_geography', 'management', 
        #     'global_facts', 'high_school_world_history', 'electrical_engineering', 
        #     'high_school_european_history', 'jurisprudence', 'high_school_physics', 
        #     'conceptual_physics', 'high_school_statistics', 'virology', 
        #     'high_school_biology', 'astronomy', 'miscellaneous']
        
        for model, style in zip(models, line_styles):
            cur_x = []
            cur_y_p = []
            cur_y_np =[]
            num_scores = []
            for task in mmlu_tasks:
                print(task)
                score_file = f'/share/edc/home/xinyi_wang/incidental-supervision/out/{task}/ngrams=3/num_docs=1/allenai/{model}/scores.json'
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
                    print(num)
                    if num > 5:
                        cur_y_p.append(p_score/num)
                        cur_y_np.append(np_score/num)
                        cur_x.append(task)
                        num_scores.append(num)
            
            xs = list(range(len(cur_y_p)))
            sorted_ids = numpy.argsort(cur_y_p)
            
            print(numpy.array(num_scores)[sorted_ids])
            
            plt.plot(xs, numpy.array(cur_y_p)[sorted_ids], style, label=f'{model}')
            plt.plot(xs, numpy.array(cur_y_np)[sorted_ids], style + '-')
            plt.xticks(xs, list(numpy.array(cur_x)[sorted_ids]), rotation=45, ha='right')
            
            plt.xlabel('MMLU tasks')
            plt.ylabel('Gradient similarity')
            plt.legend()
            plt.savefig(f'mmlu_grad_sim_{model}.pdf', bbox_inches='tight')
            plt.close()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plot grad sim')
    parser.add_argument('--task', type=str, default='translation',
                        help='Eval task', )
    parser.add_argument('--out_dir', type=str, default='out',
                        help='Output directory', )
    args = parser.parse_args()
    
    main(args)


"""

python plot_grad_sim_xinyi.py \
       --task trivia_qa \
       --out ./figures/translation/dist/final

"""