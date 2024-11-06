# %%
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import ast

from src.wimbd_ import BasePaths as PATHS, Colors
from src.wimbd_ import DataConfigs as CONFIG
from src.wimbd_ import post_filter, filter_percentile, load_results
from src.wimbd_ import ProcessTriviaQA as ptqa
from src.utils import remove_nested_lists, softmax
from src.analysis import AnalyzeNgrams as an

from datetime import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Generate a timestamp
timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")

# %%
task_coverage_pth = "./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse/3/common/task-p-coverage.pkl"
task_coverage = pd.read_pickle(task_coverage_pth)

# %%
all_models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b', 'pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m']
large_models = ['pythia-12b', 'pythia-6.9b', 'pythia-2.8b', 'pythia-1.4b']
small_models = ['pythia-410m', 'pythia-160m', 'pythia-70m', 'pythia-31m', 'pythia-14m']

N_GRAMS = 5
ALIGN_THRESH = 90
# BASE_DIR = f"./results/n-grams/trivia_qa/pile/exp_3/test-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse/"
# BASE_DIR = f"./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopTrue_onlyalphaFalse"
# BASE_DIR = f"./results/n-grams/triviaqa/pile/exp_3/validation-set/n_samples_None_fkeyFalse_rkeyFalse_fstopFalse_onlyalphaFalse"
DATASET = "examples_dfs_0-shot_common_pythia_models.pkl" # "examples_dfs_4-shot_common_models.pkl"
TASKS = "mmlu"
CORPUS = "pile"
BASE_DIR = PATHS.base_ngram_paths[TASKS][CORPUS]['base_path']
OMMIT_TASKS = False
TASKS_OMMIT = ["formal_logic"]
POST_FILTER = True

# BASE_DIR = PATHS.base_ngram_paths[TASKS]['base_path']
BASE_PATH = os.path.join(BASE_DIR, f"{N_GRAMS}")
METHOD = "0-shot_common"
BASE_PATH_COMMON = os.path.join(BASE_PATH, "common")
BASE_PATH_ALL = os.path.join(BASE_PATH, "all")
# FIG_DIR = os.path.join("/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/figures", DATASET, timestamp)
# if not os.path.exists(FIG_DIR):
#     os.makedirs(FIG_DIR)

# RESULTS_PATH = os.path.join(BASE_DIR, str(N_GRAMS), 'common', DATASET)
RESULTS_PATH = os.path.join(BASE_DIR, str(N_GRAMS), DATASET)
FIGS_PTH = f"./figures/{TASKS}"

# SUB_TASKS = ['marketing', 'management', 
#              'high_school_world_history',
#              'high_school_european_history',
#              'miscellaneous']

SUB_TASKS = ['public_relations', 'miscellaneous', 'nutrition', 'management',
            'conceptual_physics', 'professional_psychology',
            'high_school_us_history', 'high_school_psychology',
            'high_school_geography', 'high_school_world_history',
            'human_aging', 'high_school_european_history', 'virology',
            'anatomy', 'astronomy', 'computer_security', 'marketing',
            'logical_fallacies', 'international_law']

# SUB_TASKS = ['marketing',
#              'high_school_world_history',
#              'miscellaneous']

MODELS = 'open-instruct-pythia-6.9b-tulu'



print(f"BASE PATH: {BASE_PATH}")

VIEW_COLS = ['Q', 'A', 'question', 'value', 'align_score', 'em', 'answer']

# %%
# filter outliers
pickle_file = pd.read_pickle(RESULTS_PATH)
# df = pd.DataFrame(pickle_file).T.sort_values("value", ascending=False)

# filter
# open
# examples_models_pth = os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_post-processed.pkl")
examples_models_pth = RESULTS_PATH
if os.path.exists(examples_models_pth):
    with open(examples_models_pth, 'rb') as f:
        examples_models = pickle.load(f)
else:
    examples_models = ptqa.process_model_examples(pickle_file, p=0.99999)
examples_model = examples_models['pythia-12b']
colors = sns.color_palette('hls', len(examples_models))

# %%
examples_model.head()

# %%
def filter_correct_choices(df):
    def is_correct_choice(row):
        example = row['example']
        if isinstance(example, str):
            example = ast.literal_eval(example)
        choices = example['choices']
        answer_index = example['answer']
        correct_answer = choices[answer_index]
        return row['A'] == correct_answer
    
    filtered_df = df[df.progress_apply(is_correct_choice, axis=1)]
    return filtered_df

def add_accuracy_column(df):
    def is_correct_answer(row):
        example = row['example']
        if isinstance(example, str):
            example = ast.literal_eval(example)
        answer_index = example['answer']
        result = row['result']
        if isinstance(result, str):
            result = ast.literal_eval(result)
        predicted_index = np.argmax(result)
        return int(predicted_index == answer_index)

    df['accuracy'] = df.progress_apply(is_correct_answer, axis=1)
    return df

examples_models = {MODELS: examples_models[MODELS]}
# examples_models = {task: examples_models[examples_models['task'] == task] for task in examples_models['task'].unique() if task in SUB_TASKS}
examples_models = {k: filter_correct_choices(v) for k, v in examples_models.items()}
examples_models = {k: add_accuracy_column(v) for k, v in examples_models.items()}

# # save 
# examples_models_correct_pth = os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_post-processed.pkl")
# with open(examples_models_correct_pth, 'wb') as f:
#     pickle.dump(examples_models, f)


# %%
examples_models_correct_pth = os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_post-processed-aligned.pkl")
if os.path.exists(examples_models_correct_pth):
    with open(examples_models_correct_pth, 'rb') as f:
        examples_models = pickle.load(f)
else:
    from src.wimbd_ import align_e5_pairs_df

    examples_models = {k: align_e5_pairs_df(df, 'Q', 'question') for k, df in examples_models.items()}

    # save
    with open(examples_models_correct_pth, 'wb') as f:
        pickle.dump(examples_models, f)

# %%
# # save
# with open(os.path.join(BASE_PATH, f"examples_dfs_{METHOD}_post-processed.pkl"), 'wb') as f:
#     pickle.dump(examples_models, f)

# %%
examples_models['open-instruct-pythia-6.9b-tulu'].head()

# %%
examples_models_ = {model: data[data['alignment_score'] > ALIGN_THRESH] for model, data in examples_models.items()}

# %%
from src.analysis import AnalyzeNgrams as an

# coverage_path = os.path.join(BASE_PATH_COMMON, "task-coverage.pkl")
# task_cov = an.calculate_average_task_coverage(BASE_PATH_ALL, [TASKS], [N_GRAMS])

# %%
ALIGN_THRESH = 80
examples_models_ = {model: data[data['alignment_score'] > ALIGN_THRESH] for model, data in examples_models.items()}
# examples_models_['high_school_european_history'].head(20)

# %%
examples_models_tasks = {task: examples_models['open-instruct-pythia-6.9b-tulu'][examples_models['open-instruct-pythia-6.9b-tulu']['task'] == task] for task in examples_models['open-instruct-pythia-6.9b-tulu']['task'].unique()}

# %%
# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# color_palette = sns.color_palette('husl', len(examples_models))
color_palette = sns.color_palette('Set2', len(examples_models_tasks))
# color_palette = plt.get_cmap('viridis', len(examples_models_))
# color_palette = [color_palette(i) for i in range(len(examples_models_))]

is_log = False
plot_var = 'value'
savefile = os.path.join(FIGS_PTH, f"avg_em_by_log_interval_all_log{is_log}_{plot_var}_aligned{ALIGN_THRESH}.pdf")
for i, (model_name, data) in enumerate(examples_models_tasks.items()):
    print(f"name: {model_name}")
    data = an.create_bins(data, plot_var, -1, 100, 20, is_log)
    # data = ptqa.get_random_samples_by_interval(data, 85)
    ax = ptqa.plot_variable_by_interval(data, log_axis=is_log,
                                        x_column='interval', y_column='accuracy',
                                        save_pth=savefile, ax=ax, 
                                        color=color_palette[i], label=model_name,
                                        #  title=f'Performance vs. Ngrams for Different Models, {TASKS}, N={N_GRAMS}',
                                        plot_std=True, ylimits=[0,1.025])

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to create percentile bins, ensuring the column is numeric and handling NaNs
def create_percentile_bins(data, column, n_bins):
    data = data.copy()
    # Convert column to numeric, coerce errors, and drop NaNs
    data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=[column])
    data['percentile'] = pd.qcut(data[column], q=n_bins, labels=False, duplicates='drop') + 1
    return data

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

color_palette = sns.color_palette('Set2', len(examples_models_tasks))

is_log = False
plot_var = 'value'
FIGS_PTH = '.'  # Example path
ALIGN_THRESH = 1  # Example alignment threshold
savefile = os.path.join(FIGS_PTH, f"avg_em_by_log_interval_all_log{is_log}_{plot_var}_aligned{ALIGN_THRESH}.pdf")

for i, (model_name, data) in enumerate(examples_models_tasks.items()):
    print(f"name: {model_name}")

    values = data[plot_var].unique()
    if len(values) < 10:
        print(f"Skipping {model_name} due to only one value")
        continue
    else:
        print(f"Values: {values}")

    # Create percentile bins
    n_bins = 20
    data = create_percentile_bins(data, plot_var, n_bins)

    # Calculate average accuracy per percentile bin
    bin_means = data.groupby('percentile').mean().reset_index()

    # Plot the data using the percentile as the x-axis
    ax.plot(bin_means['percentile'], bin_means['accuracy'], color=color_palette[i], label=model_name)

plt.legend()
plt.xlabel('Percentile')
plt.ylabel('Accuracy')
plt.title('Performance vs. Percentile for Different Models')
plt.savefig(savefile)
plt.show()


# %%
question = "What is the medical term for high blood pressure?"
df_q = examples_model[examples_model['question'] == question]
df_q['Q']

# %%
# from src.utils import normalize_string

# # Apply the function to each row of the DataFrame
# # examples_model[['q_coverage', 'a_coverage']] = examples_model.apply(calculate_qa_coverage, axis=1)

# coverage_df = an.calculate_qa_coverage(examples_model)

# coverage_df['q_coverage'].unique()

# coverage_df[coverage_df['q_coverage'] < 0]

# %%
is_log = False
plot_var = 'q_coverage'
x_label = 'Question Coverage (p)'

examples_model_cov_int = an.create_bins(coverage_df, plot_var, 
                                        -0.1, 1.0, 5, 
                                        is_log=is_log)

ptqa.plot_variable_by_interval(examples_model_cov_int, log_axis=is_log,
                                x_column='interval', y_column='em',
                                save_pth=savefile, 
                                label='pythia-12b',
                                title=f'Performance vs. {plot_var} for Different Models N={N_GRAMS}',
                                x_label=x_label, 
                                plot_std=True)

# %%



