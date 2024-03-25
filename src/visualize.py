
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from scipy import stats
from scipy.stats import linregress
from itertools import combinations
import pandas as pd



def filter_path_1(path, keywords):
    parts = path.split(os.sep)
    keywords = set(keywords)  # Convert keywords to set for faster lookup
    filtered_parts = []
    for part in parts:
        if any(keyword in part for keyword in keywords):
            filtered_parts.append(part)
            # Once a keyword is found in part, remove it from keywords set
            keywords -= set([keyword for keyword in keywords if keyword in part])
        if not keywords:
            # All keywords found, no need to continue loop
            break
    return os.sep.join(filtered_parts)

def filter_path_2(path, keywords):
    parts = path.split(os.sep)
    filtered_keywords = []
    for keyword in keywords:
        if any(keyword in part for part in parts):
            filtered_keywords.append(keyword)
    return ' '.join(filtered_keywords)

# Define a function to find and extract the score from a score.txt file
def get_scores(folders, filename):
    scores = {}
    for folder_path in folders:
        # Search for score.txt files in the folder and its subdirectories
        # score_files = glob.glob(f'{folder_path}/**/{filename}', recursive=True)
        # seacgh for score.txt files in the folder only
        score_files = glob.glob(f'{folder_path}/{filename}', recursive=True)

        # If no score file is found, raise an exception
        if not score_files:
            raise FileNotFoundError(f'No score.txt file found in {folder_path} or its subdirectories.')

        # If more than one score file is found, raise an exception
        if len(score_files) > 1:
            raise ValueError(f'More than one score.txt file found in {folder_path} or its subdirectories.')

        # Read the score from the score file
        with open(score_files[0], 'r') as file:
            score = float(file.read().strip())  # Change to int if the score is an integer
        score_key = filter_path(folder_path, keywords=["dataset", "checkpoint"])
        scores[score_key] = score
    return scores

# Define a function to find and extract the score from a score.txt file
def get_scores_json(folders, filename, key=None):
    scores = {}
    for folder_path in folders:
        # Search for score.txt files in the folder and its subdirectories
        # score_files = glob.glob(f'{folder_path}/**/{filename}', recursive=True)
        # seacgh for score.txt files in the folder only
        score_files = glob.glob(f'{folder_path}/{filename}', recursive=True)

        # If no score file is found, raise an exception
        if not score_files:
            raise FileNotFoundError(f'No score file found in {folder_path} or its subdirectories.')

        # If more than one score file is found, raise an exception
        if len(score_files) > 1:
            raise ValueError(f'More than one score file found in {folder_path} or its subdirectories.')

        # Read the score from the score file
        with open(score_files[0], 'r') as file:
            score = json.load(file)
        if key is not None:
            score = score[key]
        score_key = filter_path(folder_path, keywords)
        scores[score_key] = score
    return scores

def plot_scores(scores_dict, title, metric, keywords, label_color_dict=None,
                std_metric=None, reverse=False, ax=None):
    # Create a new figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Get unique labels
    labels = scores_dict.keys()
    unique_labels = list(set(labels))

    # Create a color map
    color_map = plt.get_cmap('tab10')  # Choose the color map you like
    colors = color_map(np.linspace(0, 1, len(unique_labels)))

    # Create a dictionary mapping labels to colors
    if label_color_dict is None:
        label_color_dict = {label: color for label, color in zip(unique_labels, colors)}

    # Sort labels and scores in decreasing order by scores
    items = []
    for label in labels:
        score = scores_dict[label]
        std = 0
        if isinstance(score, dict):
            score = score.get(metric, 0)  # use the 'metric' value if it exists, otherwise use 0
            std = score.get(std_metric, 0) if std_metric and isinstance(score, dict) else 0
        items.append((label, score, std))
    items = sorted(items, key=lambda x: x[1], reverse=reverse)
    labels, scores, stds = zip(*items)

    # Plot the bars with colors according to their labels
    for i, (score, std) in enumerate(zip(scores, stds)):
        bar_container = ax.bar(i, score, yerr=std, color=label_color_dict[labels[i]])
        bar = bar_container.patches[0]  # Get the first (and only) bar from the container
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '{:.3f}'.format(score), 
                ha='center', va='bottom')  # Add the exact value on top of the bar

    ax.set_ylabel('Average Score')
    ax.set_title(title)

    # Create custom legend
    patches = [mpatches.Patch(color=color, label=label) for label, color in label_color_dict.items()]
    ax.legend(handles=patches)

    ax.set_xticks([])  # Remove x-axis labels
    # ax.set_ylim(min(scores) - 0.2, max(scores) + 0.2)  # Add some margin at the top and bottom
    ax.set_ylim([0, 1])
    plt.tight_layout()  # Adjust layout so labels aren't cut off

    # If ax was not provided, show the plot
    if ax is None:
        plt.show()

def calculate_average_ranking(scores_dict):
    # Initialize a dictionary to store the rankings of each model
    model_rankings = {}

    # Iterate over the tasks
    for task, task_results in scores_dict.items():
        # Sort the models by their metric in descending order and get their rankings
        sorted_models = sorted(task_results.items(), key=lambda x: x[1]['metric'], reverse=True)
        rankings = {model: rank for rank, (model, _) in enumerate(sorted_models)}

        # Add the rankings to the model_rankings dictionary
        for model, rank in rankings.items():
            if model not in model_rankings:
                model_rankings[model] = []
            model_rankings[model].append(rank + 1)

    # Calculate the average ranking of each model
    average_rankings = {model: sum(ranks) / len(ranks) for model, ranks in model_rankings.items()}

    # Sort the models by their average ranking in ascending order
    sorted_average_rankings = dict(sorted(average_rankings.items(), key=lambda item: item[1]))

    return sorted_average_rankings

def calculate_average_score(scores_dict):
    # Initialize a dictionary to store the scores of each model
    model_scores = {}

    # Iterate over the tasks
    for task, task_results in scores_dict.items():
        # Add the scores to the model_scores dictionary
        for model, result in task_results.items():
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(result['metric'])

    # Calculate the average score of each model
    average_scores = {model: sum(scores) / len(scores) for model, scores in model_scores.items()}

    # Sort the models by their average score in descending order
    sorted_average_scores = dict(sorted(average_scores.items(), key=lambda item: item[1], reverse=True))

    return sorted_average_scores


def calculate_significance(results_dict):
    # Get all combinations of models
    model_pairs = combinations(results_dict.keys(), 2)

    significance_results = {}

    # Perform a t-test for each pair
    for model1, model2 in model_pairs:
        t_stat, p_value = stats.ttest_ind(results_dict[model1], results_dict[model2])
        significance_results[(model1, model2)] = p_value

    return pd.DataFrame(significance_results, index=['p-value']).T

def plot_accuracy_line(results_average, param_sizes, order=None, colors=None, 
                       metrics=None, title=None):
    if colors is None:
        colors = {'160M': 'r', '1.4B': 'b'}
    # Get the keys (datasets) and values (average accuracies) from results_average
    datasets = list(results_average.keys())
    n_datasets = len(datasets)
    
    # Check if the last level is a dictionary of metrics
    if isinstance(next(iter(next(iter(results_average.values())).values())), dict):
        avg_accs = [[results_average[dataset][param_size][metrics['acc']] for param_size in param_sizes] for dataset in datasets]
    else:
        avg_accs = [[results_average[dataset][param_size] for param_size in param_sizes] for dataset in datasets]

    # If order is provided, use it to order the datasets and accuracies
    if order:
        sorted_datasets = order
        sorted_avg_accs = [avg_accs[datasets.index(dataset)] for dataset in order]
    # Otherwise, sort the datasets and accuracies by the average accuracy of the first parameter size
    else:
        sorted_datasets, sorted_avg_accs = zip(*sorted(zip(datasets, avg_accs), key=lambda x: x[1][0]))

    # Create a line plot
    plt.figure(figsize=(5, 5))

    # Define the x locations for the datasets
    x = np.arange(len(sorted_datasets))

    # Create the lines for each parameter size
    for i, param_size in enumerate(param_sizes):
        plt.plot(x, [sorted_avg_accs[j][i] for j in range(len(sorted_avg_accs))], 
                 color=colors[param_size], marker='o')

    plt.xticks(x, sorted_datasets, rotation=22.5)
    plt.ylabel('Average Accuracy')
    if title:
        plt.title(title)
    else:
        plt.title(f'Average Accuracy by Training Dataset (n=78)')
    plt.legend(param_sizes)

    plt.tight_layout()
    plt.show()


# def plot_model_performance(average_rankings, average_score, model_to_color, task_group):
#     param_order = ['160M', '1.4B']
#     # Create a figure with two subplots
#     fig, axs = plt.subplots(1, 2, figsize=(10, 8))

#     # Add the overarching title
#     fig.suptitle(task_group, fontsize=16)

#     # Plot average rankings
#     for i, param_size in enumerate(param_order):
#         for j, model in enumerate(sorted(average_rankings.keys())):
#             value = average_rankings[model][param_size]
#             # Adjust x-coordinate to create zig-zag pattern
#             x = i + (-1)**j * 0.15
#             # y = value + 0.1
#             axs[0].scatter(x, value, color=model_to_color[model + ' ' + param_size])
#             axs[0].annotate(model, (x, value), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8)

#     # Get a consistent order of the models
#     model_order = sorted(average_score.keys())

#     # Plot average scores
#     for i, param_size in enumerate(param_order):
#         for j, model in enumerate(model_order):
#             value = average_score[model][param_size]
#             # Adjust x-coordinate to create zig-zag pattern
#             x = i + (-1)**j * 0.15
#             axs[1].scatter(x, value, color=model_to_color[model + ' ' + param_size])
#             axs[1].annotate(model, (x, value), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8)

#     for ax, ylabel, title in zip(axs, ['Average Ranking', 'Average Score'], ['Average Ranking of Models', 'Average Score of Models']):
#         ax.set_xlabel('Models')
#         ax.set_ylabel(ylabel)
#         ax.set_title(title)
#         ax.grid(axis='y')
#         ax.set_xticks(np.arange(len(param_order)))  # Set x-ticks manually
#         ax.set_xticklabels(param_order)  # Set x-tick labels manually
#         ax.set_xlim(-0.5, len(param_order) - 0.5)  # Adjust x-axis limits

#     axs[0].invert_yaxis()

#     plt.tight_layout(pad=1.0)  # Increase padding
#     plt.subplots_adjust(top=0.88)  # Adjust the top padding after adding the title
#     plt.show()

def plot_model_performance(average_score, model_to_color, task_group, title=None):
    param_order = ['160M', '1.4B']
    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(5, 8))

    # Add the overarching title
    fig.suptitle(task_group, fontsize=16)

    # Get a consistent order of the models
    model_order = sorted(average_score.keys())

    # Plot average scores
    for i, param_size in enumerate(param_order):
        for j, model in enumerate(model_order):
            value = average_score[model][param_size]
            # Adjust x-coordinate to create zig-zag pattern
            x = i + (-1)**j * 0.15
            ax.scatter(x, value, color=model_to_color[model + ' ' + param_size])
            ax.annotate(model, (x, value), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Average Score')
    ax.set_title('Average Score of Models')
    ax.grid(axis='y')
    ax.set_xticks(np.arange(len(param_order)))  # Set x-ticks manually
    ax.set_xticklabels(param_order)  # Set x-tick labels manually
    ax.set_xlim(-0.5, len(param_order) - 0.5)  # Adjust x-axis limits

    plt.tight_layout(pad=1.0)  # Increase padding
    plt.subplots_adjust(top=0.88)  # Adjust the top padding after adding the title
    plt.show()

def plot_model_performance_horizontal(average_score, model_to_color, task_group):
    param_order = ['160M', '1.4B']
    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Add the overarching title
    fig.suptitle(task_group, fontsize=16)

    # Get a consistent order of the models
    model_order = sorted(average_score.keys())

    # Plot average scores
    for i, param_size in enumerate(param_order):
        # Get models and their scores for the current parameter size
        models_scores = [(model, average_score[model][param_size]) for model in model_order]
        # Sort models by their scores
        sorted_models_scores = sorted(models_scores, key=lambda x: x[1])
        # Get the sorted models and their scores
        sorted_models, sorted_scores = zip(*sorted_models_scores)
        for j, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
            # Adjust y-coordinate to create zig-zag pattern
            noise = random.uniform(-0.1, 0.1)  # Adjust the range as needed
            y = i + (-1)**j * 0.15 + noise
            ax.scatter(score, y, color=model_to_color[model + ' ' + param_size])
            ax.annotate(model, (score, y), xytext=(0, -5), textcoords='offset points', ha='center', va='top', fontsize=8)

    ax.set_ylabel('Models')
    ax.set_xlabel('Average Score')
    ax.set_title('Average Score of Models')
    ax.grid(axis='x')
    ax.set_yticks(np.arange(len(param_order)))  # Set y-ticks manually
    ax.set_yticklabels(param_order)  # Set y-tick labels manually
    ax.set_ylim(-0.5, len(param_order) - 0.5)  # Adjust y-axis limits

    plt.tight_layout(pad=1.0)  # Increase padding
    plt.subplots_adjust(top=0.88)  # Adjust the top padding after adding the title
    plt.show()



def plot_accuracy_vs_count_with_fit(df, models, models_ommit, 
                                    model_color_mapping, n_samples=None,
                                    x='count', y='accuracy', degree=1):
    plt.figure(figsize=(10, 8))

    for model in models:
        if model in models_ommit:
            continue

        # Filter the DataFrame for the current model
        df_model = df[df['model'] == model]

        # If n_samples is specified, sample n_samples from each x group
        if n_samples is not None:
            df_model = df_model.groupby(x, group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state=1))

        # Plot each individual point
        plt.scatter(df_model[x], df_model[y], color=model_color_mapping[model], s=7.5, label=f'{model} data')

        # Calculate the polynomial fit
        if len(df_model[x]) > 1:  # Need at least two data points to fit a line
            coeffs = np.polyfit(df_model[x], df_model[y], degree)
            poly_fn = np.poly1d(coeffs)
            # Generate x values for plotting the fit line
            x_fit = np.linspace(min(df_model[x]), max(df_model[x]), num=200)
            # Plot the polynomial fit line
            plt.plot(x_fit, poly_fn(x_fit), 
                     color=model_color_mapping[model], label=f'{model} fit',
                     lw=5)
            # Annotate the line of best fit with the model name
            plt.annotate(model, 
                         xy=(max(df_model[x]), poly_fn(max(df_model[x]))), 
                         xytext=(5, 0), 
                         textcoords='offset points', 
                         color=model_color_mapping[model], 
                         fontsize=9)

    plt.ylabel(y)
    plt.xlabel(x)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.title(f'Accuracy vs {x.capitalize()} with Polynomial Fit for Each Model')
    plt.show()

