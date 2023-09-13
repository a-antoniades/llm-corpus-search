
import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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

def plot_scores(scores_dict, title, metric, keywords, 
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