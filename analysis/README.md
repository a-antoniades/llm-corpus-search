### Task-gram Analysis

To perform the analysis in this directory, you need to have the following files:

- `data/task_gram_count/<task>/<subtask>/<ngram>.pkl`: the count of task-grams for all subtasks of each task, with desired ngram size.
- `data/wimbd/<task>/<ngram>/model_perf.pkl`: the performance of all models in interest on a task, with desired ngram size.

To plot the performance of the models on tasks v.s. the count of task-grams, use the `plot_performance.py` script.

To plot the distributional memorization/generalization scores, first compute the LLM probabilities with the `compute_llm_probs.py` script, and then use the `plot_mem.py` script.

To plot the gradient-based influence scores, first compute the gradient-based influence scores with the `compute_grad_prod.py` script, and then use the `plot_influence.py` script.

To perform the ChatGPT-base prompt optimization, use the `prompt_opt.py` script.