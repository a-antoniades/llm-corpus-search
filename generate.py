from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed
)
from datasets import load_from_disk

import json
import argparse
import re
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from datetime import datetime
import evaluate

from tqdm import tqdm
import warnings

SKIP_SPECIAL_TOKENS = False
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gpt2", type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--dataset_path", default="wikitext", type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to generate")
    parser.add_argument("--answer_choices", default=None, nargs='*', help="Answer choices for rank classification")
    parser.add_argument("--ds_key", default=None, type=str, help="Dataset key if dict")
    parser.add_argument("--text_column", default=None, type=str, help="Column to use as prompt")
    parser.add_argument("--target_options", default=None, type=str, help="Target options for rank classification")
    args = parser.parse_args()
    return args

def create_prompt(text, seperator_candidates=None):
    if seperator_candidates is None:
        separator_candidates = ["'text':", "summary", "'target':", 
                                "label:", "sentiment:",
                                "entailment, neutral, or contradiction?"]
    matches = None
    if isinstance(text, str):
        for m in separator_candidates:
            matches = list(re.finditer(m, text))
            if matches:
                break
        if matches:
            indices = [match.end() for match in matches]
        else:
            # just split string in half
            indices = [len(text) // 2]
        text_input = text[:indices[0]], text[indices[0]:]
        prompt = text_input[0]
        target = text_input[1]
    elif isinstance(text, list):
        assert len(text) == 2, "text must be a list of length 2"
        prompt = text[0]
        target = text[1]

    # add space after prompt
    # if prompt[-1] != " ":
    #     prompt += " "
    # # remove space before target
    # if target[0] == " ":
    #     target = target[1:]

    return prompt, target


# import openai
# openai.api_key = "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" # os.environ["OPENAI_KEY"]

# def extract_options(input_string):
#     # Define a regular expression pattern for the options phrase
#     pattern = r"(options are|OPTIONS|options|OPT|Available choices|Options|Select from:|Select from the following.\
#                 |Choose your answer from|Choose your answer from:|Select from:|Options:|OPTIONS:\
#                 |Select from the following|Available choices|pick from the following)\s*:\s*(.*)"
#     # Search for the pattern in the input string, ignoring case
#     matches = list(re.finditer(pattern, input_string, re.IGNORECASE))
#     if matches:
#         # Get the last match
#         match = matches[-1]
#         options_string = match.group(2)

#         # Use GPT-3.5 to filter the options
#         prompt = f"filter this prompt to only return the answer options: {options_string}"
#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo",
#             prompt=prompt,
#             max_tokens=100
#         )

#         # Get the generated text
#         options = response.choices[0].text.strip()

#         print(f"response: {response}")

#         return options
#     else:
#         return []


def extract_options(input_string):
    # Define a regular expression pattern for the options phrase
    pattern = r"(options are|OPTIONS|options|OPT|Available choices|Options|Select from:|Select from the following.|Select from|OPTIONS:|a\).\|b\.)\s*:\s*(.*)"
    # Search for the pattern in the input string, ignoring case
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        # If a match is found, split the matched string on '--', '[-]', '\n', '; ', or '+' to get individual options
        options = re.split('--|\[-\]|\n|; |\+|\(a\)\.\s|\(b\)\.\s', match.group(2))
        # Remove leading and trailing whitespace from each option
        options = [option.strip() for option in options if option.strip()]
        # Remove all words before the first delimiter option
        options = [re.split('[-+;]', option, 1)[-1].strip() for option in options]
        # Remove any text after the period following the last delimiter
        options = [re.split('\.', option, 1)[0].strip() for option in options]
        # Remove trailing semicolon from the last option
        if options and options[-1].endswith(';'):
            options[-1] = options[-1][:-1]
        # Remove any options that do not start with a letter or a number
        options = [option for option in options if option and option[0].isalnum()]
        return options
    else:
        return []

def extract_options_with_targets(input_string, target_options):
    # Define a regular expression pattern for the options phrase
    pattern = r"(options are|OPTIONS|options|OPT|Available choices|Options|Select from:|Select from the following.|Select from|OPTIONS:|a\).\|b\.)\s*:\s*(.*)"
    # Find all matches of the pattern in the input string, ignoring case
    matches = re.findall(pattern, input_string, re.IGNORECASE)
    options = []
    if matches:
        # If matches are found, check for the options that exist in the target_options dictionary
        for match in matches:
            options.extend([option for option in target_options if option in match])
    return options
    
def save_generated_samples(generated_samples, checkpoint_dir, overwrite=True):
    file_name = f"generated_samples.json"
    file_path = os.path.join(checkpoint_dir, file_name)

    # Serialize the JSON object to a string
    json_string = json.dumps(generated_samples, indent=2)
    
    # Insert a newline after the closing brace of each dictionary entry
    formatted_json_string = json_string.replace("},", "},\n")
    
    # Write the formatted string to the file
    with open(file_path, "w") as f:
        f.write(formatted_json_string)
    
    tqdm.write(f"Saved generated samples to {file_path}")


def compute_metrics(true_dec, decoded_output, **kwargs):
    # Calculate scores for each metric
    scores = {}
    for metric_name, metric_func in kwargs.items():
        try:
            scores[metric_name] = metric_func.compute(predictions=decoded_output, references=true_dec)
        except:
            # tqdm.write(f"Error computing {metric_name}")
            # tqdm.write(f"Predictions: {decoded_output}, len: {len(decoded_output)}")
            # tqdm.write(f"References: {true_dec}, len: {len(true_dec)}")
            scores[metric_name] = 0.0
    return scores


def load_scores():
    bleu_score = evaluate.load("bleu", keep_in_memory=True)
    meteor_sc = evaluate.load("meteor", keep_in_memory=True)
    return bleu_score, meteor_sc


def compute_greedy_decoding_log_prob(model, tokenizer, prompt, continuation,
                                     normalize=True):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Encode the continuation
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)

    log_prob_sum = 0.0

    with torch.no_grad():
        for token_id in continuation_ids:
            # Get the model outputs (logits) for the current sequence
            outputs = model(input_ids).logits

            # Compute the softmax to obtain probabilities
            probs = F.softmax(outputs[:, -1, :], dim=-1)

            # Compute the log probability of the next token in the continuation
            log_prob = torch.log(probs[0, token_id]).item()

            # Add the log probability to the total sum
            log_prob_sum += log_prob

            # Append the current token to the input sequence for the next step
            input_ids = torch.cat([input_ids, torch.tensor([[token_id]], dtype=torch.long)], dim=-1)

    return log_prob_sum / len(continuation_ids) if normalize else log_prob_sum


def compute_conditional_log_prob(model, tokenizer, prompt, continuation,
                                 normalize=True, device='cpu'):
    # Combine prompt and continuation
    total_sequence = prompt + continuation

    # Encode the input sequence
    input_ids = tokenizer.encode(total_sequence, return_tensors="pt", add_special_tokens=False).to(device)

    # Get the model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids).logits

    # Compute the softmax to obtain probabilities
    probs = F.softmax(outputs, dim=-1)

    # Compute the log probabilities
    log_probs = torch.log(probs)

    # Sum the log probabilities of the tokens in the continuation
    total_ids = tokenizer.encode(total_sequence, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_length = len(prompt_ids)
    continuation_length = len(total_ids) - prompt_length
    total_length = prompt_length + continuation_length

    assert total_length == len(input_ids[0]), f"lens do not match, total_length: {total_length}, input_ids: {len(input_ids[0])}"
    # tqdm.write(f"prompt_length: {prompt_length}, continuation_length: {continuation_length}, total_length: {total_length}")

    log_prob_sum = 0.0
    for idx, token_id in enumerate(total_ids[prompt_length:]):  # start from the continuation part
        log_prob_sum += log_probs[0, prompt_length + idx, token_id].item()

    return log_prob_sum / len(total_ids[prompt_length:]) if normalize else log_prob_sum


def rank_classification(model, tokenizer, prompt, true_choice, answer_choices, device='cpu'):
    len_choice = 4
    scores = {}
    # true_choice = true_choice[:len_choice]
    has_space = true_choice[0] == " "
    for choice in answer_choices:
        # add space before choice2 if true_choice has space
        if has_space and choice[0] != " ":
            choice = " " + choice
        if choice == true_choice:
            continue
        # log_prob_sum = compute_greedy_decoding_log_prob(model, tokenizer, prompt, choice)
        log_prob_sum = compute_conditional_log_prob(model, tokenizer, prompt, choice, device=device)
        scores[choice] = log_prob_sum
    true_score = compute_conditional_log_prob(model, tokenizer, prompt, true_choice, device=device)
    scores[f"true_choice"] = true_score

    # sort scores by value
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    # get rank of true choice
    true_rank = list(scores.keys()).index(f"true_choice") + 1
    return scores, true_rank, true_score

# In your main function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def load_model(model_name_or_path, device=device):
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, config=config)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        model.to(device)
        model.eval()
        return config, tokenizer, model
    
    bleu_score, meteor_sc = load_scores()

    _, tokenizer, model = load_model(args.model_name_or_path, device=device)

    if args.target_options is not None:
        with open(args.target_options, "r") as f:
            target_options = json.load(f)
    
    dataset = load_from_disk(args.dataset_path)
    if args.ds_key is not None:
        dataset = dataset[args.ds_key]
        target_options = target_options[args.ds_key]
    text_column = args.text_column if args.text_column is not None else \
                'text' if 'text' in dataset.column_names else \
                'prompt' if 'prompt' in dataset.column_names else \
                'inputs' if 'inputs' in dataset.column_names else None
    if text_column is None:
        raise ValueError(f"No valid column name found in dataset. {dataset.column_names}")


    generated_samples = []
    all_logits = []
    model_name = '.'.join(args.model_name_or_path.split("/")[-4:])
    dataset_name = '.'.join(args.dataset_path.split("/")[-3:])
    val_ds = args.dataset_path.split("/")[-1]
    if "models" not in args.model_name_or_path:
        args.model_name_or_path = os.path.join("models", args.model_name_or_path)
    save_dir = os.path.join(args.model_name_or_path, "inference", 
                            val_ds, f"ds_key:{args.ds_key}", f"n_samples_{args.num_samples}")
    # if os.path.isdir(save_dir):
    #     logger.info(f"Save dir {save_dir} already exists. Exiting...")
    #     return  # exit program
    
    # # if the first row has "template_type", filter the required template types
    # if 'template_type' in dataset[0].keys():
    #     dataset = dataset.filter(lambda row: row['template_type'] in ['zs_opt', 'fs_opt'])
    
    total_samples = args.num_samples
    sample_indices = random.sample(range(len(dataset)), len(dataset))
    generated_samples_count = 0
    ds_name = dataset_name + args.ds_key if args.ds_key is not None else dataset_name
    pbar = tqdm(total=total_samples,
                desc="Model: {}, Dataset: {}".format(model_name, ds_name))
        
    for idx, i in enumerate(sample_indices):
        if generated_samples_count >= total_samples:
            break

        row = dataset[-i]
        example = row[text_column]

        # create prompts
        if 'answer' in row.keys():
            prompt, continuation = row[text_column], row['answer']
        elif 'targets' in row.keys():
            prompt, continuation = row[text_column], row['targets']
        else:
            prompt, continuation = create_prompt(example)
        tqdm.write(f"prompt: {prompt}")
        tqdm.write(f"continuation: {continuation}")


        # get answer choices for rank classification
        if args.target_options is not None:
            answer_choices = extract_options_with_targets(example, target_options)
        if 'answer_choices' in row.keys():
            answer_choices = row['answer_choices'].split(' ||| ')
        elif args.answer_choices != None:
            answer_choices = args.answer_choices
        # elif row.get('template_type') in ['zs_opt', 'fs_opt']:
        #     answer_choices = extract_options(example)
        else:
            continue
        
        # have at least 2 answer choices
        if len(answer_choices) < 2:
            continue
                 
        # # compute conditional log prob
        # log_prob = compute_conditional_log_prob(model, tokenizer, prompt, continuation, device=device)

        # rank classification
        if answer_choices is not None:
            try:
                scores, true_rank, log_prob = rank_classification(model, tokenizer, prompt, 
                                                        continuation, answer_choices, device=device)
            except:
                continue
        else:
            print(f"answer_choices is None for example {i}")
            continue

        ## debugging
        prompt_enc = tokenizer.encode(prompt, return_tensors="pt")
        prompt_dec = tokenizer.decode(prompt_enc[0], skip_special_tokens=SKIP_SPECIAL_TOKENS)
        true_enc = tokenizer.encode(continuation, return_tensors="pt")
        true_dec = tokenizer.decode(true_enc[0], skip_special_tokens=SKIP_SPECIAL_TOKENS)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        len_continuation = len(true_dec)
        try:
            output = model.generate(input_ids, max_new_tokens=len(true_dec), do_sample=False, top_k=1, top_p=0.95, 
                                        num_return_sequences=1)
        except:
            tqdm.write(f"Error generating for example {i}")
            tqdm.write(f"Prompt: {prompt}")
            tqdm.write(f"Continuation: {continuation}")
            print(F"ERRORRRRRRRRRRRR")
            continue
            RuntimeError("Error generating for example {}".format(i))
        decoded = tokenizer.decode(output[0], skip_special_tokens=SKIP_SPECIAL_TOKENS)
        prompt_ = decoded[:len(prompt)]
        decoded_output = decoded[len(prompt_dec):len(prompt_dec)+len(true_dec)]
        decoded_output_long = decoded[len(prompt_dec):]

        tqdm.write(f"output: {decoded_output}")
        generated_samples_count += 1
        pbar.update(1)

        # compute metrics
        metrics = compute_metrics(true_dec, decoded_output,
                                  bleu_score=bleu_score, meteor_score=meteor_sc)

        # tqdm.write("--- Example {} (iter {}) ---".format(i, idx))
        # tqdm.write(f"Prompt: {prompt_}")
        # tqdm.write(f"True: {continuation}")
        # tqdm.write(f"Generated: {decoded_output_long}")
        # tqdm.write(f"Log prob: {log_prob}")
        # tqdm.write(f"BLEU score: {metrics['bleu_score']}")
        # tqdm.write(f"METEOR score: {metrics['meteor_score']}")
        # tqdm.write(f"Scores: {scores}")
        # tqdm.write(f"Rank: {true_rank}")
        # if answer_choices is not None:
            # tqdm.write(f"True rank: {true_rank}")
            # tqdm.write("Scores:")
            # for k, v in scores.items():
                # tqdm.write(f"{k}: {v}")
        generated_samples_n = {
            "prompt": prompt_dec,
            "true": true_dec,
            "generated": decoded_output_long,
            "log_prob": log_prob,
            "bleu_score": metrics['bleu_score'],
            "meteor_score": metrics['meteor_score'],
            "scores": scores if answer_choices is not None else None,
            "rank": true_rank,
            "answer_choices": answer_choices if answer_choices is not None else None
        }
        generated_samples.append(generated_samples_n)

    # get current time in month-day-hour-minute
    # now = datetime.now()
    # dt_string = now.strftime("%m-%d-%H:%M")

    # check if dir exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # save as json in checkpoint dir
    save_generated_samples(generated_samples, save_dir, overwrite=False)
    
    # calculate average log prob
    log_probs_mean = np.mean([sample["log_prob"] for sample in generated_samples])
    log_probs_std = np.std([sample["log_prob"] for sample in generated_samples])
    log_probs_samples = [sample["log_prob"] for sample in generated_samples]
    log_probs_n = len(log_probs_samples)
    log_probs = {
        "mean": log_probs_mean,
        "std": log_probs_std,
        "samples": log_probs_samples,
        "n": log_probs_n
    }
    with open(os.path.join(save_dir, "log_probs.json"), "w") as f:
        json.dump(log_probs, f)
    tqdm.write(f" -- Average log prob: {log_probs['mean']} -- ")

    # calculate average rank
    rank_samples = [sample["rank"] for sample in generated_samples if sample["rank"] is not None]
    rank_mean = np.mean(rank_samples)
    rank_std = np.std(rank_samples)
    rank_n = len(rank_samples)
    rank = {
        "mean": rank_mean,
        "std": rank_std,
        'samples': rank_samples,
        "n": rank_n
    }
    with open(os.path.join(save_dir, "rank.json"), "w") as f:
        json.dump(rank, f)
    tqdm.write(f" -- Average rank: {rank['mean']} -- ")

    # calculate average accuracy
    accuracy_samples = [1 if sample["rank"] == 1 else 0 for sample in generated_samples if sample["rank"] is not None]
    accuracy_mean = np.mean(accuracy_samples)
    accuracy_std = np.std(accuracy_samples)
    accuracy_n = len(accuracy_samples)
    accuracy = {
        "mean": accuracy_mean,
        "std": accuracy_std,
        "samples": accuracy_samples,
        "n": accuracy_n
    }
    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump(accuracy, f)
    tqdm.write(f" -- Average accuracy: {accuracy['mean']} -- ")

if __name__ == "__main__":
    # Ignore the specific warning
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")

    from transformers.utils import logging
    logging.set_verbosity_error()
    tf_logger = logging.get_logger("transformers")
    tf_logger.setLevel(logging.ERROR)

    # Get the logger for nltk
    nltk_logger = logging.get_logger('nltk')
    nltk_logger.setLevel(logging.WARNING)

    set_seed(42)
    bleu_score = evaluate.load("bleu", keep_in_memory=True)
    meteor_sc = evaluate.load("meteor", keep_in_memory=True)
    args = parse_args()
    main(args)