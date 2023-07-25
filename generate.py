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


set_seed(42)
bleu_score = evaluate.load("bleu", keep_in_memory=True)
meteor_sc = evaluate.load("meteor", keep_in_memory=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", default="gpt2", type=str, help="Path to pre-trained model or shortcut name"
    )
    parser.add_argument(
        "--dataset_path", default="wikitext", type=str, help="Path to pre-trained model or shortcut name"
    )
    parser.add_argument(
        "--num_samples", default=10, type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "--answer_choices", default=None, nargs='*', help="Answer choices for rank classification"
    )
    args = parser.parse_args()
    return args

def create_prompt(text):
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
        text_input = text[:indices[0] + 1], text[indices[0] + 1:]
        prompt = text_input[0]
        target = text_input[1]
    elif isinstance(text, list):
        assert len(text) == 2, "text must be a list of length 2"
        prompt = text[0]
        target = text[1]

    # # add space after prompt
    # if prompt[-1] != " ":
    #     prompt += " "
    # # remove space before target
    # if target[0] == " ":
    #     target = target[1:]

    return prompt, target

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
    
    print(f"Saved generated samples to {file_path}")


def compute_metrics(true_dec, decoded_output, **kwargs):
    # Calculate scores for each metric
    scores = {}
    for metric_name, metric_func in kwargs.items():
        try:
            scores[metric_name] = metric_func.compute(predictions=decoded_output, references=true_dec)
        except:
            print(f"Error computing {metric_name}")
            print(f"Predictions: {decoded_output}, len: {len(decoded_output)}")
            print(f"References: {true_dec}, len: {len(true_dec)}")
            raise Exception
    return scores

def load_scores():
    bleu_score = evaluate.load("bleu", keep_in_memory=True)
    meteor_sc = evaluate.load("meteor", keep_in_memory=True)
    return bleu_score, meteor_sc

def compute_conditional_log_prob(model, tokenizer, prompt, continuation):
    # Combine prompt and continuation
    total_sequence = prompt + continuation

    # Encode the input sequence
    input_ids = tokenizer.encode(total_sequence, return_tensors="pt")

    # Get the model outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids).logits

    # Compute the softmax to obtain probabilities
    probs = F.softmax(outputs, dim=-1)

    # Compute the log probabilities
    log_probs = torch.log(probs)

    # Sum the log probabilities of the tokens in the continuation
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
    log_prob_sum = 0.0
    for idx, token_id in enumerate(continuation_ids):
        log_prob_sum += log_probs[0, prompt_length + idx, token_id].item()

    return log_prob_sum

def compute_greedy_decoding_log_prob(model, tokenizer, prompt, continuation):
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

    return log_prob_sum


# TODO: Truncate answer choices to same length
def rank_classification(model, tokenizer, prompt, true_choice, answer_choices):
    len_choice = 4
    scores = {}
    true_choice = true_choice[:len_choice]
    has_space = true_choice[0] == " "
    print(f"answer choices: {answer_choices}")
    for choice in answer_choices:
        # add space before choice2
        if has_space and choice[0] != " ":
            choice = " " + choice
        # truncate choice to same length
        choice = choice[:4]
        if choice == true_choice:
            continue
        log_prob_sum = compute_greedy_decoding_log_prob(model, tokenizer, prompt, choice)
        scores[choice] = log_prob_sum
    true_score = compute_greedy_decoding_log_prob(model, tokenizer, prompt, true_choice)
    scores[f"true_choice"] = true_score

    # sort scores by value
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    print(f"scorezz: {scores}")

    # get rank of true choice
    true_rank = list(scores.keys()).index(f"true_choice") + 1
    return scores, true_rank

# In your main function
def main():
    args = parse_args()
    def load_model(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, config=config)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        return config, tokenizer, model
    
    bleu_score, meteor_sc = load_scores()

    _, tokenizer, model = load_model(args.model_name_or_path)
    dataset = load_from_disk(args.dataset_path)
    generated_samples = []
    all_logits = []
    for idx, i in enumerate(random.sample(range(len(dataset)), args.num_samples)):
        text_column = 'text' if 'text' in dataset.column_names else 'prompt'
        row = dataset[-i]
        example = row[text_column]

        # get answer choices for rank classification
        if args.answer_choices is None:
            answer_choices = row['answer_choices'] if 'answer_choices' in row.keys() else None
        else:
            answer_choices = args.answer_choices
        
        # create prompts
        prompt, continuation = create_prompt(example)

        print(f"Prompt: {prompt}")
        print(f"Continuation: {continuation}")

        # compute conditional log prob
        try:
            log_prob = compute_conditional_log_prob(model, tokenizer, prompt, continuation)
        except:
            print(f"Error computing log prob for example {i}")
            print(f"Prompt: {prompt}")
            print(f"Continuation: {continuation}")
            continue

        # rank classification
        if answer_choices is not None:
            scores, true_rank = rank_classification(model, tokenizer, prompt, 
                                                    continuation, answer_choices)
        else:
            scores, true_rank = None, None

        ## debugging
        prompt_enc = tokenizer.encode(prompt, return_tensors="pt")
        prompt_dec = tokenizer.decode(prompt_enc[0], skip_special_tokens=True)
        true_enc = tokenizer.encode(continuation, return_tensors="pt")
        true_dec = tokenizer.decode(true_enc[0], skip_special_tokens=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        len_continuation = len(true_dec)
        try:
            output = model.generate(input_ids, max_new_tokens=2*len(true_dec), pad_token_id=tokenizer.pad_token_id,
                                    do_sample=False, top_k=1, top_p=0.95, num_return_sequences=1)
        except:
            print(f"Error generating for example {i}")
            print(f"Prompt: {prompt}")
            print(f"Continuation: {continuation}")
            continue
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        prompt_ = decoded[:len(prompt)]
        decoded_output = decoded[len(prompt_dec):len(prompt_dec)+len(true_dec)]
        decoded_output_long = decoded[len(prompt_dec):]

        # compute metrics
        metrics = compute_metrics(true_dec, decoded_output,
                                  bleu_score=bleu_score, meteor_score=meteor_sc)

        # print and store results
        print("--- Example {} (iter {}) ---".format(i, idx))
        print(f"Prompt: {prompt_}")
        print(f"True: {continuation}")
        print(f"Generated: {decoded_output_long}")
        print(f"Log prob: {log_prob}")
        print(f"BLEU score: {metrics['bleu_score']}")
        print(f"METEOR score: {metrics['meteor_score']}")
        print(f"Scores: {scores}")
        print(f"Rank: {true_rank}")
        if answer_choices is not None:
            print(f"True rank: {true_rank}")
            print("Scores:")
            for k, v in scores.items():
                print(f"{k}: {v}")
        generated_samples_n = {
            "prompt": prompt_dec,
            "true": true_dec,
            "generated": decoded_output_long,
            "log_prob": log_prob,
            "bleu_score": metrics['bleu_score'],
            "meteor_score": metrics['meteor_score'],
            "scores": scores if answer_choices is not None else None,
            "rank": true_rank
        }
        generated_samples.append(generated_samples_n)

    # get current time in month-day-hour-minute
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H:%M")
    save_dir = os.path.join(args.model_name_or_path, "inference", dt_string)
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
    print(f" -- Average log prob: {log_probs_mean} -- ")

    # # calculate average bleu score
    # bleu_score_mean = np.mean([sample["bleu_score"] for sample in generated_samples])
    # with open(os.path.join(args.model_name_or_path, "inference", "bleu_scores.txt"), "w") as f:
    #     f.write(f"{bleu_score_mean}")
    # print(f" -- Average BLEU score: {bleu_score_mean} -- ")

    # calculate average rank
    rank_mean = np.mean([sample["rank"] for sample in generated_samples if sample["rank"] is not None])
    rank_std = np.std([sample["rank"] for sample in generated_samples if sample["rank"] is not None])
    rank_samples = [sample["rank"] for sample in generated_samples if sample["rank"] is not None]
    rank_n = len(rank_samples)
    rank = {
        "mean": rank_mean,
        "std": rank_std,
        'samples': rank_samples,
        "n": rank_n
    }
    with open(os.path.join(save_dir, "rank.json"), "w") as f:
        json.dump(rank, f)
    print(f" -- Average rank: {rank} -- ")

if __name__ == "__main__":
    main()