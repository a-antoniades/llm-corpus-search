from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    set_seed
)
from datasets import load_from_disk

# from train_gpt import ModelArguments, DataTrainingArguments
import json
import argparse
import re
import os
import random


set_seed(42)


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
    args = parser.parse_args()
    return args

def create_prompt(text, seperator="text"):
    separator_candidates = ["'text':", "summary", "'target':"]
    matches = None
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
    # print(f"Splitting text in half: {text_input}")
    # print(f"original: {text}")
    # print(f"Seperator: {seperator}")
    # print(f"Text input: {text_input}")
    prompt = text_input[0]
    continuation = text_input[1]
    return prompt, continuation

# def save_generated_samples(generated_samples, checkpoint_dir, overwrite=True):
#     n = 0
#     file_name = f"inference/generated_samples_{n}.json"
#     file_path = os.path.join(checkpoint_dir, file_name)

#     # check if dir extists
#     if not os.path.isdir(os.path.dirname(file_path)):
#         os.makedirs(os.path.dirname(file_path))
#     # check if file exists
#     if not overwrite:
#         while os.path.isfile(file_path):
#             n += 1
#             file_name = f"generated_samples_{n}.json"
#             file_path = os.path.join(checkpoint_dir, file_name)
#     with open(file_path, "w") as f:
#         json.dump(generated_samples, f)
#     print(f"Saved generated samples to {file_path}")


def save_generated_samples(generated_samples, checkpoint_dir, overwrite=True):
    n = 0
    file_name = f"inference/generated_samples_{n}.json"
    file_path = os.path.join(checkpoint_dir, file_name)

    # check if dir exists
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # check if file exists
    if not overwrite:
        while os.path.isfile(file_path):
            n += 1
            file_name = f"inference/generated_samples_{n}.json"
            file_path = os.path.join(checkpoint_dir, file_name)
    
    # Serialize the JSON object to a string
    json_string = json.dumps(generated_samples, indent=2)
    
    # Insert a newline after the closing brace of each dictionary entry
    formatted_json_string = json_string.replace("},", "},\n")
    
    # Write the formatted string to the file
    with open(file_path, "w") as f:
        f.write(formatted_json_string)
    
    print(f"Saved generated samples to {file_path}")


def main():
    args = parse_args()
    def load_model(model_name_or_path):
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        return config, tokenizer, model

    _, tokenizer, model = load_model(args.model_name_or_path)
    dataset = load_from_disk(args.dataset_path)
    generated_samples = []
    for i in random.sample(range(len(dataset)), args.num_samples):
        example = dataset[-i]['text']
        prompt, continuation = create_prompt(example, seperator="text")

        ## debugging
        prompt_enc = tokenizer.encode(prompt, return_tensors="pt")
        prompt_dec = tokenizer.decode(prompt_enc[0], skip_special_tokens=True)
        true_enc = tokenizer.encode(continuation, return_tensors="pt")
        true_dec = tokenizer.decode(true_enc[0], skip_special_tokens=True)
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id,
                                do_sample=True, top_k=50, top_p=0.95, num_return_sequences=5)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]

        # print and store results
        print("-- Example {} --".format(i))
        print(f"Prompt: {prompt}")
        print(f"True: {continuation}")
        print(f"Generated: {decoded_output}")
        generated_samples_n = {
            # "prompt": prompt,
            "prompt": prompt_dec,
            # "true": continuation,
            "true": true_dec,
            "generated": decoded_output
        }
        generated_samples.append(generated_samples_n)

    # save as json in checkpoint dir
    save_generated_samples(generated_samples, args.model_name_or_path, overwrite=False)


if __name__ == "__main__":
    main()