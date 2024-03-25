import os
import torch
import hf_olmo
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from accelerate import load_checkpoint_and_dispatch

no_split_module_classes = [
    "LlamaDecoderLayer",
    "BertLayer",
    "TransformerBlock",
    "BertEncoder",
    "BertModel",
    "T5Block",
    "GPT2Block",
]

WEIGHTS_DIR = "/share/edc/home/antonis/weights/huggingface"

def interact_with_model(model_name):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=WEIGHTS_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=WEIGHTS_DIR)

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(WEIGHTS_DIR, "models--" + model_name.replace("/", "--"), "snapshots")
    ckpt_folders = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]
    ckpt_path = ckpt_folders[-1]

    model = load_checkpoint_and_dispatch(
    model,
    ckpt_path, 
    device_map="auto", 
    no_split_module_classes=no_split_module_classes,
    offload_folder="./offload",
    )

    while True:
        # Get a prompt from the user
        prompt = input("Enter a prompt: ")

        # Encode the prompt and run it through the model
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(inputs.shape, device=device)
        outputs = model.generate(
                                 inputs, 
                                 do_sample=True,
                                 temperature=0.3,
                                 eos_token_id=tokenizer.pad_token_id,
                                 attention_mask=attention_mask,
                                 max_new_tokens=250,
                                 )

        # Decode the outputs and print the result
        result = tokenizer.decode(outputs[0])
        print(result)

def main():
    parser = argparse.ArgumentParser(description='Interact with a Hugging Face model.')
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    args = parser.parse_args()

    interact_with_model(args.model_name)

if __name__ == "__main__":
    main()