import os
import subprocess
from functools import lru_cache

DEFAULT_DATASET = 'tuluv2'
name_to_path = {
    'tuluv2': "/share/edc/home/antonis/datasets/huggingface/tuluv2_128/",
    'dolma': "/path/to/dolma/dataset"  # Add path to dolma dataset if needed
}

@lru_cache(maxsize=None)
def count_ngram(dataset_path, search_string, es=None):
    # Build the command to run the Rust CLI
    command = [
        "wimbd",
        "count",
        dataset_path,
        "--search",
        search_string
    ]

    try:
        # Add environment variables to ensure cargo binaries are found
        env = os.environ.copy()
        env["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{env['PATH']}"
        
        print(f"Running command: {' '.join(command)}")
        output = subprocess.check_output(command, universal_newlines=True, env=env)
        print(f"Raw output: {output}")

        lines = output.strip().split("\n")
        count_str = lines[-1].split("(count = ")[1].split(")")[0]
        count = int(count_str)

        return count

    except subprocess.CalledProcessError as e:
        print(f"Error running Rust CLI: {e}")
        return 0
    except Exception as e:
        print(f"Error processing output: {e}")
        print(f"Command: {' '.join(command)}")
        print(f"Output: {output if 'output' in locals() else 'No output'}")
        return 0

def count_documents_containing_phrases(dataset, phrases, es=None, all_phrases=True):
    if dataset not in name_to_path:
        raise ValueError(f"Dataset {dataset} not found in name_to_path mapping")
    
    dataset_path = name_to_path[dataset]
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist")
    
    counts = []
    for phrase in phrases:
        count = count_ngram(dataset_path, phrase, es)
        counts.append(count)

    if all_phrases:
        combined_count = min(counts)
    else:
        combined_count = max(counts)

    print(f"Phrases: {phrases}")
    print(f"Individual counts: {counts}")
    print(f"Combined count: {combined_count}")

    return combined_count

if __name__ == "__main__":
    # Example usage
    dataset = DEFAULT_DATASET

    # Single search string
    search_string = "natural language processing"
    count = count_ngram(dataset, search_string)
    print(f"Count for single search string: {count}")

    # List of search strings
    phrases = ["natural language processing", "deep learning"]
    combined_count = count_documents_containing_phrases(dataset, phrases, all_phrases=True)
    print(f"Combined count (all phrases required): {combined_count}")

    combined_count = count_documents_containing_phrases(dataset, phrases, all_phrases=False)
    print(f"Combined count (any phrase sufficient): {combined_count}")