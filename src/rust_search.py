import subprocess
from functools import lru_cache

DEFAULT_DATASET = 'tuluv2'
name_to_path = {
    # 'tuluv2': '/share/edc/home/antonis/datasets/huggingface/tuluv2_ds.json.gz',
    'tuluv2': "/share/edc/home/antonis/datasets/huggingface/tuluv2_128/"
}

@lru_cache(maxsize=None)
def count_ngram(dataset, search_string, es=None):
    # Build the command to run the Rust CLI
    command = [
        "wimbd",
        "count",
        dataset,
        "--search",
        f'"{search_string}"'
    ]

    try:
        # Run the Rust CLI command and capture the output
        output = subprocess.check_output(command, universal_newlines=True)

        # Split the output into lines
        lines = output.strip().split("\n")

        # Parse the count from the output string
        count_str = lines[0].split("(count = ")[1].split(")")[0]
        count = int(count_str)

        print(f"Search string: {search_string}")
        print(f"Count: {count}")

        return count

    except subprocess.CalledProcessError as e:
        print(f"Error running Rust CLI: {e}")
        return None

    except IndexError:
        print(f"Error parsing output: {lines}")
        return None

def count_documents_containing_phrases(dataset, phrases, es=None, all_phrases=True):
    dataset = name_to_path[dataset]
    
    counts = []
    for phrase in phrases:
        count = count_ngram(dataset, phrase, es)
        if count is None:
            return None
        counts.append(count)

    if all_phrases:
        # Return the minimum count if all phrases are required
        combined_count = min(counts)
    else:
        # Return the maximum count if any phrase is sufficient
        combined_count = max(counts)

    print(f"Phrases: {phrases}")
    print(f"All phrases required: {all_phrases}")
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