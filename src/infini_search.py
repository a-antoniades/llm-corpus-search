import requests
import time

API_ENDPOINT = "https://api.infini-gram.io/"
DEFAULT_CORPUS = "v4_dolma-v1_6_llama"

def count_ngram(corpus, query):
    payload = {
        "corpus": corpus,
        "query_type": "count",
        "query": query
    }

    response = requests.post(API_ENDPOINT, json=payload)
    result = response.json()

    if "error" in result:
        print(f"Error: {result['error']}")
        return None
    else:
        count = result["count"]
        return count

def count_documents_containing_phrases(index, phrases, es=None, all_phrases=True):
    # if es is not None:
    #     print("Warning: The 'es' parameter is not used in this implementation.")

    counts = []
    for phrase in phrases:
        count = count_ngram(index, phrase)
        if count is None:
            return 0
        counts.append(count)

    if all_phrases:
        # Return the minimum count if all phrases are required
        combined_count = min(counts)
    else:
        # Return the maximum count if any phrase is sufficient
        combined_count = max(counts)
    
    if combined_count is None:
        combined_count == 0

    print(f"Phrases: {phrases}")
    print(f"All phrases required: {all_phrases}")
    print(f"Count: {combined_count}")

    # Wait 0.1 seconds before the next request to avoid overloading
    time.sleep(0.1)

    return combined_count


if __name__ == "__main__":
    # Example usage
    index = DEFAULT_CORPUS
    n_gram = ["natural language processing", "deep learning"]

    counts = count_documents_containing_phrases(index, n_gram, all_phrases=True)
    print(f"Counts (all phrases required): {counts}")

    counts = count_documents_containing_phrases(index, n_gram, all_phrases=False)
    print(f"Counts (any phrase sufficient): {counts}")