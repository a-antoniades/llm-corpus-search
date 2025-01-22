# Configuration
DATASET="gsm8k" # or "sciq" or "mmlu"
CORPUS="dolma" # or "dolma"
N_GRAMS=3
NAME="rebuttal/wimbd/${CORPUS}/no_align/"
TYPE="wimbd" # "rust" or "infini" or "wimbd"
N_SAMPLES=1000

# Run the script
echo "Running script with ngram ${N_GRAMS} for dataset ${DATASET} and corpus ${CORPUS}"

python wimbd_search.py \
    --type ${TYPE} \
    --corpus ${CORPUS} \
    --n_grams ${N_GRAMS} \
    --dataset ${DATASET} \
    --filter_punc true \
    --filter_stopwords true \
    --filter_keywords false \
    --replace_keywords false \
    --only_alpha false \
    --align_pairs false \
    --method common \
    --name "${NAME}-${DATASET}-${MODEL}"