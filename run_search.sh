# Configuration
DATASET="triviaqa" # or "sciq" or "mmlu"
CORPUS="pile" # or "dolma"
N_GRAMS=3
NAME="exp_3/validation-set"
N_SAMPLES=20000

# Run the script
echo "Running script with ngram ${N_GRAMS}"

CUDA_VISIBLE_DEVICES='' \
python wimbd_search.py \
    --type infini \
    --corpus ${CORPUS} \
    --n_grams ${N_GRAMS} \
    --dataset ${DATASET} \
    --filter_punc true \
    --filter_stopwords true \
    --filter_keywords false \
    --replace_keywords false \
    --corpus ${CORPUS} \
    --only_alpha false \
    --method all \
    --name \"${NAME}\" \
    --method all