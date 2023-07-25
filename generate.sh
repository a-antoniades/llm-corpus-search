# intiial exp with same iterations
# DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_1/dataset_validation.arrow"

# sentiment validation set
# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_1/dataset_validation.arrow"
# export ANSWER_CHOICES=("pos" "neg" "neu")

# NLI validation set
export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI/P_1_PQA_5_promptsource_False/dataset_1/dataset_validation.arrow"
export NUM_SAMPLES=100
export ANSWER_CHOICES=("entailment" "contradiction" "neutral")
python generate.py \
    --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI/P_1_PQA_5_psource_False_prompt_False/dataset_1/23-07-23_17:56/checkpoint-46000" \
    --dataset_path $DATASET_PATH \
    --num_samples $NUM_SAMPLES \
    --answer_choices "${ANSWER_CHOICES[@]}"

# python generate.py \
#     --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_1/15-07-23_16:49/checkpoint-50000" \
#     --dataset_path $DATASET_PATH \
#     --num_samples $NUM_SAMPLES \
#     --answer_choices "${ANSWER_CHOICES[@]}"
