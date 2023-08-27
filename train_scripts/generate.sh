# intiial exp with same iterations
# DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_1/dataset_validation.arrow"

# sentiment validation set
# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_unified_labels/P_1_PQA_5_promptsource_False/dataset_1/dataset_validation.arrow"
# export ANSWER_CHOICES=("pos" "neg" "neu")


## NLI ##
# not prompted
# "/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# custom prompt
# "/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI/P_1_PQA_5_promptsource_False/dataset_1/dataset_validation.arrow"

# # sentiment
# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# export NUM_SAMPLES=1000
# export ANSWER_CHOICES=("pos" "neg" "neu")
# export MODEL=
# python generate.py \
#     --model_name_or_path $MODEL \
#     --dataset_path $DATASET_PATH \
#     --num_samples $NUM_SAMPLES \
#     --answer_choices "${ANSWER_CHOICES[@]}"




# # NLI validation set
# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# export NUM_SAMPLES=1000
# export ANSWER_CHOICES=("entailment" "contradiction" "neutral", "non-entailment")
# export MODEL=
# python generate.py \
#     --model_name_or_path $MODEL \
#     --dataset_path $DATASET_PATH \
#     --num_samples $NUM_SAMPLES \
#     --answer_choices "${ANSWER_CHOICES[@]}"


#!/bin/bash

# # NLI validation set
# models=(
# "gpt2"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI/P_1_PQA_1_psource_False_prompt_False/dataset_0.1/25-07-23_00:57/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI/P_1_PQA_1_psource_False_prompt_False/dataset_1/24-07-23_17:33/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/huggingface/merged_datasets/c4_3001124/22-07-23_16:15/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI/P_1_PQA_5_psource_False_prompt_False/dataset_0.1/26-07-23_12:05/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI/P_1_PQA_5_psource_False_prompt_False/dataset_1/23-07-23_17:56/checkpoint-50000"
# )
# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_eval_2/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# export NUM_SAMPLES=1000
# export ANSWER_CHOICES=("entailment" "contradiction" "neutral", "non-entailment")


# NLI_2 validation set
# models=(
# "gpt2-large"
# "gpt2"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/gpt2_cont/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_False_prompt_False/dataset_1/28-07-23_17:36/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2_ckpt_False/checkpoint-50000"
# "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2_ckpt_True/checkpoint-50000"
# "./models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2-large_ckpt_True/checkpoint-50000"
# )

models=(
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2-large_ckpt_False/checkpoint-50000"
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_5_psource_True_prompt_False/dataset_1/gpt2_ckpt_True/checkpoint-50000"
)

# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_validation.arrow"
export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_eval_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_validation.arrow"
export NUM_SAMPLES=1000
export ANSWER_CHOICES=("entailment" "contradiction" "neutral", "non-entailment")


# Iterate over each model
for MODEL in "${models[@]}"
do
    echo "Running for model: $MODEL"
    python ../generate.py \
        --model_name_or_path "$MODEL" \
        --dataset_path $DATASET_PATH \
        --num_samples $NUM_SAMPLES \
        --answer_choices "${ANSWER_CHOICES[@]}"
done