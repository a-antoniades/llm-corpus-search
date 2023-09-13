models=(
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2-large_ckpt_False/checkpoint-50000"
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_5_psource_True_prompt_False/dataset_1/gpt2_ckpt_True/checkpoint-50000"
)

# export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_validation.arrow"
export DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_eval_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_validation.arrow"
export NUM_SAMPLES=1000


# Iterate over each model
for MODEL in "${models[@]}"
do
    echo "Running for model: $MODEL"
    python generate.py \
        --model_name_or_path "$MODEL" \
        --dataset_path $DATASET_PATH \
        --num_samples $NUM_SAMPLES \
        
done