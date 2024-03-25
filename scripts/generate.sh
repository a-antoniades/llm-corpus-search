models=(
    # "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/gpt2-large_ckpt_False/checkpoint-50000"
    # "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/NLI_2/P_1_PQA_5_psource_True_prompt_False/dataset_1/gpt2_ckpt_True/checkpoint-50000"
    # "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_NLI/EleutherAI/pythia-160M-deduped_ckpt_False/checkpoint-70000"
    "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_NLI/EleutherAI/pythia-1.4B-deduped_ckpt_False/checkpoint-70000"
    # "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_NLI/EleutherAI/pythia-160M-deduped_ckpt_False/checkpoint-70000"
    # "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/pythia/experiment_1/huggingface/flan_v1/c4_mixed_QA/EleutherAI/pythia-160M-deduped_ckpt_False/checkpoint-70000"
)

ds_keys=(
    'NLI'
    'QA'
    'Summarization'
    'Commonsense'
    'Reading Comp.'
    'Reading Comp. w/ Commonsense'
    'Sentiment'
    'Paraphrase'    
    'Coreference'
    'Misc'
    'Math'
)

# Reverse the order
ds_keys=($(printf '%s\n' "${ds_keys[@]}" | tac))

export NUM_SAMPLES=200
export CUDA_VISIBLE_DEVICES=0
# Iterate over each model and ds_key
for MODEL in "${models[@]}"
do
    for DS_KEY in "${ds_keys[@]}"
    do
        echo "Running for model: $MODEL and dataset key: $DS_KEY"
        DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds"
        python generate.py \
            --model_name_or_path "$MODEL" \
            --dataset_path $DATASET_PATH \
            --num_samples $NUM_SAMPLES \
            --ds_key "$DS_KEY" \
            --text_column "inputs"
    done
done