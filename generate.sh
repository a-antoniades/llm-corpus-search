# intiial exp with same iterations
DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/dataset_test_with_answers.arrow"
python generate.py \
    --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_1/10-07-23_15:25/eval/checkpoint-84500" \
    --dataset_path $DATASET_PATH \
    --num_samples 250

DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/dataset_test_with_answers.arrow"
python generate.py \
    --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/models/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_0/10-07-23_15:26/save_for_later/checkpoint-145500" \
    --dataset_path $DATASET_PATH \
    --num_samples 250

