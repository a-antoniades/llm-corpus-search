DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment/dataset_1/dataset_validation.arrow"
python generate.py \
    --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/models/['sentiment', 'dataset_1']_1688427520/checkpoint-18000" \
    --dataset_path $DATASET_PATH \
    --num_samples 20