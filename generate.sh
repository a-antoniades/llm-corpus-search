DATASET_PATH="/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_1/dataset_train.arrow"
python generate.py \
    --model_name_or_path "/share/edc/home/antonis/LLM-Incidental-Supervision/models/['merged_datasets', 'dataset_0']_1688180240/checkpoint-96000" \
    --dataset_path $DATASET_PATH \
    --num_samples 20