DATASET=umls
MODE=random_walk
PATH_LEN=1
MODEL=gpt2
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=8897 train.py \
    --model_type Transformer \
    --model_name_or_path $MODEL \
    --random_initialize True \
    --mode $MODE \
    --path_len $PATH_LEN \
    --data_dir data \
    --dataset $DATASET \
    --fp16 True \
    --output_dir checkpoints/$DATASET/pretrain-$MODE-triple-$MODEL-$PATH_LEN-0615 \
    --model_max_length 512 \
    --max_steps 30000 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 200 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --entity_as_new_token True \
    --relation_as_new_token True 
    # --use_inverse_r True \
    # --weighted_r 1 \
    # --randomize_entity_name True \
    # --fsdp "full_shard auto_wrap" \
    # --tf32 True
    # CUDA_VISIBLE_DEVICES
    # --optim "adafactor" \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 16 \
    # --model_name_or_path checkpoints/$DATASET/pretrain-$MODE-new-token-$MODEL-$PATH_LEN-0614/checkpoint-10000 \
    # --resume \

# cache_dir = "./share/edc/home/antonis/datasets/huggingface"

DATASET=umls
MODE=random_walk
PATH_LEN=1
MODEL=gpt2
CUDA_VISIBLE_DEVICES=4 python train_copy.py \
    --model_type Transformer \
    --model_name_or_path $MODEL \
    --random_initialize True \
    --mode $MODE \
    --path_len $PATH_LEN \
    --data_dir data \
    --dataset $DATASET \
    --fp16 True \
    --output_dir checkpoints/$DATASET/pretrain-$MODE-triple-$MODEL-$PATH_LEN-0615 \
    --model_max_length 512 \
    --max_steps 30000 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 200 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --entity_as_new_token True \
    --relation_as_new_token True 


                "--model_type", "gpt2",
                "--model_name_or_path", "gpt2",
                "--rand_init_weights", "True",
                "--do_train",
                "--do_eval",
                "--fp16", "True",
                "--output_dir", "./models",
                "--max_steps", "10000000000",
                "--per_device_train_batch_size", "1",
                "--gradient_checkpointing",
                "--evaluation_strategy", "no",
                "--save_strategy", "steps",
                "--save_steps", "2000",
                "--save_total_limit", "200",
                "--learning_rate", "5e-4",
                "--weight_decay", "0.",
                "--warmup_steps", "1000",
                "--lr_scheduler_type", "cosine",
                "--logging_steps", "1",
                "--report_to", "wandb",
                "--dataset_name", "wikitext",
                "--dataset_config_name", "wikitext-2-raw-v1"

python train_gpt.py \
    --model_type gpt2 \
    --model_name_or_path gpt2-large \
    --rand_init_weights True \
    --do_train \
    --do_eval \
    --fp16 True \
    --output_dir ./models \
    --max_steps 10000000000 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 200 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_steps 1000 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --report_to wandb \
    --train_file "/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_1/dataset_validation.arrow" \
    --validation_file "/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_1/dataset_validation.arrow" \
    --data_config_file "/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_1/config.json" 