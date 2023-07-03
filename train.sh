# DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_0"
DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment/dataset_1"
DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
BATCH_SIZE=32
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 8 \
train_gpt.py \
   --model_type gpt2 \
   --model_name_or_path gpt2 \
   --rand_init_weights True \
   --do_train \
   --do_eval \
   --output_dir "./models" \
   --fp16 True \
   --max_steps 10000000000 \
   --per_device_train_batch_size $BATCH_SIZE \
   --gradient_checkpointing \
   --gradient_accumulation_steps 1 \
   --evaluation_strategy steps \
   --eval_steps 500 \
   --save_strategy steps \
   --save_steps 2000 \
   --load_best_model_at_end \
   --save_total_limit 2 \
   --learning_rate 5e-4 \
   --weight_decay 0.01 \
   --warmup_steps 1000 \
   --lr_scheduler_type cosine \
   --logging_steps 1 \
   --report_to wandb \
   --dataset_dir $DATASET


DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_1"
DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
BATCH_SIZE=32
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 4 \
train_gpt.py \
   --model_type gpt2 \
   --model_name_or_path gpt2 \
   --rand_init_weights True \
   --do_train \
   --do_eval \
   --output_dir "./models" \
   --fp16 True \
   --max_steps 10000000000 \
   --per_device_train_batch_size $BATCH_SIZE \
   --gradient_checkpointing \
   --gradient_accumulation_steps 1 \
   --evaluation_strategy steps \
   --eval_steps 500 \
   --save_strategy steps \
   --save_steps 2000 \
   --save_total_limit 10 \
   --learning_rate 5e-4 \
   --weight_decay 0.01 \
   --warmup_steps 1000 \
   --lr_scheduler_type cosine \
   --logging_steps 1 \
   --report_to wandb \
   --dataset_dir $DATASET