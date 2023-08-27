# FIRST TOKENIZE DATASET
DATASET="/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small"
VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
WANDB_MODE="dryrun"
BATCH_SIZE=3
REPORT_TO=False
MODEL_NAME="EleutherAI/pythia-1.4B-deduped"
EXP_PATH="./models/pythia/scrap"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
python train_gpt.py \
   --count_tokens False \
   --model_type pythia \
   --model_name_or_path $MODEL_NAME \
   --rand_init_weights True \
   --do_train \
   --do_eval \
   --output_dir $EXP_PATH \
   --fp16 True \
   --num_train_epochs 1 \
   --per_device_train_batch_size $BATCH_SIZE \
   --gradient_checkpointing \
   --gradient_accumulation_steps 10 \
   --evaluation_strategy steps \
   --eval_steps 5000 \
   --save_strategy steps \
   --save_steps 5000 \
   --load_best_model_at_end \
   --save_total_limit 250 \
   --learning_rate 2.5e-4 \
   --weight_decay 0.01 \
   --warmup_steps 2000 \
   --lr_scheduler_type constant \
   --logging_steps 1 \
   --dataset_dir $DATASET \
   --validation_dataset $VALIDATION_DATASET \
   --tokenize_only \
   --overwrite_cache \
   --wandb_mode $WANDB_MODE \
   --report_every 50000


# TRAIN ON TOKENIZED DATASET (LARGE + SMALL MODEL)
# REMEMBER: Batch size = 240. (3 * 8 GPUS * 10 Grad accum steps)
models=(
    "EleutherAI/pythia-1.4B-deduped"
    "EleutherAI/pythia-160M-deduped"
)

for MODEL_NAME in "${models[@]}"
do
   DATASET="/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_NLI"
   VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
   DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
   WANDB_MODE="run"
   BATCH_SIZE=3
   REPORT_TO="wandb"
   EXP_PATH="./models/pythia/experiment_1"
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export OMP_NUM_THREADS=8
   torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 8 \
   train_gpt.py \
       --count_tokens False \
       --model_type pythia \
       --model_name_or_path $MODEL_NAME \
       --rand_init_weights True \
       --do_train \
       --do_eval \
       --output_dir $EXP_PATH \
       --fp16 True \
       --num_train_epochs 1 \
       --per_device_train_batch_size $BATCH_SIZE \
       --gradient_checkpointing \
       --gradient_accumulation_steps 10 \
       --evaluation_strategy steps \
       --eval_steps 5000 \
       --save_strategy steps \
       --save_steps 5000 \
       --load_best_model_at_end \
       --save_total_limit 250 \
       --learning_rate 2.5e-4 \
       --weight_decay 0.01 \
       --warmup_steps 2000 \
       --lr_scheduler_type constant \
       --logging_steps 10 \
       --dataset_dir $DATASET \
       --validation_dataset $VALIDATION_DATASET \
       --report_every 50000 \
       --wandb_mode $WANDB_MODE \
       --report_to $REPORT_TO
done
