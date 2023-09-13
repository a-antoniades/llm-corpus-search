# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/dataset_0"
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment/dataset_1"
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment/P_QA_5/dataset_1"
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment/P_1_PQA_5/dataset_1"
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_val/dataset_1"
# # # C4 senitment, verbalized labels, promptsource eval
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_1"
# # # VALIDATION_DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4/P_1_PQA_5_promptsource_True/dataset_1/dataset_validation.arrow"
# # # DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/sentiment_c4_small/P_1_PQA_5_promptsource_False/dataset_1"


# DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_5_psource_True_prompt_False/dataset_1/dataset_train.arrow"
# VALIDATION_DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
# BATCH_SIZE=40
# REPORT_TO="wandb"
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export OMP_NUM_THREADS=8
# torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 6 \
# train_gpt.py \
#    --model_type gpt2 \
#    --model_name_or_path gpt2-large \
#    --rand_init_weights True \
#    --do_train \
#    --do_eval \
#    --output_dir "./models/scrap" \
#    --fp16 True \
#    --max_steps 10000000000 \
#    --per_device_train_batch_size $BATCH_SIZE \
#    --gradient_checkpointing \
#    --gradient_accumulation_steps 2 \
#    --evaluation_strategy steps \
#    --eval_steps 5000 \
#    --save_strategy steps \
#    --save_steps 5000 \
#    --load_best_model_at_end \
#    --save_total_limit 250 \
#    --learning_rate 2.5e-4 \
#    --weight_decay 0.01 \
#    --warmup_steps 1000 \
#    --lr_scheduler_type cosine \
#    --logging_steps 1 \
#    --report_to wandb \
#    --dataset_dir $DATASET \
#    --validation_dataset $VALIDATION_DATASET \
#    --report_to $REPORT_TO \
#    --report_every 50000



# DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_train.arrow"
# VALIDATION_DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_False_prompt_False/dataset_1/dataset_validation.arrow"
# DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
# BATCH_SIZE=30
# REPORT_TO="wandb"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export OMP_NUM_THREADS=6
# torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 8 \
# train_gpt.py \
#    --model_type gpt2 \
#    --model_name_or_path gpt2-large \
#    --rand_init_weights True \
#    --do_train \
#    --do_eval \
#    --output_dir "./models/scrap" \
#    --fp16 True \
#    --max_steps 10000000000 \
#    --per_device_train_batch_size $BATCH_SIZE \
#    --gradient_checkpointing \
#    --gradient_accumulation_steps 2 \
#    --evaluation_strategy steps \
#    --eval_steps 5000 \
#    --save_strategy steps \
#    --save_steps 5000 \
#    --load_best_model_at_end \
#    --save_total_limit 250 \
#    --learning_rate 2.5e-4 \
#    --weight_decay 0.01 \
#    --warmup_steps 1000 \
#    --lr_scheduler_type cosine \
#    --logging_steps 1 \
#    --report_to wandb \
#    --dataset_dir $DATASET \
#    --validation_dataset $VALIDATION_DATASET \
#    --report_to $REPORT_TO \
#    --report_every 50000





# DATASET="/share/edc/home/antonis/datasets/huggingface/merged_datasets/NLI_2/P_1_PQA_1_psource_True_prompt_False/dataset_1/dataset_train.arrow"
# VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_p_8'
# DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
# BATCH_SIZE=30
# REPORT_TO="wandb"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export OMP_NUM_THREADS=6
# torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 8 \
# train_gpt.py \
#    --count_tokens False \
#    --model_type pythia \
#    --model_name_or_path "EleutherAI/pythia-1.4B-deduped" \
#    --rand_init_weights True \
#    --do_train \
#    --do_eval \
#    --output_dir "./models/pythia" \
#    --fp16 True \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size $BATCH_SIZE \
#    --gradient_checkpointing \
#    --gradient_accumulation_steps 1 \
#    --evaluation_strategy steps \
#    --eval_steps 5000 \
#    --save_strategy steps \
#    --save_steps 5000 \
#    --load_best_model_at_end \
#    --save_total_limit 250 \
#    --learning_rate 2.5e-4 \
#    --weight_decay 0.01 \
#    --warmup_steps 1000 \
#    --lr_scheduler_type constant \
#    --logging_steps 1 \
#    --dataset_dir $DATASET \
#    --validation_dataset $VALIDATION_DATASET \
#    --report_every 50000 \
#    --streaming False

# DATASET="/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_NLI"
# VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
# DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
# WANDB_MODE="dryrun"
# BATCH_SIZE=3
# REPORT_TO=False
# MODEL_NAME="EleutherAI/pythia-160M-deduped"
# EXP_PATH="./models/pythia/scratch"
# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=32
# python train_gpt.py \
#    --count_tokens False \
#    --model_type pythia \
#    --model_name_or_path $MODEL_NAME \
#    --rand_init_weights True \
#    --do_train \
#    --do_eval \
#    --output_dir $EXP_PATH \
#    --fp16 True \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size $BATCH_SIZE \
#    --gradient_checkpointing \
#    --gradient_accumulation_steps 10 \
#    --evaluation_strategy steps \
#    --eval_steps 5000 \
#    --save_strategy steps \
#    --save_steps 5000 \
#    --load_best_model_at_end \
#    --save_total_limit 250 \
#    --learning_rate 2.5e-4 \
#    --weight_decay 0.01 \
#    --warmup_steps 2000 \
#    --lr_scheduler_type constant \
#    --logging_steps 1 \
#    --dataset_dir $DATASET \
#    --validation_dataset $VALIDATION_DATASET \
#    --tokenize_only \
#    --overwrite_cache \
#    --wandb_mode $WANDB_MODE \
#    --report_every 50000
   
DATASET="/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_NLI"
VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
WANDB_MODE="run"
BATCH_SIZE=4
REPORT_TO="wandb"
MODEL_NAME="EleutherAI/pythia-160M-deduped"
EXP_PATH="./models/pythia/experiment_1"
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export OMP_NUM_THREADS=8
torchrun --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node 6 \
train_gpt.py \
   --count_tokens False \
   --model_type pythia \
   --model_name_or_path $MODEL_NAME \
   --rand_init_weights True \
   --resume_from_checkpont True \
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