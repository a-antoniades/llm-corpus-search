datasets=(
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_QA"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_Summarization"
    "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_QA_NLI_Summarization_Commonsense"

)

# iterate over each dataset
for DATASET in "${datasets[@]}"
do  

    # VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
    # DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
    # WANDB_MODE="run"
    # BATCH_SIZE=1
    # REPORT_TO="dryrun"
    # MODEL_NAME="EleutherAI/pythia-2.8B-deduped"
    # EXP_PATH="./models/pythia/experiment_1"
    # export CUDA_VISIBLE_DEVICES=0
    # export OMP_NUM_THREADS=8
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
    #    --gradient_accumulation_steps 30 \
    #    --evaluation_strategy steps \
    #    --eval_steps 10000 \
    #    --save_strategy steps \
    #    --save_steps 10000 \
    #    --load_best_model_at_end \
    #    --save_total_limit 250 \
    #    --learning_rate 2.5e-4 \
    #    --weight_decay 0.01 \
    #    --warmup_steps 2000 \
    #    --lr_scheduler_type constant \
    #    --logging_steps 10 \
    #    --dataset_dir $DATASET \
    #    --validation_dataset $VALIDATION_DATASET \
    #    --report_every 50000 \
    #    --wandb_mode $WANDB_MODE \
    #    --tokenize_only \
    #    --report_to $REPORT_TO

    VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
    DEEPSPEED_CONFIG_PATH='/share/edc/home/antonis/LLM-Incidental-Supervision/incidental-supervision/ds_config.json'
    DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
    WANDB_MODE="run"
    BATCH_SIZE=30
    REPORT_TO="wandb"
    MODEL_NAME="EleutherAI/pythia-2.8B-deduped"
    EXP_PATH="./models/pythia/experiment_1"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export OMP_NUM_THREADS=8
    deepspeed \
    train_gpt.py \
       --count_tokens False \
       --model_type pythia \
       --model_name_or_path $MODEL_NAME \
       --rand_init_weights True \
       --resume True \
       --do_train \
       --do_eval \
       --output_dir $EXP_PATH \
       --fp16 True \
       --num_train_epochs 1 \
       --per_device_train_batch_size $BATCH_SIZE \
       --gradient_checkpointing \
       --gradient_accumulation_steps 1 \
       --evaluation_strategy steps \
       --eval_steps 10000 \
       --save_strategy steps \
       --save_steps 10000 \
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
       --report_to $REPORT_TO \
       --deepspeed $DEEPSPEED_CONFIG_PATH  # Add DeepSpeed config file path

done
