datasets=(
    "/share/edc/home/antonis/datasets/huggingface/flan_v1/ds_c4_small"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_QA"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_QA_NLI_Summarization_Commonsense"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_Commonsense"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_Summarization"
    # "/share/edc/home/antonis/datasets/huggingface/flan_v1/c4_mixed_NLI"
)

# --resume_from_checkpoint $EXP_PATH/$MODEL_NAME/checkpoint-50000 \

# iterate over each dataset
for DATASET in "${datasets[@]}"
do  
    VALIDATION_DATASET='/share/edc/home/antonis/datasets/huggingface/flan_v1_task_ds_n_5000'
    DATASET_TYPE=$(echo "$DATASET" | awk -F/ '{print $(NF-1) "/" $NF}')
    WANDB_MODE="dryrun"
    BATCH_SIZE=30
    REPORT_TO="wandb"
    MODEL_NAME="EleutherAI/pythia-160M-deduped"
    EXP_PATH="./models/pythia/experiment_2"
    TCONF_PATH="./models/pythia/experiment_1/configs/config_160M.yml"
    COUNT_TOKENS=False
    MODEL_TYPE="pythia"
    RAND_INIT_WEIGHTS=True
    RESUME=True
    DO_TRAIN="--do_train"
    DO_EVAL="--do_eval"
    FP16=True
    NUM_TRAIN_EPOCHS=1
    GRADIENT_CHECKPOINTING="--gradient_checkpointing"
    GRADIENT_ACCUMULATION_STEPS=2
    REPORT_EVERY=50000
    TOKENIZE_ONLY="--tokenize_only"
    RDZV_BACKEND="c10d"
    RDZV_ENDPOINT="localhost:0"
    NPROC_PER_NODE=4
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    # python train_gpt.py \
    #     --tconf_path $TCONF_PATH \
    #     --count_tokens $COUNT_TOKENS \
    #     --model_type $MODEL_TYPE \
    #     --model_name_or_path $MODEL_NAME \
    #     --rand_init_weights $RAND_INIT_WEIGHTS \
    #     --resume $RESUME \
    #     --output_dir $EXP_PATH \
    #     --fp16 $FP16 \
    #     --num_train_epochs $NUM_TRAIN_EPOCHS \
    #     --per_device_train_batch_size $BATCH_SIZE \
    #     --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    #     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    #     --dataset_dir $DATASET \
    #     --validation_dataset $VALIDATION_DATASET \
    #     --report_every $REPORT_EVERY \
    #     --wandb_mode "dryrun" \
    #     --tokenize_only $TOKENIZE_ONLY \
    #     --report_to $REPORT_TO

    torchrun --rdzv_backend $RDZV_BACKEND --rdzv_endpoint $RDZV_ENDPOINT --nproc_per_node $NPROC_PER_NODE \
    train_gpt.py \
        --tconf_path $TCONF_PATH \
        --count_tokens $COUNT_TOKENS \
        --model_type $MODEL_TYPE \
        --model_name_or_path $MODEL_NAME \
        --rand_init_weights $RAND_INIT_WEIGHTS \
        --resume $RESUME \
        --output_dir $EXP_PATH \
        --fp16 $FP16 \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --per_device_train_batch_size $BATCH_SIZE \
        --gradient_checkpointing $GRADIENT_CHECKPOINTING \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --dataset_dir $DATASET \
        --validation_dataset $VALIDATION_DATASET \
        --report_every $REPORT_EVERY \
        --wandb_mode "dryrun" \
        --report_to $REPORT_TO
done