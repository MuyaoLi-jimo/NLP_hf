#!/bin/bash 

batch=256
epoch=15
learning_rate=1.4e-4
version="11-13-01-test"
model="roberta-base"
dataset_name="restaurant_sup" #"laptop_sup","acl_sup",'agnews_sup','restaurant_sup'

WANDB_NAME="$model-adapter-$dataset_name-$version-e$epoch-2"
export WANDB_MODE="offline"
export WANDB_API_KEY="998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d"
export WANDB_PROJECT="train_BERT"



python train.py --dataset_name $dataset_name \
    --model_name_or_path "/home/mc_lmy/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b" \
    --use_adapter true \
    --reduction_factor 96 \
    --do_train true \
    --do_eval true \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "epoch" \
    --load_best_model_at_end \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --output_dir "./model/$model" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --report_to "wandb" \
    --run_name $WANDB_NAME 