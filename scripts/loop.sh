#!/bin/bash

batch=128
epoch=15
learning_rate=5e-5
version="11-14-01-test"
model="bert-base-uncased"
datasets=("restaurant_sup" "acl_sup" "agnews_sup")

export WANDB_MODE="offline"
export WANDB_API_KEY="998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d"
export WANDB_PROJECT="train_BERT"

for dataset_name in "${datasets[@]}"
do
    for number in {1..5}
    do
        WANDB_NAME="$model-$dataset_name-$version-e$epoch-number$number"
        
        echo "Training $model on $dataset_name attempt $number"

        CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name $dataset_name \
            --model_name_or_path $model \
            --do_train true \
            --do_eval true \
            --seed 2024 \
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
            --output_dir "./model/$model-$dataset_name-run$number" \
            --logging_strategy "steps" \
            --logging_steps 1 \
            --report_to "wandb" \
            --run_name $WANDB_NAME
    done
done