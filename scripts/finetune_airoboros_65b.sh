export WANDB_API_KEY=[redacted]
export WANDB_PROJECT=airoboros-65b-gpt4-qlora

# Requirements:
# 1. Get the base model:
#    git clone https://hf.co/decapoda-research/llama-65b-hf
# 2. Get the dataset:
#    wget -O instructions.jsonl https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.4.1/resolve/main/instructions.jsonl
# 3. Replace tokenizer_config.json and special_tokens_map.json with corrected versions.
#    cp special_tokens_map.json /workspace/llama-65b-hf/
#    cp tokenizer_config.json /workspace/llama-65b-hf/

# This can optionally be python flash_qlora.py
python qlora.py \
    --model_name_or_path /workspace/llama-65b-hf \
    --output_dir /workspace/$WANDB_PROJECT-checkpoints \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_strategy steps \
    --data_seed 11422 \
    --save_steps 500 \
    --save_total_limit 10 \
    --evaluation_strategy "no" \
    --eval_dataset_size 2 \
    --max_new_tokens 1800 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --dataset /workspace/instructions.jsonl \
    --dataset_format airoboros \
    --model_max_len 2048 \
    --per_device_train_batch_size 2 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 11422 \
    --report_to wandb \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing
