export WANDB_API_KEY=[redacted]
export WANDB_PROJECT=airoboros-65b-gpt4-qlora

# git clone https://hf.co/decapoda-research/llama-65b-hf
wget -O instructions.jsonl https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.1/resolve/main/instructions.jsonl
wget -O /workspace/llama-65b-hf/tokenizer_config.json https://huggingface.co/jondurbin/airoboros-7b-gpt4/resolve/main/tokenizer_config.json
wget -O /workspace/llama-65b-hf/special_tokens_map.json https://huggingface.co/jondurbin/airoboros-7b-gpt4/resolve/main/special_tokens_map.json

python qlora.py \
    --model_name_or_path /workspace/llama-65b-hf \
    --output_dir /workspace/airoboros-65b-gpt4-qlora \
    --max_steps 1000 \
    --logging_steps 1 \
    --save_strategy steps \
    --data_seed 73771 \
    --save_steps 25 \
    --save_total_limit 5 \
    --evaluation_strategy "no" \
    --eval_dataset_size 2 \
    --max_new_tokens 2048 \
    --dataloader_num_workers 3 \
    --group_by_length \
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
    --gradient_checkpointing \
    --dataset instructions.jsonl \
    --dataset_format airoboros \
    --source_max_len 2048 \
    --target_max_len 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 73771 \
    --report_to wandb
