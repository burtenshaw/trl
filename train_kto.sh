# if a python env named venv does not exist, make one

# pip install .
# pip install accelerate wandb 
# pip install -U peft

python examples/scripts/kto.py \
    --model_name_or_path=microsoft/phi-2 \
    --per_device_train_batch_size 8 \
    --max_steps 1000 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="phi-2-kto" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16