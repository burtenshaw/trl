# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the following command with some example arguments.
In general, the optimal configuration for KTO will be similar to that of DPO:

# regular:
python examples/scripts/kto.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/kto.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="kto_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass, field
from typing import Optional

# from datasets import Dataset, load_dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
# import wandb

# from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config

# wandb.init(
#     project="kto",
# )


# # Define and parse arguments.
# @dataclass
# class ScriptArguments:
#     """
#     The arguments for the KTO training script.
#     """

#     # debugging
#     sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})


# if __name__ == "__main__":
#     parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
#     script_args, kto_args, model_args = parser.parse_args_into_dataclasses()
#     # kto_args.desirable_weight = 0.22
#     # kto_args.generate_during_eval = True
    
#     # 1. load a pretrained model
#     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
#     model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 2. Load the dataset
#     repo_id = "argilla/ultrafeedback-binarized-preferences-cleaned-kto"
#     dataset = load_dataset(repo_id, split="train")
#     dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["test"]
    
#     # 4. initialize the KTO trainer
#     kto_trainer = KTOTrainer(
#         model=model,
#         ref_model=model_ref,
#         args=kto_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         tokenizer=tokenizer,
#         peft_config=get_peft_config(model_args),
#     )

#     # 5. train and save the model
#     kto_trainer.train()
#     kto_trainer.save_model(kto_args.output_dir)
