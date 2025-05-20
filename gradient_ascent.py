
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# to run the script, use the command: 
# 1. export CUDA_VISIBLE_DEVICES=4,5
# 2. accelerate launch --num_processes 2 run.py( only for multi gpu use accelerate)
# 3. python run.py (for single gpu)
# Note: Always remember to select the gpu either with os and os.environ in code or export CUDA_VISIBLE_DEVICES


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, default_data_collator
from config import Config
from peft import  LoraConfig, get_peft_model
from collators import grad_ascent_collator, cyclic_gd_collator
from data_module import SingleDataset, DualDataset
from utils import find_all_linear_names,
from forget_loss import GATrainer, GradDiffTrainer
from accelerate import Accelerator
import pandas as pd

accelerator = Accelerator()

cfg = Config()

# loading the paths

print('loading the paths to forget, retain and test set')
forget = pd.read_csv(cfg.forget_path) #cfg.forget_path
retain = pd.read_csv(cfg.retain_path) #cfg.retain_path
forget_path = cfg.forget_path
retain_path = cfg.retain_path
test_path = cfg.test_path


print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,)

config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = find_all_linear_names(model),
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

print(f"{LoraConfig.target_modules}")
# wrapping the model with the LoRA configuration

model = get_peft_model(model, config)
model.print_trainable_parameters()
#model.generation_config.do_sample = True
model.config.use_cache = False


dataset = SingleDataset(forget_df = forget,
                                tokenizer = tokenizer,
                                max_length = 256) 


training_args = TrainingArguments(
    output_dir = cfg.save_dir,
    overwrite_output_dir= True,
    learning_rate = cfg.lr,
    per_device_train_batch_size= cfg.batch_size,
    num_train_epochs= cfg.num_epochs,
    weight_decay = cfg.weight_decay,
    logging_dir = f'{cfg.save_dir}/logs',
    eval_strategy= 'no',
    label_names = ['labels'],
    bf16 = True,
    gradient_accumulation_steps=1,
    #save_only_model=True,
    report_to = 'wandb',
)

trainer = GATrainer(
        model = model, 
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = grad_ascent_collator,
        )

trainer.train()
accelerator.wait_for_everyone()
model.save_pretrained(cfg.save_dir)
tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
