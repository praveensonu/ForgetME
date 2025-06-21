import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from config import Config
from peft import  LoraConfig, get_peft_model
from data_module import SingleDataset
from collators import custom_data_collator_forget
from utils import find_all_linear_names
from trainer import GATrainer
from accelerate import Accelerator
import pandas as pd
from template import LLAMA3_CHAT_TEMPLATE


accelerator = Accelerator()

cfg = Config()

print('loading the forget csv file. For Gradient Ascent we only need Forget file')
forget = pd.read_csv(cfg.forget_path) 

# ---- Loading the model -----
print(f"\nLoading the Tokenizer {cfg.model_id}")
tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = cfg.access_token)
tokenizer.pad_token = tokenizer.eos_token


print(f"\nLoading the Model {cfg.model_id}")
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, 
                                             torch_dtype = torch.bfloat16, 
                                             token=cfg.access_token,)

# ----- LoRA config ---------
config = LoraConfig(
        r = cfg.LoRA_r,
        lora_alpha = cfg.LoRA_alpha,
        lora_dropout= cfg.LoRA_dropout,
        target_modules = find_all_linear_names(model),
        bias = 'none',
        task_type = 'CAUSAL_LM',
    )

print(f"{config.target_modules}")

# ------- wrapping the model with the LoRA configuration
model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False


# ------- creating template format for tokenization --------
def make_template_format(df):
    df['question'] = df['question'].apply(lambda x : LLAMA3_CHAT_TEMPLATE.format(question = x))
    df['answer'] = df['answer'].apply(lambda x : x + tokenizer.eos_token)
    return df

forget = make_template_format(forget)
print('forget question and answer\n',forget['question'][0], forget['answer'][0])


# ------- Training Arguments ---------

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
    gradient_accumulation_steps= cfg.gradient_accumulation_steps,
)


  # ------- dataset for the gradient ascent method ----- 
print('\n\ncreating the dataset for gradient ascent')
dataset = SingleDataset(forget_data = forget,
                        tokenizer = tokenizer,
                        max_length = 256) 

trainer = GATrainer(
    model = model, 
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer,
    data_collator = custom_data_collator_forget,
    )

trainer.train()

accelerator.wait_for_everyone()

if training_args.local_rank <= 0: 
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"Rank {training_args.local_rank}: Tokenizer saved.")
else:
    tokenizer.save_pretrained(cfg.save_dir)
print(f'\nForget LoRA adapter saved at {cfg.save_dir}')
model.save_pretrained(cfg.save_dir)

