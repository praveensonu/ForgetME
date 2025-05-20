from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import PreTrainedTokenizer, default_data_collator
from typing import Tuple
import math
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
import itertools
import random


# this function takes a tokenizer, max_length: int, question: str, answer: str. Returns their input_ids, labels, attention_mask.
# Internally it also applies tokenizer.apply_chat_template, which is only used for chat based or instruct based models. 
# the code also pads tokens to max_length.

def convert_raw_data_to_model_qa(tokenizer, max_length,  question, answer):
    question = str(question)
    answer = str(answer)
    
    messages = [{"role": "user", "content": question}]
    new_question = tokenizer.apply_chat_template(
        messages,
        tokenizer = False,
        add_generataion_prompt=True
    )
    
    full_text = str(new_question) + answer
    num_question_tokens = len(tokenizer.tokenize(str(new_question), add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)



# this function creates a pytorch dataset for the given dataframe or csv path. Return a tuple of input_ids, labels, attention_mask
# this class is for gradient ascent

class SingleDataset(Dataset):
    def __init__(self, forget_df, 
                 tokenizer, 
                 max_length=256, 
                 question_key = 'question',
                 answer_key = 'answer'):
        """
        Initializes the dataset for gradient ascent finetuning
        
        Args:
            data_path (str): path to the data file. csv file containing columns 'question' and 'answer'
            tokenizer (transformers.PreTrainedTokenizer): tokenizer to process the input
            max_length (int, optional): maximum sequence length for tokenization. Defaults to 512.
            template_format (str, optional): format template for structuring input
        """
        self.forget = forget_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx][self.qk]
        answer = self.data.iloc[idx][self.ak]
        return convert_raw_data_to_model_qa(
            tokenizer=self.tokenizer, 
            max_length=self.max_length, 
            question=question, 
            answer=answer,
        )


# this class is for gradient difference, but does cyclic rotation of forget and retain (based on max length of the files)
class DualDataset(Dataset): 
    """
    Dataset class for creating data for forget and retain (used by gradient difference)
    
    Args:
        forget_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for forgetting
        retain_data (pd.DataFrame): DataFrame containing 'question' and 'answer' columns for retaining
        tokenizer: tokenizer instance to process text
        max_length (int): maximum sequence length
        template_format (str, optional): format template for structuring input
    
    Returns:
        Tuple of forget and retain samples:
        (
            (forget_input_ids, forget_labels, forget_attention_mask),
            (retain_input_ids, retain_labels, retain_attention_mask)
        )
    """
    def __init__(self, forget_data, retain_data, tokenizer, max_length, template_format=None,
                 question_key = 'question',
                 answer_key = 'answer'):
        self.forget = forget_data.reset_index(drop=True)
        self.retain = retain_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key
    def __len__(self):
        return max(len(self.forget), len(self.retain))
    
    def __getitem__(self, idx):
        # Cyclic rotation of data
        forget_idx = idx % len(self.forget)
        retain_idx = idx % len(self.retain)

        forget_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.forget.iloc[forget_idx][self.qk],
            self.forget.iloc[forget_idx][self.ak],
        )

        retain_data = convert_raw_data_to_model_qa(
            self.tokenizer, self.max_length,
            self.retain.iloc[retain_idx][self.qk],
            self.retain.iloc[retain_idx][self.ak],
        )

        return (forget_data, retain_data)



