from transformers import Trainer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from accelerate import Accelerator
from typing import Dict, Union, Any, Optional, List, Tuple, Type
from torch.utils.data import DataLoader, SequentialSampler
from collators import dpo_retain_collator
from torch.utils.data import DataLoader, DistributedSampler




accelerator = Accelerator()

# not use ful for now
def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss



class GATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        """
        Computes the gradient ascent loss for the model
        """
        #if self.loss_type == 'grad_ascent':
        # unpack the forget inputs
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        # forward pass
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        forget_loss = outputs.loss * -1 # gradient ascent is negating the loss

        loss = forget_loss
        return (loss, outputs) if return_outputs else loss
