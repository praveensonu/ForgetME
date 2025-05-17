# ForgetME


## How to Run 


### Setup 

- Install dependencies.
    - Make a clean python environment
    - use nvidia-smi command in terminal to check for the cuda version in your laptop
    - Download pytorch based on the cuda version - https://pytorch.org/
    - Next install, transformers (less than version 5, preferred version 4.6 upto 5.0)
    - install, pandas, numpy, peft, accelerate
- Give the forget and retain path in the config.py file. Also add the hf token if you are using permissions required model. My suggestion, use a smaller model like 1B
- Note: Always use instruct based models. DO NOT use base model for now.
- Once installed everything, check torch and cuda availability. you can use torch.cuda.is_available() in python or cli.
- Now use export CUDA_VISIBLE_DEVICES=0 for device selection.
- python run.py


### Key things: 

- data_module file basically manipulates the data and generates pytorch dataset.
- utils has only two useful functions for now. find_linear_names and create single dataset.
- forget_loss file actually has the loss function that computes gradient ascent.
