class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.loss_type      = 'grad_ascent' # (vanilla_grad_diff, grad_ascent, batch_grad_diff, ails_grad_diff, van_npo, dpo, retain_npo)
        self.model_id       =  #'meta-llama/Meta-Llama-3.1-8B-Instruct'
        self.ft_model       = '' # not important for now
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 2 # change this to any 2 power n number 
        self.gradient_accumulation_steps = 1 
        self.num_epochs     = 10 # used 10 for dpo
        self.overwrite_dir  = True
        self.weight_decay   = 0.01 
        self.exp_type       = 'grad_ascent' # (vanilla_grad_diff, grad_ascent, batch_grad_diff, ails_grad_diff, van_npo, dpo, retain_npo)
        self.save_dir       = f'outputs/wpu_{self.exp_type}_model' # we store the unlearnt model
        self.access_token   = '' # write your hf key
        self.retriever_model= 'paraphrase-MiniLM-L6-v2' # not important
        self.forget_path    = 'dpo_forget_idk.csv' # forget data path
        self.retain_path    = 'full_retain_qa.csv'  # retain data path
        self.results_path   = f'' # this is for evaluation results not important now
        self.test_path      = '' # for evaluation
 
