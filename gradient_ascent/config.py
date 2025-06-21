class Config:
    def __init__(self):
        super(Config, self).__init__()
        # pre_unlearning - for pre-unlearning evaluation
        # gradient based [grad_ascent,gd_1_1seq, gd_1_1random, gd_direct, gd_indirect, gd_balanced, gd_cyclic, gd_melu]
        # dpo based [dpo, dpo_1_1_seq, dpo_1_1random, dpo_direct, dpo_indirect, dpo_balanced, dpo_cyclic, dpo_melu]
        # npo based [npo, npo_1_1_seq, npo_1_1random, npo_direct, npo_indirect, npo_balanced, npo_cyclic, npo_melu]

        self.loss_type      = 'gradient_ascent' # change this with the experiment types provided above
        self.access_token   = '' # please add your huggingface access token
        self.model_id       = 'meta-llama/Meta-Llama-3.1-8B-Instruct' # finetuned model path
        self.LoRA_r         = 8
        self.LoRA_alpha     = 16
        self.LoRA_dropout   = 0.05
        self.lr             = 2e-05
        self.LoRa_targets   = ['v_proj', 'k_proj', 'up_proj', 'o_proj', 'gate_proj', 'q_proj', 'down_proj']
        self.batch_size     = 4
        self.gradient_accumulation_steps = 1
        self.num_epochs     = 4
        self.overwrite_dir  = True
        self.weight_decay   = 0.01
        self.max_length     = 256
        self.exp_type       = self.loss_type 
        self.save_dir       = f'outputs/{self.exp_type}_model' # we store the unlearnt model here
        self.forget_path    = './data/dpo_forget_idk.csv' 


