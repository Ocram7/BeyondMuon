# GPT-2 124M OpenWebText config: AdamS, p = 1/2.
# $ torchrun --standalone --nproc_per_node=8 train.py config/gpt-124M-NS/AdamS.py

# experiment-specific settings
tb_run_name = 'gpt2-124M-AdamS'
optimizer_variant = "AdamS"

lr_matrix = 1e-2
lr_vector = 3e-4
min_lr_matrix = 3e-6
min_lr_vector = 3e-6

split_qkv_updates = True

# shared logging/output
tb_log = True
tb_run_project = 'owt'
out_dir = f'out/{tb_run_project}/{tb_run_name}'

# shared batch and training schedule
# 30 batch size * 1024 block size * 16 gradaccum = 491,520 tokens
batch_size = 6 * 5
block_size = 1024
gradient_accumulation_steps = 8 * 2

# this makes total number of tokens be 100B
max_iters = 200000
lr_decay_iters = 200000
warmup_iters = 2000

# shared evaluation/logging
eval_interval = 1000
eval_iters = 200
log_interval = 10

# shared optimizer settings
weight_decay = 0.0

# shared model settings
attn_type = "slow_attn"
qknorm_type = "identity"
