"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py config/gpt-124M-NS/AdamS.py --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py config/gpt-124M-NS/AdamS.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py config/gpt-124M-NS/AdamS.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py config/gpt-124M-NS/AdamS.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from optimizer_factory import build_optimizers

# -----------------------------------------------------------------------------
# default GPT-2 124M OpenWebText config values

# run lifecycle and output
out_dir = "out"
init_from = "scratch"  # 'scratch' or 'resume'
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval

# tensorboard logging
tb_log = False
tb_run_project = "owt"
tb_run_name = "gpt2"  # 'run' + str(time.time())
tb_out = "tensorboard_out"

# evaluation and console logging
eval_interval = 1000
eval_iters = 200
log_interval = 1

# data and batch shape
dataset = "openwebtext"
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes

# model dimensions and core architecture
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# attention implementation
attn_type = "sdpa_attn"
qknorm_type = "identity"

# optimizer family and optimizer hyperparameters
optimizer_variant = "Adam"  # Adam/AdamS/AdamQ/AdamZ or mSGD/mSGDS/mSGDQ/mSGDZ
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
sgd_momentum = 0.9

# learning rate schedule
lr_matrix = None  # max learning rate for matrix parameters
lr_vector = None  # max learning rate for vector parameters
min_lr_matrix = 6e-5  # minimum learning rate for matrix parameters
min_lr_vector = 6e-5  # minimum learning rate for vector parameters
max_iters = 600000  # total number of training iterations
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla

# Newton-Schulz optimizer controls
ns_iters = 15
split_qkv_updates = False

# distributed optimizer and DDP
use_zero = True
backend = "nccl"  # 'nccl', 'gloo', etc.

# system and runtime
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and (isinstance(v, (int, float, bool, str)) or v is None)
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

if lr_matrix is None or lr_vector is None:
    raise ValueError("Configs must set lr_matrix and lr_vector")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    print(
        f"ddp_world_size: {ddp_world_size}, "
        f"gradient_accumulation_steps: {gradient_accumulation_steps}"
    )
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

import torch._dynamo

torch._dynamo.config.suppress_errors = True

if tb_log and master_process:
    # Some TensorBoard builds still ship protobuf stubs that are incompatible
    # with newer protobuf runtimes. The pure-Python parser keeps logging usable.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(os.path.join(tb_out, tb_run_project, tb_run_name))

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
print(f"==> use dtype is {dtype}")
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # Recreate np.memmap every batch to avoid a known long-iteration memory leak.
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    attn_type=attn_type,
    qknorm_type=qknorm_type,
)
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError("init_from must be 'scratch' or 'resume'")
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
print(f"==> optimizer_variant : {optimizer_variant}")
optimizer = build_optimizers(
    model,
    weight_decay=weight_decay,
    lr_matrix=lr_matrix,
    lr_vector=lr_vector,
    betas=(beta1, beta2),
    device_type=device_type,
    optimizer_variant=optimizer_variant,
    use_zero=use_zero,
    ns_iters=ns_iters,
    split_qkv_updates=split_qkv_updates,
    sgd_momentum=sgd_momentum,
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


# training loop
X, Y = get_batch("train")  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:
    matrix_lr = get_lr(iter_num, lr_matrix, min_lr_matrix)
    vector_lr = get_lr(iter_num, lr_vector, min_lr_vector)
    optimizer.set_lrs(matrix_lr, vector_lr)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        if use_zero and ddp and iter_num > 0:
            optimizer.consolidate_state_dict(to=0)

        if master_process:
            losses = estimate_loss()
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if tb_log:
                writer.add_scalar("iter", iter_num, iter_num)
                writer.add_scalar("train/lr_vector", vector_lr, iter_num)
                writer.add_scalar("train/lr_matrix", matrix_lr, iter_num)
                writer.add_scalar("train/loss", losses["train"], iter_num)
                writer.add_scalar("train/mfu", running_mfu * 100, iter_num)
                writer.add_scalar("val/loss", losses["val"], iter_num)

            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # Sync DDP gradients only on the final micro-step.
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        optimizer.unscale_(scaler)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if iter_num % eval_interval == 0 and master_process and tb_log:
            writer.add_scalar("train/grad_norm", grad_norm, iter_num)

    # step the optimizer and scaler if training in fp16
    optimizer.step(scaler)
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # Scale up to undo the division above, approximating the true total loss.
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: lr_matrix {matrix_lr:.4f}, lr_vector {vector_lr:.4f}, "
            f"loss {lossf:.4f}, "
            f"time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
