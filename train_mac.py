import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)               # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_QK_NORM = False
NEURAL_MEM_ADD_VALUE_RESIDUAL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = WINDOW_SIZE // 2 # set smaller for more granularity for learning rate / momentum etc
SLIDING_WINDOWS = True
WEIGHT_TIE_MEMORY_MODEL = True  # set to have memory MLP shared across layers
STORE_ATTN_POOL_CHUNKS = True # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
KV_RECON_LOSS_WEIGHT = 0.

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

# perf related

USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 384,
    depth = 8,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    aux_kv_recon_loss_weight = KV_RECON_LOSS_WEIGHT,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    weight_tie_memory_model = WEIGHT_TIE_MEMORY_MODEL,
    neural_memory_add_value_residual = NEURAL_MEM_ADD_VALUE_RESIDUAL,
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    ),
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR
    )
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
    data_train, data_val = np.split(data, [int(90e6)])
    data_train, data_val = map(torch.from_numpy, (data_train, data_val))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss, (ar_loss, kv_recon_losses) = model(next(train_loader), return_loss = True, return_loss_breakdown = True)
        loss.backward()

    print(f'training loss: {ar_loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(loss = ar_loss.item()))

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss, (ar_loss, _) = model(next(val_loader), return_loss = True, return_loss_breakdown = True)
            print(f'validation loss: {ar_loss.item()}')

    if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = True)
        output_str = decode_tokens(sample[0])
        print(output_str)
