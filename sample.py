import os
import random
import gzip
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

# -------------
# Import the same references and classes as in your training script
# Make sure titans_pytorch is visible on your PYTHONPATH or in the same folder.
# -------------
from titans_pytorch import MemoryAsContextTransformer  # must match your local import
from adam_atan2_pytorch import AdoptAtan2  

# -----------------
# Configurable constants
# -----------------
SAVE_DIR = './saved_models'
SAVE_FILENAME = 'mac_transformer.pt'
CHECKPOINT_PATH = os.path.join(SAVE_DIR, SAVE_FILENAME)

DATA_PATH       = './data/enwik8.gz' # path to enwik8 data if you want to sample prime text from validation
SEQ_LEN         = 512                # chunk length used during training
PRIME_LENGTH    = 100                # how many tokens from data to "prime" the model with
GENERATE_LENGTH = 512                # how many new tokens to generate
NUM_LONGTERM_MEM = 4
NUM_PERSIST_MEM  = 4
NEURAL_MEM_LAYERS = (2, 4)           # same as in your training
WINDOW_SIZE      = 32
NEURAL_MEM_SEGMENT_LEN = WINDOW_SIZE // 2
KV_RECON_LOSS_WEIGHT   = 0.0
LEARNED_MEM_MODEL_WEIGHTS = True
USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN       = True

# -------------
# Helpers
# -------------
def decode_token(token: int) -> str:
    """
    Convert an integer token (0..255) into a readable character,
    forcing it to be at least ASCII 32 so that control chars do not appear directly.
    """
    return chr(max(32, token))

def decode_tokens(tokens: torch.Tensor) -> str:
    """
    Turn a sequence of integer tokens into a string.
    """
    return ''.join(decode_token(t.item()) for t in tokens)

# -------------
# Optional: Use the same text sampler dataset if you want to pick prime text from val
# -------------
class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

def load_enwik8_val():
    """
    Loads the enwik8 validation portion (after first 90e6 bytes).
    Returns a PyTorch tensor of dtype long on CPU by default.
    """
    with gzip.open(DATA_PATH, 'rb') as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8)
    # split into train and val
    data_train, data_val = np.split(data, [int(90e6)])
    data_val = torch.from_numpy(data_val).long()
    return data_val

# -------------
# Simple ancestral sampling function if you do NOT have model.sample()
# (If your MemoryAsContextTransformer includes a .sample method, you can skip this.)
# -------------
@torch.no_grad()
def generate_tokens(
    model: MemoryAsContextTransformer,
    prime_tokens: torch.Tensor,
    generate_length: int = 512,
    temperature: float = 1.0,
    min_p: float = 0.1
) -> torch.Tensor:
    """
    Ancestral sampling: at each step, feed the tokens through the model, sample the next token.
    Applies a simple "min_p" filter to avoid very low-prob tokens.
    """
    device = next(model.parameters()).device

    # ensure shape [batch=1, seq_len]
    prime_tokens = prime_tokens.unsqueeze(0).to(device)  # shape (1, prime_length)

    out = prime_tokens.clone()
    for _ in tqdm.tqdm(range(generate_length), desc="Generating"):
        logits = model(out, disable_flex_attn=True)  # (batch=1, seq_len, vocab_size=256)
        next_token_logits = logits[:, -1, :]        # last time-step's logits => shape (1, 256)

        # do a min-p filter
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        top_prob = probs.max(dim=-1, keepdim=True).values
        mask = probs < (min_p * top_prob)  # mask out everything below min_p * top_prob
        next_token_logits = next_token_logits.masked_fill(mask, float('-inf'))

        # sample from the adjusted distribution
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
        out = torch.cat([out, next_token], dim=-1)
    return out.squeeze(0)  # return shape (seq_len + generate_length,)

# -------------
# Main sampling entry point
# -------------
def main():
    # 1) Instantiate the same model architecture as your training script
    model = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 384,
        depth = 8,
        segment_len = WINDOW_SIZE,
        num_persist_mem_tokens = NUM_PERSIST_MEM,
        num_longterm_mem_tokens = NUM_LONGTERM_MEM,
        neural_memory_layers = NEURAL_MEM_LAYERS,
        neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
        neural_mem_gate_attn_output = True,
        aux_kv_recon_loss_weight = KV_RECON_LOSS_WEIGHT,
        use_flex_attn = USE_FLEX_ATTN,
        sliding_window_attn = True,
        neural_memory_kwargs = dict(
            dim_head = 64,
            heads = 4,
            attn_pool_chunks = True,
            use_accelerated_scan = USE_ACCELERATED_SCAN,
            learned_mem_model_weights = LEARNED_MEM_MODEL_WEIGHTS,
            default_model_kwargs = dict(
                depth = 2,
            )
        )
    ).cuda()

    # 2) Load your trained checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found.")
    print(f"Loading model weights from {CHECKPOINT_PATH} ...")
    state_dict = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Optionally load some data to get a prime text from the validation set
    #    (Or you can manually define your prime tokens)
    data_val = load_enwik8_val()
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)

    # pick some random slice of data to prime
    prime_slice = random.choice(val_dataset)  # shape (SEQ_LEN+1,) on CPU
    prime_tokens = prime_slice[:PRIME_LENGTH]

    # decode the prime
    prime_str = decode_tokens(prime_tokens)
    print("\n========  PRIME TEXT  =======")
    print(prime_str)
    print("=============================\n")

    # 4) Generate new tokens
    generated = generate_tokens(
        model,
        prime_tokens = prime_tokens,
        generate_length = GENERATE_LENGTH,
        temperature = 1.0,
        min_p = 0.1
    )

    # 5) Decode and print
    gen_str = decode_tokens(generated[PRIME_LENGTH:])  # skip the prime tokens
    print("\n========  GENERATED TEXT  =======")
    print(gen_str)
    print("===============================\n")

if __name__ == "__main__":
    main()
