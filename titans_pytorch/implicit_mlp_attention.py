import torch
from torch import nn, cat, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils._pytree import tree_map

from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# functions

def exists(v):
    return v is not None

# classes

class ImplicitMLPAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        implicit_mlp_hiddens: tuple[int, ...] = (),
        implicit_mlp_activation = nn.SiLU(),
        prenorm = True
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim) if prenorm else nn.Identity()

        dim_inner = dim_head * heads
        self.to_queries = nn.Linear(dim, dim_inner, bias = False)

        # keys and values

        self.rotary_embed = RotaryEmbedding(dim_head)

        # each key value forms an implicit weight (memory) of (dim_key, dim_values)
        # chaining them would then be the implicit MLP from TTT / Titans

        mlp_dims = (dim_head, *implicit_mlp_hiddens, dim_head)

        self.keys = ModuleList([])
        self.values = ModuleList([])

        for dim_in, dim_out in zip(mlp_dims[:-1], mlp_dims[1:]):

            dim_keys_inner = dim_in * heads
            dim_values_inner = dim_out * heads

            keys = nn.Linear(dim, dim_keys_inner, bias = False)
            values = nn.Linear(dim, dim_values_inner, bias = False)

            self.keys.append(keys)
            self.values.append(values)

        self.implicit_mlp_activation = implicit_mlp_activation

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        tokens,
        cache = None
    ):
        batch, seq_len, device = *tokens.shape[:2], tokens.device

        tokens = self.norm(tokens)

        q = self.to_queries(tokens)

        keys = [fn(tokens) for fn in self.keys]
        values = [fn(tokens) for fn in self.values]

        # split heads for input as well as all keys, values that form the implicit weights

        q, keys, values = tree_map(self.split_heads, (q, keys, values))

        # cache

        if exists(cache):
            cache_keys, cache_values = cache

            keys = [cat(args, dim = -2) for args in zip(cache_keys, keys)]
            values = [cat(args, dim = -2) for args in zip(cache_values, values)]

        # attend

        def attend(q, k, v):
            q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # implicit memory mlp

        out = q

        for i, (key, value) in enumerate(zip(keys, values), start = 1):
            is_last = i == len(keys)

            out = attend(out, key, value)

            if not is_last:
                out = self.implicit_mlp_activation(out)

        # merge heads

        out = self.merge_heads(out)

        return self.to_out(out), (keys, values)

# 3 layers implicit MLP attention - 64 -> 128 -> 128 -> 64 w/ relu

if __name__ == '__main__':

    implicit_mlp_attn = ImplicitMLPAttention(
        512,
        dim_head = 64,
        implicit_mlp_hiddens = (128, 128),
        implicit_mlp_activation = nn.ReLU()
    )

    tokens = torch.randn(1, 1024, 512)

    out, cache = implicit_mlp_attn(tokens)
    out, cache = implicit_mlp_attn(tokens, cache = cache)

    assert out.shape == tokens.shape
