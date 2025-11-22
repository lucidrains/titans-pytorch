from __future__ import annotations

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
        mlp_hiddens: tuple[int, ...],
        *,
        activation = nn.SiLU(),
        heads = 8,
        talking_heads = True,
        prenorm = True,
        keys_rmsnorm = True # https://openreview.net/forum?id=HkztQWZfl2
    ):
        super().__init__()
        assert isinstance(mlp_hiddens, tuple) and len(mlp_hiddens) >= 2
        dim_mlp_in, *dim_mlp_inner, dim_mlp_out = mlp_hiddens

        self.norm = nn.RMSNorm(dim) if prenorm else nn.Identity()

        dim_query_inner = dim_mlp_in * heads
        self.to_queries = nn.Linear(dim, dim_query_inner, bias = False)

        # keys and values

        self.rotary_embed = RotaryEmbedding(min(mlp_hiddens)) # just use the minimum dimension, the rest is partially rotaried

        # each key value forms an implicit weight (memory) of (dim_key, dim_values)
        # chaining them would then be the implicit MLP from TTT / Titans

        self.keys = ModuleList([])
        self.key_norms = ModuleList([])

        self.values = ModuleList([])

        for dim_in, dim_out in zip(mlp_hiddens[:-1], mlp_hiddens[1:]):

            dim_keys_inner = dim_in * heads
            dim_values_inner = dim_out * heads

            keys = nn.Linear(dim, dim_keys_inner, bias = False)
            key_norms = nn.RMSNorm(dim_in) if keys_rmsnorm else nn.Identity()

            values = nn.Linear(dim, dim_values_inner, bias = False)

            self.keys.append(keys)
            self.key_norms.append(key_norms)

            self.values.append(values)

        self.activation = activation

        # talking head - Shazeer et al.

        self.talking_heads = nn.Identity()

        if talking_heads and len(dim_mlp_inner) > 0:
            self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            nn.init.dirac_(self.talking_heads.weight)

        # split merging of heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_mlp_out * heads, dim, bias = False)

    def forward(
        self,
        tokens,
        cache = None,
        return_kv_cache = False
    ):
        batch, seq_len, device = *tokens.shape[:2], tokens.device

        tokens = self.norm(tokens)

        queries = self.to_queries(tokens)

        keys = [fn(tokens) for fn in self.keys]
        values = [fn(tokens) for fn in self.values]

        # split heads for input as well as all keys, values that form the implicit weights

        queries, keys, values = tree_map(self.split_heads, (queries, keys, values))

        # maybe norm all keys

        keys = [norm(k) for norm, k in zip(self.key_norms, keys)]

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

        out = queries

        for i, (key, value) in enumerate(zip(keys, values), start = 1):
            is_last = i == len(keys)

            out = attend(out, key, value)

            if not is_last:
                out = self.talking_heads(out)
                out = self.activation(out)

        # merge heads

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, (keys, values)

# 3 layers implicit MLP attention - 64 -> 128 -> 128 -> 64 w/ relu

if __name__ == '__main__':

    implicit_mlp_attn = ImplicitMLPAttention(
        512,
        (64, 128, 128, 64),
        activation = nn.ReLU()
    )

    tokens = torch.randn(1, 1024, 512)

    out, cache = implicit_mlp_attn(tokens)
    out, cache = implicit_mlp_attn(tokens, cache = cache)

    assert out.shape == tokens.shape
