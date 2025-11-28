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

class NestedAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        prenorm = True,
        keys_rmsnorm = True # https://openreview.net/forum?id=HkztQWZfl2
    ):
        super().__init__()

        self.norm = nn.RMSNorm(dim) if prenorm else nn.Identity()

        dim_inner = dim_head * heads
        self.to_queries = nn.Linear(dim, dim_inner, bias = False)

        # keys and values

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.to_keys = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_values = nn.Linear(dim, dim_inner * 3, bias = False)

        self.key_norms = ModuleList([nn.RMSNorm(dim_head) for _ in range(3)])
        self.nested_key_norm = nn.RMSNorm(dim_head)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        tokens,
        cache = None,
        return_kv_cache = False
    ):
        batch, seq_len, device = *tokens.shape[:2], tokens.device

        tokens = self.norm(tokens)

        queries = self.to_queries(tokens)

        keys = self.to_keys(tokens).chunk(3, dim = -1)
        values = self.to_values(tokens).chunk(3, dim = -1)

        # split heads for input as well as all keys, values that form the implicit weights

        queries, keys, values = tree_map(self.split_heads, (queries, keys, values))

        # maybe norm all keys

        keys = [norm(k) for norm, k in zip(self.key_norms, keys)]

        # cache

        if exists(cache):
            (cache_keys, cache_values), (cache_nested_keys, cache_nested_values) = cache

            keys = [cat(args, dim = -2) for args in zip(cache_keys, keys)]
            values = [cat(args, dim = -2) for args in zip(cache_values, values)]

        # attend

        def attend(q, k, v):
            q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

            return F.scaled_dot_product_attention(q, k, v, is_causal = True)

        # nested attention

        nq, nk, nv = [attend(queries, key, value) for key, value in zip(keys, values)]

        nk = self.nested_key_norm(nk)

        if exists(cache):
            nk = cat((cache_nested_keys, nk), dim = -2)
            nv = cat((cache_nested_values, nv), dim = -2)

        out = attend(nq, nk, nv)

        # merge heads

        out = self.merge_heads(out)

        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, ((keys, values), (nk, nv))

if __name__ == '__main__':

    nested_attn = NestedAttention(512)

    tokens = torch.randn(1, 1024, 512)

    out1, cache = nested_attn(tokens)
    out2, cache = nested_attn(tokens[:, -1:], cache = cache)

    assert out1.shape == tokens.shape
    assert out2.shape == (1, 1, 512)
