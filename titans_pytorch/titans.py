from __future__ import annotations
import math
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad

from tensordict import TensorDict

from titans_pytorch.associative_scan import (
    associative_scan,
    binary_operator,
    pad_at_dim
)

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
"""

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

# softclamping gradients

def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    t, inverse = pack_one_with_inverse(t, 'bn *')

    norm = t.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)

# multi head rmsnorm

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# classes

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth
    ):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim, dim)) for _ in range(depth)])

    def forward(
        self,
        x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.silu(x)

            x = x @ weight

        return x

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # queries
            nn.Parameter(torch.randn(dim, dim)), # keys
            nn.Parameter(torch.randn(dim, dim)), # values
            nn.Parameter(torch.randn(dim, dim * 2)), # ff w1
            nn.Parameter(torch.randn(dim * 2, dim)), # ff w2
        ])

    def forward(self, x):

        assert x.shape[-2] > 1, 'chunk size needs to be greater than 1 for using attention as memory'

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = F.normalize(x @ wq, dim = -1)
        k = F.normalize(x @ wk, dim = -1)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        x = x + attn_out

        h = F.silu(x @ ffw1)
        out = h @ ffw2

        return out

# main neural memory

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size = 1,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable = default_adaptive_step_transform,
        pre_rmsnorm = True,
        post_rmsnorm = True,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        default_model_kwargs: dict = dict(
            depth = 2
        )
    ):
        super().__init__()
        dim_head = default(dim_head, dim)

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads

        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        # memory mlp

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # the memory is the weights of the model

        self.memory_model = model

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), weighted_loss.mean()

        self.per_sample_grad_fn = vmap(grad(forward_and_loss, has_aux = True), in_dims = (None, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = LinearNoBias(dim, dim_inner)

        # keys and values for storing to the model

        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)
        self.store_memory_loss_fn = store_memory_loss_fn

        # empty memory embed

        self.empty_memory_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.empty_memory_embed, std = 0.02)

        # learned adaptive learning rate and momentum
        # todo - explore mlp layerwise learned lr / momentum

        self.to_momentum = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        self.to_adaptive_step = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n')
        )

        self.adaptive_step_transform = adaptive_step_transform

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # weight decay factor

        self.to_decay_factor = nn.Sequential(
            Reduce('b (n c) ... -> b n ...', 'mean', c = chunk_size),
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

    def init_weights_and_momentum(self):
        params = TensorDict(dict(self.memory_model.named_parameters()))

        init_weights = params.clone().zero_()
        init_momentum = params.clone().zero_()

        return init_weights, init_momentum

    def init_empty_memory_embed(self, batch, seq_len):
        return repeat(self.empty_memory_embed, 'd -> b n d', b = batch, n = seq_len)

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]],
        return_aux_kv_loss = False
    ):

        seq = self.store_norm(seq)

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        seq_len, chunk_size = seq.shape[-2], self.chunk_size
        round_down_seq_len = round_down_multiple(seq_len, self.chunk_size)

        seq = seq[:, :round_down_seq_len]

        # curr weights + past weights, in the case that the initial weights are learned

        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        past_state = tuple(TensorDict(d) for d in past_state)
        past_weights, past_momentum = past_state

        curr_weights = curr_weights + past_weights

        # pack batch and sequence dimension

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        adaptive_momentum = self.to_momentum(seq).sigmoid()
        decay_factor = self.to_decay_factor(seq).sigmoid()

        # keys and values

        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # maybe multi head

        keys, values = map(self.split_heads, (keys, values))

        batch = keys.shape[0]

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b (n c) d -> (b n) c d', c = self.chunk_size) for t in (keys, values))

        adaptive_lr = rearrange(adaptive_lr, 'b (n c) -> (b n) c', c = self.chunk_size)

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, aux_kv_recon_loss = self.per_sample_grad_fn(dict(curr_weights), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.apply(lambda t: -t)

        # determine scan function

        def default_associative_scan(gates, inputs):
            _, outputs = associative_scan(binary_operator, (gates, inputs))
            return outputs

        if self.use_accelerated_scan:
            from accelerated_scan.triton import scan as triton_scan
            from accelerated_scan.warp import scan as warp_scan

            scan = triton_scan if seq.is_cuda else warp_scan

            def accelerate_scan_fn(gates, inputs):
                gates = gates.expand_as(inputs)
                gates, inputs = tuple(rearrange(t, 'b n d -> b d n') for t in (gates, inputs))

                seq_len = gates.shape[-1]
                next_power_two_seq_len = 2 ** max(5, int(math.ceil(math.log2(seq_len))))

                gates = F.pad(gates, (0, next_power_two_seq_len - seq_len))
                inputs = F.pad(inputs, (0, next_power_two_seq_len - seq_len))

                outputs = scan(gates.contiguous(), inputs.contiguous())

                outputs = outputs[..., :seq_len]
                outputs = rearrange(outputs, 'b d n -> b n d')
                return outputs

            scan_fn = accelerate_scan_fn
        else:
            scan_fn = default_associative_scan

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        next_momentum = TensorDict()
        updates = TensorDict()

        for param_name, surprise in surprises.items():

            surprise, inverse_pack = pack_one_with_inverse(surprise, 'b n *')

            # derive momentum with associative scan - eq (10)

            momentum = scan_fn(adaptive_momentum, surprise) # momentum is S / surprise in the paper

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = scan_fn(1. - decay_factor, momentum) # momentum is S / surprise in the paper

            updates[param_name] = inverse_pack(update)
            next_momentum[param_name] = inverse_pack(momentum)

        # compute the next weight per batch

        last_update = updates.apply(lambda t: t[:, -1])

        next_state = (curr_weights + last_update, next_momentum)

        if not return_aux_kv_loss:
            return updates, next_state

        return updates, next_state, aux_kv_recon_loss.mean()

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
    ):
        chunk_size = self.chunk_size
        batch, seq_len = seq.shape[:2]

        seq = self.retrieve_norm(seq)

        assert seq_len >= chunk_size

        seq = seq[:, (chunk_size - 1):]
        curtailed_seq_len = seq.shape[-2]

        next_seq_len = round_up_multiple(curtailed_seq_len, chunk_size)

        padding = next_seq_len - curtailed_seq_len
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        if exists(past_weights):
            past_weights = TensorDict(past_weights)
            assert past_weights.keys() == curr_weights.keys()

            curr_weights = curr_weights + past_weights

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_heads(queries)

        # fetch values from memory model

        curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        queries = rearrange(queries, 'b (n c) d -> (b n) c d', c = chunk_size)

        # forward functional call

        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # reconstitute batch dimension

        values = rearrange(values, '(b h n) c d -> b h (n c) d', b = batch, h = self.heads)

        values = self.multihead_rmsnorm(values)

        # maybe gate

        if exists(self.retrieve_gate):
            values = values * self.retrieve_gate(seq)

        # maybe merge heads and combine

        values = self.merge_heads(values)

        values = self.combine_heads(values)

        # restore, pad with empty memory embed

        empty_memory_embeds = self.init_empty_memory_embed(values.shape[0], chunk_size - 1)
        values = torch.cat((empty_memory_embeds, values), dim = -2)

        return values[:, :seq_len]

    def forward(
        self,
        seq,
        store_seq = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        return_aux_kv_loss = False
    ):
        batch, seq_len = seq.shape[:2]

        if seq_len < self.chunk_size:
            return self.init_empty_memory_embed(batch, seq_len)

        if exists(past_state):
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            past_state = self.init_weights_and_momentum()

        store_seq = default(store_seq, seq)

        updates, next_memories, aux_kv_recon_loss = self.store_memories(store_seq, past_state, return_aux_kv_loss = True)

        past_weights, _ = past_state

        retrieved = self.retrieve_memories(seq, past_weights + updates)

        if not return_aux_kv_loss:
            return retrieved

        return retrieved, aux_kv_recon_loss
