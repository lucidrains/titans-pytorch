from __future__ import annotations
from typing import Callable

import math
from functools import partial

import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList
from torch.func import functional_call, vmap, grad

from tensordict import TensorDict

from titans_pytorch.associative_scan import (
    associative_scan,
    binary_operator,
    pad_at_dim
)

import einx
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
"""

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

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

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

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
        self.gamma = Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

# chunk pooling

class AveragePool(Module):
    def __init__(
        self,
        chunk_size
    ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)
        return reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        """
        taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.to_attn_logits = nn.Linear(dim, dim)

        # default to average pool

        nn.init.zeros_(self.to_attn_logits.weight)
        nn.init.zeros_(self.to_attn_logits.bias)

    def forward(
        self,
        x,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.chunk_size)

        x = rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)

        attn_logits = self.to_attn_logits(x)

        attn = attn_logits.softmax(dim = -2)

        return reduce(x * attn, 'b n c d -> b n d', 'sum')

# classes

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth
    ):
        super().__init__()
        self.weights = ParameterList([Parameter(torch.randn(dim, dim)) for _ in range(depth)])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

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

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):
        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.silu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):
        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.silu(x)

            x = x @ weight1 @ weight2

        return x

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # queries
            nn.Parameter(torch.randn(dim, dim)), # keys
            nn.Parameter(torch.randn(dim, dim)), # values
            nn.Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            nn.Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        wq, wk, wv, ffw1, ffw2 = self.weights

        q = F.normalize(x @ wq, dim = -1)
        k = F.normalize(x @ wk, dim = -1)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        x = x + attn_out

        h = F.silu(x @ ffw1)
        out = h @ ffw2

        return out

# associative scan wrapper

class AssocScan(Module):
    def __init__(
        self,
        use_accelerated = False
    ):
        super().__init__()
        self.use_accelerated = use_accelerated

    def forward(self, gates, inputs):

        if not self.use_accelerated:
            _, outputs = associative_scan(binary_operator, (gates, inputs))
            return outputs

        from accelerated_scan.triton import scan as triton_scan
        from accelerated_scan.warp import scan as warp_scan

        scan = triton_scan if gates.is_cuda else warp_scan

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

        return accelerate_scan_fn(gates, inputs)

# main neural memory

def default_adaptive_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1)

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1e-2,
        per_parameter_lr_modulation = False, # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation = 1e1, # max of 10.
        attn_pool_chunks = False,
        momentum = True,
        pre_rmsnorm = True,
        post_rmsnorm = True,
        qk_rmsnorm = False,
        learned_mem_model_weights = True,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        default_model_kwargs: dict = dict(
            depth = 2
        )
    ):
        super().__init__()
        dim_head = default(dim_head, dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # associative scan

        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)

        # norms

        self.retrieve_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        self.multihead_rmsnorm = MultiheadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()

        self.q_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiheadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()

        # maybe multi-headed

        dim_inner = dim_head * heads

        self.heads = heads

        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        # memory mlp

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        if not learned_mem_model_weights:
            model.requires_grad_(False)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # the memory is the weights of the model

        self.memory_model = model

        self.num_memory_parameter_tensors = len(set(model.parameters()))

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

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        self.to_keys_values = Sequential(LinearNoBias(dim, dim_inner * 2), activation)
        self.store_memory_loss_fn = store_memory_loss_fn

        # empty memory embed

        self.empty_memory_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.empty_memory_embed, std = 0.02)

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate and momentum

        self.to_momentum = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        ) if momentum else None

        self.to_adaptive_step = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n')
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            LinearNoBias(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # weight decay factor

        self.to_decay_factor = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def init_weights_and_momentum(self, zero_weights = False):
        params = TensorDict(dict(self.memory_model.named_parameters()))

        init_weights = params
        init_momentum = params.clone().zero_()

        if zero_weights:
            init_weights = params.clone().zero_()

        return init_weights, init_momentum

    def init_empty_memory_embed(self, batch, seq_len):
        return repeat(self.empty_memory_embed, 'd -> b n d', b = batch, n = seq_len)

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]],
        return_aux_kv_loss = False,
        chunk_size = None
    ):
        seq_len, chunk_size = seq.shape[-2], default(chunk_size, self.store_chunk_size)

        # handle edge case

        if seq_len < chunk_size:
            past_weight, _ = past_state
            return TensorDict(past_weight).clone().zero_(), self.zero

        seq = self.store_norm(seq)

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)

        seq = seq[:, :round_down_seq_len]

        # get the weights of the memory network

        past_state = tuple(TensorDict(d) for d in past_state)
        curr_weights, past_momentum = past_state

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation)
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # keys and values

        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # maybe multi head

        keys, values = map(self.split_heads, (keys, values))

        batch = keys.shape[0]

        # maybe qk rmsnorm

        keys = self.k_norm(keys)

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b (n c) d -> (b n) c d', c = chunk_size) for t in (keys, values))

        adaptive_lr = rearrange(adaptive_lr, 'b (n c) -> (b n) c', c = chunk_size)

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, aux_kv_recon_loss = self.per_sample_grad_fn(dict(curr_weights), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.apply(lambda t: -t)

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        next_momentum = TensorDict() if has_momentum else None
        updates = TensorDict()

        for param_name, surprise in surprises.items():

            surprise, inverse_pack = pack_one_with_inverse(surprise, 'b n *')

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                update = self.assoc_scan(adaptive_momentum, surprise) # momentum is S / surprise in the paper
                momentum = update

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1. - decay_factor, update)

            updates[param_name] = inverse_pack(update)

            if has_momentum:
                next_momentum[param_name] = inverse_pack(momentum)

        # compute the next weight per batch

        last_update = updates.apply(lambda t: t[:, -1])

        if not return_aux_kv_loss:
            return updates

        return updates, aux_kv_recon_loss.mean()

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
        chunk_size = None
    ):
        chunk_size = default(chunk_size, self.retrieve_chunk_size)
        batch, seq_len = seq.shape[:2]

        seq = self.retrieve_norm(seq)

        if seq_len < chunk_size:
            return self.init_empty_memory_embed(batch, seq_len)

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

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

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
        return_aux_kv_loss = False,
        chunk_size = None,
        store_chunk_size = None
    ):
        batch, seq_len = seq.shape[:2]

        if seq_len < self.retrieve_chunk_size:
            out = self.init_empty_memory_embed(batch, seq_len)

            if not return_aux_kv_loss:
                return out

            return out, self.zero

        if exists(past_state):
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            past_state = self.init_weights_and_momentum()

        store_seq = default(store_seq, seq)
        store_chunk_size = default(store_chunk_size, chunk_size)

        updates, aux_kv_recon_loss = self.store_memories(store_seq, past_state, chunk_size = store_chunk_size, return_aux_kv_loss = True)

        past_weights, _ = past_state

        retrieved = self.retrieve_memories(seq, past_weights + updates, chunk_size = chunk_size)

        if not return_aux_kv_loss:
            return retrieved

        return retrieved, aux_kv_recon_loss
