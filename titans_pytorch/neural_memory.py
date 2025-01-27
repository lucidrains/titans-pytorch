from __future__ import annotations
from typing import Callable

import math
from functools import partial
from collections import namedtuple

import torch
from torch import nn, cat, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList
from torch.func import functional_call, vmap, grad

from tensordict import TensorDict

from titans_pytorch.associative_scan import (
    associative_scan,
    binary_operator,
    pad_at_dim
)

from titans_pytorch.memory_models import(
    MemoryMLP
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

NeuralMemCache = namedtuple('NeuralMemCache', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

# functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_shape(td):
    return {k: v.shape for k, v in td.items()}

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

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
    if is_empty_tensor(t):
        return t

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

# associative scan wrapper

class AssocScan(Module):
    def __init__(
        self,
        use_accelerated = False
    ):
        super().__init__()
        self.use_accelerated = use_accelerated

    def forward(
        self,
        gates,
        inputs,
        prev = None,
        remove_prev = None
    ):
        remove_prev = default(remove_prev, exists(prev))

        inputs, inverse_pack_weight_shape = pack_one_with_inverse(inputs, 'b n *')
        gates, _ = pack_one_with_inverse(gates, 'b n *')

        if exists(prev):
            prev, _ = pack_one_with_inverse(prev, 'b *')

        if exists(prev):
            inputs, _ = pack([prev, inputs], 'b * d')
            gates = pad_at_dim(gates, (1, 0), value = 1., dim = -2)

        if not self.use_accelerated:
            _, out = associative_scan(binary_operator, (gates, inputs))

            if remove_prev:
                out = out[:, 1:]

            return inverse_pack_weight_shape(out)

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

        out = accelerate_scan_fn(gates, inputs)

        if remove_prev:
            out = out[:, 1:]

        return inverse_pack_weight_shape(out)

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
        batch_size = None,
        dim_head = None,
        heads = 1,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        per_parameter_lr_modulation = False, # allow outer network to control learning rate per weight matrix of memory network
        max_mem_layer_modulation = 1., # max of 10.
        attn_pool_chunks = False,
        momentum = True,
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        default_model_kwargs: dict = dict(
            depth = 2
        )
    ):
        super().__init__()
        dim_head = default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)

        self.retrieve_chunk_size, self.store_chunk_size = pair(chunk_size)

        # batch size

        if exists(batch_size):
            assert divisible_by(batch_size, self.store_chunk_size)

        self.batch_size = batch_size

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

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrieve_gate = Sequential(
            LinearNoBias(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        # memory model

        if not exists(model):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        # validate memory model

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')

            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

        # the memory is the weights of the model

        self.memory_model = model

        self.num_memory_parameter_tensors = len(set(model.parameters()))

        self.init_weight_shape = dict_get_shape(dict(model.named_parameters()))

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum()

        # two functions

        grad_fn = grad(forward_and_loss)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        self.to_keys_values = Sequential(LinearNoBias(dim, dim_inner * 2), activation)
        self.store_memory_loss_fn = store_memory_loss_fn

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
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        ) if momentum else None

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n')
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # weight decay factor

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # inits

        if exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_adaptive_step_bias)

        if exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        # maybe use accelerated scan

        self.use_accelerated_scan = use_accelerated_scan

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def init_weights(
        self,
        batch,
    ):
        weights = TensorDict(dict(self.memory_model.named_parameters()))
        weights = repeat_dict_values(weights, '... -> bh ...', bh = batch * self.heads)
        return weights

    def init_momentum(
        self,
        batch,
    ):
        weights = TensorDict(dict(self.memory_model.named_parameters()))
        zeros = weights.clone().zero_()
        zeros = repeat_dict_values(zeros, '... -> bh ...', bh = batch * self.heads)
        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0
    ):
        batch, seq_len, heads, chunk_size = *seq.shape[:2], self.heads, self.store_chunk_size

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[:, :round_down_seq_len], seq[:, round_down_seq_len:]

        next_seq_len_index = seq_index + round_down_seq_len

        # init weights if needed
        # weights of the memory network

        if not exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)

        # allow for neural memory of a previous layer to influence surprise of current layer

        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)

        # derive learned hparams for optimization of memory network

        seq = self.store_norm(seq)

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
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

        keys, values = tuple(rearrange(t, 'b h (n c) d -> (b h n) c d', c = chunk_size) for t in (keys, values))

        adaptive_lr = rearrange(adaptive_lr, 'b (n c) -> (b n) c', c = chunk_size)

        # flatten batch and time if surprise depends on previous layer memory model

        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.apply(lambda t: -t)

        # past states

        if not exists(past_state):
            # minibatch_init_weight corresponds to W0 in figure 7 of TTT paper

            minibatch_init_weight = weights
            init_momentum = self.init_momentum(batch)

            past_state = (minibatch_init_weight, init_momentum)

        past_last_update, past_last_momentum = past_state

        # early return if sequence length less than chunk size

        if num_chunks == 0:
            updates = rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
            next_store_state = NeuralMemCache(next_seq_len_index, weights, remainder, past_state, updates)

            output = (updates, next_store_state)

            return output

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        next_momentum = TensorDict() if has_momentum else None
        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update), (_, last_momentum) in zip(surprises.items(), past_last_update.items(), past_last_momentum.items()):

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                update = self.assoc_scan(adaptive_momentum, surprise, prev = last_momentum) # momentum is S / surprise in the paper
                momentum = update
                next_last_momentum[param_name] = momentum[:, -1]

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)
            next_last_update[param_name] = update[:, -1]

            updates[param_name] = update

            if has_momentum:
                next_momentum[param_name] = momentum

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        next_store_state = NeuralMemCache(next_seq_len_index, weights, remainder, next_state, updates)

        # returns

        output = (updates, next_store_state)

        return output

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor],
    ):
        chunk_size = self.retrieve_chunk_size
        batch, seq_len = seq.shape[:2]

        seq = self.retrieve_norm(seq)

        needs_pad = chunk_size > 1

        seq = pad_at_dim(seq, (1, 0), dim = 1)
        seq_len_plus_one = seq.shape[-2]

        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)

        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        curr_weights = TensorDict(past_weights)

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

        # fetch values from memory model

        if dict_get_shape(curr_weights) != self.init_weight_shape:
            curr_weights = rearrange_dict_values(curr_weights, 'b n ... -> (b n) ...')

        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

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

        values = values[:, 1:(seq_len + 1)]

        return values

    @torch.no_grad()
    def forward_inference(
        self,
        token: Tensor,
        state: NeuralMemCache | None = None,
    ):
        # unpack previous state

        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_states, updates = state

        curr_seq_len = seq_index + 1
        batch = token.shape[0]

        if token.ndim == 2:
            token = rearrange(token, 'b d -> b 1 d')

        assert token.shape[1] == 1

        # increment the sequence cache which is at most the chunk size

        cache_store_seq = safe_cat((cache_store_seq, token), dim = -2)

        # early return empty memory, when no memories are stored for steps < first chunk size

        if curr_seq_len < self.chunk_size:
            retrieve = self.retrieve_memories(token, weights, chunk_size = 1)

            output = retrieve, NeuralMemCache(curr_seq_len, weights, cache_store_seq, past_states, updates)

            return output

        # store if storage sequence cache hits the chunk size

        next_states = past_states
        store_seq_cache_len = cache_store_seq.shape[-2]

        if not exists(updates):
            updates = weights.clone().zero_()
            updates = repeat_dict_values(updates, '... -> b 1 ...', b = batch)
        else:
            updates = updates.apply(lambda t: t[:, -1:])

        if store_seq_cache_len == self.chunk_size:

            next_updates, store_state = self.store_memories(
                cache_store_seq,
                weights,
                past_state = past_states,
            )

            updates = next_updates
            cache_store_seq = None
            next_states = store_state.states

        # retrieve

        retrieved = self.retrieve_memories(token, updates, chunk_size = 1)

        # next state tuple

        next_store_state = NeuralMemCache(curr_seq_len, weights, cache_store_seq, next_states, updates)

        return retrieved, next_store_state

    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemCache | None = None,
    ):
        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state

        assert not exists(cache_store_seq) or is_empty_tensor(cache_store_seq)

        # store

        store_seq = default(store_seq, seq)

        # functions

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, self.batch_size

        need_update_weights = exists(batch_size)

        # determine split sizes and when to update

        if need_update_weights:
            update_after_final_store = divisible_by(seq_index + store_seq_len, batch_size)

            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = divisible_by(seq_range, batch_size)

            indices = seq_range[batch_boundary] - seq_index

            indices = F.pad(indices, (1, 0), value = 0)

            if indices[-1] != store_seq_len:
                indices = F.pad(indices, (0, 1), value = store_seq_len)

            split_sizes = (indices[1:] - indices[:-1]).tolist()

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False

        # accumulate updates

        updates = None

        def accum_updates(past_updates, future_updates):
            if not exists(past_updates):
                return future_updates

            return TensorDict({param_name: cat((past_update[:, :-1], future_update), dim = 1) for (param_name, past_update), (_, future_update) in zip(past_updates.items(), future_updates.items())})

        # loop through chunks of store sequences

        store_seqs = store_seq.split(split_sizes, dim = -2)

        for ind, store_seq_chunk in enumerate(store_seqs):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index = seq_index,
                past_state = past_state,
            )

            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states

            updates = accum_updates(updates, next_updates)

            if is_last and not update_after_final_store:
                continue

            # update weights once batch size is fulfilled

            last_update, _ = past_state

            weights = last_update

            next_neural_mem_state = list(next_neural_mem_state)
            next_neural_mem_state[1] = last_update
            next_neural_mem_state = NeuralMemCache(*next_neural_mem_state)

        # retrieve

        retrieved = self.retrieve_memories(
            seq,
            updates
        )

        return retrieved, next_neural_mem_state
