from __future__ import annotations
from typing import Callable

import math
from functools import partial
from itertools import zip_longest
from collections import namedtuple

import torch
from torch import nn, stack, cat, is_tensor, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module, Parameter, ParameterList, ParameterDict
from torch.func import functional_call, vmap, grad
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from tensordict import TensorDict

from assoc_scan import AssocScan

from titans_pytorch.memory_models import(
    MemoryMLP,
    ResidualNorm
)

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
h - heads
bh - batch and heads
n - sequence
d - feature dimension
c - intra-chunk
w - num memory network weight parameters
o - momentum orders
u - key / value updates - allowing a token to emit multiple key / values
"""

LinearNoBias = partial(Linear, bias = False)

# neural mem state related

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    state = tree_map(lambda t: t.detach() if is_tensor(t) else t, tuple(state))
    return NeuralMemState(*state)

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

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

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

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

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

# spectral norming the surprise update w/ newton schulz matrix iter
# Keller Jordan et al. from OSS w/ nanogpt, now being used for two works, Atlas and 'TTT done right'

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)

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
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1, # whether a single token can do multiple updates to the memory model
        qkv_receives_diff_views = False, # to address an issue raised by a phd student (who will be credited if experiments are green). basically the issue raised is that the memory MLP is only learning Wk @ Wv linear mapping and that may not be expressive enough. we will use hyper connections to allow the network to choose different previous layer inputs as keys / values and see if that does anything
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        spectral_norm_surprises = False,
        gated_transition = False,
        mem_model_norm_add_residual = True, # by default, layernorm output and add residual as proposed in TTT paper, but could be removed
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
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

        # key values receiving different views

        self.qkv_receives_diff_views = qkv_receives_diff_views

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
        self.split_kv_heads = Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)

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

        if mem_model_norm_add_residual:
            model = ResidualNorm(dim = dim_head, model = model)

        self.memory_model = model

        mem_model_params = dict(model.named_parameters())

        self.num_memory_parameter_tensors = len(mem_model_params)

        self.memory_model_parameter_names = [*mem_model_params.keys()]

        memory_model_parameters = [*mem_model_params.values()]

        if per_head_learned_parameters:
            memory_model_parameters = [repeat(p, '... -> h ...', h = heads) for p in memory_model_parameters]

        self.init_weight_shape = [p.shape for p in memory_model_parameters]

        self.memory_model_parameters = ParameterList(memory_model_parameters)
        self.per_head_learned_parameters = per_head_learned_parameters

        # the chunk size within the paper where adaptive step, momentum, weight decay are shared

        self.chunk_size = chunk_size

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, loss_weights, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) - v|²
            weighted_loss = loss * loss_weights
            return weighted_loss.sum(), loss

        # two functions

        grad_fn = grad(forward_and_loss, has_aux = True)

        self.per_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))

        # queries for retrieving from the model

        self.to_queries = Sequential(LinearNoBias(dim, dim_inner), activation)

        # keys and values for storing to the model

        assert num_kv_per_token > 0

        self.to_keys = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.to_values = Sequential(
            LinearNoBias(dim, dim_inner * num_kv_per_token),
            activation,
        )

        self.store_memory_loss_fn = store_memory_loss_fn

        self.num_kv_per_token = num_kv_per_token

        # `chunk_size` refers to chunk size used for storing to memory model weights

        chunk_size = self.store_chunk_size

        # whether to use averaging of chunks, or attention pooling

        assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

        if not attn_pool_chunks:
            self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
        else:
            self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)

        # learned adaptive learning rate

        self.to_adaptive_step = Sequential(
            nn.Linear(dim, heads * num_kv_per_token),
            Rearrange('b n (h u) -> (b h) (n u)', u = num_kv_per_token)
        )

        if not exists(adaptive_step_transform):
            adaptive_step_transform = partial(default_adaptive_step_transform, max_lr = default_step_transform_max_lr)

        self.adaptive_step_transform = adaptive_step_transform

        # momentum related

        self.to_momentum = Sequential(
            nn.Linear(dim, heads * momentum_order),
            Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
        ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = Sequential(
                nn.Linear(dim, heads * momentum_order),
                Rearrange('b n (h o) -> o (b h) n', h = heads),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        # per layer learning rate modulation

        self.to_layer_modulation = Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        # learned weight residual

        self.to_learned_weight_residual_mix = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        # allow for softclamp the gradient norms for storing memories

        self.max_grad_norm = max_grad_norm

        # spectral norming the surprises before update, a la Muon from Jordan et al.

        self.spectral_norm_surprises = spectral_norm_surprises

        # weight decay factor

        self.to_decay_factor = Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> (b h) n 1')
        )

        # learned transition, as seeing instability when decreasing neural mem batch size
        # perhaps it can slowly learn to adjust from early residual to fully transitioning to new weights every batch size

        self.transition_gate = nn.Parameter(tensor(-5.)) if gated_transition else None

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

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(
        self,
        batch,
    ):
        if self.per_head_learned_parameters:
            weights = repeat_dict_values(self.memory_model_parameter_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch * self.heads)

        return weights

    def init_momentum(
        self,
        batch,
    ):
        zeros = self.memory_model_parameter_dict.clone().zero_()

        if self.per_head_learned_parameters:
            zeros = repeat_dict_values(zeros, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = repeat_dict_values(zeros, '... -> o bh ...', bh = batch * self.heads, o = self.momentum_order)

        return zeros

    def store_memories(
        self,
        seq,
        weights: dict[str, Tensor] | None = None,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: Tensor | None = None,
        return_surprises = True
    ):
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        # shapes and variables

        heads, chunk_size, num_updates = self.heads, self.store_chunk_size, self.num_kv_per_token

        # curtail sequence by multiple of the chunk size
        # only a complete chunk of the sequence provides the memory for the next chunk

        round_down_seq_len = round_down_multiple(seq_len, chunk_size)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :], seq[..., round_down_seq_len:, :]

        next_seq_len_index = seq_index + round_down_seq_len

        # init weights if needed
        # weights of the memory network

        if not exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)

        # allow for neural memory of a previous layer to influence surprise of current layer

        weights_for_surprise = repeat_dict_values(weights, 'b ... -> b n ...', n = num_chunks)

        # initial norm

        seq = self.store_norm(seq)

        # handle keys and values coming from different sequences from hyper connection

        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        # derive learned hparams for optimization of memory network

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        # keys and values

        keys = self.to_keys(seq)
        values = self.to_values(values_seq)

        # maybe multi head

        keys, values = map(self.split_kv_heads, (keys, values))

        # maybe keys rmsnorm

        keys = self.k_norm(keys)

        # take care of chunking

        keys, values = tuple(rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates) for t in (keys, values))

        # adaptive lr

        adaptive_lr = rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_updates)

        # optionally a storing memories mask can be passed in. if False, will set the learning rate to 0. for those positions

        if exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)

            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        # maybe add previous layer weight

        assert xnor(exists(self.to_learned_weight_residual_mix), exists(prev_weights))

        if exists(prev_weights):

            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks

            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq)
                mix = rearrange(mix, 'b h n -> (b h) n')
                prev_weights = prev_weights.apply(lambda t: einx.multiply('bh n, bh n ... -> bh n ...', mix, t))

            weights_for_surprise = weights_for_surprise + prev_weights

        # flatten batch and time if surprise depends on previous layer memory model

        weights_for_surprise = rearrange_dict_values(weights_for_surprise, 'b n ... -> (b n) ...')

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, unweighted_mem_model_loss = self.per_sample_grad_fn(dict(weights_for_surprise), keys, adaptive_lr, values)

        grads = TensorDict(grads)

        # surprises

        adaptive_lr = rearrange(adaptive_lr, '(b h n) c -> b h (n c)', b = batch, h = heads)
        unweighted_mem_model_loss = rearrange(unweighted_mem_model_loss, '(b h n) c -> b h (n c)', b = batch, h = heads)

        # maybe softclamp grad norm

        if exists(self.max_grad_norm):
            grads = grads.apply(lambda t: softclamp_grad_norm(t, self.max_grad_norm))

        # restore batch and sequence dimension

        grads = rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)

        # maybe per layer modulation

        if need_layer_lr_mod:
            grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

        # negative gradients, adaptive lr already applied as loss weight

        surprises = grads.mul(-1)

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
            next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)

            output = (updates, next_store_state)

            if not return_surprises:
                return output

            return (*output, (unweighted_mem_model_loss, adaptive_lr))

        # momentum + weight decay - momentum is the new contribution, as most linear RNNs have learned forgetting gates

        updates = TensorDict()

        next_last_update = TensorDict()
        next_last_momentum = TensorDict()

        for (param_name, surprise), (_, last_update) in zip(surprises.items(), past_last_update.items()):

            update = surprise

            # derive momentum with associative scan - eq (10)

            if has_momentum:
                momentum = surprise

                momentums = [] # stores all momentum orders starting with first, to generalize to Nth order momentum

                last_momentum = past_last_momentum[param_name]

                # go from first order momentum all the way to the Nth

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum) # momentum is S / surprise in the paper

                    momentums.append(momentum)

                momentums = stack(momentums)

                next_last_momentum[param_name] = momentums[:, :, -1] # momentums shape is Float['o bh n 1']

                if learned_combine and self.learned_combine_include_zeroth:
                    # add the original surprise if learned combination of momentums
                    momentums = cat((rearrange(surprise, '... -> 1 ...'), momentums), dim = 0)

                if not learned_combine:
                    update = momentums[-1]
                else:
                    update = einsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')

            # maybe spectral norm surprises

            if self.spectral_norm_surprises:
                update = newtonschulz5(update)

            # use associative scan again for learned forgetting (weight decay) - eq (13)

            update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)

            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        # determine next state for the storing of memories

        next_state = (next_last_update, next_last_momentum)

        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        # return updates to neural memory at all chunked timesteps + neural mem cache / state to be fed back

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: dict[str, Tensor],
    ):
        chunk_size = self.retrieve_chunk_size

        weights_have_expanded_shape = dict_get_value_shapes(weights) != self.init_weight_shape

        batch, seq_len = seq.shape[:2]

        # auto infer single token decoding, if there are only 1 set of weights and 1 token

        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1

        is_single_token_decode = is_one_token and is_one_weight

        if is_single_token_decode:
            chunk_size = 1

        # padding related, for chunked processing

        need_pad = chunk_size > 1 or not is_one_weight

        if need_pad:
            seq = pad_at_dim(seq, (1, 0), dim = 1)

        seq_len_plus_one = seq.shape[-2]

        next_seq_len = round_up_multiple(seq_len_plus_one, chunk_size)

        padding = next_seq_len - seq_len_plus_one
        seq = pad_at_dim(seq, (0, padding), dim = 1)

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        weights = TensorDict(weights)

        # pre norm

        seq = self.retrieve_norm(seq)

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # maybe multihead

        queries = self.split_heads(queries)

        # maybe qk rmsnorm

        queries = self.q_norm(queries)

        # fetch values from memory model

        if weights_have_expanded_shape:
            weights = rearrange_dict_values(weights, 'b n ... -> (b n) ...')

        queries = rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)

        # forward functional call

        values = functional_call(self.memory_model, dict(weights), queries)

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

        if need_pad:
            values = values[:, 1:]

        return values[:, :seq_len]

    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None
    ):
        is_multi_input = self.qkv_receives_diff_views

        # handle single token

        if seq.ndim == 2 or (is_multi_input and seq.ndim == 3):
            seq = rearrange(seq, '... b d -> ... b 1 d')

        is_single_token = seq.shape[-2] == 1

        # if different views for qkv, then

        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        # handle previous state init

        if not exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state

        # store

        store_seq = default(store_seq, seq)

        # take care of cache

        if exists(cache_store_seq):
            store_seq = safe_cat((cache_store_seq, store_seq))

        # compute split sizes of sequence
        # for now manually update weights to last update at the correct boundaries

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, default(ttt_batch_size, self.batch_size)

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

        if exists(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)

        # whether to allow network to slowly adjust from initial weight throughout (residual path) to fully updating weights every batch

        surprises = (None, None)
        gate = None

        if exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index = seq_index,
                past_state = past_state,
                prev_weights = prev_weights,
                mask = maybe_store_mask,
                return_surprises = True
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states

            updates = accum_updates(updates, next_updates)

            surprises = tuple(safe_cat(args, dim = -1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            # update weights once batch size is fulfilled

            last_update, last_momentum = past_state

            if exists(gate):
                last_update = TensorDict({param_name: one_weight.lerp(one_last_update, gate) for (param_name, one_weight), (_, one_last_update) in zip(weights.items(), last_update.items())})

            past_state = (last_update, last_momentum)

            # set weights to the last updated weights for the last minibatch

            weights = last_update

            next_neural_mem_state = next_neural_mem_state._replace(
                weights = weights,
                states = past_state,
            )

        next_neural_mem_state = next_neural_mem_state._replace(updates = updates)

        # retrieve

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = rearrange_dict_values(last_update, 'b ... -> b 1 ...')

        retrieved = self.retrieve_memories(
            retrieve_seq,
            updates
        )

        # maybe detach

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        # returning

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
