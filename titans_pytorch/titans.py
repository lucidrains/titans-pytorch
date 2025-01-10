from __future__ import annotations
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.func import functional_call, vmap, grad_and_value

from tensordict import TensorDict

from titans_pytorch.associative_scan import (
    associative_scan,
    binary_operator
)

import einx
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

"""
ein notation:
b - batch
n - sequence
d - feature dimension
c - intra-chunk
"""

# constants

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_multiple(seq, mult):
    return seq // mult * mult

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

# classes

class MLP(Module):
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

# main neural memory

def default_loss_fn(pred, target):
    return (pred - target).pow(2).mean(dim = -1).sum()

class NeuralMemory(Module):
    def __init__(
        self,
        dim,
        model: Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn
    ):
        super().__init__()

        if not exists(model):
            model = MLP(dim, depth = 4)

        assert not exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        # the memory is the weights of the model

        self.memory_model = model

        # prepare function for per sample gradients from model above, using torch.func

        def forward_and_loss(params, inputs, target):
            pred = functional_call(self.memory_model, params, inputs)
            loss = self.store_memory_loss_fn(pred, target) # simple mse loss in paper - eq (12) - |M(k) == v|²
            return loss

        self.per_sample_grad_and_value_fn = vmap(grad_and_value(forward_and_loss), in_dims = (None, 0, 0))

        # queries for retrieving from the model

        self.to_queries = LinearNoBias(dim, dim)

        # keys and values for storing to the model

        self.to_keys_values = LinearNoBias(dim, dim * 2)
        self.store_memory_loss_fn = store_memory_loss_fn

        # learned adaptive learning rate and momentum
        # todo - explore mlp layerwise learned lr / momentum

        self.to_momentum = LinearNoBias(dim, 1)
        self.to_adaptive_step = nn.Sequential(LinearNoBias(dim, 1), Rearrange('... 1 -> ...'))
        self.to_decay_factor = LinearNoBias(dim, 1) # weight decay factor

    def init_weights_and_momentum(self):
        params = TensorDict(dict(self.memory_model.named_parameters()))

        init_weights = params.clone().zero_()
        init_momentum = params.clone().zero_()

        return init_weights, init_momentum

    def store_memories(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]]
    ):

        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        past_state = tuple(TensorDict(d) for d in past_state)
        past_weights, past_momentum = past_state

        curr_weights = curr_weights + past_weights

        # pack batch and sequence dimension

        batch = seq.shape[0]

        adaptive_lr = self.to_adaptive_step(seq).tanh() * 0.5 + 0.5.

        adaptive_momentum = self.to_momentum(seq).sigmoid()
        decay_factor = self.to_decay_factor(seq).sigmoid()

        # keys and values

        seq = rearrange(seq, 'b n d -> (b n) d')
        keys, values = self.to_keys_values(seq).chunk(2, dim = -1)

        # get grads and extra auxiliary loss (for backwarding through qkv projection in base neural memory module)

        grads, aux_store_loss = self.per_sample_grad_and_value_fn(dict(curr_weights), keys, values)

        grads = TensorDict(grads)

        # restore batch and sequence dimension

        grads = grads.apply(lambda t: rearrange(t, '(b n) ... -> b n ...', b = batch))

        # multiply gradients with learned adaptive step size

        surprises = grads.apply(lambda t: einx.multiply('b n ..., b n -> b n ...', t, -adaptive_lr))

        # derive momentum with associative scan - eq (10)

        next_momentum = TensorDict()

        for param_name, surprise in surprises.items():
            surprise, inverse_pack = pack_one_with_inverse(surprise, 'b n *')

            _, momentum = associative_scan(binary_operator, (adaptive_momentum, surprise)) # momentum is S / surprise in the paper

            momentum = inverse_pack(momentum)

            next_momentum[param_name] = momentum

        # use associative scan again for learned forgetting (weight decay) - eq (13)

        updates = TensorDict()

        for param_name, momentum in next_momentum.items():
            momentum, inverse_pack = pack_one_with_inverse(momentum, 'b n *')

            _, update = associative_scan(binary_operator, (1. - decay_factor, momentum)) # momentum is S / surprise in the paper

            update = inverse_pack(update)

            updates[param_name] = update

        # compute the next weight per batch

        last_update = updates.apply(lambda t: t[:, -1])

        next_state = (curr_weights + last_update, next_momentum)

        return updates, next_state, aux_store_loss.mean()

    def retrieve_memories(
        self,
        seq,
        past_weights: dict[str, Tensor] | None = None,
    ):
        batch = seq.shape[0]

        # the parameters of the memory model stores the memories of the key / values
        # when the MLP has only 1 weight matrix, it is equivalent to `kv` fast weight memories from linear attention literature (recall fetching of memories is q @ (kv)) / schmidhuber's paper

        curr_weights = TensorDict(dict(self.memory_model.named_parameters()))

        if exists(past_weights):
            past_weights = TensorDict(past_weights)
            assert past_weights.keys() == curr_weights.keys()

            curr_weights = curr_weights + past_weights

        # sequence Float['b n d'] to queries

        queries = self.to_queries(seq)

        # fetch values from memory model

        curr_weights = curr_weights.apply(lambda t: rearrange(t, 'b n ... -> (b n) ...'))
        queries = rearrange(queries, 'b n d -> (b n) 1 d')

        # forward functional call

        values = functional_call(self.memory_model, dict(curr_weights), queries)

        # reconstitute batch dimension

        values = rearrange(values, '(b n) 1 d -> b n d', b = batch)

        return values

    def forward(
        self,
        seq,
        past_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
        return_next_memories = False
    ):
        batch = seq.shape[0]

        if exists(past_state):
            past_state = tuple(TensorDict(d) for d in past_state)

        if not exists(past_state):
            past_state = self.init_weights_and_momentum()

        updates, next_memories, aux_kv_mse_loss = self.store_memories(seq, past_state)

        past_weights, _ = past_state

        retrieved = self.retrieve_memories(seq, past_weights + updates)

        if not return_next_memories:
            return retrieved

        return retrieved, next_memories, aux_kv_mse_loss
