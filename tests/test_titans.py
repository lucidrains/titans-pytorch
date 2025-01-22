import torch
from torch import nn

import pytest
from titans_pytorch import NeuralMemory
from titans_pytorch.mac_transformer import flex_attention, SegmentedAttention, MemoryAsContextTransformer

def exists(v):
    return v is not None

@pytest.mark.parametrize('seq_len', (32, 1024, 77))
@pytest.mark.parametrize('silu', (False, True))
@pytest.mark.parametrize('learned_mem_model_weights', (False, True))
@pytest.mark.parametrize('attn_pool_chunks', (False, True))
@pytest.mark.parametrize('momentum', (False, True))
@pytest.mark.parametrize('qk_rmsnorm', (False, True))
@pytest.mark.parametrize('max_grad_norm', (None, 2.))
@pytest.mark.parametrize('per_parameter_lr_modulation', (False, True))
def test_titans(
    seq_len,
    silu,
    learned_mem_model_weights,
    attn_pool_chunks,
    momentum,
    qk_rmsnorm,
    max_grad_norm,
    per_parameter_lr_modulation
):
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        activation = nn.SiLU() if silu else None,
        attn_pool_chunks = attn_pool_chunks,
        max_grad_norm = max_grad_norm,
        momentum = momentum,
        qk_rmsnorm = qk_rmsnorm,
        per_parameter_lr_modulation = per_parameter_lr_modulation,
        learned_mem_model_weights = learned_mem_model_weights
    )

    seq = torch.randn(2, seq_len, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_titans_attn_memory():
    from titans_pytorch.titans import MemoryAttention

    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
        model = MemoryAttention(
            dim = 384
        )
    )

    seq = torch.randn(2, 1024, 384)
    retrieved = mem(seq)

    assert seq.shape == retrieved.shape

def test_retrieve_store_diff_seq():
    mem = NeuralMemory(
        dim = 384,
        chunk_size = (64, 32),
    )

    retrieve_seq = torch.randn(2, 64 * 64, 384)
    store_seq = torch.randn(2, 64 * 32, 384)

    retrieved = mem(retrieve_seq, store_seq = store_seq)

    assert retrieve_seq.shape == retrieved.shape

def test_overriding_chunk_size():
    mem = NeuralMemory(
        dim = 384,
        chunk_size = 64,
    )

    seq = torch.randn(2, 128 * 16, 384)
    store_seq = torch.randn(2, 128 * 8, 384)

    retrieved = mem(seq, store_seq, chunk_size = 16, store_chunk_size = 8)

    assert seq.shape == retrieved.shape

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('num_persist_mem_tokens', (0, 16))
@pytest.mark.parametrize('num_longterm_mem_tokens', (0, 16))
@pytest.mark.parametrize('neural_mem_gate_attn_output', (False, True))
def test_mac(
    seq_len,
    num_persist_mem_tokens,
    num_longterm_mem_tokens,
    neural_mem_gate_attn_output
):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        num_persist_mem_tokens = num_persist_mem_tokens,
        num_longterm_mem_tokens = num_longterm_mem_tokens,
        segment_len = 128,
        neural_mem_gate_attn_output = neural_mem_gate_attn_output
    )

    x = torch.randint(0, 256, (1, seq_len))

    logits = transformer(x)
    assert logits.shape == (1, seq_len, 256)

@pytest.mark.parametrize('sliding', (False, True))
def test_mac_sampling(sliding):
    transformer = MemoryAsContextTransformer(
        num_tokens = 256,
        dim = 256,
        depth = 2,
        segment_len = 32,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 0,
        sliding_window_attn = sliding,
        neural_memory_layers = (),
        neural_mem_gate_attn_output = False
    )

    ids = torch.randint(0, 256, (1, 1023))

    # after much training

    sampled = transformer.sample(ids[:, :4], 53, use_cache = False, temperature = 0.)
    sampled_with_cache = transformer.sample(ids[:, :4], 53, use_cache = True, temperature = 0.)

    assert torch.allclose(sampled, sampled_with_cache)

@pytest.mark.parametrize('seq_len', (1023, 17))
@pytest.mark.parametrize('sliding', (True, False))
def test_flex(
    seq_len,
    sliding
):
    if not (torch.cuda.is_available() and exists(flex_attention)):
        pytest.skip()

    attn = SegmentedAttention(
        dim = 512,
        segment_len = 32,
        num_persist_mem_tokens = 1,
        num_longterm_mem_tokens = 1,
        use_flex_attn = True,
        sliding = sliding
    ).cuda()

    seq = torch.randn(1, seq_len, 512).cuda()

    out_flex, _ = attn(seq)
    out_non_flex, _ = attn(seq, disable_flex_attn = True)

    assert torch.allclose(out_flex, out_non_flex, atol = 1e-5)
