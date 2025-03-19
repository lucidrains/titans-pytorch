import torch
from torch import nn, cat, chunk
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, Callable, List, Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field
from functools import partial, lru_cache
import math
import gc
import time
from contextlib import contextmanager
import warnings
import logging
from enum import Enum, auto
import threading
import queue
import os

from einops import rearrange, repeat, reduce

# Advanced optimizations detection
TORCH_2_AVAILABLE = torch.__version__ >= "2.0.0"
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")
XFORMERS_AVAILABLE = False
FLASH_ATTN_AVAILABLE = False
TRITON_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext")

# Try to import optional dependencies
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
    logger.info("xFormers available and enabled")
except ImportError:
    logger.info("xFormers not available")

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention available and enabled")
except ImportError:
    logger.info("Flash Attention not available")

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    logger.info("Triton available and enabled")
except ImportError:
    logger.info("Triton not available")

# Device detection
def get_available_devices():
    """Returns a list of available compute devices"""
    devices = []
    if torch.cuda.is_available():
        devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices

AVAILABLE_DEVICES = get_available_devices()
logger.info(f"Available devices: {AVAILABLE_DEVICES}")

# Performance monitoring utilities
@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    start = time.time()
    yield
    end = time.time()
    logger.debug(f"{name} took {end - start:.3f} seconds")

def estimate_memory_usage(tensor_sizes, dtype=torch.float16):
    """Estimate the memory usage of tensors with given sizes"""
    bytes_per_element = torch.empty(1, dtype=dtype).element_size()
    total_elements = sum(math.prod(size) for size in tensor_sizes)
    return total_elements * bytes_per_element / (1024 ** 3)  # Return in GB

def memory_stats():
    """Return current GPU memory usage stats"""
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        current_mem = torch.cuda.memory_allocated() / (1024 ** 3)
        return f"Max: {max_mem:.2f} GB, Current: {current_mem:.2f} GB"
    return "No CUDA available"

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Enhanced Performance Configuration
@dataclass
class PerformanceConfig:
    """Advanced configuration for performance optimizations"""
    # Precision options
    use_mixed_precision: bool = True
    default_dtype: torch.dtype = torch.float16
    compute_dtype: torch.dtype = torch.float32
    layernorm_dtype: torch.dtype = torch.float32
    
    # Memory management options
    use_gradient_checkpointing: bool = True
    optimize_memory: bool = True
    max_memory_usage: Optional[float] = None  # Maximum memory usage in GB
    
    # Acceleration options
    use_xformers: bool = XFORMERS_AVAILABLE
    use_flash_attention: bool = FLASH_ATTN_AVAILABLE
    use_triton: bool = TRITON_AVAILABLE
    use_torch_compile: bool = TORCH_COMPILE_AVAILABLE
    compile_mode: str = TORCH_COMPILE_MODE
    
    # Quantization options
    quantization: Optional[str] = None  # None, 'dynamic', 'static', 'qat', 'awq', 'gptq'
    quantization_bits: int = 8
    
    # Distribution options
    distribute_method: str = "auto"  # 'auto', 'tensor', 'pipeline', 'expert', 'none'
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Context window specific options
    use_sliding_window: bool = True
    sliding_window_size: int = 8192
    use_local_attention: bool = True
    local_attention_window: int = 4096
    use_sparse_attention: bool = True
    sparse_attention_pattern: str = "fixed"  # 'fixed', 'adaptive', 'random'
    
    # Algorithm options
    activation_function: str = "swiglu"  # 'gelu', 'swiglu', 'squared_relu', 'swish'
    normalization: str = "rms_norm"  # 'layer_norm', 'rms_norm'
    
    # Memory hierarchy options
    enable_memory_hierarchy: bool = True
    num_memory_levels: int = 3
    
    # Initialize based on environment
    def __post_init__(self):
        # Override settings based on environment variables
        env_var_mapping = {
            "ULTRACONTEXT_MIXED_PRECISION": ("use_mixed_precision", lambda x: x.lower() == "true"),
            "ULTRACONTEXT_GRADIENT_CHECKPOINTING": ("use_gradient_checkpointing", lambda x: x.lower() == "true"),
            "ULTRACONTEXT_MAX_MEMORY": ("max_memory_usage", float),
            "ULTRACONTEXT_QUANTIZATION": ("quantization", str),
            "ULTRACONTEXT_QUANTIZATION_BITS": ("quantization_bits", int),
            "ULTRACONTEXT_DISTRIBUTE": ("distribute_method", str),
            "ULTRACONTEXT_SLIDING_WINDOW": ("use_sliding_window", lambda x: x.lower() == "true"),
            "ULTRACONTEXT_SLIDING_WINDOW_SIZE": ("sliding_window_size", int),
            "ULTRACONTEXT_LOCAL_ATTENTION": ("use_local_attention", lambda x: x.lower() == "true"),
            "ULTRACONTEXT_LOCAL_WINDOW": ("local_attention_window", int),
            "ULTRACONTEXT_ACTIVATION": ("activation_function", str),
            "ULTRACONTEXT_NORM": ("normalization", str),
        }
        
        for env_var, (attr_name, conversion) in env_var_mapping.items():
            if env_var in os.environ:
                try:
                    setattr(self, attr_name, conversion(os.environ[env_var]))
                    logger.info(f"Set {attr_name} to {getattr(self, attr_name)} from environment")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {e}")
        
        # Adjust settings based on available hardware
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram < 16:  # Low VRAM (<16GB)
                self.sliding_window_size = min(self.sliding_window_size, 4096)
                self.local_attention_window = min(self.local_attention_window, 2048)
                if not self.quantization:
                    self.quantization = "dynamic"
            elif total_vram < 48:  # Medium VRAM (16-48GB)
                self.sliding_window_size = min(self.sliding_window_size, 16384)
                self.local_attention_window = min(self.local_attention_window, 8192)
            # High VRAM devices keep default settings
        else:
            # CPU-only optimizations
            self.quantization = "dynamic"
            self.sliding_window_size = min(self.sliding_window_size, 4096)
            self.local_attention_window = min(self.local_attention_window, 2048)
            self.use_mixed_precision = False

# Default performance configuration
DEFAULT_PERF_CONFIG = PerformanceConfig()

# Memory tracking and profiling
class MemoryTracker:
    """Tracks memory usage during model execution"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.checkpoints = {}
        self.peak_usage = 0
        
    def checkpoint(self, name):
        """Record a memory checkpoint"""
        if not self.enabled or not torch.cuda.is_available():
            return
        
        usage = torch.cuda.memory_allocated() / (1024**3)
        self.checkpoints[name] = usage
        self.peak_usage = max(self.peak_usage, usage)
        
    def report(self):
        """Generate a memory usage report"""
        if not self.enabled or not torch.cuda.is_available():
            return "Memory tracking disabled or CUDA unavailable"
        
        report_lines = ["Memory Usage Report:"]
        report_lines.append(f"Peak memory usage: {self.peak_usage:.2f} GB")
        report_lines.append("\nCheckpoints:")
        
        previous = 0
        for name, usage in sorted(self.checkpoints.items(), key=lambda x: x[1]):
            delta = usage - previous
            report_lines.append(f"  {name}: {usage:.2f} GB (Δ {delta:.2f} GB)")
            previous = usage
            
        return "\n".join(report_lines)

# Activation functions optimized for 100M context windows
class ActivationFunctions:
    """Optimized activation functions for large context windows"""
    @staticmethod
    def gelu(x):
        """Standard GELU activation"""
        return F.gelu(x)
    
    @staticmethod
    def swiglu(x, y):
        """SwiGLU activation - better for deeper models"""
        return x * F.silu(y)
    
    @staticmethod
    def squared_relu(x):
        """Squared ReLU - efficient and works well in transformers"""
        return torch.pow(F.relu(x), 2)
    
    @staticmethod
    def swish(x):
        """Swish activation"""
        return x * torch.sigmoid(x)
    
    @staticmethod
    def get_activation_fn(name: str) -> Callable:
        """Get activation function by name"""
        if name == "gelu":
            return ActivationFunctions.gelu
        elif name == "swiglu":
            return ActivationFunctions.swiglu
        elif name == "squared_relu":
            return ActivationFunctions.squared_relu
        elif name == "swish":
            return ActivationFunctions.swish
        else:
            raise ValueError(f"Unknown activation function: {name}")

# Optimized normalization layers
class RMSNorm(Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm"""
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))
        self.use_mixed_precision = perf_config.use_mixed_precision
        self.compute_dtype = perf_config.layernorm_dtype
        
        if TRITON_AVAILABLE and perf_config.use_triton:
            self.forward = self._triton_forward
        else:
            self.forward = self._standard_forward
    
    def _standard_forward(self, x):
        """Standard RMSNorm implementation"""
        input_dtype = x.dtype
        
        if self.use_mixed_precision:
            x = x.to(self.compute_dtype)
        
        # RMS normalization formula: x * w / sqrt(mean(x²) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x * self.weight
        
        # Return to input dtype if using mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x
    
    def _triton_forward(self, x):
        """Triton-accelerated RMSNorm implementation"""
        # This would use a custom Triton kernel for RMSNorm
        # For now, we'll use the standard implementation
        return self._standard_forward(x)

class LayerNorm(Module):
    """Enhanced Layer Normalization with performance optimizations"""
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-5,
        bias: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))
        self.bias = Parameter(torch.zeros(dim)) if bias else None
        self.use_mixed_precision = perf_config.use_mixed_precision
        self.compute_dtype = perf_config.layernorm_dtype
        
        if TRITON_AVAILABLE and perf_config.use_triton:
            self.forward = self._triton_forward
        else:
            self.forward = self._standard_forward
    
    def _standard_forward(self, x):
        """Standard LayerNorm implementation"""
        input_dtype = x.dtype
        
        if self.use_mixed_precision:
            x = x.to(self.compute_dtype)
        
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        
        # Return to input dtype if using mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x
    
    def _triton_forward(self, x):
        """Triton-accelerated LayerNorm implementation"""
        # This would use a custom Triton kernel for LayerNorm
        # For now, we'll use the standard implementation
        return self._standard_forward(x)

def get_norm_class(norm_type: str):
    """Get normalization class by type"""
    if norm_type == "layer_norm":
        return LayerNorm
    elif norm_type == "rms_norm":
        return RMSNorm
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# Memory-efficient attention implementations
class HierarchicalAttention(Module):
    """
    Hierarchical attention for extremely long contexts.
    
    Uses a multi-level approach:
    1. Local attention for neighboring tokens
    2. Sparse global attention for important tokens
    3. Compressed attention for summarizing distant context
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 1024,
        global_tokens: int = 256,
        dropout: float = 0.0,
        causal: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.causal = causal
        
        # Determine inner dim
        inner_dim = num_heads * head_dim
        
        # Projections for queries, keys, values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Global token selection network
        self.token_scorer = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention implementations to use
        self.use_flash_attention = perf_config.use_flash_attention and FLASH_ATTN_AVAILABLE
        self.use_xformers = perf_config.use_xformers and XFORMERS_AVAILABLE
        
        # Memory-efficient options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
    
    def _checkpoint_fn(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
    
    def _select_global_tokens(self, x):
        """Select the most important tokens for global attention"""
        batch_size, seq_len, _ = x.shape
        
        # Score each token for importance
        scores = self.token_scorer(x).squeeze(-1)  # [batch, seq_len]
        
        # For causal models, mask out future tokens
        if self.causal:
            mask = torch.ones_like(scores, dtype=torch.bool).triu_(1)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Select top-k tokens
        num_tokens = min(self.global_tokens, seq_len)
        _, indices = torch.topk(scores, num_tokens, dim=-1)  # [batch, num_tokens]
        
        # Gather global tokens
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(-1, num_tokens)
        global_tokens = x[batch_indices, indices]  # [batch, num_tokens, dim]
        
        return global_tokens, indices
    
    def _local_attention(self, q, k, v, mask=None):
        """Compute attention within local windows"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Safety check
        if self.use_flash_attention and self.causal and seq_len <= 2048:
            # Use Flash Attention for causal attention when sequence length is manageable
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            out = flash_attn_func(q, k, v, causal=True)
            return out
        
        # Reshape for window attention
        windows = seq_len // self.window_size
        if windows == 0:  # Sequence is shorter than window size
            # Use standard attention for short sequences
            return self._standard_attention(q, k, v, mask)
            
        # Reshape to [batch, num_heads, windows, window_size, head_dim]
        q_windows = q.view(batch_size, num_heads, windows, self.window_size, head_dim)
        k_windows = k.view(batch_size, num_heads, windows, self.window_size, head_dim)
        v_windows = v.view(batch_size, num_heads, windows, self.window_size, head_dim)
        
        # Compute attention within each window
        attn_scores = torch.matmul(q_windows, k_windows.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply causal mask within windows if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(self.window_size, self.window_size, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v_windows)
        
        # Reshape back to [batch, num_heads, seq_len, head_dim]
        out = out.view(batch_size, num_heads, seq_len, head_dim)
        
        return out
    
    def _global_attention(self, q, global_k, global_v):
        """Compute attention between all queries and global tokens"""
        # q: [batch, num_heads, seq_len, head_dim]
        # global_k, global_v: [batch, num_heads, num_global, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, global_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, global_v)
        
        return out
    
    def _standard_attention(self, q, k, v, mask=None):
        """Standard attention implementation for fallback"""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(q.size(-2), k.size(-2), dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        
        return out
        
    def forward(self, x):
        """
        Forward pass for hierarchical attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Select global tokens (most important for cross-sequence attention)
        global_tokens, global_indices = self._select_global_tokens(x)
        
        # Step 2: Project inputs to queries, keys, values
        q = self._checkpoint_fn(self.to_q, x)
        k = self._checkpoint_fn(self.to_k, x)  
        v = self._checkpoint_fn(self.to_v, x)
        
        # Project global tokens to keys and values
        global_k = self._checkpoint_fn(self.to_k, global_tokens)
        global_v = self._checkpoint_fn(self.to_v, global_tokens)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        global_k = global_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        global_v = global_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute local window attention
        local_out = self._local_attention(q, k, v)
        
        # Step 4: Compute global token attention
        global_out = self._global_attention(q, global_k, global_v)
        
        # Step 5: Combine local and global attention
        out = local_out + global_out
        
        # Reshape back to [batch, seq_len, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Project back to original dimension
        out = self._checkpoint_fn(self.to_out, out)
        
        return out

class StreamingAttention(Module):
    """
    Streaming attention for processing extremely long contexts by windowing.
    
    This attention mechanism processes the input in chunks, maintaining a state
    of previous tokens, which is especially useful for autoregressive generation
    with very long contexts.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 4096,
        max_kv_cache: int = 100000,
        dropout: float = 0.0,
        causal: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.max_kv_cache = max_kv_cache
        self.causal = causal
        
        # Determine inner dim
        inner_dim = num_heads * head_dim
        
        # Projections for queries, keys, values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # KV cache for streaming mode
        self.kv_cache = None
        self.kv_cache_size = 0
        
        # Attention implementations to use
        self.use_flash_attention = perf_config.use_flash_attention and FLASH_ATTN_AVAILABLE
        
        # Memory-efficient options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward, 
                mode=perf_config.compile_mode,
                fullgraph=False  # State changes are not compatible with fullgraph
            )
    
    def _checkpoint_fn(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
    
    def _init_kv_cache(self, batch_size, device):
        """Initialize empty KV cache"""
        self.kv_cache = [
            # K cache: [batch, num_heads, max_kv_cache, head_dim]
            torch.zeros(batch_size, self.num_heads, self.max_kv_cache, self.head_dim, device=device),
            # V cache: [batch, num_heads, max_kv_cache, head_dim]
            torch.zeros(batch_size, self.num_heads, self.max_kv_cache, self.head_dim, device=device)
        ]
        self.kv_cache_size = 0
    
    def _update_kv_cache(self, k, v):
        """
        Update the KV cache with new keys and values
        
        Args:
            k, v: New keys and values of shape [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Updated keys and values including cached values
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Initialize cache if not exists
        if self.kv_cache is None:
            self._init_kv_cache(batch_size, k.device)
        
        # Handle overwrites if cache is full
        new_cache_size = self.kv_cache_size + seq_len
        if new_cache_size > self.max_kv_cache:
            # If exceeding, shift cache and make room for new tokens
            shift_amount = new_cache_size - self.max_kv_cache
            self.kv_cache[0] = torch.roll(self.kv_cache[0], shifts=-shift_amount, dims=2)
            self.kv_cache[1] = torch.roll(self.kv_cache[1], shifts=-shift_amount, dims=2)
            self.kv_cache_size = self.max_kv_cache - seq_len
        
        # Update cache
        self.kv_cache[0][:, :, self.kv_cache_size:self.kv_cache_size+seq_len] = k
        self.kv_cache[1][:, :, self.kv_cache_size:self.kv_cache_size+seq_len] = v
        
        # Create views into the cache
        k_with_cache = self.kv_cache[0][:, :, :self.kv_cache_size+seq_len]
        v_with_cache = self.kv_cache[1][:, :, :self.kv_cache_size+seq_len]
        
        # Update cache size
        self.kv_cache_size += seq_len
        
        return k_with_cache, v_with_cache
    
    def _compute_attention(self, q, k, v):
        """
        Compute attention scores and apply to values
        
        Args:
            q: Queries of shape [batch, num_heads, seq_len, head_dim]
            k: Keys of shape [batch, num_heads, kv_len, head_dim]
            v: Values of shape [batch, num_heads, kv_len, head_dim]
            
        Returns:
            Output tensor of shape [batch, num_heads, seq_len, head_dim]
        """
        # Try using Flash Attention when available
        if self.use_flash_attention:
            try:
                q, k, v = map(lambda t: t.contiguous(), (q, k, v))
                return flash_attn_func(q, k, v, causal=self.causal)
            except Exception as e:
                logger.warning(f"Flash Attention failed, falling back to standard: {e}")
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed
        if self.causal:
            q_len, k_len = q.size(-2), k.size(-2)
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=q.device),
                diagonal=k_len - q_len + 1  # +1 because we want the diagonal to be visible
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        
        return out
        
    def forward(self, x, use_cache=True, clear_cache=False):
        """
        Forward pass for streaming attention
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            use_cache: Whether to use and update the KV cache
            clear_cache: Whether to clear the KV cache before processing
            
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Clear cache if requested
        if clear_cache and self.kv_cache is not None:
            self.kv_cache = None
            self.kv_cache_size = 0
        
        # Project inputs to queries, keys, values
        q = self._checkpoint_fn(self.to_q, x)
        k = self._checkpoint_fn(self.to_k, x)
        v = self._checkpoint_fn(self.to_v, x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache if using streaming mode
        if use_cache:
            k, v = self._update_kv_cache(k, v)
        
        # Compute attention
        out = self._compute_attention(q, k, v)
        
        # Reshape back to [batch, seq_len, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Project back to original dimension
        out = self._checkpoint_fn(self.to_out, out)
        
        return out

    def clear_cache(self):
        """Clear the KV cache"""
        self.kv_cache = None
        self.kv_cache_size = 0

# Memory-efficient memory MLP modules
class UltraMemoryMLP(Module):
    """
    Ultra-efficient memory MLP for processing long contexts.
    
    Features:
    - Gradient checkpointing
    - Mixed precision handling
    - Multiple activation functions
    - Low-rank factorization
    - Activation checkpointing
    """
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 4.0,
        activation: str = "swiglu",
        dropout: float = 0.0,
        use_bias: bool = False,
        factorized: bool = True,
        factorization_rank: Optional[int] = None,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        
        # Determine intermediate dimensions
        if activation == "swiglu":
            # For SwiGLU we need 2x the features for gate and value paths
            hidden_dim = int(dim * expansion_factor * 2)
        else:
            hidden_dim = int(dim * expansion_factor)
        
        # Use factorization if requested
        self.factorized = factorized
        if factorized:
            # Determine rank for factorization (default to dim / 8)
            rank = factorization_rank or max(1, dim // 8)
            
            # Create factorized weights
            self.up_proj1 = nn.Linear(dim, rank, bias=False)
            self.up_proj2 = nn.Linear(rank, hidden_dim, bias=use_bias)
            
            self.down_proj1 = nn.Linear(hidden_dim, rank, bias=False)
            self.down_proj2 = nn.Linear(rank, dim, bias=use_bias)
        else:
            # Standard projections
            self.up_proj = nn.Linear(dim, hidden_dim, bias=use_bias)
            self.down_proj = nn.Linear(hidden_dim, dim, bias=use_bias)
        
        # Activation function
        self.activation_name = activation
        if activation == "swiglu":
            self.activation = lambda x: ActivationFunctions.swiglu(x[..., :hidden_dim//2], x[..., hidden_dim//2:])
        else:
            self.activation = ActivationFunctions.get_activation_fn(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Performance options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        self.use_mixed_precision = perf_config.use_mixed_precision
        self.compute_dtype = perf_config.compute_dtype
        
        # Initialization
        self._reset_parameters()
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
    
    def _reset_parameters(self):
        """Initialize parameters with better scaling"""
        if self.factorized:
            # Initialize factorized weights with special scaling
            nn.init.normal_(self.up_proj1.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.up_proj2.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj1.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj2.weight, mean=0.0, std=0.02 / math.sqrt(2.0))
            
            # Zero-init bias for better training stability
            if hasattr(self.up_proj2, 'bias') and self.up_proj2.bias is not None:
                nn.init.zeros_(self.up_proj2.bias)
            if hasattr(self.down_proj2, 'bias') and self.down_proj2.bias is not None:
                nn.init.zeros_(self.down_proj2.bias)
        else:
            # Initialize standard weights
            nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02 / math.sqrt(2.0))
            
            # Zero-init bias for better training stability
            if hasattr(self.up_proj, 'bias') and self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
            if hasattr(self.down_proj, 'bias') and self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)
    
    def _checkpoint_fn(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
        
    def forward(self, x):
        """Forward pass with performance optimizations"""
        orig_dtype = x.dtype
        
        # Cast to compute dtype if using mixed precision
        if self.use_mixed_precision:
            x = x.to(self.compute_dtype)
        
        if self.factorized:
            # Factorized forward pass
            x_up = self._checkpoint_fn(self.up_proj1, x)
            x_up = self._checkpoint_fn(self.up_proj2, x_up)
        else:
            # Standard forward pass
            x_up = self._checkpoint_fn(self.up_proj, x)
        
        # Apply activation (checkpointed)
        if self.use_checkpointing and self.training:
            x_act = checkpoint(self.activation, x_up)
        else:
            x_act = self.activation(x_up)
        
        # Apply dropout if specified
        if self.dropout is not None:
            x_act = self.dropout(x_act)
        
        if self.factorized:
            # Factorized down projection
            x_down = self._checkpoint_fn(self.down_proj1, x_act)
            x_down = self._checkpoint_fn(self.down_proj2, x_down)
        else:
            # Standard down projection
            x_down = self._checkpoint_fn(self.down_proj, x_act)
        
        # Restore original dtype if using mixed precision
        if self.use_mixed_precision:
            x_down = x_down.to(orig_dtype)
        
        return x_down

# Memory module with hierarchical summarization
class SummarizationMemoryLayer(Module):
    """
    Memory layer with hierarchical summarization abilities
    
    This layer maintains summaries at different levels of granularity,
    which helps with efficient processing of ultra-long contexts.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        summary_ratio: int = 4,
        num_levels: int = 3,
        dropout: float = 0.0,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.summary_ratio = summary_ratio
        self.num_levels = num_levels
        
        # Determine inner dim
        inner_dim = num_heads * head_dim
        
        # Projections for queries, keys, values
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Summarization layers
        self.summarizers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            for _ in range(num_levels)
        ])
        
        # Relevance predictors for summary levels
        self.relevance_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1)
            )
            for _ in range(num_levels)
        ])
        
        # Normalization for summaries
        norm_class = get_norm_class(perf_config.normalization)
        self.summary_norms = nn.ModuleList([
            norm_class(dim, perf_config=perf_config)
            for _ in range(num_levels)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Memory-efficient options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
    
    def _checkpoint_fn(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
    
    def _create_summaries(self, x):
        """
        Create hierarchical summaries of the input
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            List of summary tensors at different levels
        """
        batch_size, seq_len, _ = x.shape
        summaries = []
        
        current_x = x
        for level in range(self.num_levels):
            # Skip if sequence is too short for this level
            if current_x.size(1) <= 1:
                break
                
            # Calculate summary length for this level
            summary_len = max(1, current_x.size(1) // self.summary_ratio)
            
            # Create summary by average pooling followed by projection
            if summary_len < current_x.size(1):
                # Reshape to get groups of tokens to summarize
                padding = (self.summary_ratio - (current_x.size(1) % self.summary_ratio)) % self.summary_ratio
                if padding > 0:
                    # Pad the sequence if needed
                    current_x = F.pad(current_x, (0, 0, 0, padding))
                
                # Reshape and pool
                pooled = current_x.view(batch_size, -1, self.summary_ratio, self.dim)
                pooled = pooled.mean(dim=2)  # [batch, summary_len, dim]
                
                # Apply summarization transform
                summary = self._checkpoint_fn(self.summarizers[level], pooled)
                summary = self.summary_norms[level](summary)
                
                summaries.append(summary)
                current_x = summary
            else:
                # If we can't reduce further, stop
                break
        
        return summaries
    
    def _compute_relevance(self, q, summaries):
        """
        Compute relevance of each summary level to the query
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, dim]
            summaries: List of summary tensors
            
        Returns:
            List of relevance weights for each summary level
        """
        relevance_weights = []
        
        for level, summary in enumerate(summaries):
            # Project query for relevance prediction
            q_relevance = self._checkpoint_fn(self.relevance_predictors[level], q)  # [batch, seq_len, 1]
            relevance_weights.append(torch.sigmoid(q_relevance))
            
        return relevance_weights
            
    def forward(self, x):
        """
        Forward pass with hierarchical summarization
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Create hierarchical summaries
        summaries = self._create_summaries(x)
        
        # Step 2: Project inputs to queries, keys, values
        q = self._checkpoint_fn(self.to_q, x)
        k = self._checkpoint_fn(self.to_k, x)
        v = self._checkpoint_fn(self.to_v, x)
        
        # Step 3: Compute relevance of each summary level
        relevance_weights = self._compute_relevance(x, summaries)
        
        # Step 4: Project summaries to keys and values
        summary_k_list = []
        summary_v_list = []
        
        for summary in summaries:
            summary_k = self._checkpoint_fn(self.to_k, summary)
            summary_v = self._checkpoint_fn(self.to_v, summary)
            summary_k_list.append(summary_k)
            summary_v_list.append(summary_v)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 5: Compute standard attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        standard_out = torch.matmul(attn_probs, v)
        
        # Step 6: Compute attention with each summary level
        summary_outputs = []
        
        for i, (summary_k, summary_v, relevance) in enumerate(zip(summary_k_list, summary_v_list, relevance_weights)):
            # Reshape summary for attention
            s_len = summary_k.size(1)
            summary_k = summary_k.view(batch_size, s_len, self.num_heads, self.head_dim).transpose(1, 2)
            summary_v = summary_v.view(batch_size, s_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention with summary
            summary_scores = torch.matmul(q, summary_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            summary_probs = F.softmax(summary_scores, dim=-1)
            summary_probs = self.dropout(summary_probs)
            summary_out = torch.matmul(summary_probs, summary_v)
            
            # Apply relevance weighting
            relevance = relevance.view(batch_size, 1, seq_len, 1)
            summary_outputs.append(summary_out * relevance)
        
        # Step 7: Combine standard and summary outputs
        if summary_outputs:
            # Combine summaries
            combined_summaries = torch.stack(summary_outputs).sum(dim=0)
            # Weighted combination with standard output (relevance should sum to <= 1)
            relevance_sum = torch.stack(relevance_weights).sum(dim=0).view(batch_size, 1, seq_len, 1)
            out = standard_out * (1 - relevance_sum) + combined_summaries
        else:
            out = standard_out
        
        # Reshape back to [batch, seq_len, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Project back to original dimension
        out = self._checkpoint_fn(self.to_out, out)
        
        return out

# Residual wrapper with advanced norm
class AdvancedResidualBlock(Module):
    """
    Enhanced residual block with advanced normalization and flexible structure
    
    Features:
    - Pre-norm and post-norm options
    - Multiple normalization implementations
    - Adaptive weighting of residual connections
    - Gradient checkpointing
    """
    def __init__(
        self,
        dim: int,
        layer: Module,
        pre_norm: bool = True,
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        adaptive_scaling: bool = False,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        # Normalization setup
        norm_class = get_norm_class(perf_config.normalization)
        self.norm = norm_class(dim, perf_config=perf_config)
        
        # Main layer
        self.layer = layer
        
        # Structure config
        self.pre_norm = pre_norm
        self.adaptive_scaling = adaptive_scaling
        
        # Residual weight
        if adaptive_scaling:
            self.residual_weight = Parameter(torch.ones(1) * residual_scale)
        else:
            self.register_buffer('residual_weight', torch.ones(1) * residual_scale)
            
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Memory-efficient options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
    
    def _checkpoint_fn(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
        
    def forward(self, x):
        if self.pre_norm:
            # Pre-normalization architecture
            normed_x = self.norm(x)
            layer_output = self._checkpoint_fn(self.layer, normed_x)
            
            if self.dropout is not None:
                layer_output = self.dropout(layer_output)
                
            # Residual connection with scaling
            return x + layer_output * self.residual_weight
        else:
            # Post-normalization architecture
            layer_output = self._checkpoint_fn(self.layer, x)
            
            if self.dropout is not None:
                layer_output = self.dropout(layer_output)
                
            # Residual connection with scaling
            result = x + layer_output * self.residual_weight
            return self.norm(result)

# Token Eviction and Compression Module
class TokenEvictionLayer(Module):
    """
    Token eviction and compression for extremely long contexts
    
    Features:
    - Adaptive token selection based on importance scoring
    - Compression via merging of less important tokens
    - Dynamic adjustment based on sequence length
    """
    def __init__(
        self,
        dim: int,
        target_ratio: float = 0.5,  # Target compression ratio
        min_seq_len: int = 2048,    # Minimum length before compressing
        scorer_hidden_dim: int = 128,
        strategy: str = "evict",  # "evict" or "merge"
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.target_ratio = target_ratio
        self.min_seq_len = min_seq_len
        self.strategy = strategy
        
        # Token importance scoring network
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, scorer_hidden_dim),
            nn.GELU(),
            nn.Linear(scorer_hidden_dim, 1)
        )
        
        # For merge strategy
        if strategy == "merge":
            # Token merging projections
            self.merge_projection = nn.Linear(dim * 2, dim)
        
        # Memory-efficient options
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
    def _score_tokens(self, x):
        """Score tokens for importance"""
        return self.importance_scorer(x).squeeze(-1)
    
    def _evict_tokens(self, x, scores, keep_ratio):
        """Evict tokens based on scores"""
        batch_size, seq_len, _ = x.shape
        
        # Determine number of tokens to keep
        num_keep = max(1, int(seq_len * keep_ratio))
        
        # Select top-k indices
        _, indices = torch.topk(scores, num_keep, dim=1)
        
        # Sort indices for easier attention masking later
        indices = torch.sort(indices, dim=1)[0]
        
        # Gather kept tokens
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(-1, num_keep)
        kept_tokens = x[batch_indices, indices]
        
        return kept_tokens, indices
    
    def _merge_tokens(self, x, scores, keep_ratio):
        """Merge less important tokens with neighbors"""
        batch_size, seq_len, _ = x.shape
        
        # Determine number of tokens to keep (must be even for convenient merging)
        num_keep = max(2, int(seq_len * keep_ratio))
        num_keep = num_keep + (num_keep % 2)  # Ensure even
        
        # Target number of merges
        num_merges = (seq_len - num_keep) // 2
        
        if num_merges <= 0:
            return x, None  # No merging needed
        
        # Find token pairs to merge (lowest scoring pairs)
        # Create scores for adjacent token pairs
        pair_scores = scores[:, :-1] + scores[:, 1:]
        
        # Select lowest scoring pairs for merging
        _, merge_indices = torch.topk(pair_scores, num_merges, dim=1, largest=False)
        
        # Create new sequence by merging selected pairs
        merged_tokens = []
        merge_masks = []
        
        for b in range(batch_size):
            # Track which tokens have been merged
            merged_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
            
            # Mark tokens to be merged
            for idx in merge_indices[b]:
                merged_mask[idx:idx+2] = True
            
            # Create list of tokens and pairs to be merged
            to_merge = []
            remaining_indices = []
            
            i = 0
            while i < seq_len:
                if merged_mask[i] and i+1 < seq_len and merged_mask[i+1]:
                    # This is a pair to be merged
                    to_merge.append((i, i+1))
                    i += 2
                else:
                    # This token remains as is
                    remaining_indices.append(i)
                    i += 1
            
            # Compile new sequence: first add unchanged tokens
            new_tokens = [x[b, i].unsqueeze(0) for i in remaining_indices]
            
            # Then add merged token pairs
            for i, j in to_merge:
                pair = torch.cat([x[b, i], x[b, j]]).unsqueeze(0)
                merged = self.merge_projection(pair)
                new_tokens.append(merged)
            
            # Combine into new sequence
            batch_merged = torch.cat(new_tokens, dim=0)
            merged_tokens.append(batch_merged)
            merge_masks.append(merged_mask)
        
        # Stack results from all batches
        merged_tokens = torch.stack([t for t in merged_tokens])
        merge_masks = torch.stack(merge_masks)
        
        return merged_tokens, merge_masks
        
    def forward(self, x):
        """
        Forward pass with token eviction or merging
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            - Compressed tensor of shape [batch_size, new_seq_len, dim]
            - Indices of kept tokens or merge mask for attention mask adjustments
        """
        batch_size, seq_len, _ = x.shape
        
        # Skip if sequence is already short
        if seq_len <= self.min_seq_len:
            return x, None
        
        # Score tokens for importance
        if self.use_checkpointing and self.training:
            scores = checkpoint(self._score_tokens, x)
        else:
            scores = self._score_tokens(x)
        
        # Calculate adaptive compression ratio based on sequence length
        # For longer sequences, we compress more aggressively
        base_ratio = self.target_ratio
        length_factor = min(1.0, self.min_seq_len / seq_len)
        keep_ratio = max(0.1, base_ratio * length_factor)
        
        # Apply selected compression strategy
        if self.strategy == "evict":
            kept_tokens, indices = self._evict_tokens(x, scores, keep_ratio)
            return kept_tokens, indices
        elif self.strategy == "merge":
            merged_tokens, merge_mask = self._merge_tokens(x, scores, keep_ratio)
            return merged_tokens, merge_mask
        else:
            raise ValueError(f"Unknown compression strategy: {self.strategy}")

# Sliding Window Processor
class SlidingWindowModule(Module):
    """
    Process extremely long sequences using sliding windows
    
    Features:
    - Fixed-size window processing
    - Automatic chunking of long sequences
    - Optional overlapping windows with aggregation
    """
    def __init__(
        self,
        base_module: Module,
        window_size: int = 4096,
        overlap: int = 128,
        aggregation: str = "attention",  # "average", "concat", "attention"
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.base_module = base_module
        self.window_size = window_size
        self.overlap = overlap
        self.aggregation = aggregation
        
        assert overlap < window_size, "Overlap must be smaller than window size"
        
        # For attention-based aggregation in overlapping regions
        if aggregation == "attention":
            module_dim = getattr(base_module, "dim", None)
            if module_dim is None:
                raise ValueError("Base module must have a 'dim' attribute for attention aggregation")
                
            self.overlap_attention = nn.Sequential(
                nn.Linear(module_dim, module_dim // 4),
                nn.GELU(),
                nn.Linear(module_dim // 4, 1)
            )
    
    def _process_chunks(self, chunks):
        """Process each window chunk"""
        results = []
        for chunk in chunks:
            result = self.base_module(chunk)
            results.append(result)
        return results
    
    def _aggregate_results(self, results, original_len):
        """Aggregate results from all windows"""
        if self.aggregation == "concat":
            # Simple concatenation (no overlapping windows)
            output = torch.cat(results, dim=1)
            # Trim to original length
            return output[:, :original_len]
        
        elif self.aggregation == "average":
            # Average overlapping regions
            batch_size, _, hidden_dim = results[0].shape
            device = results[0].device
            
            # Initialize output and weight tensors
            step_size = self.window_size - self.overlap
            total_length = (len(results) - 1) * step_size + self.window_size
            output = torch.zeros(batch_size, total_length, hidden_dim, device=device)
            weights = torch.zeros(batch_size, total_length, 1, device=device)
            
            # Fill output tensor with weighted segments
            for i, result in enumerate(results):
                start_idx = i * step_size
                end_idx = start_idx + self.window_size
                
                # Add result to output
                output[:, start_idx:end_idx] += result
                # Increase weight count
                weights[:, start_idx:end_idx] += 1
            
            # Normalize by weights
            output = output / (weights + 1e-8)
            
            # Trim to original length
            return output[:, :original_len]
        
        elif self.aggregation == "attention":
            # Attention-based aggregation in overlapping regions
            batch_size, _, hidden_dim = results[0].shape
            device = results[0].device
            
            # Initialize output tensor
            step_size = self.window_size - self.overlap
            total_length = (len(results) - 1) * step_size + self.window_size
            output = torch.zeros(batch_size, total_length, hidden_dim, device=device)
            
            # Process non-overlapping regions directly
            for i, result in enumerate(results):
                start_idx = i * step_size
                
                if i == 0:
                    # First window: copy fully except last overlap
                    output[:, :self.window_size-self.overlap] = result[:, :self.window_size-self.overlap]
                elif i == len(results) - 1:
                    # Last window: copy fully except first overlap
                    end_pos = start_idx + self.window_size
                    output[:, start_idx+self.overlap:end_pos] = result[:, self.overlap:]
                else:
                    # Middle windows: copy middle section
                    end_non_overlap = start_idx + self.window_size - self.overlap
                    output[:, start_idx+self.overlap:end_non_overlap] = result[:, self.overlap:self.window_size-self.overlap]
            
            # Process overlapping regions with attention
            for i in range(len(results) - 1):
                current_result = results[i]
                next_result = results[i+1]
                
                # Indices for overlapping region
                start_overlap = (i+1) * step_size - self.overlap
                end_overlap = (i+1) * step_size
                
                # Get overlapping parts from both windows
                current_overlap = current_result[:, -self.overlap:]
                next_overlap = next_result[:, :self.overlap]
                
                # Score overlapping tokens
                current_scores = self.overlap_attention(current_overlap)
                next_scores = self.overlap_attention(next_overlap)
                
                # Normalize scores
                scores = torch.cat([current_scores, next_scores], dim=1)
                attention = F.softmax(scores, dim=1)
                
                # Split attention weights
                current_attention = attention[:, :self.overlap].unsqueeze(-1)
                next_attention = attention[:, self.overlap:].unsqueeze(-1)
                
                # Weighted combination
                overlap_output = current_overlap * current_attention + next_overlap * next_attention
                
                # Update output
                output[:, start_overlap:end_overlap] = overlap_output
            
            # Trim to original length
            return output[:, :original_len]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
    def forward(self, x):
        """
        Process input using sliding windows
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Processed tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # If sequence fits in a single window, process directly
        if seq_len <= self.window_size:
            return self.base_module(x)
        
        # Otherwise, break into overlapping chunks
        chunks = []
        step_size = self.window_size - self.overlap
        
        for i in range(0, seq_len, step_size):
            end_idx = min(i + self.window_size, seq_len)
            start_idx = max(0, end_idx - self.window_size)
            
            chunk = x[:, start_idx:end_idx]
            chunks.append(chunk)
            
            # If we've processed the whole sequence, stop
            if end_idx == seq_len:
                break
        
        # Process all chunks
        results = self._process_chunks(chunks)
        
        # Aggregate results
        output = self._aggregate_results(results, seq_len)
        
        return output

# Main UltraContext module for integration
class UltraContextModule(Module):
    """
    Main UltraContext module combining all advanced techniques
    
    Features:
    - Multi-level memory hierarchy
    - Token eviction and compression
    - Sliding window processing
    - Hierarchical summarization
    - Optimized for 100M token windows
    """
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        memory_module: str = "hierarchical",
        mlp_module: str = "ultra",
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        adaptive_scaling: bool = True,
        use_memory_hierarchy: bool = True,
        token_compression: Optional[str] = "evict",
        window_size: int = 8192,
        max_tokens: int = 100_000_000,  # 100M tokens
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.max_tokens = max_tokens
        
        # Base components
        norm_class = get_norm_class(perf_config.normalization)
        
        # Set up attention module based on type
        if memory_module == "streaming":
            # Streaming attention for extremely long contexts
            attn_layer = StreamingAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                max_kv_cache=min(max_tokens, 1_000_000),  # Cap KV cache at 1M tokens
                dropout=dropout,
                causal=True,
                perf_config=perf_config
            )
        elif memory_module == "hierarchical":
            # Hierarchical attention for better scaling
            attn_layer = HierarchicalAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                global_tokens=min(dim * 2, 256),  # Scale with model size
                dropout=dropout,
                causal=True,
                perf_config=perf_config
            )
        elif memory_module == "summarization":
            # Memory with hierarchical summarization
            attn_layer = SummarizationMemoryLayer(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                summary_ratio=4,
                num_levels=min(3, int(math.log2(max_tokens // window_size)) + 1),
                dropout=dropout,
                perf_config=perf_config
            )
        else:
            raise ValueError(f"Unknown memory module type: {memory_module}")
        
        # Set up MLP module based on type
        if mlp_module == "ultra":
            mlp_layer = UltraMemoryMLP(
                dim=dim,
                expansion_factor=mlp_ratio,
                activation=perf_config.activation_function,
                dropout=dropout,
                factorized=True,
                perf_config=perf_config
            )
        else:
            raise ValueError(f"Unknown MLP module type: {mlp_module}")
        
        # Create residual blocks
        self.attn_block = AdvancedResidualBlock(
            dim=dim,
            layer=attn_layer,
            pre_norm=True,
            dropout=dropout,
            adaptive_scaling=adaptive_scaling,
            perf_config=perf_config
        )
        
        self.mlp_block = AdvancedResidualBlock(
            dim=dim,
            layer=mlp_layer,
            pre_norm=True,
            dropout=dropout,
            adaptive_scaling=adaptive_scaling,
            perf_config=perf_config
        )
        
        # Optional token compression/eviction
        if token_compression:
            self.token_processor = TokenEvictionLayer(
                dim=dim,
                target_ratio=0.5,
                min_seq_len=window_size,
                strategy=token_compression,
                perf_config=perf_config
            )
        else:
            self.token_processor = None
        
        # Apply sliding window if sequence is very long
        if perf_config.use_sliding_window:
            # Create composite module
            base_module = nn.Sequential(self.attn_block, self.mlp_block)
            
            # Wrap with sliding window
            self.processor = SlidingWindowModule(
                base_module=base_module,
                window_size=window_size,
                overlap=window_size // 8,  # 12.5% overlap
                aggregation="attention",
                perf_config=perf_config
            )
        else:
            # Use standard sequential processing
            self.processor = nn.Sequential(self.attn_block, self.mlp_block)
        
        # Final normalization
        self.final_norm = norm_class(dim, perf_config=perf_config)
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
    
    def forward(self, x, use_compression=True):
        """
        Forward pass with ultra-long context handling
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            use_compression: Whether to use token compression/eviction
            
        Returns:
            Processed tensor of shape [batch_size, new_seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply token compression/eviction if needed and enabled
        if self.token_processor is not None and use_compression and seq_len > self.token_processor.min_seq_len:
            x, indices = self.token_processor(x)
        
        # Process with main module
        x = self.processor(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        return x
    
    def clear_cache(self):
        """Clear any cached state"""
        # Find and clear StreamingAttention caches
        for module in self.modules():
            if isinstance(module, StreamingAttention):
                module.clear_cache()
                
# Create factory function for UltraContext networks
def create_ultracontext_network(
    dim: int,
    depth: int = 1,
    memory_type: str = "hierarchical",  # "streaming", "hierarchical", "summarization"
    mlp_type: str = "ultra",            # Currently only "ultra" supported
    window_size: int = 8192,
    max_tokens: int = 100_000_000,      # 100M tokens
    use_compression: bool = True,
    compression_type: str = "evict",    # "evict", "merge", None
    perf_config: Optional[PerformanceConfig] = None
) -> nn.Module:
    """
    Create an UltraContext network for extremely long context windows
    
    Args:
        dim: Model dimension
        depth: Number of UltraContext layers
        memory_type: Type of memory module
        mlp_type: Type of MLP module
        window_size: Size of local attention window
        max_tokens: Maximum number of tokens to support
        use_compression: Whether to use token compression
        compression_type: Type of compression to use
        perf_config: Performance configuration
        
    Returns:
        Module that can handle extremely long contexts
    """
    perf_config = perf_config or DEFAULT_PERF_CONFIG
    
    # For extremely deep networks, adjust settings
    if depth > 10:
        # Use RMS Norm for better stability
        perf_config.normalization = "rms_norm"
        # Use SwiGLU for better gradient flow
        perf_config.activation_function = "swiglu"
    
    # Create layers
    layers = []
    for _ in range(depth):
        layer = UltraContextModule(
            dim=dim,
            memory_module=memory_type,
            mlp_module=mlp_type,
            window_size=window_size,
            max_tokens=max_tokens,
            token_compression=compression_type if use_compression else None,
            perf_config=perf_config
        )
        layers.append(layer)
    
    # Create sequential model if multiple layers
    if len(layers) == 1:
        return layers[0]
    else:
        return nn.Sequential(*layers)

# Utility function to measure throughput
def measure_throughput(model, batch_size=1, seq_len=4096, dim=768, device="cuda", warmup=5, repeats=10):
    """
    Measure throughput of a model in tokens per second
    
    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_len: Sequence length
        dim: Hidden dimension
        device: Device to use
        warmup: Number of warmup iterations
        repeats: Number of measurement iterations
        
    Returns:
        Tuple of (throughput, latency)
    """
    # Create dummy input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    elapsed = end_time - start_time
    latency_ms = (elapsed / repeats) * 1000
    tokens_per_second = (batch_size * seq_len * repeats) / elapsed
    
    return tokens_per_second, latency_ms
