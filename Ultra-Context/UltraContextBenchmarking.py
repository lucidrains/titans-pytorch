import torch
import torch.nn as nn
import time
import os
import argparse
from typing import List, Dict, Optional, Union, Tuple
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
import csv
from contextlib import contextmanager
import gc
import psutil
import sys
import platform

# Import UltraContext components
try:
    from ultracontext.core import (
        DEFAULT_PERF_CONFIG, 
        PerformanceConfig,
        UltraContextModule,
        create_ultracontext_network
    )
    from ultracontext.memory import HierarchicalMemoryManager
    from ultracontext.processing import (
        ContextualCompressor,
        RetrievalAugmentedProcessor,
        HierarchicalProcessingModule
    )
    from ultracontext.integration import (
        UltraContextConfig,
        UltraContextAPI,
        UltraContextWrapper,
        efficient_inference_mode
    )
except ImportError:
    print("UltraContext package not installed. Using local imports.")
    # Fallback to local imports
    from integration import (
        UltraContextConfig,
        UltraContextAPI,
        DEFAULT_PERF_CONFIG,
        PerformanceConfig,
        efficient_inference_mode
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext.benchmark")

#######################################
# Memory Usage Utilities
#######################################

@contextmanager
def track_memory_usage():
    """Context manager to track peak memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    try:
        yield
    finally:
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Log memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            logger.info(f"Peak GPU memory: {gpu_mem:.2f} MB")
        
        logger.info(f"CPU memory change: {mem_after - mem_before:.2f} MB")

def log_system_info():
    """Log system information to help with benchmark reproducibility"""
    logger.info("System Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Processor: {platform.processor()}")
    
    # PyTorch info
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA device count: {torch.cuda.device_count()}")
    
    # Memory info
    mem = psutil.virtual_memory()
    logger.info(f"  Total system memory: {mem.total / (1024**3):.2f} GB")
    logger.info(f"  Available memory: {mem.available / (1024**3):.2f} GB")

#######################################
# Benchmark Configurations
#######################################

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks"""
    # Model dimensions
    dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    
    # Context sizes to test
    context_sizes: List[int] = field(default_factory=lambda: [
        1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
    ])
    
    # Chunk sizes (tokens per batch)
    chunk_sizes: List[int] = field(default_factory=lambda: [1024, 4096, 16384])
    
    # Repetitions for accurate timing
    num_repetitions: int = 3
    
    # Framework configurations to test
    frameworks: List[str] = field(default_factory=lambda: [
        "ultracontext", "baseline", "kv_cache"
    ])
    
    # UltraContext configurations to test
    compression_modes: List[str] = field(default_factory=lambda: [
        "none", "low", "medium", "high"
    ])
    
    # Integration modes to test
    integration_modes: List[str] = field(default_factory=lambda: [
        "extension", "replacement", "hybrid"
    ])
    
    # Memory configurations to test
    memory_configs: List[str] = field(default_factory=lambda: [
        "hierarchical", "kv_only", "none"
    ])
    
    # Output directory for results
    output_dir: str = "benchmark_results"
    
    def __post_init__(self):
        """Ensure output directory exists"""
        os.makedirs(self.output_dir, exist_ok=True)

#######################################
# Benchmark Models
#######################################

class BaselineModel(nn.Module):
    """Baseline model with standard attention"""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention with maximum context support
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + attn_out
        
        # FFN with residual connection
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x

class KVCacheModel(nn.Module):
    """Model with standard KV-cache attention"""
    def __init__(self, dim=768, num_heads=12, max_cache_len=100_000):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_cache_len = max_cache_len
        
        # Projections for KV cache
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Initialize KV cache
        self.clear_kv_cache()
        
    def clear_kv_cache(self):
        """Clear KV cache"""
        self.k_cache = None
        self.v_cache = None
        self.current_kv_len = 0
        
    def _update_kv_cache(self, k, v):
        """Update KV cache with new keys and values"""
        batch_size, seq_len, _ = k.shape
        
        # Initialize cache if needed
        if self.k_cache is None:
            self.k_cache = k.clone()
            self.v_cache = v.clone()
            self.current_kv_len = seq_len
            return k, v
            
        # Check if cache is full
        if self.current_kv_len + seq_len > self.max_cache_len:
            # Shift to make room
            shift_amt = min(seq_len, self.current_kv_len)
            self.k_cache = torch.cat([self.k_cache[:, shift_amt:], k], dim=1)
            self.v_cache = torch.cat([self.v_cache[:, shift_amt:], v], dim=1)
            self.current_kv_len = self.max_cache_len - shift_amt + seq_len
        else:
            # Append to cache
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)
            self.current_kv_len += seq_len
            
        return self.k_cache, self.v_cache
        
    def forward(self, x, mask=None, use_cache=True):
        batch_size, seq_len, _ = x.shape
        
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        
        # Project to query, key, value
        q = self.q_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache if enabled
        if use_cache:
            k_with_cache, v_with_cache = self._update_kv_cache(
                k.transpose(1, 2), v.transpose(1, 2)
            )
            k = k_with_cache.transpose(1, 2)
            v = v_with_cache.transpose(1, 2)
        
        # Compute attention scores
        attn_output = self._attention(q, k, v)
        
        # Project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        attn_output = self.o_proj(attn_output)
        
        # Residual connection
        x = x + attn_output
        
        # FFN with residual connection
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x
    
    def _attention(self, q, k, v):
        """Compute attention scores"""
        # Scale query
        q = q * (self.head_dim ** -0.5)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply causal mask
        batch_size, num_heads, q_len, k_len = attn_scores.shape
        causal_mask = torch.triu(
            torch.ones((q_len, k_len), device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, -1e9)
        
        # Apply attention
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        return attn_output

#######################################
# Benchmark Functions
#######################################

def create_ultra_config(
    benchmark_config: BenchmarkConfig,
    max_context: int,
    compression_mode: str,
    integration_mode: str,
    memory_config: str
) -> UltraContextConfig:
    """Create UltraContext configuration based on benchmark parameters"""
    # Base settings
    dim = benchmark_config.dim
    num_heads = benchmark_config.num_heads
    head_dim = benchmark_config.head_dim
    
    # Determine window sizes based on context length
    if max_context <= 10_000:
        active_window = 4096
        sliding_window = 2048
    elif max_context <= 100_000:
        active_window = 8192
        sliding_window = 4096
    elif max_context <= 1_000_000:
        active_window = 16384
        sliding_window = 8192
    else:
        active_window = 32768
        sliding_window = 16384
    
    # Determine compression settings
    if compression_mode == "none":
        use_token_compression = False
        compression_ratio = 1.0
        memory_compression_ratio = 1.0
    elif compression_mode == "low":
        use_token_compression = True
        compression_ratio = 2.0
        memory_compression_ratio = 4.0
    elif compression_mode == "medium":
        use_token_compression = True
        compression_ratio = 4.0
        memory_compression_ratio = 8.0
    elif compression_mode == "high":
        use_token_compression = True
        compression_ratio = 8.0
        memory_compression_ratio = 16.0
    else:
        raise ValueError(f"Unknown compression mode: {compression_mode}")
    
    # Determine memory settings
    if memory_config == "none":
        use_hierarchical_memory = False
    elif memory_config == "kv_only":
        use_hierarchical_memory = True
        memory_levels = 1
    elif memory_config == "hierarchical":
        use_hierarchical_memory = True
        memory_levels = 3
    else:
        raise ValueError(f"Unknown memory config: {memory_config}")
    
    # Create configuration
    config = UltraContextConfig(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_context_length=max_context,
        active_window_size=active_window,
        sliding_window_size=sliding_window,
        use_hierarchical_memory=use_hierarchical_memory,
        memory_compression_ratio=memory_compression_ratio,
        memory_levels=memory_levels if use_hierarchical_memory else 0,
        use_token_compression=use_token_compression,
        compression_ratio=compression_ratio,
        compression_strategies=["prune", "merge", "summarize"],
        integration_mode=integration_mode,
        position_encoding="adaptive",
        use_retrieval_augmentation=use_hierarchical_memory
    )
    
    return config

def benchmark_ultracontext(
    benchmark_config: BenchmarkConfig,
    max_context: int,
    chunk_size: int,
    compression_mode: str,
    integration_mode: str,
    memory_config: str,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Benchmark UltraContext with specific configuration"""
    logger.info(f"Benchmarking UltraContext - Context: {max_context}, "
               f"Chunk: {chunk_size}, Compression: {compression_mode}, "
               f"Integration: {integration_mode}, Memory: {memory_config}")
    
    # Create model
    model = BaselineModel(
        dim=benchmark_config.dim,
        num_heads=benchmark_config.num_heads
    ).to(device)
    
    # Create UltraContext configuration
    ultra_config = create_ultra_config(
        benchmark_config,
        max_context,
        compression_mode,
        integration_mode,
        memory_config
    )
    
    # Integrate UltraContext
    model_with_ultra = UltraContextAPI.integrate(model, ultra_config)
    
    # Prepare benchmark results
    results = {
        "framework": "ultracontext",
        "max_context": max_context,
        "chunk_size": chunk_size,
        "compression_mode": compression_mode,
        "integration_mode": integration_mode,
        "memory_config": memory_config,
        "prefill_time": [],
        "token_time": [],
        "memory_usage": [],
        "tokens_seen": 0,
        "tokens_in_memory": 0
    }
    
    try:
        # Benchmark prefill phase
        for _ in range(benchmark_config.num_repetitions):
            # Clear context
            UltraContextAPI.clear_context(model_with_ultra)
            
            # Create input
            prefill_size = min(chunk_size, max_context)
            x = torch.randn(1, prefill_size, benchmark_config.dim, device=device)
            
            # Track time and memory
            with track_memory_usage():
                with efficient_inference_mode():
                    start_time = time.time()
                    
                    # Process with UltraContext
                    outputs = UltraContextAPI.process(
                        model_with_ultra,
                        x,
                        is_prefill=True
                    )
                    
                    # Sync device if CUDA
                    if device == "cuda":
                        torch.cuda.synchronize()
                        
                    elapsed = time.time() - start_time
                    
                    # Record results
                    results["prefill_time"].append(elapsed)
                    
                    # Get current memory usage
                    if hasattr(model_with_ultra.ultra_context, "get_context_size"):
                        ctx_size = model_with_ultra.ultra_context.get_context_size()
                        results["tokens_seen"] = ctx_size.get("total_tokens", 0)
                        results["tokens_in_memory"] = ctx_size.get("active_tokens", 0) + ctx_size.get("compressed_tokens", 0)
                        
                    # Record memory usage
                    if device == "cuda":
                        memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
                    else:
                        process = psutil.Process(os.getpid())
                        memory_used = process.memory_info().rss / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
            
            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()
                
        # Benchmark token generation phase (simulate processing 100 tokens)
        token_times = []
        
        for _ in range(100):  # 100 tokens
            # Create input (single token)
            token = torch.randn(1, 1, benchmark_config.dim, device=device)
            
            with efficient_inference_mode():
                start_time = time.time()
                
                # Process with UltraContext
                outputs = UltraContextAPI.process(
                    model_with_ultra,
                    token,
                    is_prefill=False
                )
                
                # Sync device if CUDA
                if device == "cuda":
                    torch.cuda.synchronize()
                    
                elapsed = time.time() - start_time
                token_times.append(elapsed)
                
        # Record token generation time (average)
        results["token_time"] = token_times
        
        # Clear context for cleanup
        UltraContextAPI.clear_context(model_with_ultra)
    
    except Exception as e:
        # Record error
        logger.error(f"Error in UltraContext benchmark: {e}")
        results["error"] = str(e)
    
    # Compute aggregate metrics
    if "prefill_time" in results and results["prefill_time"]:
        results["avg_prefill_time"] = sum(results["prefill_time"]) / len(results["prefill_time"])
        results["prefill_tokens_per_sec"] = prefill_size / results["avg_prefill_time"]
    
    if "token_time" in results and results["token_time"]:
        results["avg_token_time"] = sum(results["token_time"]) / len(results["token_time"])
        results["tokens_per_sec"] = 1.0 / results["avg_token_time"]
    
    if "memory_usage" in results and results["memory_usage"]:
        results["avg_memory_usage"] = sum(results["memory_usage"]) / len(results["memory_usage"])
        
    # Clean up
    del model_with_ultra
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def benchmark_baseline(
    benchmark_config: BenchmarkConfig,
    max_context: int,
    chunk_size: int,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Benchmark baseline model without UltraContext"""
    logger.info(f"Benchmarking baseline - Context: {max_context}, Chunk: {chunk_size}")
    
    # Create model
    model = BaselineModel(
        dim=benchmark_config.dim,
        num_heads=benchmark_config.num_heads
    ).to(device)
    
    # Prepare benchmark results
    results = {
        "framework": "baseline",
        "max_context": max_context,
        "chunk_size": chunk_size,
        "prefill_time": [],
        "token_time": [],
        "memory_usage": []
    }
    
    try:
        # Benchmark prefill phase
        for _ in range(benchmark_config.num_repetitions):
            # Create input
            prefill_size = min(chunk_size, max_context)
            x = torch.randn(1, prefill_size, benchmark_config.dim, device=device)
            
            # Track time and memory
            with track_memory_usage():
                with torch.no_grad():
                    start_time = time.time()
                    
                    # Process with baseline model
                    outputs = model(x)
                    
                    # Sync device if CUDA
                    if device == "cuda":
                        torch.cuda.synchronize()
                        
                    elapsed = time.time() - start_time
                    
                    # Record results
                    results["prefill_time"].append(elapsed)
                    
                    # Record memory usage
                    if device == "cuda":
                        memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
                    else:
                        process = psutil.Process(os.getpid())
                        memory_used = process.memory_info().rss / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
            
            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()
                
        # Benchmark token generation phase (simulate processing 1 token)
        # Note: Without context management, each token would require full context
        token = torch.randn(1, prefill_size + 1, benchmark_config.dim, device=device)
        
        with torch.no_grad():
            start_time = time.time()
            
            # Process with baseline model
            outputs = model(token)
            
            # Sync device if CUDA
            if device == "cuda":
                torch.cuda.synchronize()
                
            elapsed = time.time() - start_time
            
            # Record token generation time
            results["token_time"].append(elapsed)
        
    except Exception as e:
        # Record error
        logger.error(f"Error in baseline benchmark: {e}")
        results["error"] = str(e)
    
    # Compute aggregate metrics
    if "prefill_time" in results and results["prefill_time"]:
        results["avg_prefill_time"] = sum(results["prefill_time"]) / len(results["prefill_time"])
        results["prefill_tokens_per_sec"] = prefill_size / results["avg_prefill_time"]
    
    if "token_time" in results and results["token_time"]:
        results["avg_token_time"] = sum(results["token_time"]) / len(results["token_time"])
        results["tokens_per_sec"] = (prefill_size + 1) / results["avg_token_time"]
    
    if "memory_usage" in results and results["memory_usage"]:
        results["avg_memory_usage"] = sum(results["memory_usage"]) / len(results["memory_usage"])
        
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def benchmark_kv_cache(
    benchmark_config: BenchmarkConfig,
    max_context: int,
    chunk_size: int,
    device="cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Benchmark model with KV cache"""
    logger.info(f"Benchmarking KV cache - Context: {max_context}, Chunk: {chunk_size}")
    
    # Create model
    model = KVCacheModel(
        dim=benchmark_config.dim,
        num_heads=benchmark_config.num_heads,
        max_cache_len=max_context
    ).to(device)
    
    # Prepare benchmark results
    results = {
        "framework": "kv_cache",
        "max_context": max_context,
        "chunk_size": chunk_size,
        "prefill_time": [],
        "token_time": [],
        "memory_usage": []
    }
    
    try:
        # Benchmark prefill phase
        for _ in range(benchmark_config.num_repetitions):
            # Clear KV cache
            model.clear_kv_cache()
            
            # Create input
            prefill_size = min(chunk_size, max_context)
            x = torch.randn(1, prefill_size, benchmark_config.dim, device=device)
            
            # Track time and memory
            with track_memory_usage():
                with torch.no_grad():
                    start_time = time.time()
                    
                    # Process with KV cache model
                    outputs = model(x, use_cache=True)
                    
                    # Sync device if CUDA
                    if device == "cuda":
                        torch.cuda.synchronize()
                        
                    elapsed = time.time() - start_time
                    
                    # Record results
                    results["prefill_time"].append(elapsed)
                    
                    # Record memory usage
                    if device == "cuda":
                        memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
                    else:
                        process = psutil.Process(os.getpid())
                        memory_used = process.memory_info().rss / (1024 * 1024)  # MB
                        results["memory_usage"].append(memory_used)
            
            # Clear cache
            if device == "cuda":
                torch.cuda.empty_cache()
                
        # Benchmark token generation phase (simulate processing 100 tokens)
        token_times = []
        
        for _ in range(100):  # 100 tokens
            # Create input (single token)
            token = torch.randn(1, 1, benchmark_config.dim, device=device)
            
            with torch.no_grad():
                start_time = time.time()
                
                # Process with KV cache model
                outputs = model(token, use_cache=True)
                
                # Sync device if CUDA
                if device == "cuda":
                    torch.cuda.synchronize()
                    
                elapsed = time.time() - start_time
                token_times.append(elapsed)
                
        # Record token generation time (average)
        results["token_time"] = token_times
        
        # Clear KV cache for cleanup
        model.clear_kv_cache()
    
    except Exception as e:
        # Record error
        logger.error(f"Error in KV cache benchmark: {e}")
        results["error"] = str(e)
    
    # Compute aggregate metrics
    if "prefill_time" in results and results["prefill_time"]:
        results["avg_prefill_time"] = sum(results["prefill_time"]) / len(results["prefill_time"])
        results["prefill_tokens_per_sec"] = prefill_size / results["avg_prefill_time"]
    
    if "token_time" in results and results["token_time"]:
        results["avg_token_time"] = sum(results["token_time"]) / len(results["token_time"])
        results["tokens_per_sec"] = 1.0 / results["avg_token_time"]
    
    if "memory_usage" in results and results["memory_usage"]:
        results["avg_memory_usage"] = sum(results["memory_usage"]) / len(results["memory_usage"])
        
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def run_benchmarks(benchmark_config: BenchmarkConfig, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Run all benchmarks based on configuration"""
    results = []
    
    # Log system information
    log_system_info()
    
    # For each context size and chunk size, run the benchmarks
    for context_size in tqdm(benchmark_config.context_sizes, desc="Context Sizes"):
        for chunk_size in tqdm(benchmark_config.chunk_sizes, desc="Chunk Sizes", leave=False):
            # Skip if chunk size is larger than context size
            if chunk_size > context_size:
                continue
                
            # Run baseline benchmark
            if "baseline" in benchmark_config.frameworks:
                try:
                    baseline_results = benchmark_baseline(
                        benchmark_config,
                        context_size,
                        chunk_size,
                        device
                    )
                    results.append(baseline_results)
                    
                    # Save partial results
                    save_results(results, benchmark_config)
                except Exception as e:
                    logger.error(f"Error in baseline benchmark: {e}")
            
            # Run KV cache benchmark
            if "kv_cache" in benchmark_config.frameworks:
                try:
                    kv_cache_results = benchmark_kv_cache(
                        benchmark_config,
                        context_size,
                        chunk_size,
                        device
                    )
                    results.append(kv_cache_results)
                    
                    # Save partial results
                    save_results(results, benchmark_config)
                except Exception as e:
                    logger.error(f"Error in KV cache benchmark: {e}")
            
            # Run UltraContext benchmarks with different configurations
            if "ultracontext" in benchmark_config.frameworks:
                for compression_mode in benchmark_config.compression_modes:
                    for integration_mode in benchmark_config.integration_modes:
                        for memory_config in benchmark_config.memory_configs:
                            try:
                                ultra_results = benchmark_ultracontext(
                                    benchmark_config,
                                    context_size,
                                    chunk_size,
                                    compression_mode,
                                    integration_mode,
                                    memory_config,
                                    device
                                )
                                results.append(ultra_results)
                                
                                # Save partial results
                                save_results(results, benchmark_config)
                            except Exception as e:
                                logger.error(f"Error in UltraContext benchmark: {e}")
    
    return results

def save_results(results, benchmark_config: BenchmarkConfig):
    """Save benchmark results to disk"""
    # Create output directory
    os.makedirs(benchmark_config.output_dir, exist_ok=True)
    
    # Save raw results as JSON
    json_path = os.path.join(benchmark_config.output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary as CSV
    csv_path = os.path.join(benchmark_config.output_dir, "benchmark_summary.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Framework", "Context Size", "Chunk Size", 
            "Compression", "Integration", "Memory Config",
            "Prefill Time", "Prefill Tokens/sec", 
            "Token Time", "Tokens/sec",
            "Memory Usage (MB)"
        ])
        
        # Write rows
        for result in results:
            # Skip if error
            if "error" in result:
                continue
                
            writer.writerow([
                result.get("framework", ""),
                result.get("max_context", ""),
                result.get("chunk_size", ""),
                result.get("compression_mode", ""),
                result.get("integration_mode", ""),
                result.get("memory_config", ""),
                result.get("avg_prefill_time", ""),
                result.get("prefill_tokens_per_sec", ""),
                result.get("avg_token_time", ""),
                result.get("tokens_per_sec", ""),
                result.get("avg_memory_usage", "")
            ])
    
    logger.info(f"Results saved to {json_path} and {csv_path}")

#######################################
# Visualization Functions
#######################################

def plot_benchmark_results(benchmark_config: BenchmarkConfig):
    """Plot benchmark results"""
    # Load results
    json_path = os.path.join(benchmark_config.output_dir, "benchmark_results.json")
    
    if not os.path.exists(json_path):
        logger.error(f"Results file {json_path} not found")
        return
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Create output directory for plots
    plots_dir = os.path.join(benchmark_config.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Prefill speed vs context size
    plt.figure(figsize=(12, 8))
    
    frameworks = set()
    for result in results:
        framework = result.get("framework", "")
        if framework not in frameworks:
            frameworks.add(framework)
    
    for framework in frameworks:
        # Group by context size
        context_sizes = set()
        for result in results:
            if result.get("framework") == framework:
                context_sizes.add(result.get("max_context"))
        
        context_sizes = sorted(context_sizes)
        prefill_speeds = []
        
        for context_size in context_sizes:
            # Get average prefill speed for this context size
            speeds = []
            for result in results:
                if (result.get("framework") == framework and 
                    result.get("max_context") == context_size and
                    "prefill_tokens_per_sec" in result):
                    speeds.append(result.get("prefill_tokens_per_sec"))
            
            if speeds:
                prefill_speeds.append(sum(speeds) / len(speeds))
            else:
                prefill_speeds.append(0)
        
        plt.plot(context_sizes, prefill_speeds, marker='o', label=framework)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Prefill Speed (tokens/second)')
    plt.title('Prefill Speed vs Context Size')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prefill_speed_vs_context.png"))
    
    # Plot 2: Token generation speed vs context size
    plt.figure(figsize=(12, 8))
    
    for framework in frameworks:
        # Group by context size
        context_sizes = set()
        for result in results:
            if result.get("framework") == framework:
                context_sizes.add(result.get("max_context"))
        
        context_sizes = sorted(context_sizes)
        token_speeds = []
        
        for context_size in context_sizes:
            # Get average token speed for this context size
            speeds = []
            for result in results:
                if (result.get("framework") == framework and 
                    result.get("max_context") == context_size and
                    "tokens_per_sec" in result):
                    speeds.append(result.get("tokens_per_sec"))
            
            if speeds:
                token_speeds.append(sum(speeds) / len(speeds))
            else:
                token_speeds.append(0)
        
        plt.plot(context_sizes, token_speeds, marker='o', label=framework)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Token Generation Speed (tokens/second)')
    plt.title('Token Generation Speed vs Context Size')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "token_speed_vs_context.png"))
    
    # Plot 3: Memory usage vs context size
    plt.figure(figsize=(12, 8))
    
    for framework in frameworks:
        # Group by context size
        context_sizes = set()
        for result in results:
            if result.get("framework") == framework:
                context_sizes.add(result.get("max_context"))
        
        context_sizes = sorted(context_sizes)
        memory_usages = []
        
        for context_size in context_sizes:
            # Get average memory usage for this context size
            usages = []
            for result in results:
                if (result.get("framework") == framework and 
                    result.get("max_context") == context_size and
                    "avg_memory_usage" in result):
                    usages.append(result.get("avg_memory_usage"))
            
            if usages:
                memory_usages.append(sum(usages) / len(usages))
            else:
                memory_usages.append(0)
        
        plt.plot(context_sizes, memory_usages, marker='o', label=framework)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Context Size')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "memory_usage_vs_context.png"))
    
    # Plot 4: UltraContext comparison - different compression modes
    if "ultracontext" in frameworks:
        plt.figure(figsize=(12, 8))
        
        compression_modes = set()
        for result in results:
            if result.get("framework") == "ultracontext":
                compression = result.get("compression_mode")
                if compression:
                    compression_modes.add(compression)
        
        for compression in sorted(compression_modes):
            # Group by context size
            context_sizes = set()
            for result in results:
                if (result.get("framework") == "ultracontext" and
                    result.get("compression_mode") == compression):
                    context_sizes.add(result.get("max_context"))
            
            context_sizes = sorted(context_sizes)
            token_speeds = []
            
            for context_size in context_sizes:
                # Get average token speed for this context size and compression
                speeds = []
                for result in results:
                    if (result.get("framework") == "ultracontext" and 
                        result.get("compression_mode") == compression and
                        result.get("max_context") == context_size and
                        "tokens_per_sec" in result):
                        speeds.append(result.get("tokens_per_sec"))
                
                if speeds:
                    token_speeds.append(sum(speeds) / len(speeds))
                else:
                    token_speeds.append(0)
            
            plt.plot(context_sizes, token_speeds, marker='o', label=f"Compression: {compression}")
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Context Size (tokens)')
        plt.ylabel('Token Generation Speed (tokens/second)')
        plt.title('UltraContext: Token Speed vs Compression Mode')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ultracontext_compression_comparison.png"))
    
    # Plot 5: UltraContext comparison - different memory configurations
    if "ultracontext" in frameworks:
        plt.figure(figsize=(12, 8))
        
        memory_configs = set()
        for result in results:
            if result.get("framework") == "ultracontext":
                memory_config = result.get("memory_config")
                if memory_config:
                    memory_configs.add(memory_config)
        
        for memory_config in sorted(memory_configs):
            # Group by context size
            context_sizes = set()
            for result in results:
                if (result.get("framework") == "ultracontext" and
                    result.get("memory_config") == memory_config):
                    context_sizes.add(result.get("max_context"))
            
            context_sizes = sorted(context_sizes)
            memory_usages = []
            
            for context_size in context_sizes:
                # Get average memory usage for this context size and memory config
                usages = []
                for result in results:
                    if (result.get("framework") == "ultracontext" and 
                        result.get("memory_config") == memory_config and
                        result.get("max_context") == context_size and
                        "avg_memory_usage" in result):
                        usages.append(result.get("avg_memory_usage"))
                
                if usages:
                    memory_usages.append(sum(usages) / len(usages))
                else:
                    memory_usages.append(0)
            
            plt.plot(context_sizes, memory_usages, marker='o', label=f"Memory: {memory_config}")
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Context Size (tokens)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('UltraContext: Memory Usage vs Memory Configuration')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ultracontext_memory_comparison.png"))
    
    logger.info(f"Plots saved to {plots_dir}")

#######################################
# Main Function
#######################################

def main():
    """Run benchmarks"""
    parser = argparse.ArgumentParser(description="UltraContext Benchmarks")
    parser.add_argument("--dim", type=int, default=768,
                       help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--context_sizes", type=str, default="1000,10000,100000,1000000",
                       help="Comma-separated list of context sizes to test")
    parser.add_argument("--chunk_sizes", type=str, default="1024,4096,16384",
                       help="Comma-separated list of chunk sizes to test")
    parser.add_argument("--num_repetitions", type=int, default=3,
                       help="Number of repetitions for timing")
    parser.add_argument("--frameworks", type=str, default="ultracontext,baseline,kv_cache",
                       help="Comma-separated list of frameworks to test")
    parser.add_argument("--compression_modes", type=str, default="none,low,medium,high",
                       help="Comma-separated list of compression modes to test")
    parser.add_argument("--integration_modes", type=str, default="extension,replacement,hybrid",
                       help="Comma-separated list of integration modes to test")
    parser.add_argument("--memory_configs", type=str, default="hierarchical,kv_only,none",
                       help="Comma-separated list of memory configurations to test")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--plot_only", action="store_true",
                       help="Only plot existing results, don't run benchmarks")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run benchmarks on")
    args = parser.parse_args()
    
    # Parse list arguments
    context_sizes = [int(x) for x in args.context_sizes.split(",")]
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    frameworks = args.frameworks.split(",")
    compression_modes = args.compression_modes.split(",")
    integration_modes = args.integration_modes.split(",")
    memory_configs = args.memory_configs.split(",")
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        dim=args.dim,
        num_heads=args.num_heads,
        context_sizes=context_sizes,
        chunk_sizes=chunk_sizes,
        num_repetitions=args.num_repetitions,
        frameworks=frameworks,
        compression_modes=compression_modes,
        integration_modes=integration_modes,
        memory_configs=memory_configs,
        output_dir=args.output_dir
    )
    
    # Run benchmarks or plot only
    if args.plot_only:
        plot_benchmark_results(benchmark_config)
    else:
        results = run_benchmarks(benchmark_config, device=args.device)
        save_results(results, benchmark_config)
        plot_benchmark_results(benchmark_config)

if __name__ == "__main__":
    main()
