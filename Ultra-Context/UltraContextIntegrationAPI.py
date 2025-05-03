import torch
import torch.nn as nn
import logging
import os
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
import warnings
from contextlib import contextmanager
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext")

# Import UltraContext components
from ultracontext.core import (
    DEFAULT_PERF_CONFIG, 
    PerformanceConfig,
    UltraContextModule,
    create_ultracontext_network,
    get_norm_class
)

from ultracontext.memory import (
    HierarchicalMemoryManager,
    AdvancedHierarchicalMemoryManager,
    MemorySummarizer,
    export_memory_state,
    restore_memory_state,
    MemorySystemBenchmark,
    StreamingMemoryManager
)

from ultracontext.processing import (
    ContextualCompressor,
    RetrievalAugmentedProcessor,
    HierarchicalProcessingModule,
    TokenStreamProcessor
)

# Memory-efficient context manager
@contextmanager
def efficient_inference_mode():
    """Context manager for memory-efficient inference"""
    # Set up efficient inference
    orig_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    
    # Clear cache before processing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        yield
    finally:
        # Restore original state
        torch.set_grad_enabled(orig_grad_enabled)
        
        # Clear cache after processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Unified Configuration
@dataclass
class UltraContextConfig:
    """
    Unified configuration for UltraContext
    
    Parameters:
        model_dim: Model hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per attention head (if None, calculated from model_dim and num_heads)
        max_context_length: Maximum number of tokens in context
        active_window_size: Size of the active context window for processing
        sliding_window_size: Size of sliding window for attention
        use_memory_system: Whether to use hierarchical memory system
        memory_compression_ratio: Compression ratio for memory system
        use_token_compression: Whether to use token compression
        compression_ratio: Target compression ratio for token compression
        compression_strategies: List of compression strategies to use
        integration_mode: How to integrate with existing model ("extension", "replacement", "hybrid")
        position_encoding: Type of position encoding to use
        use_retrieval_augmentation: Whether to enable retrieval augmentation
        persistence_path: Path for storing persistent memory state
        auto_save_interval: Interval (in seconds) for auto-saving memory state
        perf_config: Performance configuration
    """
    # Model dimensions
    model_dim: int = 768
    num_heads: int = 12
    head_dim: Optional[int] = None
    
    # Context window settings
    max_context_length: int = 100_000_000  # 100M tokens
    active_window_size: int = 16384
    sliding_window_size: int = 8192
    
    # Memory and processing settings
    use_memory_system: bool = True
    memory_compression_ratio: float = 8.0
    use_token_compression: bool = True
    compression_ratio: float = 4.0
    compression_strategies: List[str] = field(default_factory=lambda: ["prune", "merge", "summarize"])
    
    # Integration settings
    integration_mode: str = "extension"  # "extension", "replacement", "hybrid"
    position_encoding: str = "adaptive"  # "absolute", "relative", "rotary", "adaptive"
    
    # Advanced features
    use_retrieval_augmentation: bool = True
    persistence_path: str = "./ultracontext_state"
    auto_save_interval: int = 600  # 10 minutes
    
    # Performance settings
    perf_config: PerformanceConfig = field(default_factory=lambda: DEFAULT_PERF_CONFIG)
    
    def __post_init__(self):
        """Validate and adjust the configuration"""
        # Calculate head_dim if not provided
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
            
        # Ensure max_context_length is reasonable
        if self.max_context_length < self.active_window_size:
            self.max_context_length = self.active_window_size
            
        # Adjust window sizes based on memory constraints
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if total_vram < 8 and self.active_window_size > 8192:
                    # Low VRAM (<8GB)
                    logger.warning(f"Low VRAM detected ({total_vram:.1f}GB), reducing window sizes")
                    self.active_window_size = 8192
                    self.sliding_window_size = 4096
            except:
                logger.warning("Failed to query GPU memory, using default window sizes")
                
        # Adjust memory and compression settings for ultra-long contexts
        if self.max_context_length > 1_000_000:
            # Very long contexts need more aggressive compression
            if self.memory_compression_ratio < 16.0:
                logger.info(f"Adjusting memory compression ratio for ultra-long context: {self.memory_compression_ratio} -> 16.0")
                self.memory_compression_ratio = 16.0
            if self.compression_ratio < 8.0:
                logger.info(f"Adjusting token compression ratio for ultra-long context: {self.compression_ratio} -> 8.0")
                self.compression_ratio = 8.0
                
    def save(self, path: str) -> None:
        """Save configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary, handling non-serializable types
        config_dict = {}
        for key, value in asdict(self).items():
            if key == "perf_config":
                config_dict[key] = asdict(value)
            else:
                config_dict[key] = value
                
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")
            
    @classmethod
    def load(cls, path: str) -> 'UltraContextConfig':
        """Load configuration from file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
            
        # Handle perf_config specially
        if "perf_config" in config_dict:
            perf_config_dict = config_dict.pop("perf_config")
            perf_config = PerformanceConfig(**perf_config_dict)
            config_dict["perf_config"] = perf_config
            
        return cls(**config_dict)
    
    @classmethod
    def from_model(cls, model, **kwargs) -> 'UltraContextConfig':
        """Create configuration from a model"""
        # Detect model dimensions
        config_kwargs = {}
        
        # First try to detect model dimensions from the model config
        if hasattr(model, "config"):
            if hasattr(model.config, "hidden_size"):
                config_kwargs["model_dim"] = model.config.hidden_size
            elif hasattr(model.config, "d_model"):
                config_kwargs["model_dim"] = model.config.d_model
            elif hasattr(model.config, "n_embd"):
                config_kwargs["model_dim"] = model.config.n_embd
                
            # Try to detect number of heads
            if hasattr(model.config, "num_attention_heads"):
                config_kwargs["num_heads"] = model.config.num_attention_heads
            elif hasattr(model.config, "n_head"):
                config_kwargs["num_heads"] = model.config.n_head
                
        # If we couldn't detect from config, try other methods
        if "model_dim" not in config_kwargs:
            # Look for embedding dimension
            for module in model.modules():
                if isinstance(module, nn.Embedding) and not isinstance(module, nn.ParameterList):
                    config_kwargs["model_dim"] = module.embedding_dim
                    break
                    
            # Look for linear layer dimension
            if "model_dim" not in config_kwargs:
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        config_kwargs["model_dim"] = module.out_features
                        break
        
        # If we still couldn't detect, use default
        if "model_dim" not in config_kwargs:
            logger.warning("Could not detect model dimensions, using default (768)")
            config_kwargs["model_dim"] = 768
            
        # Override with user provided kwargs
        config_kwargs.update(kwargs)
        
        return cls(**config_kwargs)

# Main UltraContext class that provides the unified API
class UltraContext:
    """
    Unified API for UltraContext
    
    This class provides a simple, unified interface for using UltraContext
    with any model, managing the context window, memory, and integration.
    
    Usage:
        # Create UltraContext instance
        ultra_ctx = UltraContext(model, max_context_length=1_000_000)
        
        # Integrate with model
        model = ultra_ctx.integrate()
        
        # Process text with extended context
        output = ultra_ctx.generate("Your prompt with very long context...")
    """
    def __init__(
        self, 
        model=None,
        config: Optional[UltraContextConfig] = None,
        **config_kwargs
    ):
        """
        Initialize UltraContext
        
        Args:
            model: Optional model to integrate with
            config: UltraContext configuration
            **config_kwargs: Keyword arguments for configuration
        """
        self.model = model
        
        # Create configuration
        if config is None:
            if model is not None:
                self.config = UltraContextConfig.from_model(model, **config_kwargs)
            else:
                self.config = UltraContextConfig(**config_kwargs)
        else:
            self.config = config
            
        # Initialize components
        self._initialize_components()
        
        # Integration state
        self.is_integrated = False
        self.original_forward = None
        self.context_lock = threading.RLock()
        
        # Context state and storage
        self.context_store = {}  # batch_id -> context_info
        
        # Stats tracking
        self.stats = {
            "processed_tokens": 0,
            "compressed_tokens": 0,
            "retrieved_tokens": 0,
            "prefill_times": [],
            "token_times": [],
        }
        
    def _initialize_components(self):
        """Initialize UltraContext components"""
        # Initialize memory system
        if self.config.use_memory_system:
            logger.info(f"Initializing hierarchical memory system (max tokens: {self.config.max_context_length})")
            self.memory_manager = AdvancedHierarchicalMemoryManager(
                dim=self.config.model_dim,
                l1_capacity=self.config.active_window_size,
                l2_capacity=min(1048576, self.config.active_window_size * 8),
                l3_capacity=min(10485760, self.config.max_context_length // 2),
                disk_capacity=self.config.max_context_length,
                enable_summarization=True,
                enable_streaming=True,
                enable_adaptive_policies=True,
                qos_targets={
                    "l1_latency_ms": 0.1,
                    "l2_latency_ms": 1.0,
                    "l3_latency_ms": 10.0,
                    "disk_latency_ms": 100.0,
                    "hit_rate_l1": 0.9,
                    "hit_rate_l2": 0.8,
                    "hit_rate_l3": 0.7,
                    "availability": 0.9999,
                },
                distributed_nodes=1,
                hardware_acceleration="auto",
                reliability_level="normal"
            )
        else:
            self.memory_manager = None
        
        # Initialize position encoding
        self._initialize_position_encoding()
        
        # Initialize token compression
        if self.config.use_token_compression:
            logger.info(f"Initializing token compression (ratio: {self.config.compression_ratio})")
            self.token_compressor = ContextualCompressor(
                dim=self.config.model_dim,
                target_compression_ratio=self.config.compression_ratio,
                min_tokens_before_compression=self.config.sliding_window_size // 2,
                strategies=self.config.compression_strategies,
                adaptive_compression=True,
                perf_config=self.config.perf_config
            )
        else:
            self.token_compressor = None
        
        # Initialize retrieval augmentation
        if self.config.use_retrieval_augmentation and self.memory_manager is not None:
            logger.info("Initializing retrieval augmentation")
            self.retrieval_processor = RetrievalAugmentedProcessor(
                dim=self.config.model_dim,
                memory_manager=self.memory_manager,
                fusion_type="attention",
                perf_config=self.config.perf_config
            )
        else:
            self.retrieval_processor = None
        
        # Initialize context processor
        memory_type = "streaming" if self.config.integration_mode == "replacement" else "hierarchical"
        
        logger.info(f"Initializing context processor (mode: {memory_type})")
        self.context_processor = create_ultracontext_network(
            dim=self.config.model_dim,
            depth=1,
            memory_type=memory_type,
            window_size=self.config.sliding_window_size,
            max_tokens=self.config.max_context_length,
            use_compression=self.config.use_token_compression,
            compression_type="merge" if self.config.use_token_compression else None,
            perf_config=self.config.perf_config
        )
        
        # Initialize stream processor
        self.stream_processor = TokenStreamProcessor(
            dim=self.config.model_dim,
            memory_manager=self.memory_manager,
            processor=self.context_processor,
            active_window_size=self.config.active_window_size,
            history_compression_ratio=self.config.memory_compression_ratio,
            perf_config=self.config.perf_config
        )
        
    def _initialize_position_encoding(self):
        """Initialize position encoding based on configuration"""
        from ultracontext.position import (
            AdaptiveFourierPositionEncoding,
            RelativePositionEncoding,
            RotaryPositionEncoding
        )
        
        if self.config.position_encoding == "adaptive":
            logger.info("Using Adaptive Fourier Position Encoding")
            self.position_encoding = AdaptiveFourierPositionEncoding(
                dim=self.config.model_dim,
                max_seq_len=self.config.max_context_length,
                adaptive=True,
                trainable=True
            )
        elif self.config.position_encoding == "relative":
            logger.info("Using Relative Position Encoding")
            self.position_encoding = RelativePositionEncoding(
                dim=self.config.model_dim,
                max_seq_len=self.config.max_context_length,
                num_buckets=1024,
                max_distance=self.config.max_context_length,
                heads=self.config.num_heads
            )
        elif self.config.position_encoding == "rotary":
            logger.info("Using Rotary Position Encoding")
            self.position_encoding = RotaryPositionEncoding(
                dim=self.config.model_dim,
                max_seq_len=self.config.max_context_length,
                scaling_factor=1.0
            )
        else:  # "absolute" or default
            logger.info("Using Absolute Position Embeddings")
            self.position_encoding = nn.Embedding(
                self.config.max_context_length,
                self.config.model_dim
            )
            
    def integrate(self, model=None, inplace=True):
        """
        Integrate UltraContext with a model
        
        Args:
            model: Model to integrate with (if not provided, uses model from initialization)
            inplace: Whether to modify the model in-place
            
        Returns:
            Integrated model
        """
        from ultracontext.integration import ModelIntegrator
        
        # Use provided model or fall back to initialized model
        if model is None:
            if self.model is None:
                raise ValueError("No model provided. Either pass a model to integrate() or initialize UltraContext with a model.")
            model = self.model
        else:
            self.model = model
        
        # Integrate with model
        logger.info(f"Integrating UltraContext with model (mode: {self.config.integration_mode})")
        
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        
        # Create integrator
        integrator = ModelIntegrator(self, self.config)
        
        # Integrate with model
        integrated_model = integrator.integrate_with_model(model)
        
        # Mark as integrated
        self.is_integrated = True
        
        return integrated_model
    
    def set_context(self, token_embeddings, batch_idx=0):
        """
        Set context for a specific batch index
        
        Args:
            token_embeddings: Token embeddings to use as context [batch_size, seq_len, dim]
            batch_idx: Batch index for this context
        """
        key = f"context_{batch_idx}"
        
        with self.context_lock:
            self.context_store[key] = {
                "embeddings": token_embeddings,
                "positions": torch.arange(token_embeddings.size(1), device=token_embeddings.device),
                "timestamp": time.time()
            }
            
    def get_context(self, batch_idx=0):
        """
        Get context for a specific batch index
        
        Args:
            batch_idx: Batch index to retrieve context for
            
        Returns:
            Tuple of (embeddings, positions)
        """
        key = f"context_{batch_idx}"
        
        with self.context_lock:
            if key in self.context_store:
                ctx = self.context_store[key]
                return ctx["embeddings"], ctx["positions"]
            
        return None, None
        
    def clear_context(self, batch_idx=None):
        """
        Clear context for a batch or all batches
        
        Args:
            batch_idx: Batch index to clear, or None to clear all
        """
        with self.context_lock:
            if batch_idx is not None:
                key = f"context_{batch_idx}"
                if key in self.context_store:
                    del self.context_store[key]
            else:
                self.context_store.clear()
                
        # Reset stream processor
        self.stream_processor.reset()
        
    def prefill(self, token_embeddings):
        """
        Process initial context tokens (prefill phase)
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, dim]
            
        Returns:
            Processed token embeddings
        """
        start_time = time.time()
        
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Create positions
        positions = torch.arange(
            seq_len, device=token_embeddings.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Apply position encoding
        if isinstance(self.position_encoding, nn.Embedding):
            # Handle absolute embeddings differently
            token_embeddings = token_embeddings + self.position_encoding(positions)
        else:
            token_embeddings = self.position_encoding(token_embeddings, positions)
            
        # Compress if needed and if sequence is long enough
        if self.token_compressor is not None and seq_len > self.config.sliding_window_size:
            token_embeddings, compression_info = self.token_compressor(token_embeddings)
            self.stats["compressed_tokens"] += compression_info.get("tokens_removed", 0)
            
        # Process with stream processor
        result = self.stream_processor.prefill(token_embeddings)
        
        # Store in context
        self.set_context(token_embeddings)
        
        # Update stats
        self.stats["processed_tokens"] += seq_len
        self.stats["prefill_times"].append(time.time() - start_time)
        
        return result
        
    def extend(self, token_embedding):
        """
        Process new token (extension phase)
        
        Args:
            token_embedding: New token embedding [batch_size, 1, dim]
            
        Returns:
            Processed token embedding
        """
        start_time = time.time()
        
        # Apply position encoding
        positions = torch.tensor(
            [[self.stream_processor.current_position]],
            device=token_embedding.device
        ).expand(token_embedding.size(0), -1)
        
        if isinstance(self.position_encoding, nn.Embedding):
            # Handle absolute embeddings differently
            token_embedding = token_embedding + self.position_encoding(positions)
        else:
            token_embedding = self.position_encoding(token_embedding, positions)
            
        # Process with stream processor
        result = self.stream_processor(token_embedding)
        
        # Update context
        ctx_emb, ctx_pos = self.get_context()
        if ctx_emb is not None:
            new_emb = torch.cat([ctx_emb, token_embedding], dim=1)
            new_pos = torch.cat([ctx_pos, positions], dim=1)
            self.set_context(new_emb)
            
        # Update stats
        self.stats["processed_tokens"] += 1
        self.stats["token_times"].append(time.time() - start_time)
        
        return result
        
    def forward(self, token_embeddings, is_prefill=True):
        """
        Process token embeddings
        
        Args:
            token_embeddings: Token embeddings to process
            is_prefill: Whether this is the prefill phase
            
        Returns:
            Processed token embeddings
        """
        if is_prefill:
            return self.prefill(token_embeddings)
        else:
            return self.extend(token_embeddings)
            
    def save_state(self, path=None):
        """
        Save the current state to disk
        
        Args:
            path: Path to save state to, or None to use default path
            
        Returns:
            True if successful
        """
        if path is None:
            # Use default path with timestamp
            os.makedirs(self.config.persistence_path, exist_ok=True)
            timestamp = int(time.time())
            path = os.path.join(self.config.persistence_path, f"ultracontext_state_{timestamp}.bin")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Export memory manager state if available
            if self.memory_manager is not None:
                logger.info(f"Saving memory state to {path}")
                success = export_memory_state(self.memory_manager, path)
                
                if success:
                    # Save stream processor state
                    stream_state_path = path + ".stream"
                    torch.save(self.stream_processor.state_dict(), stream_state_path)
                    
                    # Save additional metadata
                    meta_path = path + ".meta"
                    with open(meta_path, 'w') as f:
                        json.dump({
                            "stats": self.stats,
                            "timestamp": time.time(),
                            "config": asdict(self.config),
                            "version": "1.0.0"
                        }, f)
                        
                    logger.info(f"UltraContext state saved to {path}")
                    return True
                else:
                    logger.error(f"Failed to save memory state to {path}")
                    return False
            else:
                logger.warning("No memory manager available to save state")
                return False
                
        except Exception as e:
            logger.error(f"Error saving state to {path}: {e}")
            return False
            
    def load_state(self, path):
        """
        Load a previously saved state
        
        Args:
            path: Path to load state from
            
        Returns:
            True if successful
        """
        try:
            # Verify file exists
            if not os.path.exists(path):
                logger.error(f"State file not found: {path}")
                return False
                
            # Load memory manager state if available
            if self.memory_manager is not None:
                logger.info(f"Loading memory state from {path}")
                success = restore_memory_state(self.memory_manager, path)
                
                if success:
                    # Load stream processor state
                    stream_state_path = path + ".stream"
                    if os.path.exists(stream_state_path):
                        self.stream_processor.load_state_dict(torch.load(stream_state_path))
                    
                    # Load additional metadata
                    meta_path = path + ".meta"
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            if "stats" in meta:
                                self.stats = meta["stats"]
                                
                    logger.info(f"UltraContext state loaded from {path}")
                    return True
                else:
                    logger.error(f"Failed to load memory state from {path}")
                    return False
            else:
                logger.warning("No memory manager available to load state")
                return False
                
        except Exception as e:
            logger.error(f"Error loading state from {path}: {e}")
            return False
            
    def get_context_size(self):
        """
        Get current context size information
        
        Returns:
            Dictionary with context size statistics
        """
        # Get stats from stream processor
        stats = self.stream_processor.get_stats()
        
        # Add additional stats
        if self.memory_manager is not None:
            mem_stats = self.memory_manager.get_stats()
            stats.update(mem_stats)
            
        return {
            "active_tokens": stats.get("active_window_size", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "compressed_tokens": self.stats.get("compressed_tokens", 0),
            "memory_usage": stats.get("history_size", 0)
        }
        
    def run_benchmarks(self):
        """
        Run benchmarks to evaluate performance
        
        Returns:
            Dictionary with benchmark results
        """
        # Create benchmark results container
        results = {}
        
        # Run memory system benchmarks if available
        if self.memory_manager is not None:
            logger.info("Running memory system benchmarks")
            memory_results = MemorySystemBenchmark.run_all_benchmarks(
                self.memory_manager, 
                dim=self.config.model_dim
            )
            results["memory_system"] = memory_results
            
        # Run position encoding benchmarks
        logger.info("Running position encoding benchmarks")
        pos_results = self._benchmark_position_encoding()
        results["position_encoding"] = pos_results
        
        # Run token compression benchmarks if available
        if self.token_compressor is not None:
            logger.info("Running token compression benchmarks")
            compression_results = self._benchmark_token_compression()
            results["token_compression"] = compression_results
            
        # Run stream processor benchmarks
        logger.info("Running stream processor benchmarks")
        stream_results = self._benchmark_stream_processor()
        results["stream_processor"] = stream_results
        
        return results
        
    def _benchmark_position_encoding(self):
        """Benchmark position encoding performance"""
        results = {}
        device = next(self.position_encoding.parameters()).device
        
        # Test various sequence lengths
        seq_lengths = [1024, 4096, 16384, 65536]
        
        for seq_len in seq_lengths:
            if seq_len > self.config.max_context_length:
                continue
                
            # Create input
            x = torch.randn(1, seq_len, self.config.model_dim, device=device)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Warm up
            for _ in range(3):
                if isinstance(self.position_encoding, nn.Embedding):
                    _ = x + self.position_encoding(positions)
                else:
                    _ = self.position_encoding(x, positions)
                    
            # Measure
            times = []
            for _ in range(10):
                start = time.time()
                if isinstance(self.position_encoding, nn.Embedding):
                    _ = x + self.position_encoding(positions)
                else:
                    _ = self.position_encoding(x, positions)
                times.append(time.time() - start)
                
            avg_time = sum(times) / len(times)
            results[f"seq_len_{seq_len}"] = {
                "avg_time_ms": avg_time * 1000,
                "tokens_per_second": seq_len / avg_time
            }
            
        return results
        
    def _benchmark_token_compression(self):
        """Benchmark token compression performance"""
        if self.token_compressor is None:
            return {"error": "Token compressor not available"}
            
        results = {}
        device = next(self.token_compressor.parameters()).device
        
        # Test various sequence lengths
        seq_lengths = [1024, 4096, 16384, 32768]
        
        for seq_len in seq_lengths:
            if seq_len > self.config.max_context_length:
                continue
                
            # Create input
            x = torch.randn(1, seq_len, self.config.model_dim, device=device)
            
            # Warm up
            for _ in range(3):
                _ = self.token_compressor(x)
                
            # Measure
            times = []
            compression_ratios = []
            for _ in range(5):
                start = time.time()
                compressed, info = self.token_compressor(x)
                times.append(time.time() - start)
                
                # Calculate actual compression ratio
                if hasattr(info, "tokens_removed") and info.tokens_removed is not None:
                    orig_tokens = seq_len
                    compressed_tokens = orig_tokens - info.tokens_removed
                    compression_ratios.append(orig_tokens / max(1, compressed_tokens))
                    
            avg_time = sum(times) / len(times)
            avg_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
            
            results[f"seq_len_{seq_len}"] = {
                "avg_time_ms": avg_time * 1000,
                "tokens_per_second": seq_len / avg_time,
                "compression_ratio": avg_ratio
            }
            
        return results
        
    def _benchmark_stream_processor(self):
        """Benchmark stream processor performance"""
        results = {}
        device = next(self.stream_processor.parameters()).device
        
        # Test prefill with various sequence lengths
        seq_lengths = [1024, 4096, 8192]
        
        for seq_len in seq_lengths:
            if seq_len > self.config.max_context_length:
                continue
                
            # Create input
            x = torch.randn(1, seq_len, self.config.model_dim, device=device)
            
            # Reset stream processor
            self.stream_processor.reset()
            
            # Warm up
            for _ in range(2):
                _ = self.stream_processor.prefill(x)
                self.stream_processor.reset()
                
            # Measure prefill
            prefill_times = []
            for _ in range(3):
                start = time.time()
                _ = self.stream_processor.prefill(x)
                prefill_times.append(time.time() - start)
                self.stream_processor.reset()
                
            # Measure token extension
            token_x = torch.randn(1, 1, self.config.model_dim, device=device)
            _ = self.stream_processor.prefill(x)  # Prefill first
            
            token_times = []
            for _ in range(20):
                start = time.time()
                _ = self.stream_processor(token_x)
                token_times.append(time.time() - start)
                
            avg_prefill_time = sum(prefill_times) / len(prefill_times)
            avg_token_time = sum(token_times) / len(token_times)
            
            results[f"seq_len_{seq_len}"] = {
                "prefill_time_ms": avg_prefill_time * 1000,
                "prefill_tokens_per_second": seq_len / avg_prefill_time,
                "token_time_ms": avg_token_time * 1000,
                "tokens_per_second": 1 / avg_token_time
            }
            
        return results
        
    def optimize(self):
        """Optimize for better performance"""
        logger.info("Optimizing UltraContext for better performance")
        
        # Optimize memory manager if available
        if self.memory_manager is not None:
            logger.info("Optimizing memory manager")
            self.memory_manager.optimize()
            
        # Optimize stream processor
        logger.info("Optimizing stream processor")
        self.stream_processor.optimize()
        
        # Apply torch.compile if requested and available
        if self.config.perf_config.use_torch_compile and hasattr(torch, "compile"):
            logger.info("Applying torch.compile to critical components")
            
            try:
                # Compile position encoding
                if not isinstance(self.position_encoding, nn.Embedding):
                    self.position_encoding.forward = torch.compile(
                        self.position_encoding.forward,
                        mode=self.config.perf_config.compile_mode,
                        fullgraph=False
                    )
                    
                # Compile token compressor
                if self.token_compressor is not None:
                    self.token_compressor.forward = torch.compile(
                        self.token_compressor.forward,
                        mode=self.config.perf_config.compile_mode,
                        fullgraph=False
                    )
                    
                logger.info("Successfully applied torch.compile")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
        
        return True
