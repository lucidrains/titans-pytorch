import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any, Union, Callable
import math
import logging
import time
import os
import json
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc
import threading
from queue import Queue, Empty

# Assuming UltraContext components are imported
from ultracontext.core import (
    DEFAULT_PERF_CONFIG,
    PerformanceConfig,
    get_norm_class,
    UltraContextModule,
    create_ultracontext_network,
    HierarchicalAttention,
    StreamingAttention
)

from ultracontext.memory import (
    HierarchicalMemoryManager,
    PersistentTokenStorage
)

from ultracontext.processing import (
    ContextualCompressor,
    RetrievalAugmentedProcessor,
    HierarchicalProcessingModule,
    TokenStreamProcessor
)

logger = logging.getLogger("ultracontext.integration")

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

# Context state for tracking attention and patterns
class ContextState:
    """Tracks the state of the context throughout processing"""
    def __init__(self, max_size=100_000_000):
        self.max_size = max_size
        self.total_tokens = 0
        self.attention_scores = {}  # position -> average attention score
        self.token_usage = {}       # position -> usage count
        self.recent_positions = []  # Recently accessed positions
        self.key_positions = set()  # Positions identified as important
        
    def record_attention(self, positions, scores):
        """Record attention scores for positions"""
        for pos, score in zip(positions, scores):
            pos = pos.item()
            if pos in self.attention_scores:
                # Exponential moving average
                self.attention_scores[pos] = 0.9 * self.attention_scores[pos] + 0.1 * score.item()
            else:
                self.attention_scores[pos] = score.item()
                
    def record_token_access(self, positions):
        """Record token access"""
        # Add to recent positions (limited size)
        for pos in positions:
            pos = pos.item()
            self.recent_positions.append(pos)
            if len(self.recent_positions) > 1000:
                self.recent_positions.pop(0)
                
            # Update usage count
            self.token_usage[pos] = self.token_usage.get(pos, 0) + 1
            
    def identify_key_positions(self, threshold=0.7):
        """Identify key positions based on attention and usage"""
        # Find positions with high attention scores
        high_attention = {pos for pos, score in self.attention_scores.items() 
                         if score > threshold}
                         
        # Find frequently used positions
        if self.token_usage:
            avg_usage = sum(self.token_usage.values()) / len(self.token_usage)
            frequent_usage = {pos for pos, count in self.token_usage.items() 
                             if count > 2 * avg_usage}
        else:
            frequent_usage = set()
            
        # Combine criteria
        self.key_positions = high_attention.union(frequent_usage)
        
    def get_important_positions(self, max_positions=1024):
        """Get most important positions for context retention"""
        # Ensure key positions are up to date
        self.identify_key_positions()
        
        # Combine key positions with recent positions
        important = list(self.key_positions) + self.recent_positions
        
        # Filter duplicates while maintaining order
        seen = set()
        filtered = []
        for pos in reversed(important):  # Reverse to prioritize newer occurrences
            if pos not in seen:
                seen.add(pos)
                filtered.append(pos)
                
        # Return limited number of positions (most recent first)
        return filtered[:max_positions]
        
    def clear(self):
        """Reset the context state"""
        self.total_tokens = 0
        self.attention_scores.clear()
        self.token_usage.clear()
        self.recent_positions.clear()
        self.key_positions.clear()

# Configuration for UltraContext integration
@dataclass
class UltraContextConfig:
    """Configuration for UltraContext integration"""
    # Model dimensions
    dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    
    # Context window settings
    max_context_length: int = 100_000_000  # 100M tokens
    active_window_size: int = 16384
    sliding_window_size: int = 8192
    
    # Memory system settings
    use_hierarchical_memory: bool = True
    memory_compression_ratio: float = 8.0
    memory_levels: int = 3
    
    # Processing settings
    use_token_compression: bool = True
    compression_ratio: float = 4.0
    compression_strategies: List[str] = field(default_factory=lambda: ["prune", "merge", "summarize"])
    
    # Integration settings
    integration_mode: str = "extension"  # "extension", "replacement", "hybrid"
    position_encoding: str = "adaptive"  # "absolute", "relative", "rotary", "adaptive"
    
    # Advanced features
    use_retrieval_augmentation: bool = True
    use_persistent_storage: bool = False
    storage_path: Optional[str] = None
    
    # Performance settings
    perf_config: PerformanceConfig = field(default_factory=lambda: DEFAULT_PERF_CONFIG)
    
    def __post_init__(self):
        """Validate and adjust the configuration"""
        # Ensure max_context_length is reasonable
        if self.max_context_length < self.active_window_size:
            self.max_context_length = self.active_window_size
            
        # Adjust window sizes based on memory constraints
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if total_vram < 8 and self.active_window_size > 8192:
                # Low VRAM (<8GB)
                logger.warning(f"Low VRAM detected ({total_vram:.1f}GB), reducing window sizes")
                self.active_window_size = 8192
                self.sliding_window_size = 4096
                
        # Adjust memory and compression settings
        if self.max_context_length > 1_000_000:
            # Very long contexts need more aggressive compression
            self.memory_compression_ratio = max(self.memory_compression_ratio, 16.0)
            self.compression_ratio = max(self.compression_ratio, 8.0)
            
    def save(self, path):
        """Save configuration to file"""
        # Convert to dictionary, handling non-serializable types
        config_dict = {}
        for key, value in self.__dict__.items():
            if key == "perf_config":
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
                
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load(cls, path):
        """Load configuration from file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
            
        # Handle perf_config specially
        if "perf_config" in config_dict:
            perf_config_dict = config_dict.pop("perf_config")
            perf_config = PerformanceConfig(**perf_config_dict)
            config_dict["perf_config"] = perf_config
            
        return cls(**config_dict)

# Base class for position encodings
class PositionEncoding(nn.Module):
    """Base class for position encodings"""
    def __init__(self, dim, max_seq_len=100_000_000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
    def forward(self, x, positions=None):
        """
        Apply position encoding
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions
            
        Returns:
            Position-encoded embeddings
        """
        raise NotImplementedError("Subclasses must implement forward method")

# Adaptive Fourier Position Encoding for ultra-long contexts
class AdaptiveFourierPositionEncoding(PositionEncoding):
    """
    Adaptive Fourier Position Encoding for handling ultra-long contexts
    
    Features:
    - Logarithmic frequency distribution for better long-range modeling
    - Adaptive attention to learned frequency components
    - Scale-invariant position representation
    """
    def __init__(
        self, 
        dim, 
        max_seq_len=100_000_000,
        num_bands=64,
        max_frequency=10000.0,
        adaptive=True,
        trainable=True
    ):
        super().__init__(dim, max_seq_len)
        self.num_bands = num_bands
        self.max_frequency = max_frequency
        self.adaptive = adaptive
        
        # Number of frequency bands
        assert dim % (2 * num_bands) == 0, f"Dimension {dim} not divisible by {2 * num_bands}"
        
        # Initialize frequency bands (logarithmic scale)
        if trainable:
            # Trainable frequency bands
            self.frequency_bands = nn.Parameter(
                torch.exp(torch.linspace(
                    math.log(1.0), math.log(max_frequency), num_bands
                ))
            )
        else:
            # Fixed frequency bands
            self.register_buffer(
                "frequency_bands",
                torch.exp(torch.linspace(
                    math.log(1.0), math.log(max_frequency), num_bands
                ))
            )
            
        # Phase shift for each frequency band
        if trainable:
            self.phase_shifts = nn.Parameter(torch.zeros(num_bands))
        else:
            self.register_buffer("phase_shifts", torch.zeros(num_bands))
            
        # Adaptive attention to frequency bands
        if adaptive:
            self.frequency_attention = nn.Sequential(
                nn.Linear(dim, num_bands),
                nn.Softmax(dim=-1)
            )
            
    def _compute_fourier_encodings(self, positions):
        """
        Compute Fourier position encodings
        
        Args:
            positions: Position indices [batch_size, seq_len]
            
        Returns:
            Fourier encodings [batch_size, seq_len, dim]
        """
        batch_size, seq_len = positions.shape
        
        # Convert positions to float
        pos_float = positions.float()
        
        # Compute encodings for each frequency band
        band_encodings = []
        
        for band_idx in range(self.num_bands):
            # Get frequency and phase for this band
            freq = self.frequency_bands[band_idx]
            phase = self.phase_shifts[band_idx]
            
            # Compute sinusoidal encoding
            angle = pos_float * freq + phase
            sin_encoding = torch.sin(angle)
            cos_encoding = torch.cos(angle)
            
            # Interleave sin and cos
            band_encoding = torch.stack([sin_encoding, cos_encoding], dim=-1)
            band_encoding = band_encoding.view(batch_size, seq_len, 2)
            
            band_encodings.append(band_encoding)
            
        # Concatenate all band encodings
        encodings = torch.cat(band_encodings, dim=-1)
        
        # Reshape to match dimension
        encodings = encodings.view(batch_size, seq_len, self.dim)
        
        return encodings
        
    def forward(self, x, positions=None):
        """
        Apply position encoding
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions [batch_size, seq_len]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Compute Fourier encodings
        encodings = self._compute_fourier_encodings(positions)
        
        if self.adaptive:
            # Compute attention weights for frequency bands
            freq_attention = self.frequency_attention(x)  # [batch_size, seq_len, num_bands]
            
            # Reshape attention to match encodings
            attention = freq_attention.view(batch_size, seq_len, self.num_bands, 1)
            attention = attention.expand(-1, -1, -1, 2)  # Expand for sin/cos
            attention = attention.contiguous().view(batch_size, seq_len, self.dim)
            
            # Apply attention weights to encodings
            encodings = encodings * attention
            
        # Add position encodings to input embeddings
        return x + encodings

# Relative Position Encoding implementation
class RelativePositionEncoding(PositionEncoding):
    """
    Efficient relative position encoding for long contexts
    
    Features:
    - Bucket-based relative position representation
    - Efficient attention computation
    - Scales well to ultra-long contexts
    """
    def __init__(
        self,
        dim,
        max_seq_len=100_000_000,
        num_buckets=256,
        max_distance=16384,
        trainable=True,
        heads=1
    ):
        super().__init__(dim, max_seq_len)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads
        
        # Create relative position embedding table
        if trainable:
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_buckets, heads, dim // heads)
            )
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)
        else:
            self.register_buffer(
                "rel_pos_embeddings",
                torch.zeros(num_buckets, heads, dim // heads)
            )
            
    def _relative_position_bucket(self, relative_position):
        """
        Bucket relative positions for efficiency
        
        Args:
            relative_position: Relative position tensor
            
        Returns:
            Bucketed indices
        """
        # Get absolute values for symmetric bucketing
        relative_buckets = torch.abs(relative_position)
        
        # Create logarithmic buckets
        max_exact = self.num_buckets // 4
        is_small = relative_buckets < max_exact
        
        # For small distances, use exact bucketing
        relative_pos_if_small = relative_buckets
        
        # For larger distances, use logarithmic bucketing
        relative_pos_if_large = max_exact + (
            torch.log(relative_buckets.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets // 2 - max_exact)
        ).long()
        
        # Clamp to valid bucket range
        relative_pos_if_large = torch.clamp(
            relative_pos_if_large, max=self.num_buckets // 2 - 1
        )
        
        # Use small/large bucketing based on condition
        relative_buckets = torch.where(
            is_small, relative_pos_if_small, relative_pos_if_large
        )
        
        # Handle sign for asymmetric bucketing
        relative_buckets = torch.where(
            relative_position < 0,
            self.num_buckets // 2 + relative_buckets,
            relative_buckets
        )
        
        return relative_buckets
        
    def forward(self, x, positions=None):
        """
        Get position embeddings (for later use in attention)
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions [batch_size, seq_len]
            
        Returns:
            Position embeddings and position indices
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Store positions for later use in attention
        self.last_positions = positions
        
        # In relative position encoding, we don't modify the input directly
        # Instead, we compute relative position embeddings in the attention mechanism
        return x, positions
        
    def get_rel_pos_embeddings(self, q_positions, k_positions):
        """
        Get relative position embeddings for attention
        
        Args:
            q_positions: Query positions [batch_size, q_len]
            k_positions: Key positions [batch_size, k_len]
            
        Returns:
            Relative position embeddings [batch_size, heads, q_len, k_len, head_dim]
        """
        batch_size, q_len = q_positions.shape
        k_len = k_positions.shape[1]
        device = q_positions.device
        
        # Reshape positions for broadcasting
        q_pos = q_positions.unsqueeze(-1)  # [batch, q_len, 1]
        k_pos = k_positions.unsqueeze(1)   # [batch, 1, k_len]
        
        # Compute relative positions
        relative_position = q_pos - k_pos  # [batch, q_len, k_len]
        
        # Bucket relative positions
        rel_pos_bucket = self._relative_position_bucket(relative_position)  # [batch, q_len, k_len]
        
        # Get embeddings from bucket indices
        embeddings = self.rel_pos_embeddings[rel_pos_bucket]  # [batch, q_len, k_len, heads, head_dim]
        
        # Transpose for attention
        embeddings = embeddings.permute(0, 3, 1, 2, 4)  # [batch, heads, q_len, k_len, head_dim]
        
        return embeddings

# Rotary Position Embedding (RoPE) implementation
class RotaryPositionEncoding(PositionEncoding):
    """
    Rotary Position Embeddings (RoPE) for ultra-long contexts
    
    Features:
    - Theoretically unbounded position range
    - Efficient computation (applied directly to queries and keys)
    - Preserves relative position information implicitly
    """
    def __init__(
        self,
        dim,
        max_seq_len=100_000_000,
        base=10000.0,
        scaling_factor=1.0,
        trainable=False,
        interleaved=True
    ):
        super().__init__(dim, max_seq_len)
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        
        # Number of features (half the dimension for complex representation)
        self.num_features = dim // 2
        
        # Initialize frequency bands
        if trainable:
            # Trainable frequencies (uncommon but possible)
            self.inv_freq = nn.Parameter(
                1.0 / (base ** (torch.arange(0, self.num_features, 2).float() / self.num_features))
            )
        else:
            # Fixed frequencies (standard approach)
            self.register_buffer(
                "inv_freq",
                1.0 / (base ** (torch.arange(0, self.num_features, 2).float() / self.num_features))
            )
            
        # Cache for cos/sin table
        self.cos_sin_cache = {}
        
    def _get_cos_sin_cache(self, positions, device):
        """
        Get cached or compute cos/sin for positions
        
        Args:
            positions: Position indices
            device: Computation device
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        # Generate key for positions
        key = f"{positions.device}_{positions.shape}_{positions.min().item()}_{positions.max().item()}"
        
        if key in self.cos_sin_cache:
            return self.cos_sin_cache[key]
            
        # Scale positions
        positions = positions.float() * self.scaling_factor
        
        # Compute cos/sin table
        freqs = torch.outer(positions, self.inv_freq)
        cos = torch.cos(freqs).to(device)
        sin = torch.sin(freqs).to(device)
        
        # Match dimensions
        if self.interleaved:
            # Interleaved - repeat to match dimension
            cos = torch.repeat_interleave(cos, 2, dim=-1)
            sin = torch.repeat_interleave(sin, 2, dim=-1)
        else:
            # Blocked - reshape and repeat
            emb_dim = self.num_features
            cos = cos.view(*cos.shape[:-1], -1, 2 * cos.shape[-1])
            sin = sin.view(*sin.shape[:-1], -1, 2 * sin.shape[-1])
            cos = cos.repeat(*[1] * len(cos.shape[:-2]), 1, emb_dim // cos.shape[-1])
            sin = sin.repeat(*[1] * len(sin.shape[:-2]), 1, emb_dim // sin.shape[-1])
            cos = cos.view(*cos.shape[:-2], -1)
            sin = sin.view(*sin.shape[:-2], -1)
            
        # Cache results
        self.cos_sin_cache[key] = (cos, sin)
        
        return cos, sin
        
    def _rotate_half(self, x):
        """Rotate half the hidden dims of x"""
        x1 = x[..., : self.num_features]
        x2 = x[..., self.num_features : 2 * self.num_features]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, x, positions=None):
        """
        Apply rotary position encoding
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions [batch_size, seq_len]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Get cos/sin tables
        cos, sin = self._get_cos_sin_cache(
            positions.view(-1).unique().sort()[0], device
        )
        
        # Index into cos/sin tables
        cos = cos[positions.view(-1)]
        sin = sin[positions.view(-1)]
        
        # Reshape to match input
        cos = cos.view(*positions.shape, -1)
        sin = sin.view(*positions.shape, -1)
        
        # Apply rotary embeddings
        # This applies the rotation in the complex plane
        x_rope = (x * cos) + (self._rotate_half(x) * sin)
        
        return x_rope
        
    def apply_rotary_to_query_key(self, q, k, positions=None):
        """
        Apply rotary embeddings to query and key tensors
        
        Args:
            q: Query tensor [batch_size, heads, seq_len, head_dim]
            k: Key tensor [batch_size, heads, seq_len, head_dim]
            positions: Optional positions [batch_size, seq_len]
            
        Returns:
            Rotary-encoded query and key tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        
        # Generate positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
        # Get cos/sin tables
        cos, sin = self._get_cos_sin_cache(
            positions.view(-1).unique().sort()[0], device
        )
        
        # Index into cos/sin tables
        cos = cos[positions.view(-1)]
        sin = sin[positions.view(-1)]
        
        # Reshape to match query/key
        cos = cos.view(batch_size, seq_len, 1, head_dim)
        sin = sin.view(batch_size, seq_len, 1, head_dim)
        
        # Apply rotary embeddings to query and key
        q_rope = (q * cos) + (self._rotate_half_qk(q) * sin)
        k_rope = (k * cos) + (self._rotate_half_qk(k) * sin)
        
        return q_rope, k_rope
        
    def _rotate_half_qk(self, x):
        """Rotate half the hidden dims of query/key tensors"""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)

# Base UltraContext wrapper for model integration
class UltraContextWrapper(nn.Module):
    """
    Base wrapper for integrating UltraContext with existing models
    
    This class handles:
    - Initialization of all UltraContext components
    - Integration with host model's attention mechanism
    - Context management and optimization
    """
    def __init__(
        self,
        config: UltraContextConfig,
        host_model=None  # Optional host model to integrate with
    ):
        super().__init__()
        self.config = config
        self.host_model = host_model
        self.dim = config.dim
        
        # Initialize components
        self._init_components()
        
        # Set up position encoding
        self._init_position_encoding()
        
        # Context state tracking
        self.context_state = ContextState(max_size=config.max_context_length)
        
        # Streaming processor for real-time token handling
        self.stream_processor = TokenStreamProcessor(
            dim=config.dim,
            memory_manager=self.memory_manager if config.use_hierarchical_memory else None,
            processor=self.context_processor,
            active_window_size=config.active_window_size,
            history_compression_ratio=config.memory_compression_ratio,
            perf_config=config.perf_config
        )
        
        # Thread-safe storage for context
        self.context_store = {}
        self.context_lock = threading.RLock()
        
    def _init_components(self):
        """Initialize UltraContext components"""
        # Initialize memory system if requested
        if self.config.use_hierarchical_memory:
            self.memory_manager = HierarchicalMemoryManager(
                dim=self.dim,
                l1_capacity=self.config.active_window_size,
                l2_capacity=self.config.active_window_size * 8,
                l3_capacity=self.config.max_context_length,
                l2_compression_ratio=self.config.memory_compression_ratio / 2,
                l3_compression_ratio=self.config.memory_compression_ratio,
                perf_config=self.config.perf_config
            )
            
            # Add persistent storage if requested
            if self.config.use_persistent_storage and self.config.storage_path:
                self.persistent_storage = PersistentTokenStorage(
                    dim=self.dim,
                    storage_path=self.config.storage_path,
                    max_tokens=self.config.max_context_length,
                    compression_ratio=self.config.memory_compression_ratio * 2
                )
            else:
                self.persistent_storage = None
        else:
            self.memory_manager = None
            self.persistent_storage = None
            
        # Initialize token compression if requested
        if self.config.use_token_compression:
            self.token_compressor = ContextualCompressor(
                dim=self.dim,
                target_compression_ratio=self.config.compression_ratio,
                min_tokens_before_compression=self.config.sliding_window_size // 2,
                strategies=self.config.compression_strategies,
                adaptive_compression=True,
                perf_config=self.config.perf_config
            )
        else:
            self.token_compressor = None
            
        # Initialize retrieval augmentation if requested
        if self.config.use_retrieval_augmentation and self.memory_manager is not None:
            self.retrieval_processor = RetrievalAugmentedProcessor(
                dim=self.dim,
                memory_manager=self.memory_manager,
                fusion_type="attention",
                perf_config=self.config.perf_config
            )
        else:
            self.retrieval_processor = None
            
        # Initialize main context processor
        memory_type = "streaming" if self.config.integration_mode == "replacement" else "hierarchical"
        
        self.context_processor = create_ultracontext_network(
            dim=self.dim,
            depth=1,
            memory_type=memory_type,
            window_size=self.config.sliding_window_size,
            max_tokens=self.config.max_context_length,
            use_compression=self.config.use_token_compression,
            compression_type="merge" if self.config.use_token_compression else None,
            perf_config=self.config.perf_config
        )
        
    def _init_position_encoding(self):
        """Initialize position encoding based on configuration"""
        if self.config.position_encoding == "adaptive":
            self.position_encoding = AdaptiveFourierPositionEncoding(
                dim=self.dim,
                max_seq_len=self.config.max_context_length,
                adaptive=True,
                trainable=True
            )
        elif self.config.position_encoding == "relative":
            self.position_encoding = RelativePositionEncoding(
                dim=self.dim,
                max_seq_len=self.config.max_context_length,
                num_buckets=1024,
                max_distance=self.config.max_context_length,
                heads=self.config.num_heads
            )
        elif self.config.position_encoding == "rotary":
            self.position_encoding = RotaryPositionEncoding(
                dim=self.dim,
                max_seq_len=self.config.max_context_length,
                scaling_factor=1.0
            )
        else:  # "absolute" or default
            # Simple learned position embeddings
            self.position_encoding = nn.Embedding(
                self.config.max_context_length,
                self.dim
            )
            
    def _get_store_key(self, batch_idx=0):
        """Get key for context store"""
        return f"context_{batch_idx}"
        
    def set_context(self, token_embeddings, batch_idx=0):
        """
        Set context for a specific batch index
        
        Args:
            token_embeddings: Token embeddings to use as context
            batch_idx: Batch index for this context
        """
        key = self._get_store_key(batch_idx)
        
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
        key = self._get_store_key(batch_idx)
        
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
                key = self._get_store_key(batch_idx)
                if key in self.context_store:
                    del self.context_store[key]
            else:
                self.context_store.clear()
                
        # Reset stream processor
        self.stream_processor.reset()
        
        # Clear context state
        self.context_state.clear()
        
    def prefill(self, token_embeddings):
        """
        Process initial context tokens (prefill phase)
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, dim]
            
        Returns:
            Processed token embeddings
        """
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
            token_embeddings, _ = self.token_compressor(token_embeddings)
            
        # Process with stream processor
        result = self.stream_processor.prefill(token_embeddings)
        
        # Store in context
        self.set_context(token_embeddings)
        
        return result
        
    def extend(self, token_embedding):
        """
        Process new token (extension phase)
        
        Args:
            token_embedding: New token embedding [batch_size, 1, dim]
            
        Returns:
            Processed token embedding
        """
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
            
    def get_context_size(self):
        """Get current context size information"""
        stats = self.stream_processor.get_stats()
        
        return {
            "active_tokens": stats.get("active_window_size", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "compressed_tokens": stats.get("compressed_tokens", 0),
            "memory_usage": stats.get("history_size", 0)
        }

# Integrated Attention layer that supports UltraContext
class UltraContextAttention(nn.Module):
    """
    Attention layer with UltraContext integration
    
    This layer either wraps an existing attention layer or provides
    its own implementation with UltraContext capabilities.
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        dropout=0.0,
        original_attn=None,  # Original attention module to wrap
        integration_mode="extension",  # "extension", "replacement", "hybrid"
        window_size=8192,
        max_context_length=100_000_000,
        ultra_wrapper=None,  # UltraContextWrapper instance
        position_encoding=None,  # Optional position encoding module
        perf_config=DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.integration_mode = integration_mode
        self.window_size = window_size
        self.max_context_length = max_context_length
        
        # Keep original attention if provided
        self.original_attn = original_attn
        
        # Use provided wrapper if available
        self.ultra_wrapper = ultra_wrapper
        
        # Use provided position encoding if available
        self.position_encoding = position_encoding
        
        # If in replacement mode, create our own attention
        if integration_mode == "replacement" or original_attn is None:
            # Define projections
            self.to_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_k = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_v = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_out = nn.Linear(num_heads * self.head_dim, dim, bias=False)
            
            # Define attention implementation
            if perf_config.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
                self.attn_fn = lambda q, k, v, mask: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout
                )
            else:
                self.attn_fn = self._standard_attention
                self.attn_dropout = nn.Dropout(dropout)
                
        # For streaming in extension mode with very long contexts
        if integration_mode == "extension" and max_context_length > window_size * 10:
            # Create streaming attention for the extended context
            self.streaming_attn = StreamingAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=self.head_dim,
                window_size=window_size,
                max_kv_cache=min(1_000_000, max_context_length),
                dropout=dropout,
                causal=True,
                perf_config=perf_config
            )
            
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and hasattr(torch, "compile"):
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode,
                fullgraph=False  # State changes are not compatible with fullgraph
            )
            
    def _standard_attention(self, q, k, v, mask=None):
        """Standard attention implementation"""
        scale = q.shape[-1] ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Combine with values
        out = torch.matmul(attn, v)
        
        return out
        
    def _process_extended_context(self, x, ultra_context=None):
        """
        Process with extended context using UltraContext
        
        Args:
            x: Current input embeddings
            ultra_context: Optional UltraContext tensor
            
        Returns:
            Extended attention output
        """
        if ultra_context is None or self.ultra_wrapper is None:
            return None
            
        # Extract context embeddings
        context_emb, context_pos = ultra_context
        
        if context_emb is None:
            return None
            
        if hasattr(self, "streaming_attn"):
            # Use streaming attention for very long contexts
            # Combine current input with context
            combined = torch.cat([context_emb, x], dim=1)
            
            # Process with streaming attention
            out = self.streaming_attn(combined)
            
            # Extract only the current input part
            return out[:, -x.size(1):]
        else:
            # Use context processor from wrapper
            # Apply position encoding if available
            if self.position_encoding is not None:
                if isinstance(self.position_encoding, nn.Embedding):
                    # Handle absolute embeddings
                    pos_ids = torch.arange(context_emb.size(1), device=context_emb.device)
                    context_emb = context_emb + self.position_encoding(pos_ids)
                else:
                    # Handle other position encodings
                    context_emb = self.position_encoding(context_emb, context_pos)
                    
            # Use UltraContext processor
            extended_attn = self.ultra_wrapper.context_processor(x)
            
            return extended_attn
            
    def _process_with_replacement(self, x, positions=None, ultra_context=None):
        """
        Process input using replacement mode (UltraContext handles everything)
        
        Args:
            x: Input embeddings
            positions: Optional position indices
            ultra_context: Optional UltraContext information
            
        Returns:
            Processed output
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position encoding if available
        if isinstance(self.position_encoding, RotaryPositionEncoding):
            q, k = self.position_encoding.apply_rotary_to_query_key(q, k, positions)
            
        # Apply relative position encoding if available
        rel_pos_bias = None
        if isinstance(self.position_encoding, RelativePositionEncoding) and positions is not None:
            rel_pos_bias = self.position_encoding.get_rel_pos_embeddings(
                positions, positions
            )
            
        # Create causal mask
        mask = torch.ones(
            (seq_len, seq_len), device=x.device, dtype=torch.bool
        ).triu_(1).unsqueeze(0).unsqueeze(0)
        
        # Compute attention
        attn_output = self.attn_fn(q, k, v, mask)
        
        # Apply relative position bias if available
        if rel_pos_bias is not None:
            attn_output = attn_output + rel_pos_bias
            
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Project back to original dimension
        output = self.to_out(attn_output)
        
        return output
        
    def _process_with_extension(self, x, positions=None, ultra_context=None):
        """
        Process input using extension mode (Original attention + UltraContext)
        
        Args:
            x: Input embeddings
            positions: Optional position indices
            ultra_context: Optional UltraContext information
            
        Returns:
            Processed output
        """
        # Process with original attention
        if self.original_attn is not None:
            orig_output = self.original_attn(x)
        else:
            # Fallback if no original attention
            orig_output = self._process_with_replacement(x, positions)
            
        # Process with extended context
        extended_output = self._process_extended_context(x, ultra_context)
        
        if extended_output is not None:
            # Combine original and extended outputs
            # Simple averaging for now - could be made adaptive
            return (orig_output + extended_output) / 2
        else:
            return orig_output
            
    def _process_with_hybrid(self, x, positions=None, ultra_context=None):
        """
        Process input using hybrid mode (Adaptive combination)
        
        Args:
            x: Input embeddings
            positions: Optional position indices
            ultra_context: Optional UltraContext information
            
        Returns:
            Processed output
        """
        # Process with original attention
        if self.original_attn is not None:
            orig_output = self.original_attn(x)
        else:
            # Fallback if no original attention
            orig_output = self._process_with_replacement(x, positions)
            
        # Process with extended context
        extended_output = self._process_extended_context(x, ultra_context)
        
        if extended_output is None:
            return orig_output
            
        # Compute gating values (adaptive mixing)
        # Project both outputs to a common space
        if not hasattr(self, "hybrid_gate"):
            self.hybrid_gate = nn.Sequential(
                nn.Linear(self.dim * 2, self.dim // 2),
                nn.GELU(),
                nn.Linear(self.dim // 2, 1),
                nn.Sigmoid()
            )
            
        # Concatenate both outputs
        concat = torch.cat([orig_output, extended_output], dim=-1)
        
        # Compute gate values
        gates = self.hybrid_gate(concat)
        
        # Apply gating
        combined = orig_output * (1 - gates) + extended_output * gates
        
        return combined
        
    def forward(self, x, positions=None, ultra_context=None):
        """
        Forward pass with UltraContext integration
        
        Args:
            x: Input embeddings [batch_size, seq_len, dim]
            positions: Optional position indices [batch_size, seq_len]
            ultra_context: Optional UltraContext information
            
        Returns:
            Processed output [batch_size, seq_len, dim]
        """
        # Process based on integration mode
        if self.integration_mode == "replacement":
            return self._process_with_replacement(x, positions, ultra_context)
        elif self.integration_mode == "extension":
            return self._process_with_extension(x, positions, ultra_context)
        elif self.integration_mode == "hybrid":
            return self._process_with_hybrid(x, positions, ultra_context)
        else:
            # Fallback to original attention
            if self.original_attn is not None:
                return self.original_attn(x)
            else:
                return self._process_with_replacement(x, positions)

# Main integration utilities for applying UltraContext to existing models
class UltraContextIntegrator:
    """
    Utilities for integrating UltraContext with existing models
    
    Supports integration with:
    - Hugging Face Transformers models
    - PyTorch models with attention mechanisms
    - Custom model architectures
    """
    @staticmethod
    def integrate_with_model(
        model,
        config: UltraContextConfig,
        attention_layer_path="attention",
        position_encoding_path=None,
        inplace=True
    ):
        """
        Integrate UltraContext with an existing model
        
        Args:
            model: Model to integrate with
            config: UltraContext configuration
            attention_layer_path: Path to attention layer(s) in model
            position_encoding_path: Path to position encoding in model
            inplace: Whether to modify the model in-place
            
        Returns:
            Integrated model
        """
        if not inplace:
            # Create a copy of the model
            model = copy.deepcopy(model)
            
        # Create UltraContext wrapper
        ultra_wrapper = UltraContextWrapper(config, host_model=model)
        
        # Find and modify attention layers
        if attention_layer_path == "auto":
            # Automatic detection of attention layers
            attention_layers = UltraContextIntegrator._find_attention_layers(model)
        else:
            # Manual specification of attention layers
            attention_layers = UltraContextIntegrator._get_layers_by_path(model, attention_layer_path)
            
        # Find position encoding if specified
        position_encoding = None
        if position_encoding_path:
            position_encodings = UltraContextIntegrator._get_layers_by_path(model, position_encoding_path)
            if position_encodings:
                position_encoding = position_encodings[0]
                
        # Replace or wrap attention layers
        for i, attn_layer in enumerate(attention_layers):
            UltraContextIntegrator._replace_attention(
                model, attn_layer, ultra_wrapper, config, position_encoding
            )
            
        # Add UltraContext wrapper to model
        model.ultra_context = ultra_wrapper
        
        # Add helper methods to model
        model.set_ultracontext = ultra_wrapper.set_context
        model.get_ultracontext = ultra_wrapper.get_context
        model.clear_ultracontext = ultra_wrapper.clear_context
        
        return model
        
    @staticmethod
    def _find_attention_layers(model):
        """Automatically find attention layers in model"""
        attention_layers = []
        
        # Common attention layer class names
        attention_names = [
            "Attention", "MultiheadAttention", "SelfAttention",
            "FlashAttention", "AttnLayer", "MHA"
        ]
        
        # Search model modules
        for name, module in model.named_modules():
            module_name = module.__class__.__name__
            if any(attn_name in module_name for attn_name in attention_names):
                attention_layers.append((name, module))
                
        return attention_layers
        
    @staticmethod
    def _get_layers_by_path(model, path):
        """Get layers by path specification"""
        if path is None:
            return []
            
        layers = []
        
        # Handle multiple paths separated by commas
        if ',' in path:
            paths = path.split(',')
            for p in paths:
                layers.extend(UltraContextIntegrator._get_layers_by_path(model, p.strip()))
            return layers
            
        # Handle wildcard paths
        if '*' in path:
            parts = path.split('.')
            pattern = re.compile('^' + '\\.'.join([p.replace('*', '.*') for p in parts]) + '$')
            
            for name, module in model.named_modules():
                if pattern.match(name):
                    layers.append((name, module))
                    
            return layers
            
        # Handle direct path
        parts = path.split('.')
        current = model
        
        try:
            for part in parts:
                current = getattr(current, part)
            
            if isinstance(current, list):
                return [(f"{path}[{i}]", module) for i, module in enumerate(current)]
            elif isinstance(current, nn.ModuleList):
                return [(f"{path}[{i}]", module) for i, module in enumerate(current)]
            else:
                return [(path, current)]
        except (AttributeError, IndexError):
            logger.warning(f"Could not find layer at path: {path}")
            return []
            
    @staticmethod
    def _replace_attention(model, attn_info, ultra_wrapper, config, position_encoding=None):
        """Replace or wrap an attention layer"""
        name, attn_layer = attn_info
        
        # Get parent module and attribute name
        if '.' in name:
            parent_name, attr_name = name.rsplit('.', 1)
            parent = UltraContextIntegrator._get_module_by_path(model, parent_name)
        else:
            parent = model
            attr_name = name
            
        # Handle indexed access (for lists)
        if '[' in attr_name and ']' in attr_name:
            list_name, idx_str = attr_name.split('[')
            idx = int(idx_str.replace(']', ''))
            module_list = getattr(parent, list_name)
            
            # Create new attention layer
            new_attn = UltraContextAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                original_attn=attn_layer,
                integration_mode=config.integration_mode,
                window_size=config.sliding_window_size,
                max_context_length=config.max_context_length,
                ultra_wrapper=ultra_wrapper,
                position_encoding=position_encoding or ultra_wrapper.position_encoding,
                perf_config=config.perf_config
            )
            
            # Replace in list
            module_list[idx] = new_attn
        else:
            # Create new attention layer
            new_attn = UltraContextAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                original_attn=attn_layer,
                integration_mode=config.integration_mode,
                window_size=config.sliding_window_size,
                max_context_length=config.max_context_length,
                ultra_wrapper=ultra_wrapper,
                position_encoding=position_encoding or ultra_wrapper.position_encoding,
                perf_config=config.perf_config
            )
            
            # Replace attribute
            setattr(parent, attr_name, new_attn)
            
    @staticmethod
    def _get_module_by_path(model, path):
        """Get module by path"""
        parts = path.split('.')
        current = model
        
        for part in parts:
            # Handle indexed access
            if '[' in part and ']' in part:
                name, idx_str = part.split('[')
                idx = int(idx_str.replace(']', ''))
                current = getattr(current, name)[idx]
            else:
                current = getattr(current, part)
                
        return current
        
    @staticmethod
    def patch_forward_method(model, config: UltraContextConfig):
        """
        Patch model's forward method to integrate UltraContext
        
        Args:
            model: Model to patch
            config: UltraContext configuration
            
        Returns:
            Patched model
        """
        # Store original forward method
        original_forward = model.forward
        
        # Create wrapper function
        def ultracontext_forward(self, *args, **kwargs):
            # Check if UltraContext wrapper exists
            if not hasattr(self, "ultra_context"):
                return original_forward(*args, **kwargs)
                
            # Extract input embeddings
            input_embeddings = UltraContextIntegrator._extract_input_embeddings(self, args, kwargs)
            
            if input_embeddings is None:
                # Can't extract embeddings, use original forward
                return original_forward(*args, **kwargs)
                
            # Process with UltraContext wrapper
            batch_size, seq_len, _ = input_embeddings.shape
            
            # Check if this is prefill or extension
            is_prefill = seq_len > 1
            
            # Process with UltraContext
            if is_prefill:
                # Initial processing (prefill)
                self.ultra_context.prefill(input_embeddings)
            else:
                # Extending with new token
                self.ultra_context.extend(input_embeddings)
                
            # Get UltraContext for attention layers
            ultra_ctx = (None, None)
            for b in range(batch_size):
                ctx_emb, ctx_pos = self.ultra_context.get_context(b)
                if ctx_emb is not None:
                    ultra_ctx = (ctx_emb, ctx_pos)
                    break
                    
            # Store context in kwargs for attention layers
            kwargs["ultra_context"] = ultra_ctx
            
            # Run original forward
            return original_forward(*args, **kwargs)
            
        # Patch model's forward method
        model.forward = types.MethodType(ultracontext_forward, model)
        
        return model
        
    @staticmethod
    def _extract_input_embeddings(model, args, kwargs):
        """Try to extract input embeddings from arguments"""
        # Common argument names for input embeddings
        embedding_names = ["input_embeds", "inputs_embeds", "input_embeddings", "hidden_states"]
        
        # Check kwargs first
        for name in embedding_names:
            if name in kwargs and kwargs[name] is not None:
                return kwargs[name]
                
        # Check positional arguments (common in some models)
        for arg in args:
            if isinstance(arg, torch.Tensor) and len(arg.shape) == 3:
                # Likely the input embeddings
                return arg
                
        return None

# Utility functions for prefilling and streaming with UltraContext
def prefill_with_ultracontext(
    model,
    input_embeddings,
    context_state=None,
    batch_size=1,
    **kwargs
):
    """
    Process initial context (prefill phase)
    
    Args:
        model: Model with UltraContext
        input_embeddings: Input embeddings to process
        context_state: Optional context state to use
        batch_size: Batch size
        
    Returns:
        Model outputs
    """
    # Ensure model has UltraContext
    if not hasattr(model, "ultra_context"):
        raise ValueError("Model does not have UltraContext integration")
        
    # Use model's UltraContext wrapper
    wrapper = model.ultra_context
    
    # Process input embeddings
    with efficient_inference_mode():
        # Apply prefill
        processed_embeddings = wrapper.prefill(input_embeddings)
        
        # Get UltraContext for attention layers
        ctx_emb, ctx_pos = wrapper.get_context(0)
        
        # Run model forward pass with context
        outputs = model.forward(
            inputs_embeds=processed_embeddings,
            ultra_context=(ctx_emb, ctx_pos),
            **kwargs
        )
        
    return outputs

def stream_with_ultracontext(
    model,
    input_embedding,
    **kwargs
):
    """
    Process single token (streaming phase)
    
    Args:
        model: Model with UltraContext
        input_embedding: Input embedding for single token
        
    Returns:
        Model outputs
    """
    # Ensure model has UltraContext
    if not hasattr(model, "ultra_context"):
        raise ValueError("Model does not have UltraContext integration")
        
    # Use model's UltraContext wrapper
    wrapper = model.ultra_context
    
    # Process input embeddings
    with efficient_inference_mode():
        # Apply extension
        processed_embedding = wrapper.extend(input_embedding)
        
        # Get UltraContext for attention layers
        ctx_emb, ctx_pos = wrapper.get_context(0)
        
        # Run model forward pass with context
        outputs = model.forward(
            inputs_embeds=processed_embedding,
            ultra_context=(ctx_emb, ctx_pos),
            **kwargs
        )
        
    return outputs

# Example usage with Hugging Face models
def integrate_with_hf_model(model, config: UltraContextConfig, inplace=True):
    """
    Integrate UltraContext with a Hugging Face model
    
    Args:
        model: Hugging Face model
        config: UltraContext configuration
        inplace: Whether to modify the model in-place
        
    Returns:
        Integrated model
    """
    # Detect model type
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "")
    else:
        model_type = ""
        
    # Set up paths based on model type
    if "llama" in model_type.lower():
        attention_layer_path = "model.layers.*.self_attn"
        position_encoding_path = None  # Llama uses rotary position embeddings
    elif "mistral" in model_type.lower():
        attention_layer_path = "model.layers.*.self_attn"
        position_encoding_path = None  # Mistral uses rotary position embeddings
    elif "gpt" in model_type.lower():
        attention_layer_path = "transformer.h.*.attn"
        position_encoding_path = "transformer.wpe"
    elif "falcon" in model_type.lower():
        attention_layer_path = "transformer.h.*.self_attention"
        position_encoding_path = None  # Falcon uses rotary position embeddings
    elif "t5" in model_type.lower():
        attention_layer_path = "encoder.block.*.layer.0.SelfAttention"
        position_encoding_path = "encoder.embed_positions"
    elif "bert" in model_type.lower():
        attention_layer_path = "encoder.layer.*.attention.self"
        position_encoding_path = "embeddings.position_embeddings"
    else:
        # Default to automatic detection
        attention_layer_path = "auto"
        position_encoding_path = None
        
    # Integrate UltraContext
    integrated_model = UltraContextIntegrator.integrate_with_model(
        model,
        config,
        attention_layer_path=attention_layer_path,
        position_encoding_path=position_encoding_path,
        inplace=inplace
    )
    
    # Patch forward method for Hugging Face models
    UltraContextIntegrator.patch_forward_method(integrated_model, config)
    
    return integrated_model

# High-level API for using UltraContext
class UltraContextAPI:
    """
    High-level API for using UltraContext with any model
    
    Provides simple methods for:
    - Creating UltraContext configuration
    - Integrating with models
    - Processing inputs with extended context
    """
    @staticmethod
    def create_config(
        dim: int,
        max_context_length: int = 100_000_000,
        integration_mode: str = "extension",
        **kwargs
    ) -> UltraContextConfig:
        """
        Create UltraContext configuration
        
        Args:
            dim: Model dimension
            max_context_length: Maximum context length (default: 100M tokens)
            integration_mode: Integration mode (default: "extension")
            **kwargs: Additional configuration options
            
        Returns:
            UltraContext configuration
        """
        config_kwargs = {
            "dim": dim,
            "max_context_length": max_context_length,
            "integration_mode": integration_mode
        }
        
        # Add additional kwargs
        config_kwargs.update(kwargs)
        
        return UltraContextConfig(**config_kwargs)
        
    @staticmethod
    def integrate(model, config=None, **config_kwargs):
        """
        Integrate UltraContext with a model
        
        Args:
            model: Model to integrate with
            config: UltraContext configuration
            **config_kwargs: Configuration options (if config not provided)
            
        Returns:
            Integrated model
        """
        # Create config if not provided
        if config is None:
            # Try to detect model dimension
            dim = UltraContextAPI._detect_model_dimension(model)
            
            # Create configuration
            config = UltraContextAPI.create_config(dim, **config_kwargs)
            
        # Check if model is from Hugging Face
        if UltraContextAPI._is_huggingface_model(model):
            return integrate_with_hf_model(model, config)
        else:
            # Generic integration
            return UltraContextIntegrator.integrate_with_model(model, config)
            
    @staticmethod
    def process(
        model,
        input_embeddings,
        is_prefill=True,
        **kwargs
    ):
        """
        Process inputs with UltraContext
        
        Args:
            model: Model with UltraContext
            input_embeddings: Input embeddings
            is_prefill: Whether this is the prefill phase
            **kwargs: Additional arguments for model forward pass
            
        Returns:
            Model outputs
        """
        # Ensure model has UltraContext
        if not hasattr(model, "ultra_context"):
            raise ValueError("Model does not have UltraContext integration")
            
        if is_prefill:
            return prefill_with_ultracontext(model, input_embeddings, **kwargs)
        else:
            return stream_with_ultracontext(model, input_embeddings, **kwargs)
            
    @staticmethod
    def clear_context(model, batch_idx=None):
        """
        Clear UltraContext state
        
        Args:
            model: Model with UltraContext
            batch_idx: Batch index to clear (None for all)
        """
        # Ensure model has UltraContext
        if not hasattr(model, "ultra_context"):
            raise ValueError("Model does not have UltraContext integration")
            
        model.ultra_context.clear_context(batch_idx)
        
    @staticmethod
    def _detect_model_dimension(model):
        """Try to detect model dimension"""
        # Check common dimension attributes
        for attr_name in ["hidden_size", "d_model", "hidden_dim", "embed_dim", "dim"]:
            # Check model attributes
            if hasattr(model, attr_name):
                return getattr(model, attr_name)
                
            # Check model.config (for Hugging Face models)
            if hasattr(model, "config") and hasattr(model.config, attr_name):
                return getattr(model.config, attr_name)
                
        # Check first linear layer dimension
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return module.out_features
                
        # Default dimension
        return 768
        
    @staticmethod
    def _is_huggingface_model(model):
        """Check if model is from Hugging Face"""
        # Check for common Hugging Face model attributes
        if hasattr(model, "config") and hasattr(model, "model_type"):
            return True
            
        # Check model class name
        model_class = model.__class__.__name__
        hf_prefixes = ["GPT", "Llama", "T5", "Bert", "Falcon", "Mistral", "RoBERTa", "Transformer"]
        
        return any(prefix in model_class for prefix in hf_prefixes)
