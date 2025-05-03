import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger("ultracontext.position")

class PositionEncoding(nn.Module):
    """Base class for position encodings"""
    def __init__(self, dim: int, max_seq_len: int = 100_000_000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply position encoding
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions [batch_size, seq_len]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, dim]
        """
        raise NotImplementedError("Subclasses must implement forward method")

    @staticmethod
    def create_encoding(encoding_type: str, dim: int, max_seq_len: int = 100_000_000, **kwargs) -> 'PositionEncoding':
        """Factory method to create position encoding by type"""
        if encoding_type == "adaptive":
            return AdaptiveFourierPositionEncoding(dim, max_seq_len, **kwargs)
        elif encoding_type == "relative":
            return RelativePositionEncoding(dim, max_seq_len, **kwargs)
        elif encoding_type == "rotary":
            return RotaryPositionEncoding(dim, max_seq_len, **kwargs)
        elif encoding_type == "absolute":
            return nn.Embedding(max_seq_len, dim)
        else:
            logger.warning(f"Unknown position encoding type: {encoding_type}, defaulting to adaptive")
            return AdaptiveFourierPositionEncoding(dim, max_seq_len, **kwargs)


class AdaptiveFourierPositionEncoding(PositionEncoding):
    """
    Adaptive Fourier Position Encoding for handling ultra-long contexts
    
    Features:
    - Logarithmic frequency distribution for better long-range modeling
    - Adaptive attention to learned frequency components
    - Scale-invariant position representation
    
    Args:
        dim: Model dimension
        max_seq_len: Maximum sequence length
        num_bands: Number of frequency bands
        max_frequency: Maximum frequency
        adaptive: Whether to use adaptive frequency attention
        trainable: Whether frequency bands are trainable
    """
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 100_000_000,
        num_bands: int = 64,
        max_frequency: float = 10000.0,
        adaptive: bool = True,
        trainable: bool = True
    ):
        super().__init__(dim, max_seq_len)
        self.num_bands = num_bands
        self.max_frequency = max_frequency
        self.adaptive = adaptive
        
        # Number of frequency bands
        if dim % (2 * num_bands) != 0:
            # Adjust num_bands to make it divisible
            self.num_bands = dim // 2
            logger.warning(f"Adjusting num_bands to {self.num_bands} to match dimension {dim}")
        
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
    
    def _compute_fourier_encodings(self, positions: torch.Tensor) -> torch.Tensor:
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
        
        # Scale positions for very long sequences
        if self.max_seq_len > 100_000:
            # Apply log scaling for very long sequences
            scale_factor = math.log(self.max_seq_len) / self.max_seq_len
            pos_float = pos_float * scale_factor
        
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
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
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


class RelativePositionEncoding(PositionEncoding):
    """
    Efficient relative position encoding for long contexts
    
    Features:
    - Bucket-based relative position representation
    - Efficient attention computation
    - Scales well to ultra-long contexts
    
    Args:
        dim: Model dimension
        max_seq_len: Maximum sequence length
        num_buckets: Number of buckets for relative positions
        max_distance: Maximum relative distance to encode
        trainable: Whether embeddings are trainable
        heads: Number of attention heads
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100_000_000,
        num_buckets: int = 256,
        max_distance: int = 16384,
        trainable: bool = True,
        heads: int = 1
    ):
        super().__init__(dim, max_seq_len)
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads
        
        # Create relative position embedding table
        head_dim = dim // heads
        
        if trainable:
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_buckets, heads, head_dim)
            )
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)
        else:
            self.register_buffer(
                "rel_pos_embeddings",
                torch.zeros(num_buckets, heads, head_dim)
            )
        
        # Store last used positions for attention
        self.last_positions = None
            
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
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
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get position embeddings (for later use in attention)
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            positions: Optional token positions [batch_size, seq_len]
            
        Returns:
            Tuple of (input embeddings, position indices)
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
        
    def get_rel_pos_embeddings(self, q_positions: torch.Tensor, k_positions: torch.Tensor) -> torch.Tensor:
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


class RotaryPositionEncoding(PositionEncoding):
    """
    Rotary Position Embeddings (RoPE) for ultra-long contexts
    
    Features:
    - Theoretically unbounded position range
    - Efficient computation (applied directly to queries and keys)
    - Preserves relative position information implicitly
    
    Args:
        dim: Model dimension
        max_seq_len: Maximum sequence length
        base: Base value for frequency computation
        scaling_factor: Scaling factor for positions
        trainable: Whether frequencies are trainable
        interleaved: Whether to use interleaved format
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100_000_000,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        trainable: bool = False,
        interleaved: bool = True
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
        
        # Apply NTK-aware scaling for very long contexts
        if max_seq_len > 4096:
            # Apply scaling for longer contexts
            self.scaling_factor = min(self.scaling_factor, 4096 / max_seq_len * 2)
            logger.info(f"Using RoPE scaling_factor = {self.scaling_factor} for max_seq_len = {max_seq_len}")
        
    def _get_cos_sin_cache(self, positions: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x"""
        x1 = x[..., : self.num_features]
        x2 = x[..., self.num_features : 2 * self.num_features]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
    def apply_rotary_to_query_key(self, q: torch.Tensor, k: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Permute for broadcasting
        cos = cos.permute(0, 2, 1, 3)  # [batch, 1, seq_len, head_dim]
        sin = sin.permute(0, 2, 1, 3)  # [batch, 1, seq_len, head_dim]
        
        # Apply rotary embeddings to query and key
        q_rope = (q * cos) + (self._rotate_half_qk(q) * sin)
        k_rope = (k * cos) + (self._rotate_half_qk(k) * sin)
        
        return q_rope, k_rope
        
    def _rotate_half_qk(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of query/key tensors"""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat((-x2, x1), dim=-1)


class NTKScaledRotaryPositionEncoding(RotaryPositionEncoding):
    """
    NTK-aware scaled Rotary Position Embeddings for extremely long contexts
    
    This is an extension of RoPE that applies NTK-aware scaling to better handle
    extremely long sequences (millions of tokens).
    
    References:
        - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        - https://arxiv.org/abs/2104.09864
        - Blog post: https://blog.eleuther.ai/rotary-embeddings/
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100_000_000,
        base: float = 10000.0,
        scaling_type: str = "dynamic",
        scaling_factor: float = None,
        trainable: bool = False,
        interleaved: bool = True
    ):
        # Calculate appropriate scaling factor
        if scaling_factor is None:
            if scaling_type == "linear":
                # Linear scaling
                original_max_len = 4096  # Base model context length
                scaling_factor = original_max_len / max_seq_len
            elif scaling_type == "ntk":
                # NTK-aware scaling
                original_max_len = 4096
                alpha = 1 - (original_max_len / max_seq_len) * 0.5
                scaling_factor = alpha
            elif scaling_type == "dynamic":
                # Dynamic scaling based on context length
                if max_seq_len <= 4096:
                    scaling_factor = 1.0
                elif max_seq_len <= 16384:
                    scaling_factor = 0.75
                elif max_seq_len <= 65536:
                    scaling_factor = 0.5
                elif max_seq_len <= 262144:
                    scaling_factor = 0.25
                else:
                    scaling_factor = 0.1
            else:
                scaling_factor = 1.0
        
        # Initialize parent class with calculated scaling factor
        super().__init__(
            dim=dim,
            max_seq_len=max_seq_len,
            base=base,
            scaling_factor=scaling_factor,
            trainable=trainable,
            interleaved=interleaved
        )
        
        logger.info(f"Using NTK-scaled RoPE with scaling_factor = {self.scaling_factor}")
