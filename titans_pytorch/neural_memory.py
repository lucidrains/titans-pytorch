import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple, TypeVar, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, deque
import time
import logging
from functools import partial, lru_cache
import os
import uuid
import einops
import hashlib
from einops import rearrange, reduce, repeat
import numpy as np
import heapq
from contextlib import contextmanager

# Optional imports that will be used when available
try:
    from torch.utils.checkpoint import checkpoint
    HAS_CHECKPOINTING = True
except ImportError:
    HAS_CHECKPOINTING = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Configure logging
logger = logging.getLogger('UltraScaleNeuralMemory')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

##############################################
# Advanced Neural Memory Hierarchies        #
##############################################

class NeuralMemoryTier(Enum):
    """
    Hierarchical memory tiers for 100M+ token contexts with specialized functions
    """
    FOCUS = auto()        # Current working focus (highest precision, fastest access, tiny)
    ACTIVE = auto()       # Active working context (high precision, ultrafast access, small)
    FOREGROUND = auto()   # Recent important context (high precision, very fast access)
    BACKGROUND = auto()   # Supporting context (medium precision, fast access)
    EPISODIC = auto()     # Recent context sequences (medium precision, medium access speed)
    SEMANTIC = auto()     # Distilled knowledge (compressed semantic representations)
    GENERAL = auto()      # Medium-term context (lower precision, compressed)
    CATEGORICAL = auto()  # Categorized historical information (specialized compression)
    ARCHIVAL = auto()     # Historical context (highly compressed, indexed)
    REFERENCE = auto()    # External knowledge (compressed symbolic pointers)
    CONSOLIDATED = auto() # Distilled and integrated knowledge (min. size, max. information)
    OFFLOADED = auto()    # External storage (retrieval via learned indices)

class NeuralCognitiveSignals(Enum):
    """Memory management signals for token importance and attention allocation"""
    SALIENCE = auto()     # Information importance/relevance 
    NOVELTY = auto()      # New or unexpected information
    RECENCY = auto()      # Recent activation/access
    FREQUENCY = auto()    # Access frequency/pattern
    COHERENCE = auto()    # Contextual fit/alignment
    CAUSALITY = auto()    # Causal significance
    UNCERTAINTY = auto()  # Model uncertainty about token
    PREDICTION = auto()   # Prediction error/surprise
    EMOTION = auto()      # Sentiment/emotional loading
    UTILITY = auto()      # Task-relevance utility

@dataclass
class TokenMetadata:
    """Rich metadata for token management and retrieval"""
    # Core identifiers
    token_id: int
    position: int
    creation_time: float
    sequence_id: Optional[str] = None
    
    # Memory management
    tier: NeuralMemoryTier = NeuralMemoryTier.ACTIVE
    importance: float = 0.5  # Overall importance score
    last_access_time: float = 0.0
    access_count: int = 0
    access_pattern: List[float] = field(default_factory=list)
    
    # Cognitive signals (specialized importance indicators)
    cognitive_signals: Dict[NeuralCognitiveSignals, float] = field(default_factory=dict)
    
    # Compression state
    compression_ratio: float = 1.0  # 1.0 = no compression
    precision_bits: int = 32        # Current precision
    compressed: bool = False
    
    # Content properties
    content_hash: Optional[str] = None
    semantic_vector: Optional[Tensor] = None
    token_type: int = 0
    entity_references: List[str] = field(default_factory=list)
    
    # Retrieval aids
    search_keys: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Relational information
    related_tokens: Dict[int, float] = field(default_factory=dict)  # token_id -> strength
    causal_predecessors: List[int] = field(default_factory=list)
    causal_successors: List[int] = field(default_factory=list)
    
    # Offloading
    offloaded: bool = False
    offload_path: Optional[str] = None
    
    def update_cognitive_signals(self, signals: Dict[NeuralCognitiveSignals, float]) -> None:
        """Update the cognitive signals and recalculate importance"""
        self.cognitive_signals.update(signals)
        
        # Recalculate overall importance (weighted average of signals)
        # Weights could be learned or set heuristically
        weights = {
            NeuralCognitiveSignals.SALIENCE: 0.25,
            NeuralCognitiveSignals.NOVELTY: 0.15,
            NeuralCognitiveSignals.RECENCY: 0.20,
            NeuralCognitiveSignals.FREQUENCY: 0.10,
            NeuralCognitiveSignals.COHERENCE: 0.10,
            NeuralCognitiveSignals.CAUSALITY: 0.10,
            NeuralCognitiveSignals.UNCERTAINTY: 0.03,
            NeuralCognitiveSignals.PREDICTION: 0.03,
            NeuralCognitiveSignals.EMOTION: 0.02,
            NeuralCognitiveSignals.UTILITY: 0.02,
        }
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for signal, value in self.cognitive_signals.items():
            if signal in weights:
                weighted_sum += value * weights[signal]
                weight_total += weights[signal]
        
        if weight_total > 0:
            self.importance = weighted_sum / weight_total
        else:
            # Default when no weighted signals are available
            self.importance = 0.5

    def update_access(self, current_time: float) -> None:
        """Update access statistics"""
        self.last_access_time = current_time
        self.access_count += 1
        
        # Keep track of last 5 access times for pattern recognition
        self.access_pattern.append(current_time)
        if len(self.access_pattern) > 5:
            self.access_pattern = self.access_pattern[-5:]
        
        # Update recency signal
        self.update_cognitive_signals({
            NeuralCognitiveSignals.RECENCY: 1.0,  # Just accessed
            NeuralCognitiveSignals.FREQUENCY: min(1.0, self.access_count / 10.0)  # Normalize frequency
        })
    
    def compute_content_hash(self, embedding: Tensor) -> str:
        """Compute a content hash for deduplication"""
        # Convert tensor to bytes and hash
        tensor_bytes = embedding.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()

class NeuroSymbolicCompression:
    """
    Advanced neural-symbolic compression system with semantic preservation
    for extreme compression of tokens while maintaining retrievability
    """
    
    def __init__(
        self,
        model_dim: int,
        base_ratio: float = 0.5,
        distance_factor: float = 0.3,
        max_ratio: float = 0.001,  # 0.1% of original size for distant tokens
        block_size: int = 1024,
        enable_mixed_precision: bool = True,
        enable_semantic_compression: bool = True,
        semantic_dim: int = 64,
        precision_schedule: List[int] = [32, 16, 8, 4, 2, 1],
        device: Optional[torch.device] = None,
        kmeans_codebook_size: int = 1024,
        enable_neural_compression: bool = True
    ):
        self.model_dim = model_dim
        self.base_ratio = base_ratio
        self.distance_factor = distance_factor
        self.max_ratio = max_ratio
        self.block_size = block_size
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_semantic_compression = enable_semantic_compression
        self.semantic_dim = semantic_dim
        self.precision_schedule = precision_schedule
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kmeans_codebook_size = kmeans_codebook_size
        self.enable_neural_compression = enable_neural_compression
        
        # Statistics tracking
        self.compression_stats = defaultdict(list)
        
        # Initialize semantic encoder for knowledge distillation
        if enable_semantic_compression:
            self.semantic_encoder = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, semantic_dim)
            ).to(device)
            
            self.semantic_decoder = nn.Sequential(
                nn.Linear(semantic_dim, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, model_dim)
            ).to(device)
        else:
            self.semantic_encoder = None
            self.semantic_decoder = None
            
        # Neural compression components
        if enable_neural_compression:
            # Sparse autoencoder for ultra-compression
            self.neural_encoder = nn.Sequential(
                nn.Linear(model_dim, model_dim // 4),
                nn.LayerNorm(model_dim // 4),
                nn.GELU(),
                nn.Linear(model_dim // 4, model_dim // 16),
                nn.LayerNorm(model_dim // 16),
            ).to(device)
            
            self.neural_decoder = nn.Sequential(
                nn.Linear(model_dim // 16, model_dim // 4),
                nn.LayerNorm(model_dim // 4),
                nn.GELU(),
                nn.Linear(model_dim // 4, model_dim),
            ).to(device)
        else:
            self.neural_encoder = None
            self.neural_decoder = None
            
        # Create and initialize codebooks for vector quantization
        self.initialize_codebooks()
        
    def initialize_codebooks(self):
        """Initialize vector quantization codebooks"""
        # Primary codebook for main token compression
        self.primary_codebook = nn.Parameter(
            torch.randn(self.kmeans_codebook_size, self.model_dim) * 0.01
        )
        
        # Semantic codebook for semantic compression
        if self.enable_semantic_compression:
            self.semantic_codebook = nn.Parameter(
                torch.randn(self.kmeans_codebook_size // 4, self.semantic_dim) * 0.01
            )
        
        # Specialized codebooks for different token types/regions
        self.specialized_codebooks = nn.ParameterDict({
            "entity": nn.Parameter(torch.randn(self.kmeans_codebook_size // 8, self.model_dim) * 0.01),
            "numeric": nn.Parameter(torch.randn(self.kmeans_codebook_size // 8, self.model_dim) * 0.01),
            "syntactic": nn.Parameter(torch.randn(self.kmeans_codebook_size // 8, self.model_dim) * 0.01),
        })
        
    def calculate_compression_ratio(
        self, 
        token_distance: int, 
        importance: float = 0.5,
        tier: NeuralMemoryTier = NeuralMemoryTier.GENERAL
    ) -> float:
        """
        Calculate compression ratio based on token distance, importance, and tier
        
        Args:
            token_distance: Distance in tokens from current position
            importance: Token importance (0-1)
            tier: Memory tier for contextual compression adjustment
            
        Returns:
            Compression ratio (1.0 = no compression, 0.001 = 99.9% compression)
        """
        # Base ratio from distance with exponential decay
        tier_factor = {
            NeuralMemoryTier.FOCUS: 1.0,
            NeuralMemoryTier.ACTIVE: 0.95,
            NeuralMemoryTier.FOREGROUND: 0.8,
            NeuralMemoryTier.BACKGROUND: 0.6,
            NeuralMemoryTier.EPISODIC: 0.4,
            NeuralMemoryTier.SEMANTIC: 0.3,
            NeuralMemoryTier.GENERAL: 0.2,
            NeuralMemoryTier.CATEGORICAL: 0.15,
            NeuralMemoryTier.ARCHIVAL: 0.1,
            NeuralMemoryTier.REFERENCE: 0.05,
            NeuralMemoryTier.CONSOLIDATED: 0.02,
            NeuralMemoryTier.OFFLOADED: 0.01,
        }.get(tier, 0.2)
        
        # Calculate distance-based ratio 
        if token_distance < 1000:
            # Recent tokens get minimal compression
            distance_ratio = 1.0
        elif token_distance < 10000:
            # Gradual compression within first 10K tokens
            distance_ratio = self.base_ratio * math.exp(-self.distance_factor * token_distance / 10000)
        elif token_distance < 100000:
            # More aggressive for 10K-100K range
            distance_ratio = self.base_ratio * math.exp(-self.distance_factor * token_distance / 10000) * 0.5
        elif token_distance < 1000000:
            # Very aggressive for 100K-1M range
            distance_ratio = self.base_ratio * math.exp(-self.distance_factor * token_distance / 10000) * 0.2
        else:
            # Maximum compression beyond 1M distance
            distance_ratio = self.base_ratio * math.exp(-self.distance_factor * token_distance / 10000) * 0.1
        
        # Adjust by importance (higher importance = less compression)
        importance_factor = 0.2 + (importance * 0.8)  # Range: 0.2-1.0
        
        # Combine all factors
        final_ratio = max(self.max_ratio, distance_ratio * importance_factor * tier_factor)
        
        return final_ratio
    
    def determine_precision(self, token_distance: int, importance: float) -> int:
        """
        Determine bit precision based on token distance and importance
        
        Args:
            token_distance: Distance in tokens from current position
            importance: Token importance (0-1)
            
        Returns:
            Bit precision (32, 16, 8, 4, 2, or 1)
        """
        if not self.enable_mixed_precision:
            return 32
        
        # Increase precision based on importance
        importance_boost = math.floor(importance * 3)  # 0-2 levels boost
            
        # Base precision selection on distance and boost by importance
        if token_distance < 1000:
            precision_idx = 0  # 32-bit
        elif token_distance < 10000:
            precision_idx = 1  # 16-bit
        elif token_distance < 100000:
            precision_idx = 2  # 8-bit
        elif token_distance < 1000000:
            precision_idx = 3  # 4-bit
        elif token_distance < 10000000:
            precision_idx = 4  # 2-bit
        else:
            precision_idx = 5  # 1-bit
            
        # Apply importance boost (but don't exceed highest precision)
        precision_idx = max(0, precision_idx - importance_boost)
        
        return self.precision_schedule[precision_idx]

    def _compress_neural(
        self, 
        tensor: Tensor,
        ratio: float
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Compress using neural networks for adaptive compression"""
        if not self.enable_neural_compression or self.neural_encoder is None:
            # Fall back to SVD if neural compression not available
            return self._compress_svd(tensor, ratio)
            
        # Get original metadata
        original_shape = tensor.shape
        
        # If tensor is not already on the target device, move it
        tensor = tensor.to(self.device)
        
        # Apply neural encoder
        encoded = self.neural_encoder(tensor)
        
        # Apply sparsity based on ratio (only keep top k% values)
        k = max(1, int(encoded.numel() * ratio))
        values, indices = torch.topk(encoded.abs().flatten(), k)
        threshold = values.min()
        
        # Create sparse mask
        mask = encoded.abs() >= threshold
        sparse_encoded = encoded * mask
        
        return {
            "encoded": sparse_encoded,
            "mask": mask
        }, {
            "algorithm": "neural",
            "original_shape": original_shape,
            "ratio": ratio,
            "sparsity": 1.0 - (k / encoded.numel())
        }
    
    def _decompress_neural(
        self, 
        compressed_data: Dict[str, Tensor],
        metadata: Dict[str, Any]
    ) -> Tensor:
        """Decompress neural-compressed tensor"""
        encoded = compressed_data["encoded"]
        
        # Apply neural decoder
        decoded = self.neural_decoder(encoded)
        
        # Reshape to original shape if needed
        original_shape = metadata.get("original_shape")
        if original_shape and tuple(decoded.shape) != tuple(original_shape):
            decoded = decoded.reshape(original_shape)
            
        return decoded
        
    def _compress_semantic(
        self, 
        tensor: Tensor,
        ratio: float
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Distill semantic information into compact representation"""
        if not self.enable_semantic_compression or self.semantic_encoder is None:
            # Fall back to SVD if semantic compression not available
            return self._compress_svd(tensor, ratio)
            
        # Get original metadata
        original_shape = tensor.shape
        
        # If tensor is not already on the target device, move it
        tensor = tensor.to(self.device)
        
        # Apply semantic encoder
        semantic = self.semantic_encoder(tensor)
        
        # Quantize the semantic vector for extreme compression
        if hasattr(self, 'semantic_codebook'):
            # Compute distances to codebook vectors
            distances = torch.cdist(semantic.unsqueeze(0), self.semantic_codebook.unsqueeze(0))[0]
            
            # Find nearest codebook vector
            indices = torch.argmin(distances, dim=1)
            
            return {
                "indices": indices,
            }, {
                "algorithm": "semantic_vq",
                "original_shape": original_shape,
                "ratio": ratio
            }
        else:
            return {
                "semantic": semantic,
            }, {
                "algorithm": "semantic",
                "original_shape": original_shape,
                "ratio": ratio
            }
    
    def _decompress_semantic(
        self, 
        compressed_data: Dict[str, Tensor],
        metadata: Dict[str, Any]
    ) -> Tensor:
        """Expand semantic representation back to full embedding"""
        algorithm = metadata.get("algorithm", "")
        
        if algorithm == "semantic_vq":
            # Get the codebook vectors from indices
            indices = compressed_data["indices"]
            semantic = self.semantic_codebook[indices]
        else:
            semantic = compressed_data["semantic"]
        
        # Apply semantic decoder
        decoded = self.semantic_decoder(semantic)
        
        # Reshape to original shape if needed
        original_shape = metadata.get("original_shape")
        if original_shape and tuple(decoded.shape) != tuple(original_shape):
            decoded = decoded.reshape(original_shape)
            
        return decoded
        
    def _compress_vector_quantization(
        self, 
        tensor: Tensor,
        ratio: float,
        token_type: int = 0
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Compress using vector quantization with learned codebooks"""
        # Get original metadata
        original_shape = tensor.shape
        
        # If tensor is not already on the target device, move it
        tensor = tensor.to(self.device)
        
        # Select appropriate codebook based on token type
        if token_type == 1 and "entity" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["entity"]
        elif token_type == 2 and "numeric" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["numeric"]
        elif token_type == 3 and "syntactic" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["syntactic"]
        else:
            codebook = self.primary_codebook
        
        # Compute distances to codebook vectors
        distances = torch.cdist(tensor.unsqueeze(0), codebook.unsqueeze(0))[0]
        
        # Find nearest codebook vector
        indices = torch.argmin(distances, dim=1)
        
        return {
            "indices": indices,
            "codebook_id": token_type if token_type in [1, 2, 3] else 0
        }, {
            "algorithm": "vq",
            "original_shape": original_shape,
            "ratio": ratio
        }
    
    def _decompress_vector_quantization(
        self, 
        compressed_data: Dict[str, Tensor],
        metadata: Dict[str, Any]
    ) -> Tensor:
        """Decompress vector-quantized tensor"""
        indices = compressed_data["indices"]
        codebook_id = compressed_data.get("codebook_id", 0)
        
        # Select the appropriate codebook
        if codebook_id == 1 and "entity" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["entity"]
        elif codebook_id == 2 and "numeric" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["numeric"]
        elif codebook_id == 3 and "syntactic" in self.specialized_codebooks:
            codebook = self.specialized_codebooks["syntactic"]
        else:
            codebook = self.primary_codebook
        
        # Get the codebook vectors
        reconstructed = codebook[indices]
        
        # Reshape to original shape if needed
        original_shape = metadata.get("original_shape")
        if original_shape and tuple(reconstructed.shape) != tuple(original_shape):
            reconstructed = reconstructed.reshape(original_shape)
            
        return reconstructed
    
    def _compress_svd(
        self, 
        tensor: Tensor, 
        ratio: float
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Compress tensor using truncated SVD with adaptive rank"""
        original_shape = tensor.shape
        
        # For tensors with more than 2 dimensions, reshape to 2D
        if tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
            
        # Compute SVD
        try:
            U, S, V = torch.svd(tensor)
            
            # Determine rank based on compression ratio
            max_rank = min(tensor.shape)
            
            # Ensure we keep at least one component
            rank = max(1, min(max_rank, int(max_rank * ratio)))
            
            # Truncate to specified rank
            U_truncated = U[:, :rank]
            S_truncated = S[:rank]
            V_truncated = V[:, :rank]
            
            return {
                "U": U_truncated,
                "S": S_truncated,
                "V": V_truncated
            }, {
                "algorithm": "svd",
                "original_shape": original_shape,
                "rank": rank,
                "ratio": ratio
            }
        except Exception as e:
            logger.warning(f"SVD compression failed: {e}, falling back to original tensor")
            return {"data": tensor}, {"algorithm": "none", "error": str(e)}
            
    def _compress_quantization(
        self, 
        tensor: Tensor, 
        ratio: float,
        bits: int
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Compress tensor using ultra-low bit quantization"""
        original_shape = tensor.shape
        dtype = tensor.dtype
        
        # Ensure bits is valid (1, 2, 4, 8, 16)
        bits = min(16, max(1, bits))
        
        # Compute min and max
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Handle special case where min == max
        if min_val == max_val:
            return {"data": torch.zeros_like(tensor), "value": min_val.unsqueeze(0)}, {
                "algorithm": "const",
                "original_shape": original_shape
            }
            
        scale = (max_val - min_val) / ((1 << bits) - 1)
        scale = max(scale, 1e-10)  # Avoid division by zero
        
        # Quantize
        tensor_normalized = ((tensor - min_val) / scale).round().clamp(0, (1 << bits) - 1)
        
        if bits <= 8:
            quantized = tensor_normalized.to(torch.uint8)
        else:
            quantized = tensor_normalized.to(torch.int16)
        
        return {
            "data": quantized,
            "min": min_val.unsqueeze(0),
            "scale": scale.unsqueeze(0)
        }, {
            "algorithm": "quant",
            "original_shape": original_shape,
            "original_dtype": dtype,
            "bits": bits,
            "ratio": ratio
        }
        
    def _compress_sparse(
        self, 
        tensor: Tensor, 
        ratio: float
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """Compress tensor using extreme sparsity"""
        original_shape = tensor.shape
        
        # Determine sparsity based on ratio (higher ratio = more values kept)
        target_nonzeros = max(1, int(tensor.numel() * ratio))
        
        # Find threshold for top values
        tensor_flat = tensor.abs().reshape(-1)
        
        # Find the kth largest value as threshold
        if target_nonzeros < tensor_flat.numel():
            threshold_idx = tensor_flat.numel() - target_nonzeros
            threshold_value = torch.kthvalue(tensor_flat, threshold_idx).values
            
            # Create sparse tensor - keep only values above threshold
            mask = tensor.abs() > threshold_value
            indices = mask.nonzero()
            values = tensor[mask]
        else:
            # Keep all values
            indices = tensor.nonzero()
            values = tensor[indices[:, 0], indices[:, 1]] if tensor.dim() == 2 else tensor[tuple(indices.t())]
            
        return {
            "indices": indices,
            "values": values
        }, {
            "algorithm": "sparse",
            "original_shape": original_shape,
            "ratio": ratio,
            "sparsity": 1.0 - (values.numel() / tensor.numel())
        }
    
    def _compress_noop(
        self, 
        tensor: Tensor
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """No-op compression (just returns the original tensor)"""
        return {"data": tensor}, {"algorithm": "noop", "original_shape": tensor.shape}

    def compress_token(
        self, 
        embedding: Tensor,
        token_distance: int,
        importance: float = 0.5,
        token_type: int = 0,
        tier: NeuralMemoryTier = NeuralMemoryTier.GENERAL
    ) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        """
        Compress a token embedding with optimal algorithm selection
        
        Args:
            embedding: Token embedding tensor
            token_distance: Distance from current position
            importance: Token importance score (0-1)
            token_type: Type of token (used for specialized compression)
            tier: Memory tier for contextual compression
            
        Returns:
            Tuple of (compressed_data, metadata)
        """
        # Determine compression parameters
        ratio = self.calculate_compression_ratio(token_distance, importance, tier)
        precision = self.determine_precision(token_distance, importance)
        
        # For very close tokens, skip compression
        if token_distance < 1000 and tier in [NeuralMemoryTier.FOCUS, NeuralMemoryTier.ACTIVE]:
            return self._compress_noop(embedding)
        
        # Select compression algorithm based on distance and token properties
        if token_distance > 1000000 or tier in [NeuralMemoryTier.CONSOLIDATED, NeuralMemoryTier.SEMANTIC]:
            # For extremely distant tokens, use semantic compression
            if self.enable_semantic_compression:
                return self._compress_semantic(embedding, ratio)
            else:
                # Fallback to vector quantization for extreme distances
                return self._compress_vector_quantization(embedding, ratio, token_type)
        elif token_distance > 100000 or tier in [NeuralMemoryTier.CATEGORICAL, NeuralMemoryTier.ARCHIVAL]:
            # For very distant tokens, use vector quantization
            return self._compress_vector_quantization(embedding, ratio, token_type)
        elif token_distance > 10000 or tier in [NeuralMemoryTier.EPISODIC, NeuralMemoryTier.GENERAL]:
            # For moderately distant tokens, use quantization
            return self._compress_quantization(embedding, ratio, precision)
        elif token_distance > 1000 or tier in [NeuralMemoryTier.BACKGROUND]:
            # For semi-distant tokens, use SVD or neural compression
            if self.enable_neural_compression and importance > 0.7:
                return self._compress_neural(embedding, ratio)
            else:
                return self._compress_svd(embedding, ratio)
        else:
            # For close tokens, use no compression or light SVD
            if ratio > 0.9:
                return self._compress_noop(embedding)
            else:
                return self._compress_svd(embedding, ratio)
    
    def decompress_token(
        self, 
        compressed_data: Dict[str, Tensor],
        metadata: Dict[str, Any]
    ) -> Tensor:
        """Decompress a token embedding"""
        algorithm = metadata.get("algorithm", "none")
        
        if algorithm == "noop":
            return compressed_data["data"]
        elif algorithm == "svd":
            # Reconstruct from SVD components
            U = compressed_data["U"]
            S = compressed_data["S"]
            V = compressed_data["V"]
            
            # Reconstruct
            reconstructed = torch.matmul(U * S.unsqueeze(0), V.t())
            
            # Reshape to original shape if needed
            original_shape = metadata.get("original_shape")
            if original_shape and tuple(reconstructed.shape) != tuple(original_shape):
                reconstructed = reconstructed.reshape(original_shape)
                
            return reconstructed
        elif algorithm == "quant":
            # Dequantize
            data = compressed_data["data"]
            min_val = compressed_data["min"]
            scale = compressed_data["scale"]
            
            # Convert to float and rescale
            dequantized = data.float() * scale + min_val
            
            # Reshape if needed
            original_shape = metadata.get("original_shape")
            if original_shape and tuple(dequantized.shape) != tuple(original_shape):
                dequantized = dequantized.reshape(original_shape)
                
            # Convert back to original dtype
            original_dtype = metadata.get("original_dtype", torch.float32)
            return dequantized.to(original_dtype)
        elif algorithm == "sparse":
            # Reconstruct sparse tensor
            indices = compressed_data["indices"]
            values = compressed_data["values"]
            original_shape = metadata["original_shape"]
            
            # Create output tensor
            output = torch.zeros(original_shape, device=indices.device, dtype=values.dtype)
            
            # Populate with values
            if indices.shape[0] > 0:  # Check if there are any indices
                output.index_put_(tuple(indices.t()), values)
                
            return output
        elif algorithm == "const":
            # Constant value tensor
            value = compressed_data["value"]
            shape = metadata["original_shape"]
            return torch.full(shape, value, device=value.device, dtype=value.dtype)
        elif algorithm == "neural":
            # Neural network decompression
            return self._decompress_neural(compressed_data, metadata)
        elif algorithm == "semantic" or algorithm == "semantic_vq":
            # Semantic decompression
            return self._decompress_semantic(compressed_data, metadata)
        elif algorithm == "vq":
            # Vector quantization decompression
            return self._decompress_vector_quantization(compressed_data, metadata)
        else:
            # Unknown algorithm
            logger.warning(f"Unknown compression algorithm: {algorithm}")
            if "data" in compressed_data:
                return compressed_data["data"]
            else:
                # Try to return the first tensor found as a fallback
                for k, v in compressed_data.items():
                    if isinstance(v, torch.Tensor):
                        return v
                
                # Last resort fallback
                logger.error("Failed to decompress token with unknown algorithm")
                return torch.zeros(metadata.get("original_shape", (1, self.model_dim)), 
                                  device=self.device)

class UltraScaleMemoryManager:
    """
    Ultra-scale memory manager optimized for 100M+ token context windows with
    neural-symbolic hybrid approach, extreme efficiency, and intelligent memory policies
    """
    
    def __init__(
        self,
        model_dim: int,
        # Memory capacity parameters
        max_focus_tokens: int = 128,             # Tiny working memory
        max_active_tokens: int = 1024,           # Current context window
        max_foreground_tokens: int = 8192,       # Recent important context
        max_background_tokens: int = 32768,      # Supporting context
        max_episodic_tokens: int = 131072,       # Episodic memory capacity
        max_semantic_tokens: int = 65536,        # Semantic memory capacity
        max_general_tokens: int = 524288,        # General memory capacity
        max_categorical_tokens: int = 1048576,   # Categorical memory capacity
        max_archival_tokens: int = 4194304,      # Archival memory capacity 
        max_reference_tokens: int = 2097152,     # Reference memory capacity
        max_consolidated_tokens: int = 1048576,  # Consolidated memory capacity
        max_memory_resident_tokens: int = 8388608,  # Maximum tokens in GPU/CPU
        max_total_tokens: int = 100 * 1000 * 1000,  # 100M tokens total capacity
        
        # Memory management parameters
        offload_path: str = "./ultra_memory_offload",
        enable_tensor_compression: bool = True,
        enable_semantic_compression: bool = True,
        compression_schedules: bool = True,
        token_pruning: bool = True,
        token_deduplication: bool = True,
        pruning_threshold: float = 0.1,         # Threshold for pruning unimportant tokens
        min_pruning_distance: int = 16384,      # Don't prune tokens closer than this
        enable_checkpointing: bool = True,
        checkpoint_interval: int = 100000,
        
        # Indexing and retrieval
        enable_vector_index: bool = True,        # Enable semantic vector indexing
        vector_index_sample_rate: float = 0.1,   # Percentage of tokens to index
        vector_dim: int = 64,                    # Dimension for retrieval vectors
        
        # Hardware management
        device: Optional[torch.device] = None,
        offload_device: Optional[torch.device] = None,
        enable_mixed_precision: bool = True,
        precision_bits: int = 16,
        
        # Advanced memory policies
        memory_consolidation_interval: int = 1000000,  # Token interval for consolidation
        semantic_extraction_threshold: float = 0.7,    # Importance threshold for semantic extraction
        cognitive_context_size: int = 4096,            # Size of active cognitive context
        
        # Task-oriented memory
        enable_task_memory: bool = True,              # Enable specialized task memory
        task_memory_size: int = 16384,                # Number of task-specific tokens
        
        # Auto-scaling
        auto_scaling: bool = True,                    # Auto-scale memory tiers
        min_free_space_ratio: float = 0.1             # Min free space to maintain
    ):
        # Store core parameters
        self.model_dim = model_dim
        
        # Memory capacity parameters
        self.max_focus_tokens = max_focus_tokens
        self.max_active_tokens = max_active_tokens
        self.max_foreground_tokens = max_foreground_tokens
        self.max_background_tokens = max_background_tokens
        self.max_episodic_tokens = max_episodic_tokens
        self.max_semantic_tokens = max_semantic_tokens
        self.max_general_tokens = max_general_tokens
        self.max_categorical_tokens = max_categorical_tokens
        self.max_archival_tokens = max_archival_tokens
        self.max_reference_tokens = max_reference_tokens
        self.max_consolidated_tokens = max_consolidated_tokens
        self.max_memory_resident_tokens = max_memory_resident_tokens
        self.max_total_tokens = max_total_tokens
        
        # Memory management parameters
        self.offload_path = offload_path
        self.enable_tensor_compression = enable_tensor_compression
        self.enable_semantic_compression = enable_semantic_compression
        self.compression_schedules = compression_schedules
        self.token_pruning = token_pruning
        self.token_deduplication = token_deduplication
        self.pruning_threshold = pruning_threshold
        self.min_pruning_distance = min_pruning_distance
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        
        # Hardware management
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.offload_device = offload_device or torch.device('cpu')
        self.enable_mixed_precision = enable_mixed_precision
        self.precision_bits = precision_bits
        
        # Vector indexing parameters
        self.enable_vector_index = enable_vector_index
        self.vector_index_sample_rate = vector_index_sample_rate
        self.vector_dim = vector_dim
        
        # Advanced memory policies
        self.memory_consolidation_interval = memory_consolidation_interval
        self.semantic_extraction_threshold = semantic_extraction_threshold
        self.cognitive_context_size = cognitive_context_size
        
        # Task-oriented memory
        self.enable_task_memory = enable_task_memory
        self.task_memory_size = task_memory_size
        
        # Auto-scaling settings
        self.auto_scaling = auto_scaling
        self.min_free_space_ratio = min_free_space_ratio
        
        # Initialize compression engine
        self.compression_engine = NeuroSymbolicCompression(
            model_dim=model_dim,
            base_ratio=0.5,
            distance_factor=0.3,
            max_ratio=0.001,  # 0.1% of original size for distant tokens
            enable_mixed_precision=enable_mixed_precision,
            enable_semantic_compression=enable_semantic_compression,
            semantic_dim=vector_dim,
            precision_schedule=[32, 16, 8, 4, 2, 1],
            device=self.device
        )
        
        # Initialize memory storage for each tier using OrderedDict for consistent iteration
        self.memory_tiers = {tier: OrderedDict() for tier in NeuralMemoryTier}
        
        # Token metadata storage
        self.token_metadata = {}  # token_id -> TokenMetadata
        
        # Create semantic vector index if enabled
        if enable_vector_index:
            self.init_vector_index()
        else:
            self.vector_index = None
            
        # Create task memory if enabled
        if enable_task_memory:
            self.task_memory = {}
            self.task_memory_index = None
            self.current_task_id = None
        else:
            self.task_memory = None
            
        # Position tracking
        self.current_position = 0
        self.last_checkpoint_position = 0
        
        # Current sequence tracking
        self.current_sequence_id = str(uuid.uuid4())
        self.sequence_boundaries = {}  # sequence_id -> (start_pos, end_pos)
        
        # Memory usage tracking
        self.total_memory_usage = 0
        self.memory_usage_by_tier = {tier: 0 for tier in NeuralMemoryTier}
        
        # Token access pattern tracking
        self.recent_access_patterns = deque(maxlen=1000)
        
        # Content hash to token mapping for deduplication
        self.content_hash_map = {}  # hash -> token_id
        
        # Create offload directory if needed
        if not os.path.exists(offload_path):
            os.makedirs(offload_path, exist_ok=True)
            
        # Performance counters
        self.stats = {
            "token_additions": 0,
            "token_retrievals": 0,
            "token_promotions": 0,
            "token_offloads": 0,
            "token_loads": 0,
            "token_consolidations": 0,
            "token_prunings": 0,
            "token_deduplications": 0,
            "memory_maintenance_runs": 0
        }
        
        logger.info(f"Initialized UltraScaleMemoryManager for 100M+ context")
        
    def init_vector_index(self):
        """Initialize vector index for semantic search"""
        self.vector_index = None
        
        # Try to use FAISS for vector indexing if available
        if HAS_FAISS:
            try:
                # Create a flat L2 index for fast cosine similarity
                self.vector_index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for normalized vectors
                
                # Move to GPU if available
                if self.device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'):
                    res = faiss.StandardGpuResources()
                    self.vector_index = faiss.index_cpu_to_gpu(res, 0, self.vector_index)
                    
                # Index to token mapping
                self.vector_index_to_token = []
                
                logger.info("FAISS vector index initialized for semantic retrieval")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS index: {e}, falling back to basic retrieval")
                self.vector_index = None
        else:
            logger.info("FAISS not available, using basic vector retrieval instead")
    
    def add_tokens(
        self, 
        token_embeddings: Tensor,
        token_ids: Optional[List[int]] = None,
        token_types: Optional[List[int]] = None,
        importance_scores: Optional[Tensor] = None,
        sequence_id: Optional[str] = None,
        cognitive_signals: Optional[Dict[NeuralCognitiveSignals, Tensor]] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[int]:
        """
        Add token embeddings to memory with rich metadata
        
        Args:
            token_embeddings: Tensor of token embeddings (seq_len, dim)
            token_ids: Optional list of token IDs (will be generated if None)
            token_types: Optional list of token types
            importance_scores: Optional importance scores
            sequence_id: Optional sequence identifier for grouped tokens
            cognitive_signals: Optional dict of cognitive signals for each token
            task_id: Optional task identifier for specialized memory
            metadata: Optional additional metadata for the tokens
            
        Returns:
            List of token IDs that were added
        """
        seq_len = token_embeddings.shape[0]
        
        # Move to device if needed
        if token_embeddings.device != self.device:
            token_embeddings = token_embeddings.to(self.device)
        
        # Generate token IDs if not provided
        if token_ids is None:
            token_ids = list(range(self.current_position, self.current_position + seq_len))
            
        # Use default importance if not provided
        if importance_scores is None:
            importance_scores = torch.ones(seq_len, device=self.device) * 0.5
            
        # Use default token types if not provided
        if token_types is None:
            token_types = [0] * seq_len
            
        # Use current sequence ID if not provided
        if sequence_id is None:
            sequence_id = self.current_sequence_id
            
        # Update sequence boundaries
        if sequence_id not in self.sequence_boundaries:
            self.sequence_boundaries[sequence_id] = (self.current_position, self.current_position + seq_len)
        else:
            start, _ = self.sequence_boundaries[sequence_id]
            self.sequence_boundaries[sequence_id] = (start, self.current_position + seq_len)
            
        # Process cognitive signals if provided
        if cognitive_signals is None:
            cognitive_signals = {}
            
        # Convert cognitive signals to per-token dict
        token_cognitive_signals = []
        for i in range(seq_len):
            signals_dict = {}
            for signal, values in cognitive_signals.items():
                if isinstance(values, torch.Tensor) and values.shape[0] == seq_len:
                    signals_dict[signal] = values[i].item()
                else:
                    # Use default value
                    signals_dict[signal] = 0.5
            token_cognitive_signals.append(signals_dict)
            
        # Update task reference
        if task_id is not None:
            self.current_task_id = task_id
            
        # Store tokens in active tier
        added_token_ids = []
        for i in range(seq_len):
            token_id = token_ids[i]
            embed = token_embeddings[i:i+1]  # Keep dimension
            importance = importance_scores[i].item()
            
            # Check for deduplication if enabled
            if self.token_deduplication:
                content_hash = hashlib.md5(embed.cpu().numpy().tobytes()).hexdigest()
                
                if content_hash in self.content_hash_map:
                    existing_token_id = self.content_hash_map[content_hash]
                    if existing_token_id in self.token_metadata:
                        # Increment the reference counter for the existing token
                        self.token_metadata[existing_token_id].importance = max(
                            self.token_metadata[existing_token_id].importance,
                            importance
                        )
                        
                        # Update access time
                        self.token_metadata[existing_token_id].update_access(time.time())
                        
                        # Skip adding this duplicate token
                        self.stats["token_deduplications"] += 1
                        continue
                else:
                    # Add hash to map
                    self.content_hash_map[content_hash] = token_id
            
            # Create token metadata
            token_meta = TokenMetadata(
                token_id=token_id,
                position=self.current_position + i,
                creation_time=time.time(),
                sequence_id=sequence_id,
                tier=NeuralMemoryTier.ACTIVE,
                importance=importance,
                last_access_time=time.time(),
                token_type=token_types[i],
                compression_ratio=1.0,
                precision_bits=self.precision_bits if self.enable_mixed_precision else 32,
                cognitive_signals=token_cognitive_signals[i],
                compressed=False
            )
            
            # Add any custom metadata
            if metadata:
                for k, v in metadata.items():
                    if hasattr(token_meta, k):
                        setattr(token_meta, k, v[i] if isinstance(v, (list, tuple)) and i < len(v) else v)
            
            # Store metadata
            self.token_metadata[token_id] = token_meta
            
            # Store embedding in active tier
            self.memory_tiers[NeuralMemoryTier.ACTIVE][token_id] = {
                "embedding": embed,
                "compressed": False
            }
            
            # Update memory usage
            self.total_memory_usage += embed.nelement() * embed.element_size()
            self.memory_usage_by_tier[NeuralMemoryTier.ACTIVE] += embed.nelement() * embed.element_size()
            
            # Add to vector index if enabled and selected for indexing
            if self.enable_vector_index and self.vector_index is not None:
                if random.random() < self.vector_index_sample_rate:
                    if self.compression_engine.semantic_encoder is not None:
                        # Use semantic encoder to get retrieval vector
                        with torch.no_grad():
                            semantic_vector = self.compression_engine.semantic_encoder(embed).cpu().numpy()
                            # Normalize for cosine similarity
                            norm = np.linalg.norm(semantic_vector)
                            if norm > 0:
                                semantic_vector = semantic_vector / norm
                                
                        # Store the semantic vector
                        token_meta.semantic_vector = torch.from_numpy(semantic_vector).to(self.device)
                        
                        # Add to index
                        try:
                            self.vector_index.add(semantic_vector.reshape(1, -1))
                            self.vector_index_to_token.append(token_id)
                        except Exception as e:
                            logger.warning(f"Failed to add to vector index: {e}")
            
            # Add to task memory if enabled
            if self.enable_task_memory and task_id is not None:
                if task_id not in self.task_memory:
                    self.task_memory[task_id] = set()
                self.task_memory[task_id].add(token_id)
            
            # Add to final list of added tokens
            added_token_ids.append(token_id)
            self.stats["token_additions"] += 1
            
        # Update current position
        self.current_position += seq_len
        
        # Run memory maintenance
        self._maintain_memory_tiers()
        
        # Checkpoint if needed
        if self.enable_checkpointing and (self.current_position - self.last_checkpoint_position >= self.checkpoint_interval):
            self._create_checkpoint()
            self.last_checkpoint_position = self.current_position
            
        return added_token_ids
        
    def retrieve_tokens(
        self, 
        token_positions: List[int],
        return_compressed: bool = False,
        return_metadata: bool = True
    ) -> Union[Tensor, Tuple[Tensor, List[TokenMetadata]]]:
        """
        Retrieve token embeddings by position
        
        Args:
            token_positions: List of token positions to retrieve
            return_compressed: Whether to return compressed representations
            return_metadata: Whether to return token metadata
            
        Returns:
            If return_metadata is True: Tuple of (embeddings, metadata)
            Otherwise: Just embeddings tensor
        """
        embeddings = []
        metadata_list = []
        
        for pos in token_positions:
            # Find token ID for this position
            token_id = None
            for tid, meta in self.token_metadata.items():
                if meta.position == pos:
                    token_id = tid
                    break
                    
            if token_id is None:
                # Token not found - use zero embedding
                embeddings.append(torch.zeros(1, self.model_dim, device=self.device))
                metadata_list.append(None)
                continue
                
            # Get metadata
            meta = self.token_metadata[token_id]
            tier = meta.tier
            
            # Update access stats
            meta.update_access(time.time())
            self.recent_access_patterns.append((token_id, time.time()))
            
            # Try to get from memory tiers
            if token_id in self.memory_tiers[tier]:
                token_data = self.memory_tiers[tier][token_id]
                
                if token_data["compressed"] and not return_compressed:
                    # Decompress
                    embedding = self.compression_engine.decompress_token(
                        token_data["compressed_data"], token_data["metadata"])
                else:
                    embedding = token_data["embedding"]
                    
                embeddings.append(embedding)
                metadata_list.append(meta if return_metadata else None)
                
                # Consider promoting frequently accessed tokens
                if meta.access_count > 5 and tier not in [NeuralMemoryTier.FOCUS, NeuralMemoryTier.ACTIVE, NeuralMemoryTier.FOREGROUND]:
                    self._promote_token(token_id)
                    
                self.stats["token_retrievals"] += 1
            else:
                # Try to load from offloaded storage
                loaded = self._load_token_from_offload(token_id)
                
                if loaded:
                    token_data = self.memory_tiers[tier][token_id]
                    
                    if token_data["compressed"] and not return_compressed:
                        # Decompress
                        embedding = self.compression_engine.decompress_token(
                            token_data["compressed_data"], token_data["metadata"])
                    else:
                        embedding = token_data["embedding"]
                        
                    embeddings.append(embedding)
                    metadata_list.append(meta if return_metadata else None)
                    self.stats["token_retrievals"] += 1
                    self.stats["token_loads"] += 1
                else:
                    # Token not found even in offload - use zero embedding
                    embeddings.append(torch.zeros(1, self.model_dim, device=self.device))
                    metadata_list.append(None)
                    
        # Stack embeddings
        if embeddings:
            result_embeddings = torch.cat(embeddings, dim=0)
            if return_metadata:
                return result_embeddings, metadata_list
            else:
                return result_embeddings
        else:
            # Return empty tensor
            empty_result = torch.zeros(0, self.model_dim, device=self.device)
            if return_metadata:
                return empty_result, []
            else:
                return empty_result
                
    def retrieve_by_sequence(
        self, 
        sequence_id: str,
        return_metadata: bool = True
    ) -> Union[Tensor, Tuple[Tensor, List[TokenMetadata]]]:
        """
        Retrieve all tokens from a given sequence
        
        Args:
            sequence_id: Sequence ID to retrieve
            return_metadata: Whether to return token metadata
            
        Returns:
            If return_metadata is True: Tuple of (embeddings, metadata)
            Otherwise: Just embeddings tensor
        """
        if sequence_id not in self.sequence_boundaries:
            logger.warning(f"Sequence ID {sequence_id} not found")
            empty_result = torch.zeros(0, self.model_dim, device=self.device)
            if return_metadata:
                return empty_result, []
            else:
                return empty_result
                
        start_pos, end_pos = self.sequence_boundaries[sequence_id]
        positions = list(range(start_pos, end_pos))
        
        return self.retrieve_tokens(positions, return_metadata=return_metadata)
    
    def retrieve_by_semantic(
        self, 
        query_embedding: Tensor,
        top_k: int = 5,
        threshold: float = 0.7,
        return_metadata: bool = True
    ) -> Union[Tensor, Tuple[Tensor, List[TokenMetadata], List[float]]]:
        """
        Retrieve tokens semantically similar to the query embedding
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of results to return
            threshold: Similarity threshold (0-1)
            return_metadata: Whether to return token metadata
            
        Returns:
            If return_metadata is True: Tuple of (embeddings, metadata, similarities)
            Otherwise: Just embeddings tensor
        """
        if not self.enable_vector_index or self.vector_index is None:
            logger.warning("Vector index not enabled or initialized")
            empty_result = torch.zeros(0, self.model_dim, device=self.device)
            if return_metadata:
                return empty_result, [], []
            else:
                return empty_result
        
        # Ensure query is on the correct device
        if query_embedding.device != self.device:
            query_embedding = query_embedding.to(self.device)
            
        # Get semantic vector
        if self.compression_engine.semantic_encoder is not None:
            with torch.no_grad():
                semantic_vector = self.compression_engine.semantic_encoder(query_embedding).cpu().numpy()
                # Normalize for cosine similarity
                norm = np.linalg.norm(semantic_vector)
                if norm > 0:
                    semantic_vector = semantic_vector / norm
        else:
            # Use raw embedding if no encoder
            semantic_vector = query_embedding.cpu().numpy()
            norm = np.linalg.norm(semantic_vector)
            if norm > 0:
                semantic_vector = semantic_vector / norm
                
        try:
            # Search the index
            D, I = self.vector_index.search(semantic_vector.reshape(1, -1), min(top_k, len(self.vector_index_to_token)))
            
            # Filter by threshold
            valid_indices = [i for i, d in zip(I[0], D[0]) if d >= threshold and i < len(self.vector_index_to_token)]
            
            # Get token IDs
            token_ids = [self.vector_index_to_token[idx] for idx in valid_indices]
            similarities = [float(D[0][i]) for i, idx in enumerate(valid_indices)]
            
            # Retrieve tokens
            embeddings = []
            metadata_list = []
            
            for tid in token_ids:
                if tid in self.token_metadata:
                    meta = self.token_metadata[tid]
                    tier = meta.tier
                    
                    # Update access stats
                    meta.update_access(time.time())
                    
                    # Get embedding
                    if tid in self.memory_tiers[tier]:
                        token_data = self.memory_tiers[tier][tid]
                        
                        if token_data["compressed"]:
                            # Decompress
                            embedding = self.compression_engine.decompress_token(
                                token_data["compressed_data"], token_data["metadata"])
                        else:
                            embedding = token_data["embedding"]
                            
                        embeddings.append(embedding)
                        metadata_list.append(meta if return_metadata else None)
                    else:
                        # Try to load from offload
                        loaded = self._load_token_from_offload(tid)
                        
                        if loaded:
                            token_data = self.memory_tiers[tier][tid]
                            
                            if token_data["compressed"]:
                                # Decompress
                                embedding = self.compression_engine.decompress_token(
                                    token_data["compressed_data"], token_data["metadata"])
                            else:
                                embedding = token_data["embedding"]
                                
                            embeddings.append(embedding)
                            metadata_list.append(meta if return_metadata else None)
            
            # Stack embeddings if any found
            if embeddings:
                result_embeddings = torch.cat(embeddings, dim=0)
                if return_metadata:
                    return result_embeddings, metadata_list, similarities[:len(embeddings)]
                else:
                    return result_embeddings
            else:
                empty_result = torch.zeros(0, self.model_dim, device=self.device)
                if return_metadata:
                    return empty_result, [], []
                else:
                    return empty_result
                    
        except Exception as e:
            logger.warning(f"Error in semantic retrieval: {e}")
            empty_result = torch.zeros(0, self.model_dim, device=self.device)
            if return_metadata:
                return empty_result, [], []
            else:
                return empty_result
    
    def get_context_window(
        self, 
        window_size: int,
        start_position: Optional[int] = None,
        return_importance: bool = False,
        return_metadata: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[TokenMetadata]]]:
        """
        Get a context window of token embeddings
        
        Args:
            window_size: Size of the context window
            start_position: Starting position (or current position - window_size if None)
            return_importance: Whether to return importance scores
            return_metadata: Whether to return token metadata
            
        Returns:
            Token embeddings tensor or tuple with additional information
        """
        if start_position is None:
            start_position = max(0, self.current_position - window_size)
            
        positions = list(range(start_position, start_position + window_size))
        
        if return_metadata:
            embeddings, metadata = self.retrieve_tokens(positions, return_metadata=True)
            if return_importance:
                importance = torch.tensor([m.importance if m is not None else 0.0 for m in metadata],
                                       device=embeddings.device)
                return embeddings, importance, metadata
            else:
                return embeddings, metadata
        else:
            if return_importance:
                embeddings, metadata = self.retrieve_tokens(positions, return_metadata=True)
                importance = torch.tensor([m.importance if m is not None else 0.0 for m in metadata],
                                       device=embeddings.device)
                return embeddings, importance
            else:
                return self.retrieve_tokens(positions, return_metadata=False)
                
    def get_recursive_memory_usage(self, obj, visited=None):
        """Recursively calculate memory usage of nested structures"""
        if visited is None:
            visited = set()
            
        obj_id = id(obj)
        if obj_id in visited:
            return 0
            
        visited.add(obj_id)
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(self.get_recursive_memory_usage(k, visited) + 
                        self.get_recursive_memory_usage(v, visited) 
                        for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self.get_recursive_memory_usage(i, visited) for i in obj)
        elif isinstance(obj, torch.Tensor):
            # Already counted by sys.getsizeof, but add any storage not counted
            if obj.is_quantized:
                # Add quantization scales and zero points
                size += obj.q_scale().numel() * obj.q_scale().element_size()
                if obj.q_zero_point() is not None:
                    size += obj.q_zero_point().numel() * obj.q_zero_point().element_size()
            
        return size
            
    def _maintain_memory_tiers(self) -> None:
        """
        Maintain memory tiers by compressing, migrating, and offloading tokens
        to keep memory usage within limits while preserving a 100M+ context
        """
        # Skip if we've just done this recently (avoid excessive maintenance)
        if self.stats["token_additions"] % 1000 != 0 and not self._is_memory_critical():
            return
            
        self.stats["memory_maintenance_runs"] += 1
        logger.debug("Running memory tier maintenance")
        
        # Process tiers from smallest/closest to largest/farthest
        
        # Step 1: FOCUS -> ACTIVE tier management
        focus_count = len(self.memory_tiers[NeuralMemoryTier.FOCUS])
        if focus_count > self.max_focus_tokens:
            # Sort tokens by importance (lowest first)
            focus_tokens = sorted(
                [(tid, self.token_metadata[tid].importance) for tid in self.memory_tiers[NeuralMemoryTier.FOCUS]],
                key=lambda x: x[1]
            )
            
            # Migrate least important tokens to ACTIVE tier
            to_migrate = focus_count - self.max_focus_tokens
            for tid, _ in focus_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.FOCUS, NeuralMemoryTier.ACTIVE)
                
        # Step 2: ACTIVE -> FOREGROUND tier management
        active_count = len(self.memory_tiers[NeuralMemoryTier.ACTIVE])
        if active_count > self.max_active_tokens:
            # Sort tokens by position (oldest first)
            active_tokens = sorted(
                [(tid, self.token_metadata[tid].position) for tid in self.memory_tiers[NeuralMemoryTier.ACTIVE]],
                key=lambda x: x[1]
            )
            
            # Migrate oldest tokens to FOREGROUND tier
            to_migrate = active_count - self.max_active_tokens
            for tid, _ in active_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.ACTIVE, NeuralMemoryTier.FOREGROUND)
                
        # Step 3: FOREGROUND -> BACKGROUND tier management
        foreground_count = len(self.memory_tiers[NeuralMemoryTier.FOREGROUND])
        if foreground_count > self.max_foreground_tokens:
            # Sort tokens by position (oldest first) AND importance (lowest first)
            # This weighted approach keeps important tokens in the foreground longer
            foreground_tokens = sorted(
                [(tid, self.token_metadata[tid].position, self.token_metadata[tid].importance) 
                 for tid in self.memory_tiers[NeuralMemoryTier.FOREGROUND]],
                key=lambda x: (x[1] - self.current_position) * (1.0 - x[2])  # Age multiplied by (1-importance)
            )
            
            # Migrate oldest tokens to BACKGROUND tier
            to_migrate = foreground_count - self.max_foreground_tokens
            for tid, _, _ in foreground_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.FOREGROUND, NeuralMemoryTier.BACKGROUND)
        
        # Step 4: BACKGROUND -> lower tiers based on specialized characteristics
        background_count = len(self.memory_tiers[NeuralMemoryTier.BACKGROUND])
        if background_count > self.max_background_tokens:
            # Sort tokens by position and importance
            background_tokens = sorted(
                [(tid, self.token_metadata[tid].position, self.token_metadata[tid].importance) 
                 for tid in self.memory_tiers[NeuralMemoryTier.BACKGROUND]],
                key=lambda x: (x[1] - self.current_position) * (1.0 - x[2])
            )
            
            # Migrate tokens to appropriate tiers based on characteristics
            to_migrate = background_count - self.max_background_tokens
            for tid, _, importance in background_tokens[:to_migrate]:
                meta = self.token_metadata[tid]
                
                # Pick appropriate tier based on token characteristics
                if importance > self.semantic_extraction_threshold:
                    # High importance tokens go to semantic memory
                    target_tier = NeuralMemoryTier.SEMANTIC
                elif meta.position > self.current_position - 1000000:
                    # Recent enough for episodic memory
                    target_tier = NeuralMemoryTier.EPISODIC
                else:
                    # Otherwise general memory
                    target_tier = NeuralMemoryTier.GENERAL
                    
                self._migrate_token(tid, NeuralMemoryTier.BACKGROUND, target_tier)
                
        # Step 5: Manage mid-tier memories (EPISODIC, SEMANTIC, GENERAL)
        
        # 5.1: EPISODIC memory management
        episodic_count = len(self.memory_tiers[NeuralMemoryTier.EPISODIC])
        if episodic_count > self.max_episodic_tokens:
            episodic_tokens = sorted(
                [(tid, self.token_metadata[tid].position) for tid in self.memory_tiers[NeuralMemoryTier.EPISODIC]],
                key=lambda x: x[1]  # Sort by position (oldest first)
            )
            
            to_migrate = episodic_count - self.max_episodic_tokens
            for tid, _ in episodic_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.EPISODIC, NeuralMemoryTier.GENERAL)
                
        # 5.2: SEMANTIC memory management
        semantic_count = len(self.memory_tiers[NeuralMemoryTier.SEMANTIC])
        if semantic_count > self.max_semantic_tokens:
            semantic_tokens = sorted(
                [(tid, self.token_metadata[tid].importance) for tid in self.memory_tiers[NeuralMemoryTier.SEMANTIC]],
                key=lambda x: x[1]  # Sort by importance (lowest first)
            )
            
            to_migrate = semantic_count - self.max_semantic_tokens
            for tid, _ in semantic_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.SEMANTIC, NeuralMemoryTier.CATEGORICAL)
                
        # 5.3: GENERAL memory management
        general_count = len(self.memory_tiers[NeuralMemoryTier.GENERAL])
        if general_count > self.max_general_tokens:
            general_tokens = sorted(
                [(tid, self.token_metadata[tid].position) for tid in self.memory_tiers[NeuralMemoryTier.GENERAL]],
                key=lambda x: x[1]  # Sort by position (oldest first)
            )
            
            to_migrate = general_count - self.max_general_tokens
            for tid, _ in general_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.GENERAL, NeuralMemoryTier.CATEGORICAL)
                
        # Step 6: Deep memory tier management (CATEGORICAL, ARCHIVAL, REFERENCE, CONSOLIDATED)
        
        # 6.1: CATEGORICAL memory management
        categorical_count = len(self.memory_tiers[NeuralMemoryTier.CATEGORICAL])
        if categorical_count > self.max_categorical_tokens:
            categorical_tokens = sorted(
                [(tid, self.token_metadata[tid].position) for tid in self.memory_tiers[NeuralMemoryTier.CATEGORICAL]],
                key=lambda x: x[1]  # Sort by position (oldest first)
            )
            
            to_migrate = categorical_count - self.max_categorical_tokens
            for tid, _ in categorical_tokens[:to_migrate]:
                self._migrate_token(tid, NeuralMemoryTier.CATEGORICAL, NeuralMemoryTier.ARCHIVAL)
                
        # 6.2: ARCHIVAL memory management
        archival_count = len(self.memory_tiers[NeuralMemoryTier.ARCHIVAL])
        if archival_count > self.max_archival_tokens:
            archival_tokens = sorted(
                [(tid, self.token_metadata[tid].position) for tid in self.memory_tiers[NeuralMemoryTier.ARCHIVAL]],
                key=lambda x: x[1]  # Sort by position (oldest first)
            )
            
            to_migrate = archival_count - self.max_archival_tokens
            for tid, _ in archival_tokens[:to_migrate]:
                # For extremely old tokens, consider consolidation or pruning
                meta = self.token_metadata[tid]
                if meta.importance < self.pruning_threshold and self.token_pruning:
                    self._prune_token(tid)
                    self.stats["token_prunings"] += 1
                else:
                    # Alternate between REFERENCE and CONSOLIDATED
                    target_tier = NeuralMemoryTier.REFERENCE if random.random() < 0.5 else NeuralMemoryTier.CONSOLIDATED
                    self._migrate_token(tid, NeuralMemoryTier.ARCHIVAL, target_tier)
         
        # Step 7: Apply token pruning for distant tokens if enabled
        if self.token_pruning:
            # Identify pruning candidates from lower tiers
            pruning_candidates = []
            for tier in [NeuralMemoryTier.CATEGORICAL, NeuralMemoryTier.ARCHIVAL, 
                        NeuralMemoryTier.REFERENCE, NeuralMemoryTier.CONSOLIDATED]:
                for tid in list(self.memory_tiers[tier].keys()):
                    meta = self.token_metadata[tid]
                    distance = self.current_position - meta.position
                    
                    if distance >= self.min_pruning_distance:
                        # Calculate pruning score (higher = more prunable)
                        # Factors: low importance, high distance, low access frequency
                        pruning_score = (
                            (1.0 - meta.importance) * 0.6 +  # Low importance (60% weight)
                            min(1.0, distance / 10000000) * 0.3 +  # Distance (30% weight)
                            max(0.0, 1.0 - (meta.access_count / 100)) * 0.1  # Low access (10% weight)
                        )
                        
                        # Only consider if score is high enough
                        if pruning_score > 0.7:
                            pruning_candidates.append((tid, pruning_score))
            
            # Sort and prune up to a percentage of tokens
            if pruning_candidates:
                # Sort by score (highest first)
                pruning_candidates.sort(key=lambda x: -x[1])
                
                # Determine how many tokens to prune
                resident_count = sum(len(self.memory_tiers[tier]) for tier in NeuralMemoryTier)
                max_prune = min(len(pruning_candidates), max(1, int(resident_count * 0.01)))  # Max 1% at a time
                
                # Prune tokens
                for tid, _ in pruning_candidates[:max_prune]:
                    self._prune_token(tid)
                    self.stats["token_prunings"] += 1
                    
        # Step 8: Offload tokens if we exceed resident memory limit
        resident_count = sum(len(self.memory_tiers[tier]) for tier in NeuralMemoryTier)
        
        if resident_count > self.max_memory_resident_tokens:
            # Focus on oldest tokens from colder tiers first
            offload_candidates = []
            for tier in [NeuralMemoryTier.CONSOLIDATED, NeuralMemoryTier.REFERENCE, 
                         NeuralMemoryTier.ARCHIVAL, NeuralMemoryTier.CATEGORICAL,
                         NeuralMemoryTier.GENERAL]:
                for tid in self.memory_tiers[tier]:
                    meta = self.token_metadata[tid]
                    distance = self.current_position - meta.position
                    last_access = meta.last_access_time
                    access_count = meta.access_count
                    
                    # Score based on distance, recency, and access frequency
                    score = distance * 0.5 + (time.time() - last_access) * 0.3 - access_count * 10
                    offload_candidates.append((tid, score))
                    
            # Sort by score (highest first - most offloadable)
            offload_candidates.sort(key=lambda x: -x[1])
            
            # Offload enough tokens to get under the limit
            to_offload = resident_count - self.max_memory_resident_tokens
            for tid, _ in offload_candidates[:to_offload]:
                self._offload_token(tid)
                self.stats["token_offloads"] += 1
                
        # Step 9: Consolidate memory if needed
        if self.current_position >= self.memory_consolidation_interval and self.current_position % self.memory_consolidation_interval == 0:
            self._consolidate_memory()
            
        # Step 10: Auto-scale memory tiers if enabled
        if self.auto_scaling and self.current_position > 10000:
            self._auto_scale_memory_tiers()
            
    def _is_memory_critical(self) -> bool:
        """Check if memory usage is critical and requires immediate maintenance"""
        # Check if any tier is over 95% capacity
        critical_tiers = [
            NeuralMemoryTier.FOCUS,
            NeuralMemoryTier.ACTIVE,
            NeuralMemoryTier.FOREGROUND,
            NeuralMemoryTier.BACKGROUND
        ]
        
        for tier in critical_tiers:
            max_size = getattr(self, f"max_{tier.name.lower()}_tokens")
            current_size = len(self.memory_tiers[tier])
            if current_size > max_size * 0.95:
                return True
                
        # Check if total resident token count is near limit
        resident_count = sum(len(self.memory_tiers[tier]) for tier in NeuralMemoryTier)
        if resident_count > self.max_memory_resident_tokens * 0.95:
            return True
            
        return False
            
    def _migrate_token(
        self, 
        token_id: int, 
        from_tier: NeuralMemoryTier, 
        to_tier: NeuralMemoryTier
    ) -> bool:
        """Migrate a token from one tier to another with appropriate compression"""
        if token_id not in self.memory_tiers[from_tier]:
            return False
            
        # Get token data and metadata
        token_data = self.memory_tiers[from_tier][token_id]
        meta = self.token_metadata[token_id]
        
        # Update memory usage tracking
        if token_data["compressed"]:
            # For compressed tokens, calculate size based on compressed data
            size = sum(t.nelement() * t.element_size() for t in token_data["compressed_data"].values())
        else:
            # For uncompressed tokens, use embedding size
            size = token_data["embedding"].nelement() * token_data["embedding"].element_size()
            
        self.memory_usage_by_tier[from_tier] -= size
        self.memory_usage_by_tier[to_tier] += size
        
        # Move token to new tier
        self.memory_tiers[to_tier][token_id] = token_data
        del self.memory_tiers[from_tier][token_id]
        
        # Update metadata
        meta.tier = to_tier
        
        # Apply compression if needed when moving to a colder tier
        if self.enable_tensor_compression and not token_data["compressed"]:
            # Tiers that require compression
            compression_tiers = [
                NeuralMemoryTier.BACKGROUND,
                NeuralMemoryTier.EPISODIC,
                NeuralMemoryTier.SEMANTIC,
                NeuralMemoryTier.GENERAL,
                NeuralMemoryTier.CATEGORICAL,
                NeuralMemoryTier.ARCHIVAL,
                NeuralMemoryTier.REFERENCE,
                NeuralMemoryTier.CONSOLIDATED
            ]
            
            if to_tier in compression_tiers:
                # Determine distance for compression ratio
                distance = self.current_position - meta.position
                self._compress_token(token_id, distance)
                
        return True
        
    def _compress_token(self, token_id: int, distance: int) -> bool:
        """Compress a token's embedding based on its distance and characteristics"""
        if token_id not in self.token_metadata:
            return False
            
        meta = self.token_metadata[token_id]
        tier = meta.tier
        
        if token_id not in self.memory_tiers[tier]:
            return False
            
        token_data = self.memory_tiers[tier][token_id]
        
        # Skip if already compressed
        if token_data["compressed"]:
            return False
            
        # Get embedding and importance
        embedding = token_data["embedding"]
        importance = meta.importance
        token_type = meta.token_type
        
        # Apply compression
        compressed_data, metadata = self.compression_engine.compress_token(
            embedding, distance, importance, token_type, tier
        )
        
        # Update token data
        old_size = embedding.nelement() * embedding.element_size()
        
        # Calculate new size
        new_size = sum(t.nelement() * t.element_size() for t in compressed_data.values())
        
        # Update memory usage
        self.total_memory_usage = self.total_memory_usage - old_size + new_size
        self.memory_usage_by_tier[tier] = self.memory_usage_by_tier[tier] - old_size + new_size
        
        # Store compressed data
        self.memory_tiers[tier][token_id] = {
            "compressed_data": compressed_data,
            "metadata": metadata,
            "compressed": True
        }
        
        # Update token metadata
        meta.compression_ratio = metadata.get("ratio", 1.0)
        meta.compressed = True
        meta.precision_bits = metadata.get("bits", 32)
        
        return True
        
    def _offload_token(self, token_id: int) -> bool:
        """Offload a token to disk storage"""
        if token_id not in self.token_metadata:
            return False
            
        meta = self.token_metadata[token_id]
        tier = meta.tier
        
        if token_id not in self.memory_tiers[tier]:
            return False
            
        # Get token data
        token_data = self.memory_tiers[tier][token_id]
        
        # Ensure token is compressed before offloading
        if not token_data["compressed"] and self.enable_tensor_compression:
            distance = self.current_position - meta.position
            self._compress_token(token_id, distance)
            token_data = self.memory_tiers[tier][token_id]
            
        # Create offload file path
        offload_path = os.path.join(self.offload_path, f"token_{token_id}.pt")
        
        try:
            # Save token data and metadata
            torch.save({
                "token_data": token_data,
                "metadata": meta
            }, offload_path)
            
            # Update memory usage
            if token_data["compressed"]:
                size = sum(t.nelement() * t.element_size() for t in token_data["compressed_data"].values())
            else:
                size = token_data["embedding"].nelement() * token_data["embedding"].element_size()
                
            self.total_memory_usage -= size
            self.memory_usage_by_tier[tier] -= size
            
            # Remove from memory
            del self.memory_tiers[tier][token_id]
            
            # Update metadata
            meta.offloaded = True
            meta.offload_path = offload_path
            
            # Update token tier to OFFLOADED
            meta.tier = NeuralMemoryTier.OFFLOADED
            
            return True
        except Exception as e:
            logger.error(f"Failed to offload token {token_id}: {e}")
            return False
            
    def _load_token_from_offload(self, token_id: int) -> bool:
        """Load a token from offload storage"""
        if token_id not in self.token_metadata:
            return False
            
        meta = self.token_metadata[token_id]
        
        # Skip if already in memory
        for tier in NeuralMemoryTier:
            if token_id in self.memory_tiers[tier]:
                return True
                
        # Get offload path
        offload_path = meta.offload_path
        if not offload_path or not os.path.exists(offload_path):
            # Try default path
            offload_path = os.path.join(self.offload_path, f"token_{token_id}.pt")
            if not os.path.exists(offload_path):
                return False
                
        try:
            # Load token data
            data = torch.load(offload_path)
            
            # Get token data and metadata
            token_data = data["token_data"]
            loaded_meta = data["metadata"]
            
            # Determine appropriate tier
            tier = loaded_meta.tier
            if tier == NeuralMemoryTier.OFFLOADED:
                # Place in ARCHIVED tier when loading
                tier = NeuralMemoryTier.ARCHIVAL
                
            # Add back to memory
            self.memory_tiers[tier][token_id] = token_data
            
            # Update memory usage
            if token_data["compressed"]:
                size = sum(t.nelement() * t.element_size() for t in token_data["compressed_data"].values())
            else:
                size = token_data["embedding"].nelement() * token_data["embedding"].element_size()
                
            self.total_memory_usage += size
            self.memory_usage_by_tier[tier] += size
            
            # Update metadata
            meta.offloaded = False
            meta.tier = tier
            
            return True
        except Exception as e:
            logger.error(f"Failed to load token {token_id} from offload: {e}")
            return False
            
    def _promote_token(self, token_id: int) -> bool:
        """Promote a token to a higher tier based on importance and usage"""
        if token_id not in self.token_metadata:
            return False
            
        meta = self.token_metadata[token_id]
        current_tier = meta.tier
        
        # Skip tokens that are already in high tiers
        if current_tier in [NeuralMemoryTier.FOCUS, NeuralMemoryTier.ACTIVE, NeuralMemoryTier.FOREGROUND]:
            return False
            
        # Determine target tier based on importance, recency and current tier
        importance = meta.importance
        recency = 1.0 - min(1.0, (self.current_position - meta.position) / 1000000)
        access_score = min(1.0, meta.access_count / 10.0)
        
        # Combined promotion score
        promotion_score = importance * 0.4 + recency * 0.4 + access_score * 0.2
        
        # Only promote if score is high enough
        if promotion_score < 0.6:
            return False
            
        # Define promotion paths
        promotion_paths = {
            NeuralMemoryTier.BACKGROUND: NeuralMemoryTier.FOREGROUND,
            NeuralMemoryTier.EPISODIC: NeuralMemoryTier.BACKGROUND,
            NeuralMemoryTier.SEMANTIC: NeuralMemoryTier.BACKGROUND,
            NeuralMemoryTier.GENERAL: NeuralMemoryTier.EPISODIC,
            NeuralMemoryTier.CATEGORICAL: NeuralMemoryTier.GENERAL,
            NeuralMemoryTier.ARCHIVAL: NeuralMemoryTier.CATEGORICAL,
            NeuralMemoryTier.REFERENCE: NeuralMemoryTier.GENERAL,
            NeuralMemoryTier.CONSOLIDATED: NeuralMemoryTier.SEMANTIC,
            NeuralMemoryTier.OFFLOADED: NeuralMemoryTier.ARCHIVAL
        }
        
        # Get target tier
        if current_tier in promotion_paths:
            target_tier = promotion_paths[current_tier]
        else:
            # No defined promotion path
            return False
            
        # Ensure token is in memory
        if current_tier == NeuralMemoryTier.OFFLOADED:
            if not self._load_token_from_offload(token_id):
                return False
                
        # Migrate to target tier
        success = self._migrate_token(token_id, current_tier, target_tier)
        
        # If token was compressed and moving to a higher tier, decompress it
        if success and target_tier in [NeuralMemoryTier.FOCUS, NeuralMemoryTier.ACTIVE, NeuralMemoryTier.FOREGROUND]:
            token_data = self.memory_tiers[target_tier][token_id]
            if token_data["compressed"]:
                # Decompress and update
                decompressed = self.compression_engine.decompress_token(
                    token_data["compressed_data"], token_data["metadata"])
                
                # Update memory usage
                old_size = sum(t.nelement() * t.element_size() for t in token_data["compressed_data"].values())
                new_size = decompressed.nelement() * decompressed.element_size()
                
                self.total_memory_usage = self.total_memory_usage - old_size + new_size
                self.memory_usage_by_tier[target_tier] = self.memory_usage_by_tier[target_tier] - old_size + new_size
                
                # Store decompressed tensor
                self.memory_tiers[target_tier][token_id] = {
                    "embedding": decompressed,
                    "compressed": False
                }
                
                # Update metadata
                meta.compressed = False
                meta.compression_ratio = 1.0
                meta.precision_bits = 32
                
        # Track promotion in stats
        if success:
            self.stats["token_promotions"] += 1
                
        return success
            
    def _prune_token(self, token_id: int) -> bool:
        """Completely remove a token from memory"""
        if token_id not in self.token_metadata:
            return False
            
        meta = self.token_metadata[token_id]
        tier = meta.tier
        
        # Remove from memory if present
        if tier != NeuralMemoryTier.OFFLOADED and token_id in self.memory_tiers[tier]:
            token_data = self.memory_tiers[tier][token_id]
            
            # Update memory usage
            if token_data["compressed"]:
                size = sum(t.nelement() * t.element_size() for t in token_data["compressed_data"].values())
            else:
                size = token_data["embedding"].nelement() * token_data["embedding"].element_size()
                
            self.total_memory_usage -= size
            self.memory_usage_by_tier[tier] -= size
            
            del self.memory_tiers[tier][token_id]
            
        # Remove from offload storage if present
        offload_path = meta.offload_path
        if offload_path and os.path.exists(offload_path):
            try:
                os.remove(offload_path)
            except Exception as e:
                logger.warning(f"Failed to remove offloaded token {token_id}: {e}")
                
        # Remove from vector index if present
        if self.enable_vector_index and self.vector_index is not None:
            try:
                if token_id in self.vector_index_to_token:
                    idx = self.vector_index_to_token.index(token_id)
                    # Note: We can't easily remove from FAISS index, so we just remove from the mapping
                    self.vector_index_to_token[idx] = -1  # Mark as invalid
            except Exception as e:
                logger.warning(f"Error updating vector index during pruning: {e}")
                
        # Remove from content hash map if present
        if self.token_deduplication:
            for hash_val, tid in list(self.content_hash_map.items()):
                if tid == token_id:
                    del self.content_hash_map[hash_val]
                    break
                    
        # Remove from task memory if present
        if self.enable_task_memory and self.task_memory:
            for task_id, tokens in self.task_memory.items():
                if token_id in tokens:
                    tokens.remove(token_id)
                    
        # Remove metadata
        del self.token_metadata[token_id]
        
        return True
        
    def _consolidate_memory(self) -> None:
        """
        Perform memory consolidation by extracting semantic information
        from groups of related tokens and creating consolidated tokens
        that represent their collective meaning
        """
        # Skip if no semantic encoder available
        if not self.enable_semantic_compression or self.compression_engine.semantic_encoder is None:
            return
            
        logger.info("Performing memory consolidation")
        
        # Target tiers for consolidation
        target_tiers = [
            NeuralMemoryTier.ARCHIVAL,
            NeuralMemoryTier.CATEGORICAL,
            NeuralMemoryTier.GENERAL
        ]
        
        # Group tokens by sequence
        sequence_tokens = defaultdict(list)
        
        for tier in target_tiers:
            for token_id in self.memory_tiers[tier]:
                meta = self.token_metadata[token_id]
                if meta.sequence_id:
                    sequence_tokens[meta.sequence_id].append(token_id)
                    
        # Process sequences with enough tokens
        consolidated_count = 0
        for sequence_id, tokens in sequence_tokens.items():
            # Only consolidate sequences with enough tokens
            if len(tokens) < 10:
                continue
                
            # Group into chunks of 10-50 tokens
            chunk_size = min(50, max(10, len(tokens) // 10))
            
            # Process in chunks
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i+chunk_size]
                if len(chunk) < 5:  # Skip very small chunks
                    continue
                    
                # Get embeddings for these tokens
                embeddings = []
                for tid in chunk:
                    for tier in target_tiers:
                        if tid in self.memory_tiers[tier]:
                            token_data = self.memory_tiers[tier][tid]
                            
                            if token_data["compressed"]:
                                embed = self.compression_engine.decompress_token(
                                    token_data["compressed_data"], token_data["metadata"])
                            else:
                                embed = token_data["embedding"]
                                
                            embeddings.append(embed)
                            break
                
                if not embeddings:
                    continue
                    
                # Stack embeddings
                chunk_embeds = torch.cat(embeddings, dim=0)
                
                # Calculate consolidated embedding using semantic encoder
                with torch.no_grad():
                    # Get semantic vectors
                    semantic_vectors = self.compression_engine.semantic_encoder(chunk_embeds)
                    
                    # Average to get consolidated representation
                    consolidated_semantic = semantic_vectors.mean(dim=0, keepdim=True)
                    
                    # Decode back to embedding space
                    consolidated_embedding = self.compression_engine.semantic_decoder(consolidated_semantic)
                
                # Create a new consolidated token
                consolidated_id = self.current_position + 1000000 + consolidated_count
                consolidated_count += 1
                
                # Calculate average importance of consolidated tokens
                avg_importance = sum(self.token_metadata[tid].importance for tid in chunk) / len(chunk)
                
                # Create metadata for consolidated token
                meta = TokenMetadata(
                    token_id=consolidated_id,
                    position=consolidated_id,  # Use ID as position for consolidated tokens
                    creation_time=time.time(),
                    sequence_id=sequence_id,
                    tier=NeuralMemoryTier.CONSOLIDATED,
                    importance=avg_importance * 1.2,  # Boost importance slightly
                    last_access_time=time.time(),
                    token_type=0,  # Generic type for consolidated tokens
                    compression_ratio=0.1,  # Assume high compression
                    precision_bits=16,
                    compressed=True,
                    tags={"consolidated", f"seq_{sequence_id}"}
                )
                
                # Add references to original tokens
                meta.related_tokens = {tid: 1.0 for tid in chunk}
                
                # Store metadata
                self.token_metadata[consolidated_id] = meta
                
                # Compress and store the consolidated token
                compressed_data, compress_meta = self.compression_engine.compress_token(
                    consolidated_embedding, 1000000, avg_importance, 0, NeuralMemoryTier.CONSOLIDATED
                )
                
                # Store in memory
                self.memory_tiers[NeuralMemoryTier.CONSOLIDATED][consolidated_id] = {
                    "compressed_data": compressed_data,
                    "metadata": compress_meta,
                    "compressed": True
                }
                
                # Update memory usage
                size = sum(t.nelement() * t.element_size() for t in compressed_data.values())
                self.total_memory_usage += size
                self.memory_usage_by_tier[NeuralMemoryTier.CONSOLIDATED] += size
                
                # Prune some of the original tokens to save space
                # Keep highest importance tokens, prune lowest
                if self.token_pruning:
                    # Sort by importance
                    sorted_chunk = sorted([(tid, self.token_metadata[tid].importance) for tid in chunk],
                                        key=lambda x: x[1])
                    
                    # Prune bottom third of tokens
                    to_prune = sorted_chunk[:len(chunk)//3]
                    for tid, _ in to_prune:
                        self._prune_token(tid)
                        self.stats["token_prunings"] += 1
                        
        # Update stats
        self.stats["token_consolidations"] += consolidated_count
        
        logger.info(f"Memory consolidation complete: {consolidated_count} consolidated tokens created")
                    
    def _auto_scale_memory_tiers(self) -> None:
        """
        Automatically adjust memory tier sizes based on usage patterns
        """
        if not self.auto_scaling:
            return
            
        # Calculate usage ratios for each tier
        usage_ratios = {}
        for tier in NeuralMemoryTier:
            max_size = getattr(self, f"max_{tier.name.lower()}_tokens", 0)
            if max_size > 0:
                current_size = len(self.memory_tiers[tier])
                usage_ratios[tier] = current_size / max_size
        
        # Find tiers that are nearly full
        full_tiers = [tier for tier, ratio in usage_ratios.items() if ratio > 0.9]
        
        # Find tiers with low utilization
        empty_tiers = [tier for tier, ratio in usage_ratios.items() if ratio < 0.3]
        
        # Skip if no scaling needed
        if not full_tiers or not empty_tiers:
            return
            
        # Calculate scaling factor (borrow 10% from each empty tier for each full tier)
        scaling_factor = 0.1
        
        # Scale up full tiers and scale down empty tiers
        for full_tier in full_tiers:
            full_tier_attr = f"max_{full_tier.name.lower()}_tokens"
            current_size = getattr(self, full_tier_attr)
            
            # Distribute capacity from empty tiers
            for empty_tier in empty_tiers:
                empty_tier_attr = f"max_{empty_tier.name.lower()}_tokens"
                empty_size = getattr(self, empty_tier_attr)
                
                # Calculate amount to transfer
                transfer_amount = int(empty_size * scaling_factor)
                
                # Scale down empty tier
                setattr(self, empty_tier_attr, empty_size - transfer_amount)
                
                # Scale up full tier
                setattr(self, full_tier_attr, current_size + transfer_amount)
                
                logger.debug(f"Auto-scaled: {transfer_amount} slots from {empty_tier.name} to {full_tier.name}")
                
                # Update current size
                current_size += transfer_amount
        
    def _create_checkpoint(self) -> str:
        """Create a checkpoint of the current memory state"""
        if not self.enable_checkpointing:
            return ""
            
        checkpoint_id = f"checkpoint_{self.current_position}_{int(time.time())}"
        checkpoint_path = os.path.join(self.offload_path, f"{checkpoint_id}.pt")
        
        try:
            # Create simplified state for checkpoint (metadata only)
            checkpoint_data = {
                "current_position": self.current_position,
                "token_metadata": self.token_metadata,
                "sequence_boundaries": self.sequence_boundaries,
                "timestamp": time.time(),
                "stats": self.stats
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Created memory checkpoint at position {self.current_position}")
            
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""
            
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore memory state from a checkpoint"""
        checkpoint_path = os.path.join(self.offload_path, f"{checkpoint_id}.pt")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
            
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path)
            
            # Restore metadata
            self.current_position = checkpoint_data["current_position"]
            self.token_metadata = checkpoint_data["token_metadata"]
            self.sequence_boundaries = checkpoint_data["sequence_boundaries"]
            
            if "stats" in checkpoint_data:
                self.stats = checkpoint_data["stats"]
                
            # tokens will be loaded on demand from offload
            
            logger.info(f"Restored memory checkpoint: {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
            
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        # Count tokens in each tier
        tier_counts = {tier.name: len(tokens) for tier, tokens in self.memory_tiers.items()}
        
        # Add offloaded count
        tier_counts["OFFLOADED"] = sum(1 for meta in self.token_metadata.values() if meta.offloaded)
        
        # Overall numbers
        total_tokens = len(self.token_metadata)
        resident_tokens = sum(len(tokens) for tokens in self.memory_tiers.values())
        
        return {
            "total_tokens": total_tokens,
            "resident_tokens": resident_tokens,
            "tier_counts": tier_counts,
            "total_memory_mb": self.total_memory_usage / (1024 * 1024),
            "memory_by_tier_mb": {tier.name: usage / (1024 * 1024) for tier, usage in self.memory_usage_by_tier.items()},
            "current_position": self.current_position,
            "performance_counters": self.stats
        }
        
    def __str__(self) -> str:
        """String representation with memory stats"""
        stats = self.get_memory_usage_stats()
        
        result = "UltraScaleMemoryManager Stats:\n"
        result += f"Total Tokens: {stats['total_tokens']:,}\n"
        result += f"Resident Tokens: {stats['resident_tokens']:,}\n"
        result += f"Current Position: {stats['current_position']:,}\n"
        result += f"Total Memory: {stats['total_memory_mb']:.2f} MB\n"
        result += "Token counts by tier:\n"
        
        for tier, count in stats['tier_counts'].items():
            result += f"  {tier}: {count:,}\n"
            
        return result

##############################################
# Advanced Attention for 100M Token Contexts #
##############################################

class HierarchicalTransformerAttention(nn.Module):
    """
    Revolutionary attention mechanism for 100M+ token context windows using a hierarchical 
    multi-resolution design with extreme sparsity, distributed block processing, and
    specialized attention patterns for different context regions.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        max_seq_len: int = 100 * 1000 * 1000,  # 100M tokens
        causal: bool = True,
        dropout: float = 0.0,
        
        # Hierarchical attention windows
        focus_window: int = 128,           # Ultra-high precision window
        active_window: int = 1024,         # Active high-precision window
        mid_window: int = 8192,            # Mid-precision window
        long_window: int = 65536,          # Low-precision long-range window
        global_tokens: int = 1024,         # Number of global tokens
        
        # Attention mechanisms
        recurrent_memory: bool = True,     # Use recurrent memory mechanisms
        multi_hop: bool = True,            # Use multi-hop attention
        num_hops: int = 3,                 # Number of hops for multi-hop attention
        relative_pos: bool = True,         # Use relative position encoding
        rotary_pos: bool = True,           # Use rotary position encoding
        
        # Efficiency options
        block_size: int = 1024,            # Block size for chunked processing
        sliding_window_stride: int = 256,  # Stride for sliding windows
        memory_efficient: bool = True,     # Use memory-efficient attention
        use_flash_attention: bool = True,  # Use flash attention if available
        precision: str = 'auto',           # 'auto', 'fp32', 'fp16', 'bf16'
        
        # Memory integration
        memory_manager: Optional[UltraScaleMemoryManager] = None,
        use_memory_manager: bool = True,    # Whether to use the memory manager
        
        # Advanced features
        group_size: int = 1,               # Number of heads per group for grouped attention
        use_linear_bias: bool = False,     # Use bias in linear projections
        scale_factor: float = 1.0,         # Scaling factor for attention
        layer_idx: int = 0,                # Layer index for position-dependent settings
        has_kv_cache: bool = True,         # Whether to use KV caching
        use_triton: bool = True,           # Use Triton kernels when available
        use_xpos: bool = True,             # Use x-positional encoding
        dynamic_scaling: bool = True       # Scale attention dynamically based on sequence length
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.layer_idx = layer_idx
        self.causal = causal
        self.max_seq_len = max_seq_len
        
        # Hierarchical window sizes
        self.focus_window = focus_window
        self.active_window = active_window
        self.mid_window = mid_window
        self.long_window = long_window
        self.global_tokens = global_tokens
        
        # Attention options
        self.recurrent_memory = recurrent_memory
        self.multi_hop = multi_hop
        self.num_hops = num_hops
        self.relative_pos = relative_pos
        self.rotary_pos = rotary_pos
        
        # Efficiency options
        self.block_size = block_size
        self.sliding_window_stride = sliding_window_stride
        self.memory_efficient = memory_efficient
        self.use_flash_attention = use_flash_attention
        self.precision = precision
        
        # Memory integration
        self.memory_manager = memory_manager
        self.use_memory_manager = use_memory_manager
        
        # Advanced features
        self.group_size = group_size
        self.use_linear_bias = use_linear_bias
        self.scale_factor = scale_factor
        self.has_kv_cache = has_kv_cache
        self.use_triton = use_triton and HAS_TRITON
        self.use_xpos = use_xpos
        self.dynamic_scaling = dynamic_scaling
        
        # Set dimension per head
        self.dim_head = dim_head or (dim // heads)
        inner_dim = self.dim_head * heads
        
        # Linear projections
        self.to_q = nn.Linear(dim, inner_dim, bias=use_linear_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=use_linear_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=use_linear_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=use_linear_bias)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        if rotary_pos:
            # Enhanced Rotary embeddings for ultra-long contexts
            self.rotary_emb = EnhancedRotaryEmbedding(
                dim=self.dim_head, 
                max_seq_len=max_seq_len,
                base=10000,
                use_xpos=use_xpos
            )
            
        if relative_pos:
            # Relative positional bias
            self.rel_pos_bias = RelativePositionalBias(
                max_distance=long_window,
                n_heads=heads
            )
            
        # For recurrent memory
        if recurrent_memory:
            # Recurrent state for capturing ultra-long dependencies
            self.init_recurrent_state()
            
        # Check for Flash Attention
        self.has_flash_attn = False
        if use_flash_attention:
            try:
                import flash_attn
                self.has_flash_attn = True
            except ImportError:
                logger.warning("flash_attn package not found, falling back to standard attention")
                
        # For hierarchical attention tracking
        self.global_token_indices = None
        self.attention_stats = {
            "focus_tokens_retrieved": 0,
            "active_tokens_retrieved": 0,
            "mid_tokens_retrieved": 0,
            "long_tokens_retrieved": 0,
            "global_tokens_retrieved": 0,
            "total_tokens_retrieved": 0
        }
            
    def init_recurrent_state(self):
        """Initialize recurrent state for long-range memory"""
        # Learned initial state
        self.recurrent_init = nn.Parameter(torch.zeros(1, self.heads, 1, self.dim_head))
        
        # State compression and expansion
        self.compress_state = nn.Sequential(
            nn.Linear(self.dim_head, self.dim_head // 4),
            nn.LayerNorm(self.dim_head // 4),
            nn.GELU()
        )
        
        self.expand_state = nn.Sequential(
            nn.Linear(self.dim_head // 4, self.dim_head),
            nn.LayerNorm(self.dim_head)
        )
        
        # Current recurrent state (will grow during usage)
        self.register_buffer('recurrent_state', None, persistent=False)
        
    def reset_recurrent_state(self, batch_size=1):
        """Reset recurrent state"""
        self.recurrent_state = self.recurrent_init.expand(batch_size, -1, -1, -1).clone()
        
    def update_recurrent_state(self, k: Tensor, v: Tensor, attention_scores: Optional[Tensor] = None):
        """Update recurrent state with new information"""
        if self.recurrent_state is None:
            batch_size = k.size(0)
            self.reset_recurrent_state(batch_size)
            
        # Compress and extract the most salient information
        if attention_scores is not None:
            # Weight by attention scores
            k_weighted = torch.matmul(attention_scores.transpose(-2, -1), k)
            v_weighted = torch.matmul(attention_scores.transpose(-2, -1), v)
            
            # Average weighted values
            k_avg = k_weighted.mean(dim=2, keepdim=True)
            v_avg = v_weighted.mean(dim=2, keepdim=True)
        else:
            # Simple average if no attention scores
            k_avg = k.mean(dim=2, keepdim=True)
            v_avg = v.mean(dim=2, keepdim=True)
            
        # Compress state
        k_compressed = self.compress_state(k_avg)
        v_compressed = self.compress_state(v_avg)
        
        # Compute weighted mean with existing state
        state_compressed = self.compress_state(self.recurrent_state)
        
        # Update compressed state with exponential moving average
        alpha = 0.9  # Weight for previous state
        state_compressed = alpha * state_compressed + (1 - alpha) * (k_compressed + v_compressed) / 2
        
        # Expand back to full dimension
        self.recurrent_state = self.expand_state(state_compressed)
        
    def get_hierarchical_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a hierarchical attention mask with multi-scale attention patterns
        for different regions of the context
        """
        # Initialize full attention mask
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
        
        # Generate focus window mask (strongest attention around immediate context)
        focus_size = min(seq_len, self.focus_window)
        for i in range(seq_len):
            start = max(0, i - focus_size // 2)
            end = min(seq_len, i + focus_size // 2 + 1)
            mask[i, start:end] = True
            
        # Generate active window mask
        active_size = min(seq_len, self.active_window)
        for i in range(seq_len):
            start = max(0, i - active_size // 2)
            end = min(seq_len, i + active_size // 2 + 1)
            mask[i, start:end] = True
            
        # Generate mid-range mask with strided pattern (attend to every n-th token)
        mid_size = min(seq_len, self.mid_window)
        stride = max(1, mid_size // 100)  # Stride for mid window
        for i in range(seq_len):
            start = max(0, i - mid_size)
            end = min(seq_len, i + 1)  # Causal: only look back
            # Strided pattern
            indices = torch.arange(start, end, stride, device=device)
            mask[i, indices] = True
            
        # Add global tokens (attend to first tokens and evenly spaced ones)
        num_global = min(seq_len, self.global_tokens)
        
        # Always attend to first N tokens
        prefix_size = min(32, num_global // 4)
        mask[:, :prefix_size] = True
        
        # Add evenly spaced global tokens throughout the sequence
        if seq_len > prefix_size:
            stride = (seq_len - prefix_size) // (num_global - prefix_size)
            if stride > 0:  # Ensure stride is at least 1
                global_indices = torch.arange(prefix_size, seq_len, stride, device=device)
                global_indices = global_indices[:num_global-prefix_size]
                mask[:, global_indices] = True
                
        # Apply causal constraint if needed
        if self.causal:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = mask & causal_mask
            
        return mask
    
    def _apply_xpos(self, q: Tensor, k: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Apply XPos (extended positional encoding) to queries and keys"""
        if not self.use_xpos or not hasattr(self, 'rotary_emb'):
            return q, k
            
        # Delegate to rotary embedding's xpos implementation
        return self.rotary_emb.apply_xpos(q, k, seq_len)
    
    def _apply_rotary(self, q: Tensor, k: Tensor, positions: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary positional embeddings to queries and keys"""
        if not self.rotary_pos or not hasattr(self, 'rotary_emb'):
            return q, k
            
        # Apply rotary embeddings
        q_rot = self.rotary_emb(q, positions)
        k_rot = self.rotary_emb(k, positions)
        
        return q_rot, k_rot
    
    def _scaled_dot_product(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        mask: Optional[Tensor] = None,
        dropout_p: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute scaled dot-product attention
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, dim_head]
            k: Key tensor of shape [batch, heads, seq_len, dim_head]
            v: Value tensor of shape [batch, heads, seq_len, dim_head]
            mask: Optional attention mask of shape [seq_len, seq_len]
            dropout_p: Dropout probability
            
        Returns:
            Tuple of (output, attention weights)
        """
        # Scale query
        scale_factor = self.scale_factor / math.sqrt(self.dim_head)
        if self.dynamic_scaling:
            # Further reduce scale for very long sequences to avoid numerical instability
            seq_scaling = 1.0 / math.sqrt(1 + math.log(max(1, k.size(2)) / 1024))
            scale_factor *= seq_scaling
            
        q = q * scale_factor
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply relative positional bias if enabled
        if self.relative_pos and hasattr(self, 'rel_pos_bias'):
            rel_pos_bias = self.rel_pos_bias(q.size(2), k.size(2))
            attn_scores = attn_scores + rel_pos_bias
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -torch.finfo(attn_scores.dtype).max)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def _chunk_attention(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None
    ) -> Tensor:
        """
        Process attention in chunks to save memory
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, dim_head]
            k: Key tensor of shape [batch, heads, seq_len, dim_head]
            v: Value tensor of shape [batch, heads, seq_len, dim_head]
            mask: Optional attention mask
            chunk_size: Size of chunks to process
            
        Returns:
            Output tensor of shape [batch, heads, seq_len, dim_head]
        """
        batch_size, heads, seq_len, dim_head = q.shape
        
        # Use provided chunk size or default
        chunk_size = chunk_size or self.block_size
        
        # Adjust chunk size based on sequence length
        if seq_len > 131072:  # 128K tokens
            chunk_size = min(chunk_size, 1024)  # Smaller chunks for very long sequences
        elif seq_len > 32768:  # 32K tokens
            chunk_size = min(chunk_size, 2048)  # Medium chunks for long sequences
        else:
            chunk_size = min(chunk_size, 4096)  # Larger chunks for shorter sequences
            
        # Ensure chunk size is reasonable
        chunk_size = min(chunk_size, seq_len)
        
        # Initialize output
        outputs = []
        
        # Process queries in chunks
        for chunk_idx in range(0, seq_len, chunk_size):
            # Define chunk boundaries
            chunk_end = min(chunk_idx + chunk_size, seq_len)
            chunk_len = chunk_end - chunk_idx
            
            # Extract chunk of queries
            q_chunk = q[:, :, chunk_idx:chunk_end]
            
            # Compute causal attention mask for this chunk if needed
            if self.causal:
                # Create chunk-specific causal mask
                if mask is not None:
                    # Use provided mask and apply causality
                    chunk_mask = mask[chunk_idx:chunk_end, :seq_len]
                    if chunk_mask.dim() == 2:
                        chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Create causal mask from scratch
                    causal_mask = torch.ones(chunk_len, seq_len, device=q.device, dtype=torch.bool)
                    
                    # Apply causality - each position in chunk can only attend to previous positions
                    for i in range(chunk_len):
                        pos = chunk_idx + i
                        causal_mask[i, pos+1:] = False
                        
                    chunk_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Just use provided mask for this chunk
                if mask is not None:
                    chunk_mask = mask[chunk_idx:chunk_end, :seq_len].unsqueeze(0).unsqueeze(0)
                else:
                    chunk_mask = None
            
            # Compute attention
            chunk_output, _ = self._scaled_dot_product(
                q_chunk, k, v, 
                mask=chunk_mask.squeeze(0).squeeze(0) if chunk_mask is not None else None,
                dropout_p=self.attn_dropout.p
            )
            outputs.append(chunk_output)
            
        # Concatenate all chunks
        return torch.cat(outputs, dim=2)
    
    def _multi_hop_attention(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply multi-hop attention for better long-range dependency modeling
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, dim_head]
            k: Key tensor of shape [batch, heads, seq_len, dim_head]
            v: Value tensor of shape [batch, heads, seq_len, dim_head]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention weights)
        """
        # Initial attention
        x, attn_weights = self._scaled_dot_product(q, k, v, mask, self.attn_dropout.p)
        
        # For additional hops
        for _ in range(self.num_hops - 1):
            # Use output of previous hop as query
            q = x
            
            # Apply another hop of attention
            x, hop_attn = self._scaled_dot_product(q, k, v, mask, self.attn_dropout.p)
            
            # Update attention weights (additive accumulation)
            attn_weights = attn_weights + hop_attn
            
        # Normalize accumulated attention weights
        attn_weights = attn_weights / self.num_hops
        
        return x, attn_weights
    
    def _hierarchical_attention(
        self, 
        q: Tensor, 
        k: Tensor, 
        v: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply hierarchical attention with multi-scale processing
        
        Args:
            q: Query tensor of shape [batch, heads, seq_len, dim_head]
            k: Key tensor of shape [batch, heads, seq_len, dim_head]
            v: Value tensor of shape [batch, heads, seq_len, dim_head]
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch, heads, seq_len, dim_head]
        """
        batch_size, heads, seq_len, dim_head = q.shape
        
        # Reset stats
        self.attention_stats = {
            "focus_tokens_retrieved": 0,
            "active_tokens_retrieved": 0,
            "mid_tokens_retrieved": 0,
            "long_tokens_retrieved": 0,
            "global_tokens_retrieved": 0,
            "total_tokens_retrieved": 0
        }
        
        # For very short sequences, just use standard attention
        if seq_len <= self.active_window:
            output, _ = self._scaled_dot_product(q, k, v, mask, self.attn_dropout.p)
            self.attention_stats["focus_tokens_retrieved"] = seq_len
            self.attention_stats["total_tokens_retrieved"] = seq_len
            return output
        
        # Generate hierarchical attention if no mask provided
        if mask is None:
            mask = self.get_hierarchical_mask(seq_len, q.device)
        
        # Process in chunks based on hierarchical mask
        output = torch.zeros_like(q)
        
        # 1. Process focus window for each position (highest precision)
        focus_size = min(seq_len, self.focus_window)
        
        for i in range(0, seq_len, self.block_size):
            # Process in blocks for efficiency
            end_idx = min(i + self.block_size, seq_len)
            block_len = end_idx - i
            
            for j in range(block_len):
                pos = i + j
                
                # Define focus window around current position
                start = max(0, pos - focus_size // 2)
                end = min(seq_len, pos + focus_size // 2 + 1)
                
                # Apply causal constraint
                if self.causal:
                    end = min(end, pos + 1)
                
                # Get tokens in focus window
                k_focus = k[:, :, start:end]
                v_focus = v[:, :, start:end]
                
                # Create position-specific mask
                pos_mask = torch.zeros(1, end-start, device=q.device, dtype=torch.bool)
                pos_mask[0, :] = True
                
                # Apply attention for this position
                q_pos = q[:, :, pos:pos+1]  # Shape: [batch, heads, 1, dim_head]
                
                # Note: for single query position, scaled dot product simplifies
                attn_scores = torch.matmul(q_pos, k_focus.transpose(-1, -2))
                attn_scores = attn_scores * (dim_head ** -0.5 * self.scale_factor)
                
                # Apply mask
                attn_scores = attn_scores.masked_fill(~pos_mask.unsqueeze(0).unsqueeze(0), 
                                                      -torch.finfo(attn_scores.dtype).max)
                
                # Apply softmax
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.attn_dropout(attn_weights)
                
                # Apply attention
                output[:, :, pos:pos+1] = torch.matmul(attn_weights, v_focus)
                
                # Update stats
                self.attention_stats["focus_tokens_retrieved"] += end - start
                
        # 2. Apply active window attention (already covered by focus window)
        # 3. Apply mid-range attention (strided)
        if seq_len > self.active_window:
            mid_size = min(seq_len, self.mid_window)
            stride = max(1, mid_size // 100)  # Stride for mid window
            
            # Identify mid-range tokens
            mid_range_indices = []
            for i in range(0, seq_len, stride):
                if i >= self.active_window:  # Only tokens beyond active window
                    mid_range_indices.append(i)
            
            # Only proceed if there are mid-range tokens
            if mid_range_indices:
                # Get mid-range keys and values
                k_mid = k[:, :, mid_range_indices]
                v_mid = v[:, :, mid_range_indices]
                
                # Apply attention for all queries
                attn_scores = torch.matmul(q, k_mid.transpose(-1, -2)) 
                attn_scores = attn_scores * (dim_head ** -0.5 * self.scale_factor)
                
                # Apply softmax
                attn_weights = F.softmax(attn_scores, dim=-1) 
                attn_weights = self.attn_dropout(attn_weights)
                
                # Apply attention with a weighting factor (less weight than focus)
                mid_output = torch.matmul(attn_weights, v_mid) * 0.5
                
                # Add to main output
                output = output + mid_output
                
                # Update stats
                self.attention_stats["mid_tokens_retrieved"] += len(mid_range_indices)
                
        # 4. Apply global tokens attention
        if self.global_tokens > 0:
            num_global = min(seq_len, self.global_tokens)
            
            # Identify global tokens: first tokens and uniformly spaced ones
            prefix_size = min(32, num_global // 4)
            prefix_indices = list(range(prefix_size))
            
            # Add evenly spaced tokens
            if seq_len > prefix_size:
                stride = (seq_len - prefix_size) // (num_global - prefix_size)
                if stride > 0:
                    strided_indices = list(range(prefix_size, seq_len, stride))[:num_global-prefix_size]
                    global_indices = prefix_indices + strided_indices
                else:
                    global_indices = prefix_indices
            else:
                global_indices = prefix_indices
                
            # Store for visualization/debugging
            self.global_token_indices = global_indices
            
            # Get global keys and values
            k_global = k[:, :, global_indices]
            v_global = v[:, :, global_indices]
            
            # Apply attention for all queries
            attn_scores = torch.matmul(q, k_global.transpose(-1, -2))
            attn_scores = attn_scores * (dim_head ** -0.5 * self.scale_factor)
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention with a weighting factor (global context influence)
            global_output = torch.matmul(attn_weights, v_global) * 0.2
            
            # Add to main output
            output = output + global_output
            
            # Update stats
            self.attention_stats["global_tokens_retrieved"] += len(global_indices)
            
        # 5. Add recurrent state if enabled
        if self.recurrent_memory and self.recurrent_state is not None:
            # Apply attention between queries and recurrent state
            attn_scores = torch.matmul(q, self.recurrent_state.transpose(-1, -2))
            attn_scores = attn_scores * (dim_head ** -0.5 * self.scale_factor)
            
            # Apply softmax
            attn_weights = F.softmax(attn_scores, dim=-1) 
            attn_weights = self.attn_dropout(attn_weights)
            
            # Apply attention with a weighting factor (recurrent memory influence)
            recurrent_output = torch.matmul(attn_weights, self.recurrent_state) * 0.1
            
            # Add to main output
            output = output + recurrent_output
            
        # Update total tokens retrieved
        self.attention_stats["total_tokens_retrieved"] = (
            self.attention_stats["focus_tokens_retrieved"] +
            self.attention_stats["active_tokens_retrieved"] +
            self.attention_stats["mid_tokens_retrieved"] +
            self.attention_stats["long_tokens_retrieved"] +
            self.attention_stats["global_tokens_retrieved"]
        )
        
        return output
    
    def forward(
        self,
        x: Tensor,
        memory: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        positions: Optional[Tensor] = None,
        use_kv_cache: bool = False,
        kv_cache: Optional[Dict[str, Tensor]] = None,
        attn_mode: str = 'hierarchical',  # 'standard', 'chunk', 'multi_hop', 'hierarchical'
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for hierarchical transformer attention
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            memory: Optional memory tensor to attend to
            mask: Optional attention mask
            positions: Optional position indices
            use_kv_cache: Whether to use KV cache
            kv_cache: Optional KV cache dict
            attn_mode: Attention mode to use
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor or tuple of (output, attention)
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate position indices if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
            
        # Initialize or get KV cache
        k_cache, v_cache = None, None
        if use_kv_cache and kv_cache is not None:
            k_cache = kv_cache.get('k', None)
            v_cache = kv_cache.get('v', None)
            
        # Project inputs to queries, keys, values
        q = self.to_q(x)
        
        # Handle memory for cross-attention
        if memory is not None:
            # Cross-attention case
            k = self.to_k(memory)
            v = self.to_v(memory)
        else:
            # Self-attention case
            k = self.to_k(x)
            v = self.to_v(x)
            
        # Update KV cache if using
        if use_kv_cache:
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
            if v_cache is not None:
                v = torch.cat([v_cache, v], dim=1)
                
            # Store updated KV cache
            if kv_cache is not None:
                kv_cache['k'] = k
                kv_cache['v'] = v
            
        # Reshape to [batch, heads, seq_len, dim_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Apply positional encodings
        if self.rotary_pos:
            q, k = self._apply_rotary(q, k, positions)
            
        if self.use_xpos:
            q, k = self._apply_xpos(q, k, seq_len)
            
        # Choose attention implementation based on mode and sequence length
        attention = None
        
        if self.has_flash_attn and x.is_cuda and seq_len <= 4096:
            # Use Flash Attention if available and sequence is not too long
            try:
                import flash_attn
                from flash_attn.flash_attention import FlashAttention
                
                flash_attn_func = FlashAttention(softmax_scale=1.0/math.sqrt(self.dim_head))
                q_4d = q.transpose(1, 2)  # [b, n, h, d]
                k_4d = k.transpose(1, 2)  # [b, n, h, d]
                v_4d = v.transpose(1, 2)  # [b, n, h, d]
                
                output = flash_attn_func(q_4d, k_4d, v_4d, causal=self.causal)[0]
                output = output.transpose(1, 2)  # back to [b, h, n, d]
            except Exception as e:
                logger.warning(f"Flash attention failed: {e}, falling back to standard attention")
                output, attention = self._scaled_dot_product(q, k, v, mask, self.attn_dropout.p)
        elif attn_mode == 'standard' or seq_len <= 1024:
            # Standard attention for short sequences
            output, attention = self._scaled_dot_product(q, k, v, mask, self.attn_dropout.p)
        elif attn_mode == 'chunk' or seq_len <= 16384:
            # Chunked attention for medium sequences
            output = self._chunk_attention(q, k, v, mask)
        elif attn_mode == 'multi_hop' and self.multi_hop:
            # Multi-hop attention
            output, attention = self._multi_hop_attention(q, k, v, mask)
        else:
            # Hierarchical attention for very long sequences
            output = self._hierarchical_attention(q, k, v, mask)
            
        # Update recurrent state if enabled
        if self.recurrent_memory and memory is None:  # Only for self-attention
            self.update_recurrent_state(k, v, attention if attention is not None else None)
            
        # Reshape output
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        # Final projection and dropout
        output = self.to_out(output)
        output = self.resid_dropout(output)
        
        if return_attention and attention is not None:
            return output, attention
        else:
            return output

class EnhancedRotaryEmbedding(nn.Module):
    """
    Enhanced rotary positional embeddings optimized for 100M+ token contexts,
    with distance-aware extended precision and hybrid log-linear scaling
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int = 100 * 1000 * 1000,
        base: int = 10000,
        use_xpos: bool = True,
        scale_factor: float = 1.0,
        use_hybrid_scaling: bool = True,
        precision_scaling: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.use_xpos = use_xpos
        self.scale_factor = scale_factor
        self.use_hybrid_scaling = use_hybrid_scaling
        self.precision_scaling = precision_scaling
        
        # Adjust base for ultra-long contexts to avoid numerical issues
        if max_seq_len > 65536:
            # Hybrid log-linear scaling for ultra-long contexts
            if use_hybrid_scaling:
                scale_base = math.log(max_seq_len / 65536) / math.log(2)
                self.hybrid_base = base * (1.0 + scale_base * 0.1)
                self.hybrid_factor = 1.0 / (1.0 + scale_base * 0.1)
            else:
                # Traditional RoPE scaling
                self.base = base * (max_seq_len / 65536) ** (self.dim / (self.dim - 2))
                
        # Create and cache the frequency tensors for efficiency
        self.precompute_freqs()
        
        # For xpos scaling
        if use_xpos:
            # XPos scale and shift parameters
            self.xpos_scale = nn.Parameter(torch.ones(1, 1, 1, dim//2))
            self.xpos_shift = nn.Parameter(torch.zeros(1, 1, 1, dim//2))
            
    def precompute_freqs(self):
        """Precompute the frequency tensors for efficiency using extended precision"""
        # Create and register theta
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Scale theta for very long contexts if using hybrid scaling
        if self.use_hybrid_scaling and hasattr(self, 'hybrid_factor'):
            theta = theta * self.hybrid_factor
            
        # Create position sequence with extended precision
        if self.max_seq_len > 1000000:
            # For ultra-long sequences, use higher precision and segment-wise calculation
            segment_size = 1000000
            num_segments = (self.max_seq_len + segment_size - 1) // segment_size
            
            all_freqs = []
            for i in range(num_segments):
                start = i * segment_size
                end = min((i + 1) * segment_size, self.max_seq_len)
                seqs = torch.arange(start, end, dtype=torch.float64)  # Higher precision
                
                if self.use_hybrid_scaling and hasattr(self, 'hybrid_base'):
                    # Apply log scaling for positions beyond certain threshold
                    hybrid_base = self.hybrid_base
                    # Linear scaling for initial positions
                    mask_linear = (seqs < 65536).to(torch.float64)
                    # Log scaling for later positions
                    mask_log = 1 - mask_linear
                    
                    # Hybrid calculation
                    scaled_seqs = seqs * mask_linear + (65536 + torch.log(seqs/65536 + 1) * 65536) * mask_log
                    freqs = torch.outer(scaled_seqs, theta.to(torch.float64))
                else:
                    # Standard scaling
                    freqs = torch.outer(seqs, theta.to(torch.float64))
                
                all_freqs.append(freqs)
                
            # Concatenate all segments
            freqs = torch.cat(all_freqs, dim=0).float()
        else:
            # For shorter sequences, use standard precision
            seqs = torch.arange(self.max_seq_len, dtype=torch.float32)
            freqs = torch.outer(seqs, theta)
        
        # Register cos and sin values as buffers
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
        
    def apply_xpos(self, q: Tensor, k: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Apply XPos (extended positional encoding) scaling"""
        if not self.use_xpos:
            return q, k
            
        # Create position indices
        device = q.device
        i = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Log position indices for better scaling with extremely long sequences
        pos_log = torch.log1p(i) / math.log(self.max_seq_len)  # [1, seq_len]
        pos_log = pos_log.unsqueeze(-1).unsqueeze(1)  # [1, 1, seq_len, 1]
        
        # Apply scaling factor to QK based on log position
        scale = (self.xpos_scale * pos_log + self.xpos_shift).exp()
        
        # Reshape scale to match q and k shapes properly
        # q: [batch, heads, seq_len, dim_head]
        # Extract and scale real and imaginary parts separately
        dim_half = q.shape[-1] // 2
        
        # Split into real and imaginary parts
        q_real, q_imag = q[..., :dim_half], q[..., dim_half:]
        k_real, k_imag = k[..., :dim_half], k[..., dim_half:]
        
        # Apply scaling
        q_real = q_real * scale
        q_imag = q_imag * scale
        k_real = k_real * scale
        k_imag = k_imag * scale
        
        # Recombine
        q_out = torch.cat([q_real, q_imag], dim=-1)
        k_out = torch.cat([k_real, k_imag], dim=-1)
        
        return q_out, k_out
        
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Apply rotary embeddings to input tensor
        
        Args:
            x: Input tensor of shape [batch, ..., seq_len, dim]
            positions: Position indices of shape [seq_len]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Extract shapes
        seq_len = positions.shape[0]
        
        # Only use positions within precomputed range
        positions = positions % self.max_seq_len
        
        # Get the cos and sin values for these positions
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        
        # Match dimensions
        while cos.dim() < x.dim() - 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
        # Add dummy dimension for dim
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)
        
        # Expand to match x's dimensions exactly
        for i in range(x.dim() - cos.dim()):
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            
        # Ensure the target shape matches exactly
        target_shape = x.shape[:-1] + (cos.shape[-1],)
        cos = cos.expand(target_shape)
        sin = sin.expand(target_shape)
        
        # Adjust for precision issues in ultra-long contexts (>10M)
        if self.precision_scaling and max(positions) > 10000000:
            # Apply precision stabilization factors
            precision_factor = 1.0 - torch.log1p(positions.float() / 10000000) * 0.1
            precision_factor = precision_factor.to(x.device).unsqueeze(-1)
            
            # Expand to match shape
            for _ in range(x.dim() - 2):
                precision_factor = precision_factor.unsqueeze(0)
                
            # Apply to cos and sin (promotes numerical stability)
            cos = cos * precision_factor
            sin = sin * precision_factor
        
        # Half of the dimensions get rotated
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply complex multiplication
        # (a+bi)(cos+sin*i) = (a*cos-b*sin) + (a*sin+b*cos)i
        result = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return result

class RelativePositionalBias(nn.Module):
    """
    Enhanced relative positional bias for ultra-long contexts,
    with hierarchical multi-scale representation
    """
    
    def __init__(
        self,
        max_distance: int = 128,
        n_heads: int = 8,
        num_buckets: int = 32,
        max_log_distance: int = 100 * 1000 * 1000,
        use_multi_scale: bool = True,
        num_scales: int = 3,
        scale_base: float = 10.0
    ):
        super().__init__()
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_log_distance = max_log_distance
        self.use_multi_scale = use_multi_scale
        self.num_scales = num_scales
        self.scale_base = scale_base
        
        # Create relative position embedding tables
        if use_multi_scale:
            # Use multiple embedding tables for different distance scales
            self.rel_pos_bias = nn.ModuleList([
                nn.Embedding(num_buckets, n_heads) for _ in range(num_scales)
            ])
            
            # Scale mixing weights
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:
            # Traditional single-scale embedding
            self.rel_pos_bias = nn.Embedding(num_buckets, n_heads)
            
    def _relative_position_bucket(
        self, 
        relative_position: Tensor, 
        scale_idx: int = 0
    ) -> Tensor:
        """Convert relative positions to bucket indices with multi-scale support"""
        # For multi-scale, adjust distance based on scale
        if self.use_multi_scale and scale_idx > 0:
            # Apply exponential scaling for higher scales
            scale_factor = self.scale_base ** scale_idx
            relative_position = torch.sign(relative_position) * (
                torch.log(torch.abs(relative_position).float() + 1) / 
                math.log(scale_factor)
            ).to(relative_position.dtype)
        
        # Handle both bidirectional ([-max_distance, max_distance]) and unidirectional ([0, max_distance])
        num_buckets = self.num_buckets
        max_exact = self.max_distance // 2
        is_small = torch.abs(relative_position) < max_exact
        
        # Use log-space buckets for longer distances
        relative_position_if_large = max_exact + (
            torch.log(torch.abs(relative_position).float() / max_exact) / 
            math.log(self.max_log_distance / max_exact) * (num_buckets // 2)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        # Select bucket index based on distance
        relative_buckets = torch.where(
            is_small,
            torch.abs(relative_position),
            relative_position_if_large
        )
        
        # For bidirectional attention, split buckets for positive and negative distances
        relative_buckets = torch.where(
            relative_position > 0,
            relative_buckets + num_buckets // 2,
            relative_buckets
        )
        
        return relative_buckets
        
    def forward(self, query_len: int, key_len: int) -> Tensor:
        """
        Compute relative positional bias
        
        Args:
            query_len: Length of query
            key_len: Length of key
            
        Returns:
            Relative positional bias of shape [1, heads, query_len, key_len]
        """
        device = self.rel_pos_bias[0].weight.device if self.use_multi_scale else self.rel_pos_bias.weight.device
        
        # Create position indices
        q_pos = torch.arange(query_len, dtype=torch.long, device=device)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device)
        
        # Compute relative positions [query_len, key_len]
        relative_position = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
        
        if self.use_multi_scale:
            # Compute biases at multiple scales
            all_biases = []
            
            for scale_idx in range(self.num_scales):
                # Get buckets for this scale
                rel_buckets = self._relative_position_bucket(relative_position, scale_idx)
                # Get bias values
                bias = self.rel_pos_bias[scale_idx](rel_buckets)  # [query_len, key_len, heads]
                # Transpose to [heads, query_len, key_len]
                bias = bias.permute(2, 0, 1)
                all_biases.append(bias)
                
            # Mix scales with learned weights
            scale_weights = F.softmax(self.scale_weights, dim=0)
            
            # Calculate weighted sum of biases
            bias = torch.stack(all_biases, dim=0) * scale_weights.view(-1, 1, 1, 1)
            bias = bias.sum(dim=0)
        else:
            # Traditional single-scale bias
            rel_buckets = self._relative_position_bucket(relative_position)
            bias = self.rel_pos_bias(rel_buckets)  # [query_len, key_len, heads]
            bias = bias.permute(2, 0, 1)  # [heads, query_len, key_len]
            
        # Add batch dimension: [1, heads, query_len, key_len]
        return bias.unsqueeze(0)

##############################################
# UltraScaleTransformer for 100M Contexts   #
##############################################

class UltraScaleTransformerBlock(nn.Module):
    """
    Ultra-optimized transformer block for 100M+ token contexts
    with specialized attention, efficient computation, and memory integration
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        norm_position: str = 'pre',  # 'pre' or 'post'
        activation: nn.Module = nn.GELU(),
        use_memory_manager: bool = True,
        memory_manager: Optional[UltraScaleMemoryManager] = None,
        max_seq_len: int = 100 * 1000 * 1000,  # 100M tokens
        use_mega_attention: bool = True,
        relative_pos: bool = True,
        rotary_pos: bool = True,
        use_xpos: bool = True,
        layer_idx: int = 0,
        block_sparse: bool = True,
        enable_quantization: bool = True,
        memory_efficient: bool = True,
        use_flash_attention: bool = True,
        use_checkpointing: bool = True,
        gate_residual: bool = True,
        use_parallel_attention: bool = True,
        use_bias: bool = False,
        sandwich_norm: bool = True,  # Extra normalization between attention and MLP
        ffn_expansion_factor: float = 2.0,  # Additional expansion for certain FFN activations
        window_size: int = 1024,
        global_tokens: int = 1024,
        adaptive_computation: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.norm_position = norm_position
        self.use_memory_manager = use_memory_manager
        self.memory_manager = memory_manager
        self.use_checkpointing = use_checkpointing and HAS_CHECKPOINTING
        self.gate_residual = gate_residual
        self.use_parallel_attention = use_parallel_attention
        self.sandwich_norm = sandwich_norm
        self.adaptive_computation = adaptive_computation
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        
        if sandwich_norm:
            self.norm_sandwich = nn.LayerNorm(dim, eps=layer_norm_eps)
            
        # Residual gating for enhanced gradient flow
        if gate_residual:
            self.gate1 = nn.Parameter(torch.ones(1))
            self.gate2 = nn.Parameter(torch.ones(1))
        
        # Attention mechanism
        if use_mega_attention:
            self.attention = HierarchicalTransformerAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                max_seq_len=max_seq_len,
                causal=True,
                dropout=dropout,
                focus_window=128,
                active_window=window_size,
                mid_window=window_size * 8,
                long_window=window_size * 64,
                global_tokens=global_tokens,
                recurrent_memory=True,
                multi_hop=True,
                relative_pos=relative_pos,
                rotary_pos=rotary_pos,
                memory_efficient=memory_efficient,
                use_flash_attention=use_flash_attention,
                memory_manager=memory_manager,
                use_memory_manager=use_memory_manager,
                use_xpos=use_xpos,
                layer_idx=layer_idx,
                use_linear_bias=use_bias
            )
        else:
            # Fallback to standard attention (not recommended for 100M+ tokens)
            self.attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
                bias=use_bias
            )
            
        # Determine FFN dimensions
        mlp_dim = int(dim * mlp_ratio)
        if activation.__class__.__name__ in ['SwiGLU', 'GeGLU', 'GEGLU', 'ReGLU']:
            # These activations need more capacity for optimal performance
            mlp_dim = int(mlp_dim * ffn_expansion_factor)
            
        # MLP block options
        if block_sparse:
            # Block-sparse MLP for high efficiency
            self.mlp = BlockSparseFFN(
                dim=dim,
                hidden_dim=mlp_dim,
                activation=activation,
                dropout=dropout,
                block_size=64,
                sparsity=0.8,
                quantize=enable_quantization,
                use_bias=use_bias
            )
        else:
            # Standard MLP
            self.mlp = MLP(
                dim=dim,
                hidden_dim=mlp_dim,
                activation=activation,
                dropout=dropout,
                use_bias=use_bias
            )
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive computation components
        if adaptive_computation:
            self.compute_controller = nn.Sequential(
                nn.Linear(dim, dim // 16),
                nn.GELU(),
                nn.Linear(dim // 16, 1),
                nn.Sigmoid()
            )
            
    def _checkpoint_forward(self, module, *inputs):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpointing and HAS_CHECKPOINTING and any(p.requires_grad for p in module.parameters()):
            return checkpoint(module, *inputs)
        else:
            return module(*inputs)
    
    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass for the transformer block
        
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            mask: Optional attention mask
            memory: Optional memory tensor for cross-attention
            
        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        # Adaptive computation - skip layer if computation controller determines it's not needed
        if self.adaptive_computation:
            # Compute a representative embedding of the sequence
            seq_embedding = x.mean(dim=1)  # [batch, dim]
            compute_score = self.compute_controller(seq_embedding)
            
            # Skip computation if score is too low (threshold depends on layer depth)
            skip_threshold = 0.1 + (0.4 * self.layer_idx / 10)  # Higher layers are more likely to be computed
            if compute_score.mean() < skip_threshold and not self.training:
                return x  # Skip this layer's computation
                
        # Choose parallel or sequential processing
        if self.use_parallel_attention:
            # Parallel attention + FFN (better throughput)
            return self._forward_parallel(x, mask, memory)
        else:
            # Sequential attention -> FFN (better for very deep models)
            return self._forward_sequential(x, mask, memory)
    
    def _forward_sequential(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None
    ) -> Tensor:
        """Sequential processing (attention followed by FFN)"""
        # Apply layer normalization based on position
        if self.norm_position == 'pre':
            # Pre-norm (better for training stability with deep models)
            attn_input = self.norm1(x)
            
            # Apply attention
            if isinstance(self.attention, HierarchicalTransformerAttention):
                if self.use_checkpointing:
                    attn_output = self._checkpoint_forward(self.attention, attn_input, memory, mask)
                else:
                    attn_output = self.attention(attn_input, memory=memory, mask=mask)
            else:
                # Standard PyTorch attention
                attn_output, _ = self.attention(
                    attn_input, attn_input, attn_input,
                    key_padding_mask=mask.logical_not() if mask is not None else None,
                    need_weights=False
                )
                
            # Residual connection with optional gating
            if self.gate_residual:
                x = x + self.dropout(attn_output) * self.gate1
            else:
                x = x + self.dropout(attn_output)
            
            # Optional sandwich norm between attention and FFN
            if self.sandwich_norm:
                mlp_input = self.norm_sandwich(x)
            else:
                # Standard pre-norm for MLP
                mlp_input = self.norm2(x)
                
            # Apply MLP with checkpointing if enabled
            if self.use_checkpointing:
                mlp_output = self._checkpoint_forward(self.mlp, mlp_input)
            else:
                mlp_output = self.mlp(mlp_input)
            
            # Residual connection with optional gating
            if self.gate_residual:
                output = x + self.dropout(mlp_output) * self.gate2
            else:
                output = x + self.dropout(mlp_output)
        else:
            # Post-norm (traditional transformer)
            # Apply attention
            if isinstance(self.attention, HierarchicalTransformerAttention):
                if self.use_checkpointing:
                    attn_output = self._checkpoint_forward(self.attention, x, memory, mask)
                else:
                    attn_output = self.attention(x, memory=memory, mask=mask)
            else:
                # Standard PyTorch attention
                attn_output, _ = self.attention(
                    x, x, x,
                    key_padding_mask=mask.logical_not() if mask is not None else None,
                    need_weights=False
                )
                
            # Residual connection and norm
            if self.gate_residual:
                x = self.norm1(x + self.dropout(attn_output) * self.gate1)
            else:
                x = self.norm1(x + self.dropout(attn_output))
            
            # Apply MLP with checkpointing if enabled
            if self.use_checkpointing:
                mlp_output = self._checkpoint_forward(self.mlp, x)
            else:
                mlp_output = self.mlp(x)
            
            # Residual connection and norm
            if self.gate_residual:
                output = self.norm2(x + self.dropout(mlp_output) * self.gate2)
            else:
                output = self.norm2(x + self.dropout(mlp_output))
            
        return output
    
    def _forward_parallel(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None
    ) -> Tensor:
        """Parallel processing (attention and FFN in parallel)"""
        if self.norm_position == 'pre':
            # Pre-norm (better for training stability with deep models)
            normed_x = self.norm1(x)
            
            # Apply attention
            if isinstance(self.attention, HierarchicalTransformerAttention):
                if self.use_checkpointing:
                    attn_output = self._checkpoint_forward(self.attention, normed_x, memory, mask)
                else:
                    attn_output = self.attention(normed_x, memory=memory, mask=mask)
            else:
                # Standard PyTorch attention
                attn_output, _ = self.attention(
                    normed_x, normed_x, normed_x,
                    key_padding_mask=mask.logical_not() if mask is not None else None,
                    need_weights=False
                )
            
            # Apply MLP in parallel with checkpointing if enabled
            if self.sandwich_norm:
                normed_x2 = self.norm_sandwich(normed_x)
            else:
                normed_x2 = self.norm2(normed_x)
                
            if self.use_checkpointing:
                mlp_output = self._checkpoint_forward(self.mlp, normed_x2)
            else:
                mlp_output = self.mlp(normed_x2)
            
            # Combine outputs with optional gating
            if self.gate_residual:
                output = x + self.dropout(attn_output) * self.gate1 + self.dropout(mlp_output) * self.gate2
            else:
                output = x + self.dropout(attn_output) + self.dropout(mlp_output)
        else:
            # Post-norm (traditional transformer)
            # Apply attention
            if isinstance(self.attention, HierarchicalTransformerAttention):
                if self.use_checkpointing:
                    attn_output = self._checkpoint_forward(self.attention, x, memory, mask)
                else:
                    attn_output = self.attention(x, memory=memory, mask=mask)
            else:
                # Standard PyTorch attention
                attn_output, _ = self.attention(
                    x, x, x,
                    key_padding_mask=mask.logical_not() if mask is not None else None,
                    need_weights=False
                )
                
            # Apply MLP in parallel
            if self.use_checkpointing:
                mlp_output = self._checkpoint_forward(self.mlp, x)
            else:
                mlp_output = self.mlp(x)
            
            # Combine outputs with optional gating
            if self.gate_residual:
                combined = x + self.dropout(attn_output) * self.gate1 + self.dropout(mlp_output) * self.gate2
            else:
                combined = x + self.dropout(attn_output) + self.dropout(mlp_output)
                
            # Apply final norm
            output = self.norm1(combined)
            
        return output

class MLP(nn.Module):
    """Standard MLP with optional SwiGLU/GeGLU activation"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
        use_bias: bool = True
    ):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 4)
        out_dim = out_dim or dim
        
        # Determine if using a gated activation like SwiGLU/GeGLU
        self.is_gated = activation.__class__.__name__ in ['SwiGLU', 'GeGLU', 'GEGLU', 'ReGLU']
        
        if self.is_gated:
            # For gated activations, we need two projections
            self.fc1_gate = nn.Linear(dim, hidden_dim, bias=use_bias)
            self.fc1_value = nn.Linear(dim, hidden_dim, bias=use_bias)
            self.activation = activation
            self.fc2 = nn.Linear(hidden_dim, out_dim, bias=use_bias)
        else:
            # Standard MLP
            self.fc1 = nn.Linear(dim, hidden_dim, bias=use_bias)
            self.activation = activation
            self.fc2 = nn.Linear(hidden_dim, out_dim, bias=use_bias)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        if self.is_gated:
            # For gated activations
            gate = self.fc1_gate(x)
            value = self.fc1_value(x)
            
            # Apply activation with gating
            if hasattr(self.activation, 'forward_gated'):
                # For activations with dedicated gated implementation
                x = self.activation.forward_gated(gate, value)
            else:
                # Manual gating
                x = self.activation(gate) * value
        else:
            # Standard MLP
            x = self.fc1(x)
            x = self.activation(x)
            
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x

class BlockSparseFFN(nn.Module):
    """
    Block-sparse feed-forward network with quantization for 
    memory-efficient processing of mega-scale contexts
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: nn.Module = nn.GELU(),
        dropout: float = 0.0,
        block_size: int = 64,
        sparsity: float = 0.8,
        quantize: bool = True,
        quantize_bits: int = 8,
        use_bias: bool = False,
        structured_sparsity: bool = True,
        residual_in_fp32: bool = True,
        two_stage: bool = True  # Use two-stage FFN for better accuracy
    ):
        super().__init__()
        hidden_dim = hidden_dim or (dim * 4)
        out_dim = out_dim or dim
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.quantize = quantize
        self.block_size = block_size
        self.residual_in_fp32 = residual_in_fp32
        self.two_stage = two_stage
        
        # Adjust hidden dimensions to be divisible by block size
        self.hidden_dim = ((hidden_dim + block_size - 1) // block_size) * block_size
        
        # First stage - dense expansion with block sparsity
        self.fc1 = BlockSparseLinear(
            dim, self.hidden_dim,
            bias=use_bias,
            block_size=block_size,
            sparsity=sparsity,
            structured_sparsity=structured_sparsity,
            enable_quantization=quantize,
            quantization_bits=quantize_bits
        )
        
        # Determine if using a gated activation
        self.is_gated = activation.__class__.__name__ in ['SwiGLU', 'GeGLU', 'GEGLU', 'ReGLU']
        
        if self.is_gated:
            # For gated activations, we need a second expansion
            self.fc1_gate = BlockSparseLinear(
                dim, self.hidden_dim,
                bias=use_bias,
                block_size=block_size,
                sparsity=sparsity,
                structured_sparsity=structured_sparsity,
                enable_quantization=quantize,
                quantization_bits=quantize_bits
            )
            
        # Activation
        self.activation = activation
        
        # Optional second stage for two-stage FFN
        if self.two_stage:
            mid_dim = self.hidden_dim // 2
            self.fc_mid = BlockSparseLinear(
                self.hidden_dim, mid_dim,
                bias=use_bias,
                block_size=block_size,
                sparsity=sparsity / 2,  # Lower sparsity for middle layer
                structured_sparsity=structured_sparsity,
                enable_quantization=quantize,
                quantization_bits=quantize_bits
            )
            self.mid_activation = activation
            self.fc2 = BlockSparseLinear(
                mid_dim, out_dim,
                bias=use_bias,
                block_size=block_size,
                sparsity=sparsity,
                structured_sparsity=structured_sparsity,
                enable_quantization=quantize,
                quantization_bits=quantize_bits
            )
        else:
            # Standard one-stage FFN
            self.fc2 = BlockSparseLinear(
                self.hidden_dim, out_dim,
                bias=use_bias,
                block_size=block_size,
                sparsity=sparsity,
                structured_sparsity=structured_sparsity,
                enable_quantization=quantize,
                quantization_bits=quantize_bits
            )
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with block sparsity and optional quantization"""
        # Store input dtype for residual
        input_type = x.dtype
        
        # For residual in fp32 (helps with stability)
        if self.residual_in_fp32:
            x_for_residual = x.float()
        
        # First projection
        if self.is_gated:
            # Gated path
            gate = self.fc1_gate(x)
            value = self.fc1(x)
            
            # Apply activation with gating
            if hasattr(self.activation, 'forward_gated'):
                # For activations with dedicated gated implementation
                h = self.activation.forward_gated(gate, value)
            else:
                # Manual gating
                h = self.activation(gate) * value
        else:
            # Standard path
            h = self.fc1(x)
            h = self.activation(h)
            
        h = self.dropout(h)
        
        # Two-stage processing if enabled
        if self.two_stage:
            h = self.fc_mid(h)
            h = self.mid_activation(h)
            h = self.dropout(h)
            
        # Final projection
        h = self.fc2(h)
        
        # Restore original dtype if needed
        if self.residual_in_fp32:
            h = h.to(input_type)
            
        return h

class UltraScaleTransformer(nn.Module):
    """
    Ultimate 100M+ context transformer with hierarchical memory architecture,
    extreme efficiency optimizations, and hybrid token processing
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 24,
        heads: int = 16,
        dim_head: Optional[int] = None,
        max_seq_len: int = 100 * 1000 * 1000,
        vocab_size: Optional[int] = None,
        max_memory_tokens: int = 100 * 1000 * 1000,
        enable_memory_manager: bool = True,
        token_retrieval_mode: str = 'auto',  # 'auto', 'exact', 'streaming', 'fixed'
        memory_offload_path: str = "./ultra_memory_offload",
        
        # Architecture options
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        use_rope: bool = True,
        use_flash_attn: bool = True,
        use_xpos: bool = True,
        relative_pos: bool = True,
        norm_position: str = 'pre',
        final_norm: bool = True,
        use_flash_ff: bool = True,
        
        # Model parallelism and efficiency
        enable_checkpointing: bool = True,
        activation: Union[str, nn.Module] = 'gelu',
        use_bias: bool = False,
        tie_emb_prj: bool = True,
        block_sparse: bool = True,
        enable_quantization: bool = True,
        memory_efficient: bool = True,
        enable_ema: bool = True,  # Use exponential moving average for weights
        
        # Advanced architecture options
        dim_expansion_factor: float = 1.0,  # Expanding dimensions in later layers
        decoupled_emb_dim: Optional[int] = None,  # Separate embedding dimension
        sandwich_norm: bool = True,  # Extra norm between attention and FFN
        memory_layers: Optional[List[int]] = None,  # Layers that use memory
        adaptive_computation: bool = True,  # Adaptively skip layers
        sparse_layers: Optional[List[int]] = None,  # Layers with extra sparsity
        
        # Memory configuration
        memory_config: Optional[Dict[str, Any]] = None,
        
        # Device settings
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head or (dim // heads)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.enable_memory_manager = enable_memory_manager
        self.token_retrieval_mode = token_retrieval_mode
        self.use_rope = use_rope
        self.memory_efficient = memory_efficient
        self.enable_ema = enable_ema
        self.adaptive_computation = adaptive_computation
        
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        # Initialize memory manager if enabled
        if enable_memory_manager:
            memory_config = memory_config or {}
            self.memory_manager = UltraScaleMemoryManager(
                model_dim=dim,
                max_total_tokens=max_memory_tokens,
                offload_path=memory_offload_path,
                enable_tensor_compression=enable_quantization,
                enable_semantic_compression=True,
                token_pruning=True,
                token_deduplication=True,
                enable_checkpointing=True,
                enable_vector_index=True,
                device=self.device,
                **memory_config
            )
        else:
            self.memory_manager = None
            
        # Determine embedding dimension (can be decoupled from model dim)
        self.emb_dim = decoupled_emb_dim or dim
        
        # Initialize token embedding
        if vocab_size is not None:
            self.token_emb = nn.Embedding(vocab_size, self.emb_dim)
            
            # Add projection if embedding dimension differs from model dimension
            if self.emb_dim != dim:
                self.emb_proj = nn.Identity()
                
        # Positional embedding options
        if use_rope:
            # Will be handled in attention layers
            self.pos_emb = None
        else:
            # Use learned positional embeddings for layers without RoPE
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
            nn.init.normal_(self.pos_emb, std=0.02)
            
        # Dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Convert activation string to module if needed
        if isinstance(activation, str):
            if activation == 'gelu':
                activation = nn.GELU()
            elif activation == 'swiglu':
                activation = nn.SiLU()  # SwiGLU will be handled in FFN
            elif activation == 'relu':
                activation = nn.ReLU()
            else:
                logger.warning(f"Unknown activation: {activation}, using GELU")
                activation = nn.GELU()
                
        # Set memory layers if not provided
        if memory_layers is None:
            # By default, use memory in 1/3 of layers
            memory_layers = list(range(depth // 3, depth, depth // 3))
            
        # Set sparse layers if not provided
        if sparse_layers is None:
            # By default, increase sparsity in later layers
            sparse_layers = list(range(depth // 2, depth))
        
        # Initialize transformer layers with progressive expansion
        self.layers = nn.ModuleList([])
        for i in range(depth):
            # Apply dimension expansion in later layers if needed
            if dim_expansion_factor > 1.0:
                # Gradually increase dimension
                layer_dim = dim if i < depth // 2 else int(dim * dim_expansion_factor)
                
                # Add projection for dimension change if needed
                if i == depth // 2 and layer_dim != dim:
                    self.layers.append(nn.Linear(dim, layer_dim, bias=use_bias))
            else:
                layer_dim = dim
                
            # Compute layer-specific configurations
            use_memory = i in memory_layers
            higher_sparsity = i in sparse_layers
            sparsity_factor = 1.5 if higher_sparsity else 1.0  # Increase sparsity in some layers
            
            # Create transformer block
            self.layers.append(UltraScaleTransformerBlock(
                dim=layer_dim,
                heads=heads,
                dim_head=self.dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                norm_position=norm_position,
                activation=activation,
                use_memory_manager=use_memory and enable_memory_manager,
                memory_manager=self.memory_manager if use_memory else None,
                max_seq_len=max_seq_len,
                use_mega_attention=True,
                relative_pos=relative_pos,
                rotary_pos=use_rope,
                use_xpos=use_xpos,
                layer_idx=i,
                block_sparse=block_sparse,
                enable_quantization=enable_quantization,
                memory_efficient=memory_efficient,
                use_flash_attention=use_flash_attention,
                use_checkpointing=enable_checkpointing,
                sandwich_norm=sandwich_norm,
                gate_residual=True,
                use_parallel_attention=(i % 2 == 0),  # Alternate between parallel and sequential
                use_bias=use_bias,
                window_size=1024 + (i * 128),  # Gradually increase window size in later layers
                global_tokens=256 + (i * 32),   # Gradually increase global tokens
                adaptive_computation=adaptive_computation
            ))
            
        # Final normalization
        if final_norm:
            self.norm_f = nn.LayerNorm(dim if dim_expansion_factor == 1.0 else layer_dim, eps=layer_norm_eps)
        else:
            self.norm_f = nn.Identity()
            
        # Final projection for language modeling
        if vocab_size is not None:
            if tie_emb_prj and self.emb_dim == dim:
                # Tie embedding and projection weights
                self.to_logits = lambda x: F.linear(x, self.token_emb.weight)
            else:
                # Separate projection layer
                self.to_logits = nn.Linear(dim if dim_expansion_factor == 1.0 else layer_dim, 
                                          vocab_size, bias=use_bias)
                
        # EMA setup if enabled
        if enable_ema:
            self.ema_params = {}
            self.ema_decay = 0.999
            
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with scaled initialization"""
        if isinstance(module, nn.Linear):
            # Use scaled initialization for better gradient flow in deep models
            nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(self.depth))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def update_ema(self):
        """Update exponential moving average of weights"""
        if not self.enable_ema:
            return
            
        # Initialize EMA parameters on first call
        if not self.ema_params:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.data.clone()
                    
        # Update EMA parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                    
    def apply_ema(self):
        """Apply EMA weights to model"""
        if not self.enable_ema or not self.ema_params:
            return
            
        # Store current parameters
        current_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                current_params[name] = param.data.clone()
                
        # Apply EMA parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in self.ema_params:
                    param.data.copy_(self.ema_params[name])
                    
        return current_params
        
    def restore_params(self, current_params):
        """Restore original parameters after using EMA weights"""
        if not current_params:
            return
            
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.requires_grad and name in current_params:
                    param.data.copy_(current_params[name])
    
    @contextmanager
    def ema_context(self):
        """Context manager for temporarily using EMA weights"""
        if not self.enable_ema:
            yield
            return
            
        current_params = self.apply_ema()
        try:
            yield
        finally:
            self.restore_params(current_params)
            
    def get_input_embeddings(self, token_ids: Tensor) -> Tensor:
        """Get embeddings from token IDs"""
        if not hasattr(self, 'token_emb'):
            raise ValueError("Model does not have token embeddings")
            
        embeddings = self.token_emb(token_ids)
        
        # Project embeddings if needed
        if self.emb_dim != self.dim:
            embeddings = self.emb_proj(embeddings)
            
        return embeddings
        
    def get_token_indices(self, token_ids: Optional[Tensor] = None, input_embeds: Optional[Tensor] = None) -> Tensor:
        """Get position indices for tokens"""
        if token_ids is not None:
            seq_len = token_ids.size(1)
        elif input_embeds is not None:
            seq_len = input_embeds.size(1)
        else:
            raise ValueError("Either token_ids or input_embeds must be provided")
            
        # Get device from inputs
        device = token_ids.device if token_ids is not None else input_embeds.device
        
        # Create position indices
        return torch.arange(seq_len, device=device)
        
    def forward(
        self,
        token_ids: Optional[Tensor] = None,
        input_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, Tensor]]] = None,
        use_kv_cache: bool = False,
        use_chunked_processing: bool = True,
        chunk_size: int = 4096,
        store_memory: bool = True,
        return_dict: bool = True,
        return_attention_stats: bool = False,
        use_ema: bool = False,
        token_memory_ids: Optional[List[int]] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the model
        
        Args:
            token_ids: Optional tensor of token IDs [batch, seq_len]
            input_embeds: Optional pre-computed embeddings [batch, seq_len, dim]
            attention_mask: Optional attention mask [batch, seq_len]
            position_ids: Optional position indices [batch, seq_len]
            past_key_values: Optional cached KV for faster generation
            use_kv_cache: Whether to use KV caching for generation
            use_chunked_processing: Whether to process long sequences in chunks
            chunk_size: Size of chunks for processing
            store_memory: Whether to store token embeddings in memory manager
            return_dict: Whether to return dict with additional info
            return_attention_stats: Whether to return attention statistics
            use_ema: Whether to use EMA weights for this forward pass
            token_memory_ids: Optional list of token IDs for memory retrieval
            
        Returns:
            Model output tensor or dict with additional information
        """
        # Use EMA weights if specified
        if use_ema and self.enable_ema:
            with self.ema_context():
                return self._forward_impl(
                    token_ids, input_embeds, attention_mask, position_ids,
                    past_key_values, use_kv_cache, use_chunked_processing,
                    chunk_size, store_memory, return_dict, return_attention_stats,
                    token_memory_ids
                )
        else:
            return self._forward_impl(
                token_ids, input_embeds, attention_mask, position_ids,
                past_key_values, use_kv_cache, use_chunked_processing,
                chunk_size, store_memory, return_dict, return_attention_stats,
                token_memory_ids
            )
    
    def _forward_impl(
        self,
        token_ids: Optional[Tensor] = None,
        input_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[Dict[int, Dict[str, Tensor]]] = None,
        use_kv_cache: bool = False,
        use_chunked_processing: bool = True,
        chunk_size: int = 4096,
        store_memory: bool = True,
        return_dict: bool = True,
        return_attention_stats: bool = False,
        token_memory_ids: Optional[List[int]] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Internal implementation of forward pass"""
        # Initialize KV cache if using
        if use_kv_cache and past_key_values is None:
            past_key_values = {}
            
        # Get embeddings
        if input_embeds is None:
            if token_ids is None:
                raise ValueError("Either token_ids or input_embeds must be provided")
                
            # Get token embeddings
            x = self.get_input_embeddings(token_ids)
        else:
            x = input_embeds
            
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        # Get position IDs if not provided
        if position_ids is None:
            position_ids = self.get_token_indices(token_ids, input_embeds)
            
        # Add positional embeddings if using learned positions
        if self.pos_emb is not None:
            max_pos = min(seq_len, self.max_seq_len)
            x = x + self.pos_emb[:, :max_pos]
            
        # Apply embedding dropout
        x = self.emb_dropout(x)
        
        # Process according to sequence length
        if use_chunked_processing and seq_len > chunk_size:
            # Process in chunks for very long sequences
            x = self._chunked_forward(
                x, attention_mask, position_ids, past_key_values,
                use_kv_cache, chunk_size, store_memory, token_memory_ids
            )
        else:
            # Store in memory manager if enabled
            if self.enable_memory_manager and self.memory_manager is not None and store_memory:
                importance = torch.ones(batch_size, seq_len, device=x.device) * 0.5
                
                for b in range(batch_size):
                    self.memory_manager.add_tokens(
                        token_embeddings=x[b],
                        importance_scores=importance[b],
                        token_ids=token_memory_ids
                    )
            
            # Standard processing
            attn_stats = []
            
            # Process through transformer layers
            for i, layer in enumerate(self.layers):
                # Skip linear dimension projection layers
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    continue
                    
                # Process through transformer block
                layer_past = past_key_values.get(i) if past_key_values else None
                
                x = layer(x, mask=attention_mask)
                
                # Collect attention statistics if requested
                if return_attention_stats and hasattr(layer, 'attention') and hasattr(layer.attention, 'attention_stats'):
                    attn_stats.append(layer.attention.attention_stats)
                    
        # Apply final normalization
        x = self.norm_f(x)
        
        # Convert to logits if needed
        if hasattr(self, 'to_logits'):
            logits = self.to_logits(x)
        else:
            logits = None
            
        if return_dict:
            outputs = {
                'last_hidden_state': x,
                'logits': logits
            }
            
            if return_attention_stats:
                outputs['attention_stats'] = attn_stats
                
            return outputs
        else:
            return logits if logits is not None else x
    
    def _chunked_forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Tensor,
        past_key_values: Optional[Dict[int, Dict[str, Tensor]]],
        use_kv_cache: bool,
        chunk_size: int,
        store_memory: bool,
        token_memory_ids: Optional[List[int]]
    ) -> Tensor:
        """Process long sequences in chunks for memory efficiency"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Determine optimal chunk size based on hardware
        device_chunks = {
            'cuda': min(8192, chunk_size),
            'cpu': min(4096, chunk_size),
            'mps': min(2048, chunk_size)
        }
        optimal_chunk_size = device_chunks.get(str(x.device.type), chunk_size)
        
        # Adjust for extremely long sequences
        if seq_len > 1000000:  # 1M tokens
            optimal_chunk_size = min(optimal_chunk_size, 1024)
        elif seq_len > 100000:  # 100K tokens
            optimal_chunk_size = min(optimal_chunk_size, 2048)
            
        # Use adjusted chunk size
        chunk_size = optimal_chunk_size
        
        # Process in chunks
        outputs = torch.zeros_like(x)
        
        for chunk_start in range(0, seq_len, chunk_size):
            # Define chunk boundaries
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_slice = slice(chunk_start, chunk_end)
            
            # Extract chunk
            x_chunk = x[:, chunk_slice, :]
            
            # Extract corresponding attention mask if provided
            if attention_mask is not None:
                mask_chunk = attention_mask[:, chunk_slice]
            else:
                mask_chunk = None
                
            # Extract position IDs for this chunk
            pos_chunk = position_ids[chunk_slice]
            
            # Store in memory manager if enabled
            if self.enable_memory_manager and self.memory_manager is not None and store_memory:
                importance = torch.ones(batch_size, chunk_end - chunk_start, device=x.device) * 0.5
                
                for b in range(batch_size):
                    self.memory_manager.add_tokens(
                        token_embeddings=x_chunk[b],
                        importance_scores=importance[b],
                        token_ids=token_memory_ids[chunk_start:chunk_end] if token_memory_ids else None
                    )
            
            # Process chunk through transformer layers
            for i, layer in enumerate(self.layers):
                # Skip linear dimension projection layers
                if isinstance(layer, nn.Linear):
                    x_chunk = layer(x_chunk)
                    continue
                    
                # Process through transformer block
                layer_past = past_key_values.get(i) if past_key_values else None
                
                x_chunk = layer(x_chunk, mask=mask_chunk)
                
            # Store processed chunk
            outputs[:, chunk_slice, :] = x_chunk
            
        return outputs
        
    def generate(
        self,
        token_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_kv_cache: bool = True,
        use_ema: bool = False
    ) -> Tensor:
        """
        Generate text using the model
        
        Args:
            token_ids: Input token IDs [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: ID of pad token
            eos_token_id: ID of EOS token
            use_kv_cache: Whether to use KV caching for faster generation
            use_ema: Whether to use EMA weights for generation
            
        Returns:
            Generated token IDs [batch, max_length]
        """
        # Ensure the model is in eval mode
        self.eval()
        
        # Use EMA weights if specified
        if use_ema and self.enable_ema:
            current_params = self.apply_ema()
            
        try:
            batch_size = token_ids.shape[0]
            
            # Clone to avoid modifying the input
            token_ids = token_ids.clone()
            
            # Initialize KV cache if using
            past_key_values = {} if use_kv_cache else None
            
            # Track sequences that have finished
            if eos_token_id is not None:
                unfinished = torch.ones(batch_size, 1, device=token_ids.device, dtype=torch.bool)
            
            # Keep generating until max length
            for _ in range(max_length - token_ids.shape[1]):
                # Get position IDs
                position_ids = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
                
                # Forward pass
                outputs = self.forward(
                    token_ids=token_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_kv_cache=use_kv_cache,
                    return_dict=True
                )
                
                # Get logits for next token prediction
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / max(temperature, 1e-7)
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for batch_idx in range(batch_size):
                        for prev_token in token_ids[batch_idx]:
                            if prev_token.item() < next_token_logits.shape[-1]:
                                next_token_logits[batch_idx, prev_token] /= repetition_penalty
                
                # Apply filtering
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')
                        
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        
                        # Shift the indices to the right to keep the first token above threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')
                        
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                # Update tokens
                token_ids = torch.cat([token_ids, next_token], dim=-1)
                
                # Check if EOS token has been generated
                if eos_token_id is not None:
                    unfinished = unfinished & (next_token != eos_token_id).bool()
                    if not unfinished.any():
                        break
                        
            # Apply padding if needed
            if pad_token_id is not None:
                # Replace EOS tokens with padding
                if eos_token_id is not None:
                    token_ids[token_ids == eos_token_id] = pad_token_id
                    
            return token_ids
            
        finally:
            # Restore original weights if EMA was used
            if use_ema and self.enable_ema:
                self.restore_params(current_params)
                
    @torch.no_grad()
    def embed_text(
        self, 
        token_ids: Tensor,
        use_ema: bool = True,
        normalize: bool = True,
        pool_method: str = 'mean'  # 'mean', 'max', 'cls'
    ) -> Tensor:
        """
        Generate text embeddings
        
        Args:
            token_ids: Input token IDs [batch, seq_len]
            use_ema: Whether to use EMA weights
            normalize: Whether to normalize outputs
            pool_method: How to pool sequence embeddings
            
        Returns:
            Text embeddings [batch, dim]
        """
        # Set model to eval mode
        self.eval()
        
        # Use EMA weights if specified
        with self.ema_context() if use_ema and self.enable_ema else nullcontext():
            # Forward pass
            outputs = self.forward(
                token_ids=token_ids,
                return_dict=True
            )
            
            hidden_states = outputs['last_hidden_state']
            
            # Pool embeddings
            if pool_method == 'cls':
                pooled = hidden_states[:, 0]
            elif pool_method == 'max':
                pooled = torch.max(hidden_states, dim=1)[0]
            else:  # 'mean' is default
                pooled = torch.mean(hidden_states, dim=1)
                
            # Normalize if requested
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
                
            return pooled

##############################################
# Utility Functions and Factory Method       #
##############################################

def create_ultra_scale_transformer(
    dim: int = 2048,
    depth: int = 32,
    heads: int = 32,
    context_size: int = 100 * 1000 * 1000,  # 100M tokens
    vocab_size: Optional[int] = None,
    mode: str = 'balanced',  # 'balanced', 'performance', 'extreme', 'efficient', or 'custom'
    memory_mode: str = 'auto',  # 'auto', 'standard', 'maximum', 'minimum'
    **kwargs
) -> UltraScaleTransformer:
    """
    Factory method to create a pre-configured UltraScaleTransformer
    
    Args:
        dim: Model dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        context_size: Maximum context size in tokens
        vocab_size: Optional vocabulary size
        mode: Configuration mode
        memory_mode: Memory configuration mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured UltraScaleTransformer instance
    """
    config = {
        'dim': dim,
        'depth': depth,
        'heads': heads,
        'max_seq_len': context_size,
        'vocab_size': vocab_size
    }
    
    # Apply memory configuration based on mode
    memory_config = {}
    
    if memory_mode == 'maximum':
        memory_config = {
            'token_pruning': False,  # Keep all tokens
            'pruning_threshold': 0.01,  # Very low pruning threshold
            'token_deduplication': True,
            'enable_vector_index': True,
            'vector_index_sample_rate': 0.2,  # Index more tokens
            'enable_offloading': True
        }
    elif memory_mode == 'minimum':
        memory_config = {
            'token_pruning': True,
            'pruning_threshold': 0.3,  # Higher pruning threshold
            'token_deduplication': True,
            'enable_vector_index': False,  # Disable vector indexing
            'enable_offloading': False,  # Disable offloading
            'max_memory_resident_tokens': context_size // 10  # Keep only 10% of tokens resident
        }
    elif memory_mode == 'standard':
        memory_config = {
            'token_pruning': True,
            'pruning_threshold': 0.1,
            'token_deduplication': True,
            'enable_vector_index': True,
            'vector_index_sample_rate': 0.1,
            'enable_offloading': True
        }
    # 'auto' uses default settings
    
    # Apply configuration based on mode
    if mode == 'performance':
        # Optimized for inference speed
        config.update({
            'dim_head': max(64, dim // heads),
            'mlp_ratio': 3.5,
            'dropout': 0.0,
            'use_flash_attn': True,
            'memory_efficient': True,
            'enable_ema': True,
            'use_bias': False,
            'enable_checkpointing': False,  # Faster inference without checkpointing
            'block_sparse': True,
            'activation': 'swiglu',  # Faster activation
            'use_rope': True,
            'use_xpos': True,
            'enable_quantization': True,  # Use quantization for speed
            'sandbox_norm': False,  # Skip extra norm for speed
            'token_retrieval_mode': 'streaming',  # Streaming mode for faster processing
            'enable_memory_manager': True,
            'memory_config': memory_config
        })
    elif mode == 'extreme':
        # Maximally capable model for highest quality
        config.update({
            'dim_head': max(64, dim // heads),
            'mlp_ratio': 4.5,
            'dropout': 0.0 if depth < 48 else 0.1,  # Use dropout for very deep models
            'use_flash_attn': True,
            'use_rope': True,
            'use_xpos': True,
            'relative_pos': True,
            'memory_efficient': True,
            'enable_ema': True,
            'use_bias': False,
            'enable_checkpointing': True,
            'block_sparse': True,
            'enable_quantization': True,
            'adaptive_computation': True,
            'sandwich_norm': True,
            'gate_residual': True,
            'token_retrieval_mode': 'exact',
            'enable_memory_manager': True,
            'memory_config': memory_config
        })
    elif mode == 'efficient':
        # Balanced efficiency for lower resource usage
        config.update({
            'dim_head': max(64, dim // heads),
            'mlp_ratio': 3.0,  # Smaller FFN
            'dropout': 0.0,
            'use_flash_attn': True,
            'use_rope': True,
            'use_xpos': True,
            'relative_pos': False,  # Skip relative position bias
            'memory_efficient': True,
            'enable_ema': False,  # Skip EMA
            'use_bias': False,
            'enable_checkpointing': True,
            'block_sparse': True,
            'enable_quantization': True,
            'adaptive_computation': True,
            'sandwich_norm': False,
            'token_retrieval_mode': 'fixed',
            'enable_memory_manager': True,
            'memory_config': memory_config
        })
    else:  # 'balanced' or 'custom'
        # Default balanced configuration
        config.update({
            'dim_head': max(64, dim // heads),
            'mlp_ratio': 4.0,
            'dropout': 0.0,
            'use_flash_attn': True,
            'use_rope': True,
            'use_xpos': True,
            'relative_pos': True,
            'memory_efficient': True,
            'enable_ema': True,
            'use_bias': False,
            'enable_checkpointing': True,
            'block_sparse': True,
            'enable_quantization': True,
            'sandwich_norm': True,
            'gate_residual': True,
            'token_retrieval_mode': 'auto',
            'enable_memory_manager': True,
            'memory_config': memory_config
        })
    
    # Override with any user-provided kwargs
    config.update(kwargs)
    
    # Create the model
    return UltraScaleTransformer(**config)

##############################################
# Example Usage                             #
##############################################

def example_usage():
    """Example of how to use the UltraScaleTransformer"""
    # Create a model with 100M token context window
    model = create_ultra_scale_transformer(
        dim=2048,  # Model dimension
        depth=32,  # Number of layers
        heads=32,  # Number of attention heads
        context_size=100_000_000,  # 100M token context
        vocab_size=32000,  # Vocabulary size
        mode='balanced'  # Configuration mode
    )
    
    # Generate random input
    batch_size = 1
    seq_len = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    token_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    # Forward pass
    outputs = model(token_ids=token_ids, return_dict=True)
    
    # Print output shape
    print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Generate text
    generated = model.generate(
        token_ids=token_ids[:, :10],  # Use first 10 tokens as prompt
        max_length=20,
        temperature=0.7,
        top_p=0.9
    )
    
    print(f"Generated tokens shape: {generated.shape}")
    
    # Get embeddings
    embeddings = model.embed_text(token_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Print memory manager stats
    if model.memory_manager is not None:
        print(model.memory_manager)

if __name__ == "__main__":
    example_usage() = nn.Linear(self.emb_dim, dim, bias=use_bias)
            else:
                self.emb_proj
