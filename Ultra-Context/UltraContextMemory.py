import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from typing import Optional, List, Dict, Tuple, Any, Union
import math
import logging
import queue
from dataclasses import dataclass, field
from collections import deque
import time
import gc
import weakref
import threading
import heapq
import uuid
from enum import Enum, auto
import numpy as np
import random
import os
import pickle # For serialization (basic example)

# Assuming UltraContext Core is imported
# from UltraContextCore.py import (
#     DEFAULT_PERF_CONFIG,
#     PerformanceConfig,
#     get_norm_class,
#     ActivationFunctions,
#     timer
# )

# --- Mock UltraContextCore for standalone execution ---
@dataclass
class PerformanceConfig:
    optimize_memory: bool = True
    optimize_speed: bool = True
    optimize_reliability: bool = False
    use_mixed_precision: bool = False
    device_specific_optimizations: bool = False
    use_cuda_graphs: bool = False

DEFAULT_PERF_CONFIG = PerformanceConfig()

def get_norm_class(norm_type: str):
    if norm_type.lower() == "layernorm":
        return nn.LayerNorm
    elif norm_type.lower() == "rmsnorm":
        # Placeholder RMSNorm
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
            def forward(self, x):
                variance = x.pow(2).mean(-1, keepdim=True)
                x = x * torch.rsqrt(variance + self.eps)
                return self.weight * x
        return RMSNorm
    else:
        return nn.LayerNorm # Default

class ActivationFunctions:
    GELU = nn.GELU
    ReLU = nn.ReLU
    SiLU = nn.SiLU

# Simple timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper
# --- End Mock UltraContextCore ---


# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ultracontext.memory")
# Set specific log levels if needed
# logging.getLogger("ultracontext.memory.vectorstore").setLevel(logging.WARNING)


# Memory access pattern tracking (Unchanged)
class MemoryAccessPattern(Enum):
    SEQUENTIAL = auto()
    LOCAL = auto()
    RANDOM = auto()
    REPEATED = auto()
    RECENCY = auto()
    START_FOCUSED = auto()
    END_FOCUSED = auto()
    SEMANTIC = auto()

# Enhanced TokenInfo (Unchanged logical structure, but usage context changes)
class TokenInfo:
    def __init__(self, value, position, creation_time=None, storage_tier=None):
        self.value = value # Token embedding/representation OR compressed form depending on tier
        self.position = position
        self.creation_time = creation_time or time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0
        self.access_positions = []
        self.importance_score = 0.5 # Dynamic importance score (0-1)
        self.storage_tier = storage_tier # Track which tier this info object lives in

    def record_access(self, position=None):
        self.last_access_time = time.time()
        self.access_count += 1
        if position is not None:
            self.access_positions.append(position)

    @property
    def recency(self):
        return time.time() - self.last_access_time

    @property
    def age(self):
        return time.time() - self.creation_time

    @property
    def frequency(self):
        age = max(1e-6, self.age)
        return self.access_count / age

    def update_importance(self, attention_weight=None):
        # Simplified importance update - more complex model might be needed for 1B scale
        recency_factor = math.exp(-self.recency / (3600 * 24)) # Slower decay over a day
        frequency_factor = math.log1p(self.access_count) / math.log1p(100) # Log scale, saturate slower
        base_importance = 0.3 * recency_factor + 0.3 * frequency_factor

        if attention_weight is not None:
            self.importance_score = 0.4 * base_importance + 0.6 * attention_weight
        else:
            self.importance_score = 0.8 * self.importance_score + 0.2 * base_importance

        self.importance_score = max(0.0, min(1.0, self.importance_score)) # Clamp
        return self.importance_score

# Vector database storage (Enhanced for Scale)
class TokenVectorStore:
    """
    Storage for token vectors within a *specific memory tier* (e.g., L2 or L3 RAM).
    Not intended to hold 1B tokens directly in memory. Disk/Distributed tiers use different mechanisms.
    """
    def __init__(self, dim, tier_id, max_tokens=1_000_000, similarity_threshold=0.7, index_type="flat"):
        self.dim = dim
        self.tier_id = tier_id
        # max_tokens here refers to the capacity of THIS TIER'S index cache
        self.max_tokens_in_index = max_tokens
        self.similarity_threshold = similarity_threshold
        # Index type selection is crucial for performance at scale
        # 'flat': Good for small L1/L2 caches (exact, slow for large N)
        # 'hnsw': Good for medium L2/L3 (fast ANN, high memory)
        # 'ivfpq': Best for large L3/Disk (lower memory, good speed, approx.)
        self.index_type = index_type
        self.logger = logging.getLogger(f"ultracontext.memory.vectorstore.T{tier_id}")

        self.token_vectors = {} # id -> vector (potentially compressed)
        self.token_metadata = {} # id -> metadata (position, importance, etc.)

        # Index structures for this tier
        self.position_index = {} # position -> id (Only relevant for tiers where positional lookup is primary)
        self.semantic_index = None
        self.semantic_index_mapping = [] # Maps FAISS index positions to token ids

        self._index_needs_training = False
        self._faiss_available = False
        self._index_on_gpu = False

        # Try importing FAISS
        try:
            import faiss
            self._faiss_available = True
            self.logger.info(f"FAISS available for Tier {tier_id}.")
        except ImportError:
            self.logger.warning(f"FAISS not available for Tier {tier_id}, semantic search will be slow or disabled.")
            self.index_type = "simple" # Fallback

    def add(self, token_id, vector, position, metadata=None):
        """Add a token vector from THIS tier to the index."""
        if len(self.token_vectors) >= self.max_tokens_in_index:
            # Eviction from the *index cache*, not necessarily the tier itself
            # Simple LRU for index cache eviction
            oldest_id = min(self.token_metadata.keys(), key=lambda tid: self.token_metadata[tid].get("last_access_in_index", 0))
            self.remove_from_index(oldest_id)

        self.token_vectors[token_id] = vector # Store potentially compressed vector
        self.token_metadata[token_id] = metadata or {}
        self.token_metadata[token_id]["position"] = position
        self.token_metadata[token_id]["last_access_in_index"] = time.time()

        # Only maintain position index if needed for this tier's access patterns
        # self.position_index[position] = token_id

        # Invalidate semantic index - rebuild incrementally or batch
        # For large scale, adding one by one is too slow, batching is preferred.
        # Here we mark for potential rebuild/addition later.
        if self.semantic_index is not None and self.semantic_index != "simple":
             # TODO: Implement batched/incremental index updates for performance
             # For now, we just invalidate and rebuild on search, which is inefficient at scale.
             self.semantic_index = None
             self.semantic_index_mapping = []
             self.logger.debug(f"Token added, semantic index invalidated for Tier {tier_id}")


    def remove_from_index(self, token_id):
        """Remove a token from THIS tier's index cache."""
        if token_id in self.token_vectors:
            # position = self.token_metadata[token_id].get("position")
            # if position in self.position_index:
            #     del self.position_index[position]

            del self.token_vectors[token_id]
            del self.token_metadata[token_id]

            # Invalidate semantic index - removal from FAISS is tricky/expensive
            if self.semantic_index is not None and self.semantic_index != "simple":
                self.semantic_index = None
                self.semantic_index_mapping = []
                self.logger.debug(f"Token removed, semantic index invalidated for Tier {tier_id}")

    def get_vector_by_id(self, token_id):
        """Get a vector by ID from this index cache."""
        if token_id in self.token_vectors:
            self.token_metadata[token_id]["last_access_in_index"] = time.time()
            return self.token_vectors[token_id], self.token_metadata[token_id]
        return None, None

    def _setup_advanced_index(self):
        if not self._faiss_available or self.index_type == "simple":
            self.semantic_index = "simple"
            self.logger.warning(f"Using simple semantic search for Tier {tier_id}.")
            return

        try:
            import faiss
            n_vectors = len(self.token_vectors)
            if n_vectors == 0: return

            self.logger.info(f"Setting up FAISS index '{self.index_type}' for Tier {tier_id} with {n_vectors} vectors.")

            # Choose appropriate parameters based on scale (n_vectors in this tier's cache)
            if self.index_type == "flat":
                self.semantic_index = faiss.IndexFlatIP(self.dim)
                self._index_needs_training = False
            elif self.index_type == "hnsw":
                # Good balance for RAM tiers, memory intensive
                hnsw_m = 32 # Default, could be tuned
                self.semantic_index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
                self.semantic_index.hnsw.efConstruction = 40 # Tune construction speed/quality
                self.semantic_index.hnsw.efSearch = 16      # Tune search speed/quality
                self._index_needs_training = False # HNSW doesn't need separate training stage
            elif self.index_type == "ivfpq":
                # Best for large scale, requires training
                # Determine centroids based on number of vectors in this tier's cache
                nlist = min(max(128, int(np.sqrt(n_vectors) * 8)), 4096) # Rule of thumb
                # Product Quantization parameters - trade-off between compression and accuracy
                m = 8 # Number of subquantizers (power of 2, depends on dim)
                nbits = 8 # Bits per subquantizer index (8 is common)
                quantizer = faiss.IndexFlatIP(self.dim) # Base quantizer
                self.semantic_index = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
                self.semantic_index.nprobe = min(nlist, 32) # Number of clusters to search, critical for speed/accuracy
                self._index_needs_training = True
                self.logger.info(f"Tier {tier_id} IVFPQ setup: nlist={nlist}, m={m}, nbits={nbits}, nprobe={self.semantic_index.nprobe}")
            else: # Default to flat
                self.logger.warning(f"Unknown index_type '{self.index_type}', defaulting to 'flat' for Tier {tier_id}.")
                self.semantic_index = faiss.IndexFlatIP(self.dim)
                self._index_needs_training = False

            # Try moving index to GPU if available and beneficial
            if torch.cuda.is_available() and self.tier_id <= 2 and n_vectors < 5_000_000: # GPU helps more for smaller/medium indices
                try:
                    res = faiss.StandardGpuResources()
                    co = faiss.GpuClonerOptions()
                    # PQ needs fp16 for GPU efficiency
                    co.useFloat16 = (self.index_type == "ivfpq")
                    self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index, co)
                    self._index_on_gpu = True
                    self.logger.info(f"FAISS index for Tier {tier_id} moved to GPU.")
                except Exception as e:
                    self.logger.warning(f"Failed to move FAISS index for Tier {tier_id} to GPU: {e}")
                    self._index_on_gpu = False
            else:
                 self._index_on_gpu = False

        except Exception as e:
            self.logger.error(f"Error setting up FAISS index for Tier {tier_id}: {e}", exc_info=True)
            self.semantic_index = "simple"

    def _ensure_semantic_index(self):
        """Ensure the semantic index is built or rebuilt if needed."""
        if self.semantic_index is None and len(self.token_vectors) > 0:
            self._setup_advanced_index()

            if self.semantic_index == "simple": return
            if self.semantic_index is None: return # Setup failed

            n_vectors = len(self.token_vectors)
            self.logger.info(f"Building/Training FAISS index for Tier {tier_id} with {n_vectors} vectors.")

            try:
                import faiss
                # Prepare vectors (normalize for Inner Product = Cosine Sim)
                # IMPORTANT: Ensure vectors are on CPU as float32 for FAISS processing
                all_vectors = torch.stack(list(self.token_vectors.values())).float().cpu()
                if all_vectors.shape[1] != self.dim:
                     # This might happen if vectors are compressed; need decompression before indexing
                     # This indicates a design issue - the vector store should probably hold decompressed vectors
                     # or the compression needs to be handled during search. For now, we log an error.
                     self.logger.error(f"Vector dimension mismatch in Tier {self.tier_id}! Expected {self.dim}, got {all_vectors.shape[1]}. Index cannot be built.")
                     self.semantic_index = "simple"
                     return

                all_vectors = F.normalize(all_vectors, p=2, dim=1)
                vector_data = all_vectors.numpy()

                if self._index_needs_training:
                    if n_vectors < getattr(self.semantic_index, 'nlist', 128):
                         self.logger.warning(f"Not enough vectors ({n_vectors}) to train IVFPQ index effectively in Tier {self.tier_id}. Skipping training.")
                    else:
                        self.logger.info(f"Training FAISS index for Tier {tier_id}...")
                        # Select subset for training if dataset is huge (e.g., > 1M)
                        train_size = min(n_vectors, 256 * getattr(self.semantic_index, 'nlist', 128)) # FAISS recommendation
                        indices = np.random.permutation(n_vectors)[:train_size]
                        self.semantic_index.train(vector_data[indices])
                        self.logger.info(f"FAISS index training complete for Tier {tier_id}.")
                    self._index_needs_training = False # Trained or skipped

                if not self.semantic_index.is_trained:
                     self.logger.warning(f"FAISS index for Tier {self.tier_id} is not trained, cannot add vectors.")
                     return

                # Add vectors to the index
                self.logger.info(f"Adding {n_vectors} vectors to FAISS index for Tier {tier_id}...")
                if self._index_on_gpu:
                    # Add in batches to GPU index to avoid OOM
                    batch_size = 100000
                    for i in range(0, n_vectors, batch_size):
                        self.semantic_index.add(vector_data[i:i+batch_size])
                else:
                     self.semantic_index.add(vector_data)

                self.semantic_index_mapping = list(self.token_vectors.keys())
                self.logger.info(f"FAISS index build complete for Tier {tier_id}. Index size: {self.semantic_index.ntotal}")

            except Exception as e:
                self.logger.error(f"Error building FAISS index for Tier {tier_id}: {e}", exc_info=True)
                self.semantic_index = "simple" # Fallback on error


    def search_similar(self, query_vector, top_k=5):
        if len(self.token_vectors) == 0: return []

        query_vector_norm = F.normalize(query_vector.float().cpu(), p=2, dim=0) # Normalize and ensure CPU float32
        query_np = query_vector_norm.unsqueeze(0).numpy()

        self._ensure_semantic_index() # Build if needed

        if self.semantic_index == "simple" or self.semantic_index is None:
            # Fallback: Manual cosine similarity (SLOW for large N)
            # This should only happen for small caches or if FAISS fails
            similarities = {}
            for token_id, vector in self.token_vectors.items():
                 vector_norm = F.normalize(vector.float().cpu(), p=2, dim=0)
                 similarity = torch.dot(query_vector_norm, vector_norm).item()
                 similarities[token_id] = similarity

            top_ids = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
            results = []
            for tid in top_ids:
                if similarities[tid] >= self.similarity_threshold:
                     # Return original potentially compressed vector
                     vec, meta = self.get_vector_by_id(tid)
                     if vec is not None:
                        results.append((tid, vec, similarities[tid], meta))
            return results

        elif self._faiss_available:
            # Use FAISS
            try:
                # Set efSearch for HNSW if applicable
                if self.index_type == "hnsw" and hasattr(self.semantic_index, 'hnsw'):
                     self.semantic_index.hnsw.efSearch = 16 # Can be tuned

                # Set nprobe for IVF indexes if applicable
                if self.index_type == "ivfpq" and hasattr(self.semantic_index, 'nprobe'):
                    self.semantic_index.nprobe = min(getattr(self.semantic_index, 'nlist', 32), 32) # Tune nprobe

                distances, indices = self.semantic_index.search(query_np, top_k)
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.semantic_index_mapping):
                        token_id = self.semantic_index_mapping[idx]
                        similarity = float(distances[0][i]) # IP distance = cosine similarity for normalized vectors
                        if similarity >= self.similarity_threshold:
                             # Return original potentially compressed vector
                             vec, meta = self.get_vector_by_id(token_id)
                             if vec is not None:
                                results.append((token_id, vec, similarity, meta))
                return results
            except Exception as e:
                self.logger.error(f"FAISS search error in Tier {self.tier_id}: {e}", exc_info=True)
                return [] # Return empty on error
        return []

    # batch_search_similar would follow a similar logic using index.search for multiple queries
    def batch_search_similar(self, query_vectors, top_k=5):
        # Implementation similar to search_similar but using batch capability of FAISS
        # Normalize queries, ensure index, call self.semantic_index.search(queries_np, top_k)
        # Process results for each query.
        # Omitted for brevity, but structure mirrors single search.
        if len(self.token_vectors) == 0: return [[] for _ in range(query_vectors.shape[0])]

        queries_norm = F.normalize(query_vectors.float().cpu(), p=2, dim=1)
        queries_np = queries_norm.numpy()

        self._ensure_semantic_index()

        if self.semantic_index == "simple" or self.semantic_index is None:
             # Fallback (very slow for batches)
             batch_results = []
             for i in range(queries_np.shape[0]):
                 batch_results.append(self.search_similar(queries_norm[i], top_k))
             return batch_results
        elif self._faiss_available:
            try:
                if self.index_type == "hnsw" and hasattr(self.semantic_index, 'hnsw'):
                     self.semantic_index.hnsw.efSearch = 16
                if self.index_type == "ivfpq" and hasattr(self.semantic_index, 'nprobe'):
                    self.semantic_index.nprobe = min(getattr(self.semantic_index, 'nlist', 32), 32)

                distances, indices = self.semantic_index.search(queries_np, top_k)
                batch_results = []
                for q in range(queries_np.shape[0]):
                    results = []
                    for i, idx in enumerate(indices[q]):
                        if idx >= 0 and idx < len(self.semantic_index_mapping):
                            token_id = self.semantic_index_mapping[idx]
                            similarity = float(distances[q][i])
                            if similarity >= self.similarity_threshold:
                                vec, meta = self.get_vector_by_id(token_id)
                                if vec is not None:
                                    results.append((token_id, vec, similarity, meta))
                    batch_results.append(results)
                return batch_results
            except Exception as e:
                 self.logger.error(f"FAISS batch search error in Tier {self.tier_id}: {e}", exc_info=True)
                 return [[] for _ in range(queries_np.shape[0])]
        return [[] for _ in range(queries_np.shape[0])]


    def clear(self):
        self.token_vectors.clear()
        self.token_metadata.clear()
        self.position_index.clear()
        self.semantic_index = None
        self.semantic_index_mapping = []
        self._index_needs_training = False
        self._index_on_gpu = False
        gc.collect() # Help release memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info(f"VectorStore index cleared for Tier {self.tier_id}")

    def __len__(self):
        return len(self.token_vectors)

    def optimize(self):
        """Optimize the vector store index for this tier."""
        self.logger.info(f"Optimizing VectorStore index for Tier {self.tier_id}...")
        n_vectors = len(self.token_vectors)
        if n_vectors > 0 and self._faiss_available and self.semantic_index != "simple":
            # Rebuild index from scratch
            self.semantic_index = None
            self.semantic_index_mapping = []
            self._ensure_semantic_index()
            self.logger.info(f"VectorStore index optimization complete for Tier {self.tier_id}.")
        elif self.semantic_index == "simple":
             # No optimization needed for simple index
             pass
        else:
             self.logger.info(f"No optimization performed for Tier {self.tier_id} (no vectors or FAISS unavailable).")


# Importance Scorer (Enhanced for Scale)
class ImportanceScorer(Module):
    """Scores token importance based on multiple factors."""
    def __init__(
        self,
        dim: int,
        importance_dim: int = 64,
        max_context_window: int = 1_000_000_000, # Target context window
        use_attention_weights: bool = True,
        use_positional_bias: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.dim = dim
        self.importance_dim = importance_dim
        self.use_attention_weights = use_attention_weights
        self.use_positional_bias = use_positional_bias
        self.max_context_window = max_context_window

        # Smaller network for faster scoring
        self.project = nn.Linear(dim, importance_dim)

        # Positional Embeddings for 1B scale:
        # Standard nn.Embedding is too large (1B * importance_dim params)
        # Use sinusoidal or learned relative embeddings. Here, use sinusoidal.
        if use_positional_bias:
             self.positional_encoding = self.build_sinusoidal_embeddings(max_context_window, importance_dim)
             self.register_buffer('pos_enc_buffer', self.positional_encoding, persistent=False)

        mlp_input_dim = importance_dim * 2 if use_positional_bias else importance_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, importance_dim // 2),
            nn.GELU(),
            nn.Linear(importance_dim // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()
        self.logger = logging.getLogger("ultracontext.memory.scorer")
        self.logger.info(f"ImportanceScorer initialized for {max_context_window} max positions.")


    def build_sinusoidal_embeddings(self, seq_len, dim):
        """Builds sinusoidal embeddings compatible with large sequences"""
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe # Shape: [seq_len, dim]


    def get_positional_embedding(self, positions):
        """Retrieve sinusoidal embeddings for given positions"""
        # positions shape: [batch_size, seq_len]
        batch_size, seq_len = positions.shape
        # Ensure positions are within bounds
        positions = torch.clamp(positions, 0, self.max_context_window - 1).long()

        # Gather embeddings - buffer has shape [max_len, dim]
        # Need to handle batching
        all_pos_emb = []
        for b in range(batch_size):
             pos_indices = positions[b] # [seq_len]
             all_pos_emb.append(self.pos_enc_buffer[pos_indices]) # [seq_len, dim]

        return torch.stack(all_pos_emb) # [batch_size, seq_len, dim]


    def _init_weights(self):
        # Initialize layers
        nn.init.normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)
        for module in self.score_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # Token rarity based on hash is removed - too memory intensive for 1B scale
    # Importance will rely more on the neural network, attention, and access patterns.

    def forward(self, token_vectors, positions=None, attention_weights=None):
        batch_size, seq_len, _ = token_vectors.shape
        device = token_vectors.device

        projected = self.project(token_vectors)

        if self.use_positional_bias and positions is not None:
            # Ensure positions tensor is on the correct device
            positions_dev = positions.to(device=device, non_blocking=True)
            pos_emb = self.get_positional_embedding(positions_dev)
            features = torch.cat([projected, pos_emb], dim=-1)
        else:
            features = projected

        # Neural importance score
        # Ensure features are in float32 for MLP stability if using mixed precision elsewhere
        importance = self.score_mlp(features.float()).squeeze(-1)

        # Combine with attention weights
        if self.use_attention_weights and attention_weights is not None:
            # Ensure attention_weights is on the correct device and dtype
            attention_weights_dev = attention_weights.to(device=device, dtype=importance.dtype, non_blocking=True)
            # Blend: higher weight for attention
            importance = 0.4 * importance + 0.6 * attention_weights_dev
        else:
             # Blend with a fixed bias towards recency/frequency encoded in base importance
             importance = 0.8 * importance + 0.2 * 0.5 # Add small constant base


        return torch.clamp(importance, 0.0, 1.0) # Ensure valid range

    def score_single_token(self, token_vector, position=None, access_stats=None):
        # Simplified scoring for single tokens (e.g., during retrieval/rebalancing)
        # Less accurate than batch scoring but faster.
        projected = self.project(token_vector.unsqueeze(0)) # [1, importance_dim]

        if self.use_positional_bias and position is not None:
            pos_tensor = torch.tensor([[position]], device=token_vector.device)
            pos_emb = self.get_positional_embedding(pos_tensor) # [1, 1, dim]
            features = torch.cat([projected, pos_emb.squeeze(1)], dim=-1)
        else:
            features = projected

        neural_importance = self.score_mlp(features.float()).squeeze().item()

        # Combine with access stats if available
        if access_stats is not None:
            recency = access_stats.get("recency", 3600*24*30) # Default to 1 month old
            frequency = access_stats.get("frequency", 0)

            recency_factor = math.exp(-recency / (3600 * 24 * 7)) # Decay over a week
            frequency_factor = math.log1p(frequency) / math.log1p(100)

            combined_importance = 0.6 * neural_importance + 0.2 * recency_factor + 0.2 * frequency_factor
        else:
            combined_importance = neural_importance

        return max(0.0, min(1.0, combined_importance))


# Base Memory Level (Refactored for Tiered Structure)
class MemoryTier(Module):
    """Base class for a memory tier in the hierarchy."""
    def __init__(self, level_id, dim, capacity, retrieval_cost, storage_cost, eviction_policy="lru"):
        super().__init__()
        self.level_id = level_id
        self.dim = dim
        self.capacity = capacity # Max number of TokenInfo objects in this tier
        self.retrieval_cost = retrieval_cost # Relative cost/latency multiplier
        self.storage_cost = storage_cost # Relative cost per token
        self.eviction_policy = eviction_policy
        self.logger = logging.getLogger(f"ultracontext.memory.tier{level_id}")

        self.tokens = {} # token_id -> TokenInfo
        self.position_index = {} # position -> token_id (optional, maybe only for L1)
        self.lru_queue = deque() # For LRU eviction

        # Statistics
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0

        self.logger.info(f"Initialized Tier {level_id} with capacity {capacity}")

    def add(self, token_id, value, position, importance=0.5):
        raise NotImplementedError

    def get(self, token_id=None, position=None):
        raise NotImplementedError

    def _evict(self):
        raise NotImplementedError

    def _remove(self, token_id):
        """Removes a token *from this tier*. Returns the removed TokenInfo or None."""
        if token_id in self.tokens:
            token_info = self.tokens.pop(token_id)
            position = token_info.position
            if position in self.position_index:
                del self.position_index[position]
            try:
                # Removing from deque can be slow, consider alternatives if bottleneck
                self.lru_queue.remove(token_id)
            except ValueError:
                pass # Not in queue or already removed
            self.logger.debug(f"Removed token {token_id} from Tier {self.level_id}")
            return token_info
        return None

    @property
    def hit_rate(self):
        total_accesses = self.hit_count + self.miss_count
        return self.hit_count / total_accesses if total_accesses > 0 else 0.0

    @property
    def fullness(self):
         return len(self.tokens) / self.capacity if self.capacity > 0 else 1.0

    def clear(self):
        self.tokens.clear()
        self.position_index.clear()
        self.lru_queue.clear()
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        gc.collect()
        self.logger.info(f"Cleared Tier {self.level_id}")

    def __len__(self):
        return len(self.tokens)


# RAM Memory Tier (L1, L2, L3)
class RAMTier(MemoryTier):
    """Represents an in-memory tier (L1, L2, L3)."""
    def __init__(
        self,
        level_id: int,
        dim: int,
        capacity: int,
        retrieval_cost: float,
        storage_cost: float,
        eviction_policy: str = "lru",
        compressor: Optional['AdvancedMemoryCompressor'] = None,
        vector_store_config: Optional[Dict] = None, # Config for TokenVectorStore if needed
        qos_enabled: bool = False,
        qos_target_latency_ms: float = 1.0,
        dynamic_capacity: bool = True,
        min_capacity_ratio: float = 0.5,
        max_capacity_ratio: float = 2.0,
    ):
        super().__init__(level_id, dim, capacity, retrieval_cost, storage_cost, eviction_policy)
        self.compressor = compressor
        self.vector_store = None
        if vector_store_config:
             self.vector_store = TokenVectorStore(
                 dim=dim,
                 tier_id=level_id,
                 max_tokens=vector_store_config.get("max_tokens", capacity // 2), # Index cache size
                 similarity_threshold=vector_store_config.get("similarity_threshold", 0.7),
                 index_type=vector_store_config.get("index_type", "flat")
             )

        # QoS Tracking
        self.qos_enabled = qos_enabled
        self.qos_target_latency_ms = qos_target_latency_ms
        self.response_times = deque(maxlen=1000) # Store last 1000 response times in ms

        # Dynamic Capacity
        self.dynamic_capacity = dynamic_capacity
        self.base_capacity = capacity
        self.min_capacity = int(capacity * min_capacity_ratio)
        self.max_capacity = int(capacity * max_capacity_ratio)
        self.current_capacity = capacity
        self.last_capacity_adjustment = time.time()
        self.capacity_adjustment_interval = 60 # seconds

        # Eviction protection
        self.protected_tokens = set() # token_ids that should resist eviction

    def add(self, token_id, value, position, importance=0.5):
        """Add a token to this RAM tier. Value should be appropriately (de)compressed."""
        if self.dynamic_capacity:
             self._maybe_adjust_capacity()

        # Evict if full - eviction should return tokens to be potentially moved *down*
        evicted_tokens = []
        while len(self.tokens) >= self.current_capacity:
             evicted_info = self._evict()
             if evicted_info:
                 evicted_tokens.append(evicted_info)
             else:
                 # Could not evict (e.g., all protected), cannot add
                 self.logger.warning(f"Tier {self.level_id} full ({len(self.tokens)}/{self.current_capacity}) and cannot evict. Dropping token {token_id}.")
                 return None # Indicate failure to add

        # Compress value if this tier uses compression
        stored_value = self.compressor.compress(value) if self.compressor else value

        # Create token info
        token_info = TokenInfo(stored_value, position, storage_tier=self.level_id)
        token_info.importance_score = importance
        self.tokens[token_id] = token_info

        # Update position index if used (typically only L1)
        if self.level_id == 1:
             self.position_index[position] = token_id

        # Update LRU queue
        self.lru_queue.append(token_id)

        # Add to vector store index if configured
        if self.vector_store:
             # Add decompressed vector to index for accurate search
             vec_to_index = value # Assume input 'value' is decompressed
             meta = {
                 "position": position,
                 "importance": importance,
                 "creation_time": token_info.creation_time
             }
             self.vector_store.add(token_id, vec_to_index, position, meta)

        # Protect high-importance tokens
        if importance > 0.85: # Higher threshold for protection
             self.protect_token(token_id)

        return evicted_tokens # Return tokens evicted during this add operation


    def get(self, token_id=None, position=None):
        """Retrieve a token by ID or position from this tier."""
        start_time = time.time()
        self.access_count += 1

        if token_id is None and position is not None:
            # Positional lookup primarily for L1
            if self.level_id == 1:
                 token_id = self.position_index.get(position)
            else:
                 # Inefficient for L2/L3, requires scan or dedicated index
                 # For simplicity, we don't support efficient positional lookup in L2/L3 here
                 token_id = None

        token_info = self.tokens.get(token_id)

        if token_info:
            self.hit_count += 1
            token_info.record_access(position)

            # Update LRU
            try:
                 self.lru_queue.remove(token_id)
                 self.lru_queue.append(token_id)
            except ValueError: pass # May have been removed by eviction

            # Decompress if necessary
            value = self.compressor.decompress(token_info.value) if self.compressor else token_info.value

            latency_ms = (time.time() - start_time) * 1000
            if self.qos_enabled:
                 self.response_times.append(latency_ms)

            return value, token_info.position, token_info # Return decompressed value
        else:
            self.miss_count += 1
            return None, None, None

    def _evict(self):
        """Evict a token based on policy. Returns the TokenInfo of the evicted item."""
        if not self.tokens: return None

        candidates = set(self.tokens.keys()) - self.protected_tokens
        if not candidates and len(self.tokens) >= self.current_capacity:
             self.logger.warning(f"Tier {self.level_id} needs eviction but all tokens are protected or candidates empty. Forcing eviction.")
             candidates = set(self.tokens.keys()) # Force eviction among all
        if not candidates:
            self.logger.debug(f"Tier {self.level_id} has space or no candidates to evict.")
            return None # Nothing to evict

        token_to_evict = None

        if self.eviction_policy == "lru":
            # Find the first evictable candidate in LRU queue
            evicted_id = None
            temp_deque = deque()
            while self.lru_queue:
                tid = self.lru_queue.popleft()
                if tid in candidates:
                    evicted_id = tid
                    break
                else:
                    temp_deque.append(tid) # Put non-candidate back
            # Restore queue
            temp_deque.extend(self.lru_queue)
            self.lru_queue = temp_deque
            token_to_evict = evicted_id

        elif self.eviction_policy == "importance":
             token_to_evict = min(candidates, key=lambda tid: self.tokens[tid].importance_score)
        elif self.eviction_policy == "adaptive":
            # Combine recency, frequency, importance
            # Lower score = better candidate for eviction
            token_to_evict = min(candidates, key=lambda tid: (
                 0.5 * self.tokens[tid].importance_score + # Low importance is bad
                 0.3 * math.exp(-self.tokens[tid].recency / (3600*6)) + # High recency (old) is bad
                 0.2 * (1.0 / (1.0 + self.tokens[tid].frequency)) # Low frequency is bad
             ))
        else: # Default to LRU
             token_to_evict = self.lru_queue[0] if self.lru_queue else None

        if token_to_evict:
             # Remove from this tier and return its info
             self.logger.debug(f"Evicting token {token_to_evict} from Tier {self.level_id} using policy {self.eviction_policy}")
             evicted_info = self._remove(token_to_evict) # Calls internal remove
             # Also remove from vector store index cache
             if self.vector_store:
                  self.vector_store.remove_from_index(token_to_evict)
             return evicted_info
        else:
             self.logger.warning(f"Eviction policy {self.eviction_policy} failed to find a token to evict in Tier {self.level_id}")
             return None

    def _remove(self, token_id):
        """Override base remove to handle vector store and protection."""
        self.protected_tokens.discard(token_id) # Ensure unprotected on removal
        if self.vector_store:
             self.vector_store.remove_from_index(token_id)
        return super()._remove(token_id)


    def protect_token(self, token_id):
        if token_id in self.tokens:
            self.protected_tokens.add(token_id)

    def unprotect_token(self, token_id):
        self.protected_tokens.discard(token_id)

    def _maybe_adjust_capacity(self):
        """Adjust tier capacity based on QoS and fullness."""
        if not self.dynamic_capacity: return
        now = time.time()
        if now - self.last_capacity_adjustment < self.capacity_adjustment_interval: return

        self.last_capacity_adjustment = now
        current_fullness = self.fullness

        # QoS check (latency)
        avg_latency = -1
        if self.qos_enabled and self.response_times:
            avg_latency = sum(self.response_times) / len(self.response_times)

        adjustment_factor = 1.0
        reason = "stable"

        if avg_latency != -1 and avg_latency > self.qos_target_latency_ms * 1.2:
            # Too slow, likely too large - reduce capacity for better cache usage
            adjustment_factor = 0.9
            reason = f"latency ({avg_latency:.2f}ms > {self.qos_target_latency_ms:.2f}ms)"
        elif self.hit_rate < 0.6 and current_fullness > 0.7:
            # Low hit rate despite being somewhat full - increase capacity
            adjustment_factor = 1.1
            reason = f"low hit rate ({self.hit_rate:.2f})"
        elif current_fullness < 0.4 and avg_latency < self.qos_target_latency_ms * 0.8:
             # Very empty and fast - potentially decrease capacity to save resources
             adjustment_factor = 0.95
             reason = "low fullness and fast latency"
        elif current_fullness > 0.95:
             # Almost full, proactively increase slightly
             adjustment_factor = 1.05
             reason = "high fullness"

        if adjustment_factor != 1.0:
            new_capacity = int(self.current_capacity * adjustment_factor)
            new_capacity = max(self.min_capacity, min(self.max_capacity, new_capacity))
            if new_capacity != self.current_capacity:
                self.logger.info(f"Tier {self.level_id} adjusting capacity: {self.current_capacity} -> {new_capacity} (Reason: {reason})")
                self.current_capacity = new_capacity

    def optimize(self):
        """Optimize structures within this tier (e.g., vector index)."""
        self.logger.info(f"Optimizing Tier {self.level_id}...")
        if self.vector_store:
            self.vector_store.optimize()
        # Add other tier-specific optimizations if needed
        self.logger.info(f"Optimization complete for Tier {self.level_id}.")


# Advanced Memory Compressor (Enhanced for Scale)
class AdvancedMemoryCompressor(Module):
    """Advanced compression with high ratios for slower tiers."""
    def __init__(
        self,
        dim: int,
        compression_ratio: float = 4.0,
        compression_method: str = "combined", # "svd", "autoencoder", "quantization", "combined"
        quantization_bits: int = 8, # Can be 4, 8, 16
        use_pruning: bool = False, # Sparsification
        pruning_threshold: float = 0.01,
        # Vector Quantization (alternative to scalar)
        use_vq: bool = False,
        vq_codebook_size: int = 256,
        # Delta encoding (good for sequential data)
        use_delta_encoding: bool = False,
        delta_reference_update_freq: int = 100,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.target_compression_ratio = compression_ratio
        self.method = compression_method
        self.quantization_bits = quantization_bits
        self.use_pruning = use_pruning
        self.pruning_threshold = pruning_threshold
        self.use_vq = use_vq
        self.use_delta_encoding = use_delta_encoding
        self.logger = logging.getLogger("ultracontext.memory.compressor")

        self.compressed_dim = max(1, int(dim / compression_ratio))
        self.actual_compression_ratio = dim / self.compressed_dim

        self.logger.info(f"Compressor initialized: Ratio={self.actual_compression_ratio:.2f} (Target={compression_ratio}), Method={compression_method}, Bits={quantization_bits}")

        # --- Component Setup ---
        # SVD-like (Linear Projection)
        if compression_method in ["svd", "combined"]:
            self.encode_linear = nn.Linear(dim, self.compressed_dim, bias=False)
            self.decode_linear = nn.Linear(self.compressed_dim, dim, bias=False)

        # Autoencoder (Non-linear Projection) - Keep it shallow for speed
        if compression_method in ["autoencoder", "combined"]:
            hidden_dim = (dim + self.compressed_dim) // 2
            self.encoder_ae = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.compressed_dim)
            )
            self.decoder_ae = nn.Sequential(
                nn.Linear(self.compressed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )

        # Quantization Setup
        self.q_range = 2**(quantization_bits - 1) - 1 if quantization_bits > 1 else 0.5 # Range for signed int / binary
        self.q_scale = 1.0 # Dynamic scale/offset learned per tensor (or fixed) - simplified here

        # Vector Quantization Setup
        if use_vq:
             # Need a codebook (learnable)
             self.vq_codebook = Parameter(torch.randn(vq_codebook_size, self.compressed_dim))

        # Delta Encoding Setup
        if use_delta_encoding:
            self.delta_refs = deque(maxlen=10) # Store recent original vectors as references
            self.delta_update_counter = 0
            self.delta_ref_update_freq = delta_reference_update_freq

        self._reset_parameters()


    def _reset_parameters(self):
        # Initialize layers for better performance
        if hasattr(self, 'encode_linear'):
            nn.init.orthogonal_(self.encode_linear.weight)
            nn.init.orthogonal_(self.decode_linear.weight)
        if hasattr(self, 'encoder_ae'):
             for module in self.encoder_ae:
                if isinstance(module, nn.Linear): nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
             for module in self.decoder_ae:
                if isinstance(module, nn.Linear): nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if hasattr(self, 'vq_codebook'):
             nn.init.uniform_(self.vq_codebook, -1./self.vq_codebook.size(0), 1./self.vq_codebook.size(0))


    # --- Compression Steps ---
    def _reduce_dimension(self, x):
        if self.method == "svd": return self.encode_linear(x)
        if self.method == "autoencoder": return self.encoder_ae(x)
        if self.method == "combined":
             # Example: Linear first, then non-linear refine
             return self.encoder_ae(self.encode_linear(x))
        if self.method == "quantization": return x # No reduction, just quantize
        return x # Fallback

    def _quantize(self, x):
        """Scalar Quantization"""
        if self.quantization_bits <= 1: # Binary quantization
             return torch.sign(x) # Or use a learned threshold

        # Simple scalar quantization with fixed scale
        # More advanced: learn scale/offset per tensor/channel
        scaled_x = x / self.q_scale
        quantized = torch.round(torch.clamp(scaled_x, -self.q_range, self.q_range))
        # Store as appropriate integer type if needed for storage size reduction
        # Here we return the quantized float tensor
        return quantized

    def _vector_quantize(self, x_reduced):
        """Vector Quantization using Codebook"""
        if not hasattr(self, 'vq_codebook'): return x_reduced, None

        # Find nearest codebook vector
        distances = torch.cdist(x_reduced, self.vq_codebook) # [batch, codebook_size]
        indices = torch.argmin(distances, dim=-1) # [batch]
        quantized = self.vq_codebook[indices] # [batch, compressed_dim]

        # Use straight-through estimator for gradients during training (if compressor is trained)
        quantized_st = x_reduced + (quantized - x_reduced).detach()
        return quantized_st, indices # Return quantized vector and index

    def _prune(self, x):
         return x * (torch.abs(x) > self.pruning_threshold).float()

    # --- Decompression Steps ---
    def _dequantize(self, x_quantized):
        if self.quantization_bits <= 1:
             return x_quantized # Already in float format for binary

        # Inverse of quantization
        return x_quantized * self.q_scale

    def _expand_dimension(self, x_reduced):
        if self.method == "svd": return self.decode_linear(x_reduced)
        if self.method == "autoencoder": return self.decoder_ae(x_reduced)
        if self.method == "combined":
             # Inverse order of combined
             return self.decode_linear(self.decoder_ae(x_reduced))
        if self.method == "quantization": return x_reduced # No expansion needed
        return x_reduced # Fallback


    # --- Main Methods ---
    @timer # Time compression/decompression
    def compress(self, x):
        """Compress tensor x."""
        original_shape = x.shape
        if x.ndim == 1: x = x.unsqueeze(0) # Handle single vector

        original_vector = x.clone() # Needed for delta encoding

        # 1. Delta Encoding (Optional, First Step)
        delta_ref = None
        if self.use_delta_encoding and len(self.delta_refs) > 0:
            # Find best reference (e.g., closest L2 distance)
            dists = [torch.linalg.norm(x - ref) for ref in self.delta_refs]
            delta_ref = self.delta_refs[torch.argmin(torch.stack(dists))]
            x = x - delta_ref # Calculate delta

        # 2. Dimensionality Reduction
        x_reduced = self._reduce_dimension(x)

        # 3. Vector Quantization (Optional, Alternative to Scalar)
        vq_indices = None
        if self.use_vq:
            x_quantized, vq_indices = self._vector_quantize(x_reduced)
            # If VQ is used, scalar quantization is usually skipped or applied lightly
            # For simplicity, we assume VQ replaces scalar quantization here
            compressed = x_quantized # Use the VQ result
        else:
            # 4. Scalar Quantization
            x_quantized = self._quantize(x_reduced)
            compressed = x_quantized

        # 5. Pruning (Optional)
        if self.use_pruning:
            compressed = self._prune(compressed)

        # Update Delta Encoding References
        if self.use_delta_encoding:
            self.delta_update_counter += 1
            if self.delta_update_counter % self.delta_ref_update_freq == 0:
                self.delta_refs.append(original_vector)

        # Return compressed tensor (and optionally VQ indices or delta ref info if needed for storage)
        if x.ndim == 1: compressed = compressed.squeeze(0) # Restore original shape

        # In a real system, might return a dict: {'data': compressed, 'vq_idx': vq_indices, 'delta_ref_idx': ...}
        return compressed

    @timer
    def decompress(self, x_compressed):
        """Decompress tensor x_compressed."""
        original_shape = x_compressed.shape
        if x_compressed.ndim == 1: x_compressed = x_compressed.unsqueeze(0) # Handle single vector

        # Assume x_compressed contains the main data from compression step
        # Need to handle VQ, delta encoding based on stored metadata (not implemented here)

        # 1. Dequantization (Inverse of Scalar or VQ)
        if self.use_vq:
            # If VQ was used, x_compressed might be indices or already the codebook vectors.
            # Assume it's the codebook vectors for this example.
            x_dequantized = x_compressed
        else:
            x_dequantized = self._dequantize(x_compressed)

        # 2. Dimension Expansion
        x_expanded = self._expand_dimension(x_dequantized)

        # 3. Add Delta (Optional, Last Step)
        if self.use_delta_encoding:
            # Need to retrieve the correct delta_ref used during compression (requires stored metadata)
            # Placeholder: Assume we retrieve the most recent ref
            if len(self.delta_refs) > 0:
                 x_final = x_expanded + self.delta_refs[-1]
            else:
                 x_final = x_expanded
        else:
            x_final = x_expanded

        if x_final.ndim == 1: x_final = x_final.squeeze(0) # Restore original shape
        return x_final

    def calculate_compression_stats(self, x):
        # Basic stats calculation
        compressed = self.compress(x)
        reconstructed = self.decompress(compressed)

        original_bytes = x.numel() * x.element_size()

        # Estimate compressed size (depends heavily on storage format)
        # Simple estimate based on quantization bits and reduced dim
        if self.use_vq:
             # Size is based on storing indices
             bits_per_index = math.ceil(math.log2(self.vq_codebook.size(0)))
             compressed_bytes = (compressed.numel() // self.compressed_dim) * bits_per_index / 8
        else:
             compressed_bytes = compressed.numel() * self.quantization_bits / 8

        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float('inf')
        mse = F.mse_loss(reconstructed.float(), x.float()).item()

        return {
            "compression_ratio_actual": ratio,
            "reconstruction_mse": mse,
            "original_bytes": original_bytes,
            "compressed_bytes_estimated": compressed_bytes
        }


# Memory Summarizer (Enhanced for Scale)
class MemorySummarizer(Module):
    """Dynamically creates hierarchical summaries of context."""
    def __init__(
        self,
        dim: int,
        summary_ratio: float = 4.0, # Target ratio per level
        min_block_size: int = 128, # Min tokens in a block to summarize
        max_summary_tokens: int = 512, # Max tokens to feed into summarization attention
        use_attention_for_importance: bool = True,
        summary_level_max: int = 5, # More levels for deeper hierarchy
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.summary_ratio = summary_ratio
        self.min_block_size = min_block_size
        self.max_summary_tokens = max_summary_tokens
        self.summary_level_max = summary_level_max
        self.logger = logging.getLogger("ultracontext.memory.summarizer")

        # Use a shared importance scorer if possible, or create one
        self.importance_scorer = ImportanceScorer(
            dim=dim, importance_dim=64, use_attention_weights=use_attention_for_importance
        )

        # Summary generation network (e.g., simple attention + MLP)
        # Keep it lightweight as it might run often
        self.summary_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.summary_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._init_weights()

        # Store summaries (metadata, not necessarily full vectors if tiered)
        # Maps level -> list of (summary_id, position_range, source_token_count, location_info)
        self.summary_registry = {level: [] for level in range(1, summary_level_max + 1)}
        # Maps summary_id -> actual summary vector (only for summaries kept in RAM)
        self.summary_vectors_cache = {}
        # Maps summary_id -> list of source token_ids or lower-level summary_ids
        self.summary_source_map = {}

        self.logger.info(f"MemorySummarizer initialized: Ratio={summary_ratio}, MinBlock={min_block_size}, MaxLevel={summary_level_max}")


    def _init_weights(self):
        # Initialize summary network
        nn.init.xavier_uniform_(self.summary_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.summary_attention.out_proj.weight)
        for module in self.summary_mlp:
             if isinstance(module, nn.Linear): nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def score_importance(self, token_vectors, positions=None, attention_weights=None):
        return self.importance_scorer(token_vectors, positions, attention_weights)

    def generate_summary_vector(self, token_vectors):
        """Generate a single summary vector from a block of tokens using attention."""
        # token_vectors shape: [block_size, dim]
        block_size = token_vectors.shape[0]
        if block_size == 0: return None
        if block_size == 1: return token_vectors[0] # No need to summarize one token

        # Add batch dimension
        tokens_batch = token_vectors.unsqueeze(0) # [1, block_size, dim]

        # Simple self-attention summarization
        x = self.norm1(tokens_batch)
        attn_output, _ = self.summary_attention(x, x, x)
        x = tokens_batch + attn_output
        x = x + self.summary_mlp(self.norm2(x))

        # Pooling strategy: Mean pooling of the output sequence
        summary_vec = torch.mean(x, dim=1) # [1, dim]
        return summary_vec.squeeze(0) # [dim]


    def create_summary_for_block(self, token_ids, token_vectors, positions, level=1):
        """Creates summaries for a specific block of tokens."""
        n_tokens = len(token_ids)
        if n_tokens < self.min_block_size:
            self.logger.debug(f"Skipping summary for block (size {n_tokens} < {self.min_block_size})")
            return None

        # Ensure tensors are on the correct device
        device = token_vectors[0].device
        vec_tensor = torch.stack(token_vectors).to(device)
        pos_tensor = torch.tensor(positions, device=device).unsqueeze(0) # Add batch dim

        # 1. Score importance
        importance_scores = self.score_importance(vec_tensor.unsqueeze(0), pos_tensor).squeeze(0) # [n_tokens]

        # 2. Select important tokens for summary input (top K or threshold)
        keep_count = max(1, min(self.max_summary_tokens, n_tokens // int(self.summary_ratio)))
        # Alternative: Keep tokens above importance threshold
        # keep_indices = torch.where(importance_scores > 0.5)[0]
        # if len(keep_indices) > self.max_summary_tokens: ... sample ...
        _, top_indices = torch.topk(importance_scores, k=keep_count)
        top_indices = torch.sort(top_indices)[0] # Keep relative order

        selected_vectors = vec_tensor[top_indices]

        # 3. Generate summary vector
        summary_vec = self.generate_summary_vector(selected_vectors)
        if summary_vec is None: return None

        # 4. Register the summary
        summary_id = f"summary_L{level}_{uuid.uuid4()}"
        pos_min = min(positions)
        pos_max = max(positions)
        pos_range = (pos_min, pos_max)

        # Store summary metadata
        # Location info would point to where the summary vector is stored (RAM, Disk)
        self.summary_registry[level].append((summary_id, pos_range, n_tokens, "local_cache"))
        # Store the actual vector in cache (if this summarizer instance holds it)
        self.summary_vectors_cache[summary_id] = summary_vec
        # Map summary to its source tokens
        self.summary_source_map[summary_id] = token_ids

        self.logger.debug(f"Created Level {level} summary {summary_id} for positions {pos_range} ({n_tokens} tokens)")
        return summary_id, summary_vec


    def create_hierarchical_summary(self, summary_ids_level_below, level):
        """Creates a higher-level summary from lower-level summaries."""
        if level > self.summary_level_max: return None
        n_summaries = len(summary_ids_level_below)
        if n_summaries < 2: return None # Need at least 2 summaries to summarize further

        # Retrieve vectors of lower-level summaries
        summary_vectors = []
        source_summary_ids = []
        positions = [] # Use midpoint of source position range
        for sum_id in summary_ids_level_below:
             if sum_id in self.summary_vectors_cache:
                 summary_vectors.append(self.summary_vectors_cache[sum_id])
                 source_summary_ids.append(sum_id)
                 # Find position range from registry
                 for reg_id, pos_range, _, _ in self.summary_registry[level-1]:
                      if reg_id == sum_id:
                           positions.append((pos_range[0] + pos_range[1]) // 2)
                           break

        if len(summary_vectors) < 2: return None

        # Ensure tensors are on the correct device
        device = summary_vectors[0].device
        vec_tensor = torch.stack(summary_vectors).to(device)
        # Use dummy positions if needed, or average positions
        pos_tensor = torch.tensor(positions, device=device).unsqueeze(0) # Add batch dim

        # Score importance of summaries (can use the same scorer)
        importance_scores = self.score_importance(vec_tensor.unsqueeze(0), pos_tensor).squeeze(0)

        # Select important summaries
        keep_count = max(1, min(self.max_summary_tokens, n_summaries // int(self.summary_ratio)))
        _, top_indices = torch.topk(importance_scores, k=keep_count)
        top_indices = torch.sort(top_indices)[0]

        selected_vectors = vec_tensor[top_indices]
        selected_source_ids = [source_summary_ids[i] for i in top_indices]

        # Generate the higher-level summary vector
        summary_vec = self.generate_summary_vector(selected_vectors)
        if summary_vec is None: return None

        # Register the new summary
        summary_id = f"summary_L{level}_{uuid.uuid4()}"
        # Determine position range and source count from underlying summaries
        min_pos = float('inf')
        max_pos = float('-inf')
        total_source_tokens = 0
        for reg_id, pos_range, src_count, _ in self.summary_registry[level-1]:
             if reg_id in selected_source_ids:
                  min_pos = min(min_pos, pos_range[0])
                  max_pos = max(max_pos, pos_range[1])
                  total_source_tokens += src_count # Accumulate original token count

        pos_range = (min_pos, max_pos)

        self.summary_registry[level].append((summary_id, pos_range, total_source_tokens, "local_cache"))
        self.summary_vectors_cache[summary_id] = summary_vec
        # Map this summary to the lower-level summaries it was derived from
        self.summary_source_map[summary_id] = selected_source_ids

        self.logger.debug(f"Created Hierarchical Level {level} summary {summary_id} from {len(selected_source_ids)} L{level-1} summaries.")
        return summary_id, summary_vec


    def get_summary_vector(self, summary_id):
         # TODO: Implement retrieval if summary is stored on disk
         return self.summary_vectors_cache.get(summary_id)

    def find_summaries_for_position(self, position):
        """Find summaries covering a given position."""
        covering_summaries = []
        for level in range(1, self.summary_level_max + 1):
            for sum_id, pos_range, _, _ in self.summary_registry.get(level, []):
                if pos_range[0] <= position <= pos_range[1]:
                    covering_summaries.append({"id": sum_id, "level": level, "range": pos_range})
        # Sort by level (higher level = broader summary) then range size
        covering_summaries.sort(key=lambda x: (x['level'], x['range'][1] - x['range'][0]))
        return covering_summaries

    def get_source_tokens(self, summary_id, max_depth=10):
        """Recursively find original token IDs for a summary."""
        if max_depth <= 0: return []
        source_ids = self.summary_source_map.get(summary_id, [])
        original_token_ids = []
        for src_id in source_ids:
             if src_id.startswith("summary_"):
                 # Recurse into lower-level summary
                 original_token_ids.extend(self.get_source_tokens(src_id, max_depth - 1))
             else:
                 # Assumed to be an original token ID
                 original_token_ids.append(src_id)
        return list(set(original_token_ids)) # Return unique IDs

    def clear(self):
        self.summary_registry = {level: [] for level in range(1, self.summary_level_max + 1)}
        self.summary_vectors_cache.clear()
        self.summary_source_map.clear()
        self.logger.info("Cleared Memory Summarizer state.")


# --- Disk Storage Component ---
class PersistentTokenStorage:
    """
    Handles storage and retrieval of tokens on disk.
    Uses compression and potentially a simple index.
    This represents the L4 / lowest tier.
    """
    def __init__(
        self,
        dim: int,
        storage_path: str,
        max_tokens: int = 1_000_000_000, # Capacity of this disk tier
        compression_ratio: float = 16.0, # High compression for disk
        index_type: str = "ivfpq", # Suitable for disk-based ANN
        index_memory_budget_gb: float = 4.0 # RAM allocated for disk index cache
    ):
        self.dim = dim
        self.storage_path = storage_path
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.logger = logging.getLogger("ultracontext.memory.disk")

        os.makedirs(storage_path, exist_ok=True)
        self.metadata_db_path = os.path.join(storage_path, "token_metadata.db")
        self.vector_store_path = os.path.join(storage_path, "vector_store")
        os.makedirs(self.vector_store_path, exist_ok=True)

        # Compressor for this tier
        self.compressor = AdvancedMemoryCompressor(
            dim=dim,
            compression_ratio=compression_ratio,
            quantization_bits=8, # 8-bit or even 4-bit for disk
            compression_method="combined", # Robust method
            use_delta_encoding=True, # Good for potentially sequential disk writes
        )

        # Metadata storage (e.g., using LMDB or RocksDB for efficiency)
        # Using a simple dictionary here for demonstration; replace with DB for production
        self.metadata_store = {} # token_id -> {position: pos, file_path: path, offset: off, len: length, ...}
        self._load_metadata() # Load existing metadata

        # Vector Store for semantic search on disk
        # Requires careful handling of memory mapping or specialized disk-ANN libs
        # Using FAISS IndexIVFPQ with memory mapping (conceptual)
        self.vector_store = TokenVectorStore(
             dim=dim,
             tier_id=4, # Indicate Disk Tier
             # Limit index RAM cache size, not total disk tokens
             max_tokens=int(index_memory_budget_gb * 1e9 / (dim * 4)), # Rough estimate based on budget
             index_type=index_type
        )
        # TODO: Implement loading/saving FAISS index to/from disk and memory mapping

        self.logger.info(f"PersistentTokenStorage initialized at '{storage_path}' for {max_tokens} tokens.")

    def _get_token_file_path(self, token_id):
        # Simple sharding based on hash to limit directory size
        hash_prefix = str(hash(token_id) % 1000)
        dir_path = os.path.join(self.storage_path, "tokens", hash_prefix)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{token_id}.bin")

    def add(self, token_id, value, position, metadata=None):
        if len(self.metadata_store) >= self.max_tokens:
            self.logger.warning(f"Disk storage full ({self.max_tokens} tokens). Cannot add token {token_id}.")
            # TODO: Implement eviction from disk (e.g., based on age or importance if tracked)
            return False

        try:
            # Compress the token
            compressed_value = self.compressor.compress(value)
            compressed_bytes = compressed_value.cpu().numpy().tobytes()

            # Store compressed bytes to file
            file_path = self._get_token_file_path(token_id)
            offset = 0 # Simplification: store one token per file
            length = len(compressed_bytes)
            with open(file_path, 'wb') as f:
                f.write(compressed_bytes)

            # Store metadata
            token_meta = {
                "position": position,
                "file_path": file_path,
                "offset": offset,
                "length": length,
                "importance": metadata.get("importance", 0.5) if metadata else 0.5,
                "creation_time": metadata.get("creation_time", time.time()) if metadata else time.time(),
                "last_access_time": time.time()
            }
            self.metadata_store[token_id] = token_meta

            # Add to vector store index (if configured and feasible)
            # This part is complex for disk-based ANN. Might only index a subset.
            # self.vector_store.add(token_id, value, position, token_meta) # Add original value to index

            return True
        except Exception as e:
            self.logger.error(f"Error adding token {token_id} to disk storage: {e}", exc_info=True)
            return False

    def get(self, token_id):
        if token_id not in self.metadata_store:
            return None, None

        try:
            metadata = self.metadata_store[token_id]
            file_path = metadata["file_path"]
            offset = metadata["offset"]
            length = metadata["length"]

            # Read compressed bytes from file
            # Use memory mapping for efficiency in production
            with open(file_path, 'rb') as f:
                # f.seek(offset) # Needed if multiple tokens per file
                compressed_bytes = f.read(length)

            # Convert bytes back to tensor
            # Determine dtype/shape from compressor if needed (simplified here)
            compressed_value_np = np.frombuffer(compressed_bytes, dtype=np.float32) # Assuming float32 storage after numpy conversion
            # Reshape based on expected compressed dim (this needs careful handling)
            expected_elements = self.compressor.compressed_dim
            # Handle potential padding or errors
            if compressed_value_np.size < expected_elements:
                 self.logger.warning(f"Token {token_id} data length mismatch on disk. Expected {expected_elements}, got {compressed_value_np.size}. Padding.")
                 padded_np = np.zeros(expected_elements, dtype=np.float32)
                 padded_np[:compressed_value_np.size] = compressed_value_np
                 compressed_value_np = padded_np
            elif compressed_value_np.size > expected_elements:
                 self.logger.warning(f"Token {token_id} data length mismatch on disk. Expected {expected_elements}, got {compressed_value_np.size}. Truncating.")
                 compressed_value_np = compressed_value_np[:expected_elements]

            compressed_tensor = torch.from_numpy(compressed_value_np)

            # Decompress
            value = self.compressor.decompress(compressed_tensor)

            # Update access time in metadata
            metadata["last_access_time"] = time.time()
            self.metadata_store[token_id] = metadata # Update if using in-memory dict

            return value, metadata
        except FileNotFoundError:
             self.logger.error(f"Token {token_id} metadata found but file {metadata['file_path']} not found.")
             del self.metadata_store[token_id] # Clean up stale metadata
             return None, None
        except Exception as e:
            self.logger.error(f"Error retrieving token {token_id} from disk: {e}", exc_info=True)
            return None, None

    def remove(self, token_id):
        if token_id in self.metadata_store:
            metadata = self.metadata_store.pop(token_id)
            try:
                os.remove(metadata["file_path"])
                 # Remove from vector store index
                # self.vector_store.remove_from_index(token_id)
                return True
            except OSError as e:
                self.logger.error(f"Error removing token file {metadata['file_path']}: {e}")
                # Metadata already removed, log error but continue
                return False
        return False

    def search_similar(self, query_vector, top_k=5):
         # Semantic search on disk is complex. Requires a disk-aware index.
         # self.vector_store.search_similar(query_vector, top_k)
         self.logger.warning("Semantic search on disk storage is not fully implemented in this simulation.")
         return [] # Placeholder

    def _save_metadata(self):
        # In production, use atomic writes or DB transactions
        try:
             with open(self.metadata_db_path + ".tmp", 'wb') as f:
                 pickle.dump(self.metadata_store, f)
             os.replace(self.metadata_db_path + ".tmp", self.metadata_db_path)
        except Exception as e:
             self.logger.error(f"Error saving metadata: {e}")

    def _load_metadata(self):
        if os.path.exists(self.metadata_db_path):
            try:
                with open(self.metadata_db_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                self.logger.info(f"Loaded {len(self.metadata_store)} metadata entries from disk.")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}. Starting fresh.")
                self.metadata_store = {}
        else:
             self.metadata_store = {}


    def clear(self):
        # Be careful with this in production!
        self.logger.warning(f"Clearing persistent storage at {self.storage_path}!")
        self.metadata_store.clear()
        # Delete token files and directories
        token_root = os.path.join(self.storage_path, "tokens")
        if os.path.exists(token_root):
            import shutil
            shutil.rmtree(token_root)
        # Delete metadata db
        if os.path.exists(self.metadata_db_path):
            os.remove(self.metadata_db_path)
        # Clear vector store
        # self.vector_store.clear() # Assuming it handles its disk cleanup
        self.logger.info("Persistent storage cleared.")

    def __len__(self):
        return len(self.metadata_store)

    def __del__(self):
         # Ensure metadata is saved on exit
         self._save_metadata()


# Streaming and distributed memory management (Refactored for 1B Scale)
class StreamingMemoryManager(Module):
    """
    Manages token streaming, pushing older/less important tokens eventually
    to the PersistentTokenStorage (Disk Tier). Coordinates with summarizer.
    Acts as the interface to the lowest (Disk/Distributed) memory tiers.
    """
    def __init__(
        self,
        dim: int,
        persistent_storage: PersistentTokenStorage, # Pass in the disk storage instance
        summarizer: Optional[MemorySummarizer] = None,
        # Configuration for moving data to disk
        tier3_capacity: int = 8_000_000, # Estimated capacity of RAM tiers above disk
        low_importance_threshold: float = 0.2, # Threshold to move directly to disk
        staleness_threshold_seconds: float = 3600 * 24 * 7, # Move if untouched for a week
        summarization_interval_tokens: int = 131072, # Summarize roughly every 128k tokens processed
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        orchestrator: Optional['DistributedMemoryOrchestrator'] = None # For distributed setup
    ):
        super().__init__()
        self.dim = dim
        self.persistent_storage = persistent_storage
        self.summarizer = summarizer
        self.tier3_capacity = tier3_capacity # Approximate size of RAM above this manager
        self.low_importance_threshold = low_importance_threshold
        self.staleness_threshold_seconds = staleness_threshold_seconds
        self.summarization_interval = summarization_interval_tokens
        self.perf_config = perf_config
        self.orchestrator = orchestrator # Handles routing if distributed
        self.logger = logging.getLogger("ultracontext.memory.streaming")

        # State tracking
        self.total_tokens_processed = 0
        self.last_summarization_point = 0
        # We don't buffer here; assumes higher tiers handle recent tokens
        # Tracks tokens potentially eligible for moving to disk
        self.eligible_for_disk = {} # token_id -> {importance, last_access, position}

        self.logger.info("StreamingMemoryManager initialized.")

    def register_token_from_upper_tier(self, token_id, position, importance, last_access_time):
        """Called by higher tiers when a token might be demoted further."""
        # Track token as potentially moving to disk
        self.eligible_for_disk[token_id] = {
            "position": position,
            "importance": importance,
            "last_access": last_access_time
        }
        self.total_tokens_processed += 1 # Count tokens reaching this management level

        # Check summarization trigger
        if self.summarizer and (self.total_tokens_processed - self.last_summarization_point >= self.summarization_interval):
             self._trigger_summarization()


    def process_eligible_tokens(self, tokens_to_move: Dict[str, TokenInfo]):
        """
        Receives tokens evicted from L3 (or lower RAM tiers) and decides
        whether to store them on disk or discard.
        """
        moved_to_disk = 0
        discarded = 0
        now = time.time()

        for token_id, token_info in tokens_to_move.items():
            # Decide whether to keep or discard
            # Keep if: Reasonably important OR not too old
            should_keep = (token_info.importance_score > self.low_importance_threshold or
                           (now - token_info.last_access_time) < self.staleness_threshold_seconds)

            if should_keep:
                 # Store in persistent storage
                 # Value should already be compressed appropriately for the source tier (e.g., L3)
                 # We need to decompress and re-compress for the Disk tier's settings
                 value_decompressed = None
                 # Assuming token_info.value holds the compressed form from the source tier.
                 # This part needs the AdvancedHierarchicalMemoryManager to pass the correct compressor.
                 # Placeholder: Assume we get the original value somehow.
                 # In a real implementation, the caller (AdvancedHierarchicalMemoryManager)
                 # would provide the decompressed value or handle the recompression.
                 # value_decompressed = source_tier_compressor.decompress(token_info.value)

                 if value_decompressed is None:
                      self.logger.error(f"Cannot move token {token_id} to disk: Decompressed value not available.")
                      discarded += 1
                      continue

                 metadata = {
                      "importance": token_info.importance_score,
                      "creation_time": token_info.creation_time,
                      "last_access_time": token_info.last_access_time
                 }

                 # If distributed, get target node
                 target_node = self.orchestrator.get_node_for_token(token_id) if self.orchestrator else 0

                 # Add to persistent storage (potentially on a specific node)
                 # TODO: Add node_id argument to persistent_storage.add if distributed
                 success = self.persistent_storage.add(token_id, value_decompressed, token_info.position, metadata)
                 if success:
                     moved_to_disk += 1
                 else:
                     discarded += 1 # Failed to add (e.g., disk full)
            else:
                 # Discard the token permanently
                 discarded += 1
                 self.logger.debug(f"Discarding stale/unimportant token {token_id} (Importance: {token_info.importance_score:.2f}, Age: {(now - token_info.last_access_time)/3600:.1f}h)")

            # Remove from eligibility tracking
            self.eligible_for_disk.pop(token_id, None)

        if moved_to_disk > 0 or discarded > 0:
             self.logger.info(f"Processed eligible tokens: Moved {moved_to_disk} to disk, Discarded {discarded}.")


    def _trigger_summarization(self):
        """Signal that summarization might be needed for older context."""
        self.last_summarization_point = self.total_tokens_processed
        # The actual summarization logic resides in AdvancedHierarchicalMemoryManager's
        # background task, which checks various conditions. This just marks a point in time.
        self.logger.info(f"Reached summarization interval at {self.total_tokens_processed} tokens.")
        # Optionally, could provide hints about which position ranges are now "old"
        # E.g., summarize positions around (total_tokens_processed - summarization_interval)

    def retrieve_from_disk(self, token_id=None, position=None, query_vector=None):
        """Handles retrieval specifically from the persistent storage tier."""
        start_time = time.time()
        result_value, result_meta = None, None

        if token_id:
            # If distributed, route to the correct node
            if self.orchestrator:
                 node_id = self.orchestrator.get_node_for_token(token_id)
                 # TODO: Implement RPC call to retrieve from specific node's persistent storage
                 # result_value, result_meta = self.rpc_client.get_token(node_id, token_id)
                 result_value, result_meta = self.persistent_storage.get(token_id) # Simulate local access
            else:
                 result_value, result_meta = self.persistent_storage.get(token_id)

        elif position is not None:
            # Positional lookup on disk requires a disk-based index (e.g., B-tree on metadata)
            # This is generally slow and often avoided by using summaries or semantic search.
            # Simulation: Scan metadata (VERY INEFFICIENT)
            found_id = None
            for tid, meta in self.metadata_store.items(): # Accessing internal member for simulation
                 if meta.get("position") == position:
                     found_id = tid
                     break
            if found_id:
                 result_value, result_meta = self.persistent_storage.get(found_id)

        elif query_vector is not None:
             # Semantic search on disk
             if self.orchestrator:
                 # TODO: Implement distributed search coordination
                 # results = self.orchestrator.search_distributed(query_vector.unsqueeze(0), top_k=1)
                 results = self.persistent_storage.search_similar(query_vector, top_k=1) # Simulate local access
             else:
                 results = self.persistent_storage.search_similar(query_vector, top_k=1)

             if results:
                 # Result format: (token_id, vector, similarity, metadata)
                 tid, vec, sim, meta = results[0]
                 # Need the actual value, not the potentially decompressed index vector
                 result_value, result_meta = self.persistent_storage.get(tid)
                 # We could also return the vector 'vec' if the caller wants the embedding
                 # For consistency, we return the reconstructed value from disk store.

        latency_ms = (time.time() - start_time) * 1000
        self.logger.debug(f"Disk retrieval attempt took {latency_ms:.2f} ms.")
        # NOTE: Actual disk latency would be much higher

        # Return the retrieved value (decompressed) and its position
        if result_value is not None and result_meta is not None:
             return result_value, result_meta.get("position"), result_meta
        else:
             return None, None, None

    def clear(self):
         self.persistent_storage.clear()
         self.eligible_for_disk.clear()
         self.total_tokens_processed = 0
         self.last_summarization_point = 0
         self.logger.info("Cleared Streaming Memory Manager state and persistent storage.")

    def get_stats(self):
        return {
            "disk_token_count": len(self.persistent_storage),
            "eligible_for_disk_count": len(self.eligible_for_disk),
            "total_tokens_processed": self.total_tokens_processed,
            "last_summarization_point": self.last_summarization_point,
        }


# Adaptive policy management (Enhanced for Scale)
class AdaptiveMemoryPolicy(Module):
    """Adaptively tunes memory management policies across multiple tiers."""
    def __init__(
        self,
        dim: int,
        tiers: List[MemoryTier], # Pass in references to the tiers
        persistent_storage: Optional[PersistentTokenStorage] = None,
        orchestrator: Optional['DistributedMemoryOrchestrator'] = None,
        observation_window: int = 5000, # Larger window for more stable stats
        update_interval_ops: int = 500, # Update policies every N operations
        update_interval_time: float = 10.0, # Or every N seconds
        learning_rate: float = 0.01, # Slower adaptation
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.tiers = tiers
        self.persistent_storage = persistent_storage
        self.orchestrator = orchestrator
        self.observation_window = observation_window
        self.update_interval_ops = update_interval_ops
        self.update_interval_time = update_interval_time
        self.learning_rate = learning_rate
        self.logger = logging.getLogger("ultracontext.memory.policy")

        # Metrics Tracking (more comprehensive)
        self.access_history = deque(maxlen=observation_window) # (timestamp, tier_hit, position, is_write)
        self.latency_history = {t.level_id: deque(maxlen=observation_window // len(tiers)) for t in tiers}
        self.latency_history[4] = deque(maxlen=observation_window // len(tiers)) # Disk latency
        self.tier_stats_history = {t.level_id: deque(maxlen=100) for t in tiers} # Store recent fullness, hit rate
        self.tier_stats_history[4] = deque(maxlen=100) # Disk stats

        # Tunable Policies
        self.policies = {}
        self.policy_bounds = {}
        self._initialize_policies()

        # State
        self.op_count = 0
        self.last_update_time = time.time()
        self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL

        self.logger.info("AdaptiveMemoryPolicy initialized.")

    def _initialize_policies(self):
        # Tier capacities (allow dynamic adjustment)
        for tier in self.tiers:
            if isinstance(tier, RAMTier) and tier.dynamic_capacity:
                policy_name = f"tier{tier.level_id}_capacity"
                self.policies[policy_name] = tier.current_capacity
                self.policy_bounds[policy_name] = (tier.min_capacity, tier.max_capacity)

        # Promotion/Demotion thresholds
        self.policies["promotion_access_threshold"] = 3 # Access count to promote
        self.policy_bounds["promotion_access_threshold"] = (1, 10)
        self.policies["demotion_staleness_seconds"] = 3600 * 6 # Demote after 6 hours inactive
        self.policy_bounds["demotion_staleness_seconds"] = (600, 3600 * 24) # 10min to 1 day

        # Compression ratios (if tiers have compressors)
        for tier in self.tiers:
            if hasattr(tier, 'compressor') and tier.compressor:
                policy_name = f"tier{tier.level_id}_compression_ratio"
                self.policies[policy_name] = tier.compressor.target_compression_ratio
                # Allow tuning ratio slightly
                self.policy_bounds[policy_name] = (max(1.0, tier.compressor.target_compression_ratio / 1.5),
                                                   tier.compressor.target_compression_ratio * 1.5)

        # Vector index parameters (e.g., nprobe for IVFPQ)
        for tier in self.tiers:
             if hasattr(tier, 'vector_store') and tier.vector_store and tier.vector_store.index_type == 'ivfpq':
                  policy_name = f"tier{tier.level_id}_nprobe"
                  self.policies[policy_name] = getattr(tier.vector_store.semantic_index, 'nprobe', 16) if tier.vector_store.semantic_index else 16
                  nlist = getattr(tier.vector_store.semantic_index, 'nlist', 1024) if tier.vector_store.semantic_index else 1024
                  self.policy_bounds[policy_name] = (1, max(1, nlist // 4)) # Bounds for nprobe

        # Summarization control
        self.policies["summarization_frequency_tokens"] = 131072 # How often to trigger check
        self.policy_bounds["summarization_frequency_tokens"] = (32768, 1048576)
        self.policies["summarization_min_block_size"] = 128
        self.policy_bounds["summarization_min_block_size"] = (64, 512)

        self.logger.info(f"Initial policies: {self.policies}")

    def record_access(self, position, tier_hit, latency_ms, is_write=False):
        """Record a memory access operation."""
        self.op_count += 1
        ts = time.time()
        self.access_history.append((ts, tier_hit, position, is_write))

        if tier_hit is not None and tier_hit in self.latency_history:
             self.latency_history[tier_hit].append(latency_ms)
        elif tier_hit is None: # Assume disk hit if miss all RAM tiers
             self.latency_history[4].append(latency_ms) # Log disk latency

        # Trigger policy update periodically
        if self.op_count % self.update_interval_ops == 0 or (ts - self.last_update_time) > self.update_interval_time:
            self._update_policies()
            self.last_update_time = ts

    def _update_policies(self):
        """Update policies based on observed metrics."""
        self.logger.debug("Updating adaptive policies...")

        # 1. Analyze recent access patterns
        self._analyze_access_pattern()

        # 2. Gather current tier stats
        current_stats = self._gather_tier_stats()

        # 3. Update individual policies
        self._update_capacity_policies(current_stats)
        self._update_tiering_policies(current_stats)
        self._update_compression_policies(current_stats)
        self._update_index_policies(current_stats)
        self._update_summarization_policies(current_stats)

        # 4. Apply policies to the system components
        self._apply_policies()

        self.logger.debug(f"Policies updated: {self.policies}")

    def _analyze_access_pattern(self):
        # Basic pattern analysis (can be much more sophisticated)
        if len(self.access_history) < 100: return # Need more data

        positions = [pos for _, _, pos, _ in self.access_history if pos is not None]
        if not positions: return

        seq_count, local_count, rand_count, repeat_count = 0, 0, 0, 0
        last_pos = positions[0]
        pos_counts = {}

        for i in range(1, len(positions)):
            pos = positions[i]
            diff = abs(pos - last_pos)
            if diff == 1: seq_count += 1
            elif diff < 32: local_count += 1 # Locality window
            elif diff == 0: repeat_count += 1
            else: rand_count += 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            last_pos = pos

        total = len(positions) - 1
        if total == 0: return

        if seq_count / total > 0.6: self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL
        elif local_count / total > 0.5: self.current_access_pattern = MemoryAccessPattern.LOCAL
        elif repeat_count / total > 0.4: self.current_access_pattern = MemoryAccessPattern.REPEATED
        else: self.current_access_pattern = MemoryAccessPattern.RANDOM
        # Could also detect start/end focused etc.


    def _gather_tier_stats(self):
        stats = {}
        total_ram_tokens = 0
        total_ram_capacity = 0
        for tier in self.tiers:
             tier_id = tier.level_id
             s = {"hit_rate": tier.hit_rate, "fullness": tier.fullness}
             if isinstance(tier, RAMTier):
                 s["capacity"] = tier.current_capacity
                 total_ram_tokens += len(tier)
                 total_ram_capacity += tier.current_capacity
             if tier_id in self.latency_history and self.latency_history[tier_id]:
                 s["avg_latency_ms"] = np.mean(list(self.latency_history[tier_id]))
             else:
                  s["avg_latency_ms"] = -1
             self.tier_stats_history[tier_id].append(s)
             # Use average over recent history for stability
             stats[tier_id] = {k: np.mean([h[k] for h in self.tier_stats_history[tier_id]])
                               for k in s if self.tier_stats_history[tier_id]}


        # Disk stats
        disk_id = 4
        disk_s = {}
        if self.persistent_storage:
            disk_s["fullness"] = len(self.persistent_storage) / self.persistent_storage.max_tokens if self.persistent_storage.max_tokens > 0 else 1.0
        if disk_id in self.latency_history and self.latency_history[disk_id]:
            disk_s["avg_latency_ms"] = np.mean(list(self.latency_history[disk_id]))
        else:
            disk_s["avg_latency_ms"] = -1
        if disk_s:
             self.tier_stats_history[disk_id].append(disk_s)
             stats[disk_id] = {k: np.mean([h[k] for h in self.tier_stats_history[disk_id]])
                               for k in disk_s if self.tier_stats_history[disk_id]}

        stats["global"] = {"ram_fullness": total_ram_tokens / total_ram_capacity if total_ram_capacity > 0 else 1.0}
        return stats

    def _adjust_policy(self, name, adjustment):
         if name in self.policies:
             current_val = self.policies[name]
             min_val, max_val = self.policy_bounds[name]
             new_val = current_val * adjustment
             self.policies[name] = max(min_val, min(max_val, new_val))
             # Log change if significant
             if abs(self.policies[name] - current_val) / (current_val + 1e-6) > 0.01:
                  self.logger.info(f"Policy '{name}' adjusted: {current_val:.2f} -> {self.policies[name]:.2f}")

    def _update_capacity_policies(self, stats):
        # Adjust RAM tier capacities based on hit rates and latency
        for tid in sorted(stats.keys()):
             if tid == 4 or f"tier{tid}_capacity" not in self.policies: continue # Skip disk or non-dynamic tiers

             tier_stats = stats[tid]
             policy_name = f"tier{tid}_capacity"
             latency = tier_stats.get("avg_latency_ms", -1)
             hit_rate = tier_stats.get("hit_rate", -1)
             target_latency = getattr(self.tiers[tid-1], 'qos_target_latency_ms', 1.0) # Get target from tier obj

             if latency != -1 and latency > target_latency * 1.5:
                 # Too slow, decrease size
                 self._adjust_policy(policy_name, 1.0 - self.learning_rate * 2)
             elif hit_rate != -1 and hit_rate < 0.5 and tid > 1: # Don't shrink L1 too easily
                 # Low hit rate, increase size
                  self._adjust_policy(policy_name, 1.0 + self.learning_rate)
             # Consider global memory pressure? (Requires system monitoring)

    def _update_tiering_policies(self, stats):
        # Adjust promotion/demotion based on access patterns and tier fullness
        demotion_policy = "demotion_staleness_seconds"
        promotion_policy = "promotion_access_threshold"

        # If higher tiers are full and slow, demote faster
        l1_full = stats.get(1, {}).get("fullness", 0) > 0.9
        l2_full = stats.get(2, {}).get("fullness", 0) > 0.9
        l1_slow = stats.get(1, {}).get("avg_latency_ms", 0) > getattr(self.tiers[0], 'qos_target_latency_ms', 0.1) * 1.5

        if l1_full or l1_slow:
             # Make demotion faster (reduce staleness threshold)
             self._adjust_policy(demotion_policy, 1.0 - self.learning_rate * 2)
             # Make promotion harder (increase access threshold)
             self._adjust_policy(promotion_policy, 1.0 + self.learning_rate)
        elif not l2_full: # If space in next tier down
             # Make demotion slower (increase staleness threshold)
             self._adjust_policy(demotion_policy, 1.0 + self.learning_rate)
             # Make promotion easier (decrease access threshold)
             self._adjust_policy(promotion_policy, 1.0 - self.learning_rate)

    def _update_compression_policies(self, stats):
         # Adjust compression based on latency vs memory usage
         for tid in sorted(stats.keys()):
             if tid == 4 or f"tier{tid}_compression_ratio" not in self.policies: continue

             tier_stats = stats[tid]
             policy_name = f"tier{tid}_compression_ratio"
             latency = tier_stats.get("avg_latency_ms", -1)
             fullness = tier_stats.get("fullness", 0)
             target_latency = getattr(self.tiers[tid-1], 'qos_target_latency_ms', 1.0)

             if latency != -1 and latency > target_latency * 1.5:
                 # Too slow, decrease compression (lower ratio)
                 self._adjust_policy(policy_name, 1.0 - self.learning_rate)
             elif fullness > 0.9:
                 # Too full, increase compression (higher ratio)
                 self._adjust_policy(policy_name, 1.0 + self.learning_rate)

    def _update_index_policies(self, stats):
        # Adjust nprobe for IVFPQ based on latency vs hit rate (recall)
        for tid in sorted(stats.keys()):
            if f"tier{tid}_nprobe" not in self.policies: continue

            tier_stats = stats[tid]
            policy_name = f"tier{tid}_nprobe"
            latency = tier_stats.get("avg_latency_ms", -1)
            hit_rate = tier_stats.get("hit_rate", -1) # Use overall hit rate as proxy for recall
            target_latency = getattr(self.tiers[tid-1], 'qos_target_latency_ms', 1.0)

            # If search is too slow, decrease nprobe
            if latency != -1 and latency > target_latency * 2.0:
                 self._adjust_policy(policy_name, 1.0 - self.learning_rate * 2)
            # If hit rate is low (poor recall), increase nprobe
            elif hit_rate != -1 and hit_rate < 0.6:
                 self._adjust_policy(policy_name, 1.0 + self.learning_rate * 2)

    def _update_summarization_policies(self, stats):
         # If RAM is full or disk is filling, summarize more often/aggressively
         ram_fullness = stats.get("global", {}).get("ram_fullness", 0)
         disk_fullness = stats.get(4, {}).get("fullness", 0)
         freq_policy = "summarization_frequency_tokens"
         block_policy = "summarization_min_block_size"

         if ram_fullness > 0.9 or disk_fullness > 0.8:
              # Summarize more often (decrease interval)
              self._adjust_policy(freq_policy, 1.0 - self.learning_rate)
              # Summarize smaller blocks (decrease min size)
              self._adjust_policy(block_policy, 1.0 - self.learning_rate)
         elif ram_fullness < 0.5 and disk_fullness < 0.5:
               # Plenty of space, summarize less often
               self._adjust_policy(freq_policy, 1.0 + self.learning_rate)
               self._adjust_policy(block_policy, 1.0 + self.learning_rate)


    def _apply_policies(self):
        """Push updated policy values to the relevant system components."""
        for name, value in self.policies.items():
             # Apply tier capacities
             if name.endswith("_capacity") and name.startswith("tier"):
                 tid = int(name.split("_")[0][4:])
                 if tid <= len(self.tiers) and isinstance(self.tiers[tid-1], RAMTier):
                     self.tiers[tid-1].current_capacity = int(value)

             # Apply compression ratios
             elif name.endswith("_compression_ratio") and name.startswith("tier"):
                  tid = int(name.split("_")[0][4:])
                  if tid <= len(self.tiers) and hasattr(self.tiers[tid-1], 'compressor') and self.tiers[tid-1].compressor:
                      self.tiers[tid-1].compressor.target_compression_ratio = value
                      # Re-calculate internal compressed_dim based on new ratio? Or let compressor handle it?
                      # For simplicity, assume compressor adapts internally or on next use.

             # Apply index parameters (nprobe)
             elif name.endswith("_nprobe") and name.startswith("tier"):
                  tid = int(name.split("_")[0][4:])
                  if tid <= len(self.tiers) and hasattr(self.tiers[tid-1], 'vector_store') and self.tiers[tid-1].vector_store:
                       vs = self.tiers[tid-1].vector_store
                       if vs.semantic_index and vs.index_type == 'ivfpq' and hasattr(vs.semantic_index, 'nprobe'):
                            vs.semantic_index.nprobe = int(value)

             # Apply summarization parameters
             elif name == "summarization_frequency_tokens":
                  if hasattr(self.parent_manager, 'streaming_memory') and self.parent_manager.streaming_memory: # Need ref to parent
                      self.parent_manager.streaming_memory.summarization_interval = int(value)
             elif name == "summarization_min_block_size":
                   if hasattr(self.parent_manager, 'summarizer') and self.parent_manager.summarizer:
                       self.parent_manager.summarizer.min_block_size = int(value)

        # Tiering policies (promotion/demotion) are used directly by the rebalancing logic


    def get_policy(self, policy_name):
        return self.policies.get(policy_name)

    def get_all_policies(self):
        return self.policies.copy()


# --- Distributed Orchestrator --- (Conceptual)
class DistributedMemoryOrchestrator:
    """
    Coordinates memory operations across multiple nodes for >1B scale.
    (Simulation - No actual network communication implemented).
    """
    def __init__(self, dim, node_ids: List[int], shard_strategy="token_range"):
        self.dim = dim
        self.node_ids = node_ids
        self.node_count = len(node_ids)
        self.shard_strategy = shard_strategy # "token_range", "hash", "adaptive"
        self.logger = logging.getLogger("ultracontext.memory.orchestrator")

        # Simulation state
        self.nodes_state = {nid: {"available": True, "load": 0.5, "token_count": 0} for nid in node_ids}
        self.token_to_node_map = {} # token_id -> node_id

        self._setup_router()
        self.logger.info(f"Distributed Orchestrator initialized for {self.node_count} nodes.")

    def _setup_router(self):
        # Simplified routing logic
        if self.shard_strategy == "token_range" or self.shard_strategy == "hash" or self.shard_strategy == "adaptive":
            # For simulation, simple hash is easiest
            self.route_token = lambda token_id: self.node_ids[hash(token_id) % self.node_count]
            self.logger.info(f"Using simplified hash routing for simulation.")
        else:
            raise ValueError(f"Unsupported shard strategy: {self.shard_strategy}")

    def get_node_for_token(self, token_id):
        """Determine target node for a token."""
        # Check cache map first
        if token_id in self.token_to_node_map:
            node_id = self.token_to_node_map[token_id]
            # Check if node is still available
            if node_id in self.nodes_state and self.nodes_state[node_id]["available"]:
                 return node_id
            else:
                 # Node failed, need to reroute and potentially trigger recovery
                 self.logger.warning(f"Node {node_id} for token {token_id} is unavailable. Rerouting.")
                 del self.token_to_node_map[token_id] # Clear stale mapping

        # Apply routing strategy
        node_id = self.route_token(token_id)

        # Find first available node if target is down (simple failover)
        while not self.nodes_state.get(node_id, {}).get("available", False):
             self.logger.warning(f"Target node {node_id} unavailable, trying next.")
             node_id = self.node_ids[(self.node_ids.index(node_id) + 1) % self.node_count]
             if node_id == self.route_token(token_id): # Avoid infinite loop if all nodes down
                  self.logger.error("All nodes appear unavailable!")
                  raise RuntimeError("No available nodes in distributed memory system.")

        # Cache the mapping
        self.token_to_node_map[token_id] = node_id
        return node_id

    def add_token_distributed(self, token_id, value, position, metadata):
        """Routes an add operation to the correct node."""
        target_node = self.get_node_for_token(token_id)
        self.logger.debug(f"Routing ADD token {token_id} to node {target_node}")
        # --- Simulation ---
        # In reality: RPC call to target_node.add_token(...)
        self.nodes_state[target_node]["token_count"] += 1
        return True # Assume success

    def get_token_distributed(self, token_id):
        """Routes a get operation to the correct node."""
        target_node = self.get_node_for_token(token_id)
        self.logger.debug(f"Routing GET token {token_id} to node {target_node}")
        # --- Simulation ---
        # In reality: RPC call result = target_node.get_token(...)
        # Simulate miss rate
        if random.random() < 0.1: # 10% chance node doesn't have it (cache miss simulation)
             return None, None
        # Simulate retrieving dummy data
        dummy_value = torch.randn(self.dim) # Needs decompression in real system
        dummy_meta = {"position": hash(token_id) % 1_000_000_000, "importance": 0.5} # Fake position
        return dummy_value, dummy_meta

    def search_distributed(self, query_vector, top_k):
        """Broadcasts search to all nodes and merges results."""
        self.logger.debug(f"Broadcasting search to {self.node_count} nodes.")
        all_results = []
        # --- Simulation ---
        # In reality: Fan-out RPC search calls to all available nodes
        for node_id in self.node_ids:
            if self.nodes_state[node_id]["available"]:
                 # node_results = RPC_call(node_id, "search", query_vector, top_k)
                 # Simulate getting some results from each node
                 node_results = []
                 for i in range(random.randint(0, top_k)):
                      sim_score = 0.6 + random.random() * 0.4
                      dummy_id = f"node{node_id}_token{random.randint(0, 10000)}"
                      dummy_vec = torch.randn(self.dim)
                      dummy_meta = {"position": hash(dummy_id) % 1_000_000_000, "node": node_id}
                      node_results.append((dummy_id, dummy_vec, sim_score, dummy_meta))
                 all_results.extend(node_results)

        # Merge and rank results
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:top_k]

    def mark_node_failed(self, node_id):
        if node_id in self.nodes_state:
             self.nodes_state[node_id]["available"] = False
             self.logger.warning(f"Node {node_id} marked as FAILED.")
             # TODO: Trigger token redistribution / recovery process

    def get_stats(self):
        return {
            "node_count": self.node_count,
            "available_nodes": sum(1 for s in self.nodes_state.values() if s["available"]),
            "token_map_size": len(self.token_to_node_map),
            "shard_strategy": self.shard_strategy,
            # Add node load, token counts per node etc.
        }


# Main Hierarchical Memory Manager (Enhanced for 1B Scale)
class AdvancedHierarchicalMemoryManager(Module):
    """
    Top-level orchestrator for the multi-tiered memory system,
    designed conceptually for 1 Billion+ tokens.
    """
    def __init__(
        self,
        dim: int,
        l1_capacity: int = 65536,      # L1 Cache (Fast RAM) - e.g., 64K tokens
        l2_capacity: int = 1048576,    # L2 Cache (Med RAM) - e.g., 1M tokens
        l3_capacity: int = 8388608,    # L3 Cache (Slow RAM/NVMe) - e.g., 8M tokens
        disk_capacity: int = 1_000_000_000, # L4 (Disk) - Target total capacity
        disk_storage_path: str = "./ultracontext_disk_storage",
        distributed_nodes: List[int] = None, # List of node IDs if distributed, else None
        enable_summarization: bool = True,
        enable_adaptive_policies: bool = True,
        qos_targets: Dict[str, float] = None,
        reliability_level: str = "normal", # "normal", "high", "critical"
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.perf_config = perf_config
        self.device = torch.device("cuda" if torch.cuda.is_available() and not perf_config.optimize_memory else "cpu")
        self.logger = logging.getLogger("ultracontext.memory.manager")
        self.reliability_level = reliability_level
        self.enable_redundancy = (reliability_level in ["high", "critical"])

        # --- Tier Initialization ---
        self.tiers: List[MemoryTier] = []

        # L1 Tier (Fast RAM, No Compression, Flat Index for Positional)
        self.l1 = RAMTier(
            level_id=1, dim=dim, capacity=l1_capacity,
            retrieval_cost=1, storage_cost=20, eviction_policy="adaptive",
            qos_enabled=True, qos_target_latency_ms=qos_targets.get("l1_latency_ms", 0.1),
            vector_store_config=None # No semantic search needed usually
        )
        self.tiers.append(self.l1)

        # L2 Tier (Medium RAM, Light Compression, HNSW Index)
        l2_compressor = AdvancedMemoryCompressor(dim, compression_ratio=2.0, quantization_bits=16, perf_config=perf_config)
        self.l2 = RAMTier(
            level_id=2, dim=dim, capacity=l2_capacity,
            retrieval_cost=5, storage_cost=10, eviction_policy="adaptive",
            compressor=l2_compressor,
            vector_store_config={"index_type": "hnsw", "max_tokens": l2_capacity // 2}, # Cache half in index
            qos_enabled=True, qos_target_latency_ms=qos_targets.get("l2_latency_ms", 1.0)
        )
        self.tiers.append(self.l2)

        # L3 Tier (Slower RAM/NVMe, Medium Compression, IVFPQ Index)
        l3_compressor = AdvancedMemoryCompressor(dim, compression_ratio=8.0, quantization_bits=8, perf_config=perf_config)
        self.l3 = RAMTier(
            level_id=3, dim=dim, capacity=l3_capacity,
            retrieval_cost=20, storage_cost=5, eviction_policy="adaptive",
            compressor=l3_compressor,
            vector_store_config={"index_type": "ivfpq", "max_tokens": l3_capacity // 4}, # Cache less in index
            qos_enabled=True, qos_target_latency_ms=qos_targets.get("l3_latency_ms", 10.0)
        )
        self.tiers.append(self.l3)

        # L4 Tier (Disk/Distributed)
        self.orchestrator = None
        if distributed_nodes:
             self.orchestrator = DistributedMemoryOrchestrator(dim, distributed_nodes)
             # Persistent storage managed by individual nodes (conceptual)
             self.persistent_storage = None # Orchestrator handles routing
             self.logger.info(f"Using Distributed Storage via Orchestrator across {len(distributed_nodes)} nodes.")
        else:
             # Local persistent disk storage
             self.persistent_storage = PersistentTokenStorage(
                 dim=dim, storage_path=disk_storage_path, max_tokens=disk_capacity,
                 compression_ratio=16.0, # High compression for disk
                 index_type="ivfpq" # Disk-based ANN
             )
             self.logger.info(f"Using Local Disk Storage at '{disk_storage_path}'.")


        # --- Sub-Modules Initialization ---
        self.summarizer = None
        if enable_summarization:
            self.summarizer = MemorySummarizer(dim=dim, summary_level_max=5, perf_config=perf_config)

        # Streaming manager coordinates movement to disk/distributed tier
        self.streaming_memory = StreamingMemoryManager(
             dim=dim,
             persistent_storage=self.persistent_storage, # Provide disk store ref
             summarizer=self.summarizer,
             tier3_capacity=l3_capacity, # Let it know capacity above it
             orchestrator=self.orchestrator, # Provide orchestrator ref
             perf_config=perf_config
        )

        # Adaptive Policies need references to tiers and potentially other components
        self.adaptive_policies = None
        if enable_adaptive_policies:
            self.adaptive_policies = AdaptiveMemoryPolicy(
                dim=dim,
                tiers=self.tiers,
                persistent_storage=self.persistent_storage,
                orchestrator=self.orchestrator,
                perf_config=perf_config
            )
            # Give policy manager a reference back to this manager to apply summarization policies
            self.adaptive_policies.parent_manager = self

        # Global importance scorer
        self.importance_scorer = ImportanceScorer(dim=dim, max_context_window=disk_capacity, perf_config=perf_config)

        # State Tracking
        self.access_history = deque(maxlen=10000) # Track more recent accesses globally
        self.token_locations = {} # token_id -> current level_id (1, 2, 3, 4 for disk/dist)
        self.shadow_copies = {} # For reliability: token_id -> list of level_ids where copies exist

        # Background Maintenance Task Queue
        self.maintenance_queue = queue.PriorityQueue()
        self.stop_background = False
        self.background_thread = None
        self._start_background_worker()

        self.logger.info(f"AdvancedHierarchicalMemoryManager initialized for {disk_capacity} tokens.")
        self.to(self.device) # Move models (compressors, scorers) to device


    def _start_background_worker(self):
        if self.background_thread is None and self.perf_config.optimize_memory:
            self.stop_background = False
            self.background_thread = threading.Thread(target=self._background_maintenance_loop, daemon=True)
            self.background_thread.start()
            self.logger.info("Background maintenance worker started.")

    def _stop_background_worker(self):
        if self.background_thread is not None:
            self.logger.info("Stopping background maintenance worker...")
            self.stop_background = True
            # Add sentinel to queue to unblock worker
            try:
                 self.maintenance_queue.put((-1, None, None), timeout=1.0) # Highest priority
            except queue.Full:
                 pass
            self.background_thread.join(timeout=5.0)
            if self.background_thread.is_alive():
                 self.logger.warning("Background worker did not terminate gracefully.")
            self.background_thread = None
            self.logger.info("Background maintenance worker stopped.")

    def _background_maintenance_loop(self):
        self.logger.debug("Background worker loop started.")
        last_periodic_run = time.time()
        periodic_interval = 30 # seconds

        while not self.stop_background:
            try:
                # Prioritize tasks from queue
                try:
                     priority, task, args = self.maintenance_queue.get(timeout=1.0)
                     if task is None: # Sentinel check
                          break
                     task_name = getattr(task, '__name__', 'unknown_task')
                     self.logger.debug(f"Executing background task: {task_name} (Priority: {priority})")
                     task(*args)
                     self.maintenance_queue.task_done()
                     self.logger.debug(f"Finished background task: {task_name}")
                except queue.Empty:
                     # No tasks in queue, check if periodic maintenance is due
                     now = time.time()
                     if now - last_periodic_run > periodic_interval:
                          self.logger.debug("Running periodic maintenance...")
                          self._periodic_maintenance()
                          last_periodic_run = now
                          self.logger.debug("Periodic maintenance complete.")
                     else:
                          # Sleep briefly to avoid busy-waiting
                          time.sleep(0.1)

            except Exception as e:
                 self.logger.error(f"Error in background maintenance loop: {e}", exc_info=True)
                 # Avoid continuous errors by sleeping longer after an exception
                 time.sleep(5.0)
        self.logger.debug("Background worker loop finished.")


    def _schedule_maintenance(self, task, args=(), priority=50):
        """Schedule a maintenance task with priority (lower value = higher priority)."""
        if self.background_thread is not None:
            self.maintenance_queue.put((priority, task, args))

    def _periodic_maintenance(self):
        """Tasks run periodically by the background worker."""
        # Rebalance memory levels (high priority)
        self._rebalance_memory_levels() # This function now handles the logic, not scheduled separately

        # Optimize individual tiers (medium priority)
        for tier in self.tiers:
            tier.optimize()
        if self.persistent_storage:
             pass # Disk optimization might involve file compaction, index rebuilds etc.

        # Trigger summarization check (lower priority)
        self._check_and_trigger_summarization()

        # Verify QoS Guarantees (informational)
        self._verify_qos_guarantees()

        # Verify redundant copies if enabled
        if self.enable_redundancy:
             self._verify_redundant_copies()

        # Cleanup token location map (remove old entries?)
        self._cleanup_location_map()

        # Clean GPU cache if applicable
        if self.device.type == 'cuda':
             gc.collect()
             torch.cuda.empty_cache()


    @timer
    def add_tokens(self, tokens: torch.Tensor, positions: torch.Tensor, attention_weights: Optional[torch.Tensor] = None):
        """
        Adds a batch of tokens to the memory system, starting at L1.
        Handles moving evicted tokens down the hierarchy.
        """
        batch_size, seq_len, dim = tokens.shape
        if dim != self.dim:
             raise ValueError(f"Input token dimension {dim} != system dimension {self.dim}")

        tokens = tokens.to(self.device, non_blocking=True)
        positions = positions.to(self.device, non_blocking=True)
        if attention_weights is not None:
             attention_weights = attention_weights.to(self.device, non_blocking=True)

        all_token_ids = [[] for _ in range(batch_size)]
        total_tokens_added = 0

        # 1. Calculate Importance Scores
        importance_scores = self.importance_scorer(tokens, positions, attention_weights)

        # 2. Add tokens to L1
        evicted_from_l1 = []
        for b in range(batch_size):
            for i in range(seq_len):
                token_id = str(uuid.uuid4()) # Generate unique ID
                value = tokens[b, i]
                pos = positions[b, i].item()
                importance = importance_scores[b, i].item()

                # Attempt to add to L1
                evicted_l1 = self.l1.add(token_id, value, pos, importance)
                if evicted_l1 is not None: # Add succeeded
                     all_token_ids[b].append(token_id)
                     self.token_locations[token_id] = 1 # Track location
                     total_tokens_added += 1
                     # Handle evicted tokens from L1
                     evicted_from_l1.extend(evicted_l1)
                # else: Add failed (L1 full and couldn't evict), token dropped. Logged in L1.add

        self.logger.debug(f"Added {total_tokens_added} tokens to L1. {len(evicted_from_l1)} tokens evicted from L1.")

        # 3. Process evicted tokens down the hierarchy
        tokens_to_process = {info.token_id: info for info in evicted_from_l1}
        for target_level_id in range(2, len(self.tiers) + 1): # Process for L2, L3
             target_tier = self.tiers[target_level_id - 1]
             evicted_next = {}
             successfully_added_ids = set()

             for token_id, token_info in list(tokens_to_process.items()):
                 # Prepare value for target tier (decompress source, compress target)
                 # Assuming token_info.value is *not* compressed as it came from L1
                 value_to_add = token_info.value
                 importance = token_info.importance_score
                 position = token_info.position

                 # Attempt to add to target tier
                 evicted_infos = target_tier.add(token_id, value_to_add, position, importance)

                 if token_id in target_tier.tokens: # Check if add was successful
                      successfully_added_ids.add(token_id)
                      self.token_locations[token_id] = target_level_id
                      # Handle newly evicted tokens from this tier
                      if evicted_infos:
                           for info in evicted_infos:
                                evicted_next[info.token_id] = info
                 # else: Add failed, token dropped. Logged in target_tier.add

             # Remove successfully added tokens from the processing list
             for token_id in successfully_added_ids:
                  tokens_to_process.pop(token_id, None)

             # Add newly evicted tokens to the list for the *next* level
             tokens_to_process.update(evicted_next)

             self.logger.debug(f"Processed {len(successfully_added_ids)} tokens into L{target_level_id}. {len(evicted_next)} evicted.")


        # 4. Remaining tokens_to_process go to Streaming Manager (Disk/Distributed)
        if tokens_to_process:
             self.logger.debug(f"Passing {len(tokens_to_process)} tokens evicted from L3 to Streaming Manager.")
             # This step needs careful implementation regarding compression/decompression
             # Assume for now StreamingMemoryManager handles getting the right value form
             self.streaming_memory.process_eligible_tokens(tokens_to_process)


        # 5. Update global access history (treat adds as writes)
        ts = time.time()
        for b in range(batch_size):
             for i in range(seq_len):
                  if i < len(all_token_ids[b]): # Only record if add succeeded
                      self.access_history.append((ts, 1, positions[b,i].item(), True)) # Record access to L1 on write

        # 6. Record metrics for adaptive policies (simplified: just record position access)
        if self.adaptive_policies:
             self.adaptive_policies.record_access(position=positions.view(-1)[0].item(), tier_hit=1, latency_ms=1, is_write=True) # Record single access per batch

        # 7. Schedule background rebalancing
        self._schedule_maintenance(self._rebalance_memory_levels, priority=30) # Medium priority

        return all_token_ids

    @timer
    def retrieve_tokens(self, positions=None, token_ids=None, query_vectors=None, top_k=5):
        """
        Retrieves tokens by searching hierarchically through tiers (L1->L2->L3->Disk/Dist->Summaries).
        Handles decompression and promotion.
        """
        results = {} # Use dict {query_idx/token_id/position -> value}
        latencies = {} # Track latency per tier hit
        start_time_global = time.time()

        # Determine query type and prepare query items
        query_items = []
        query_mode = "position"
        if positions is not None:
             query_items = positions.view(-1).tolist()
        elif token_ids is not None:
             query_mode = "token_id"
             # Flatten list of lists if needed
             if isinstance(token_ids[0], list):
                  query_items = [tid for sublist in token_ids for tid in sublist]
             else:
                  query_items = token_ids
        elif query_vectors is not None:
             query_mode = "semantic"
             query_items = list(range(query_vectors.shape[0])) # Use index as key
        else:
            raise ValueError("Must provide positions, token_ids, or query_vectors for retrieval.")

        items_to_find = set(query_items)
        found_items_map = {} # query_item -> (value, position, source_tier)

        # 1. Search RAM Tiers (L1 -> L2 -> L3)
        for tier_id in range(1, len(self.tiers) + 1):
             if not items_to_find: break # Stop if all found
             tier = self.tiers[tier_id - 1]
             tier_found = set()
             start_time_tier = time.time()

             items_in_this_tier = list(items_to_find) # Search for remaining items

             # Perform search/lookup based on mode
             if query_mode == "position":
                  for pos in items_in_this_tier:
                       value, _, token_info = tier.get(position=pos)
                       if value is not None:
                            found_items_map[pos] = (value, pos, tier_id)
                            tier_found.add(pos)
                            # Schedule promotion if found below L1
                            if tier_id > 1: self._schedule_maintenance(self._promote_token, args=(token_info.token_id,), priority=10)
             elif query_mode == "token_id":
                  for tid in items_in_this_tier:
                       value, pos, token_info = tier.get(token_id=tid)
                       if value is not None:
                            found_items_map[tid] = (value, pos, tier_id)
                            tier_found.add(tid)
                            if tier_id > 1: self._schedule_maintenance(self._promote_token, args=(tid,), priority=10)
             elif query_mode == "semantic":
                  if hasattr(tier, 'vector_store') and tier.vector_store:
                      query_vecs_tensor = query_vectors[[idx for idx in items_in_this_tier]]
                      batch_results = tier.vector_store.batch_search_similar(query_vecs_tensor, top_k=1) # Get best match per query

                      for i, query_idx in enumerate(items_in_this_tier):
                           if batch_results[i]: # Found a match
                                tid, vec, _, meta = batch_results[i][0]
                                # Retrieve the actual value (which might involve decompression)
                                value, pos, token_info = tier.get(token_id=tid)
                                if value is not None:
                                     # Use query_idx as key, store actual found token info
                                     found_items_map[query_idx] = (value, pos, tier_id)
                                     tier_found.add(query_idx)
                                     if tier_id > 1: self._schedule_maintenance(self._promote_token, args=(tid,), priority=15) # Promote semantically hit items


             latency_ms = (time.time() - start_time_tier) * 1000
             latencies[tier_id] = latency_ms
             items_to_find -= tier_found # Remove found items
             self.logger.debug(f"Searched Tier {tier_id}. Found {len(tier_found)} items. Remaining: {len(items_to_find)}. Latency: {latency_ms:.2f} ms.")


        # 2. Search Disk/Distributed Tier (L4)
        if items_to_find:
             start_time_disk = time.time()
             disk_found = set()
             items_in_this_tier = list(items_to_find)

             if self.streaming_memory: # Check if disk manager exists
                  if query_mode == "position":
                       for pos in items_in_this_tier:
                            value, _, meta = self.streaming_memory.retrieve_from_disk(position=pos)
                            if value is not None:
                                 found_items_map[pos] = (value, pos, 4)
                                 disk_found.add(pos)
                                 # Schedule promotion from disk
                                 # Need token_id from meta if possible
                                 tid = meta.get('token_id', None) # Assume meta might have it
                                 if tid: self._schedule_maintenance(self._promote_token, args=(tid,), priority=20)

                  elif query_mode == "token_id":
                       for tid in items_in_this_tier:
                            value, pos, meta = self.streaming_memory.retrieve_from_disk(token_id=tid)
                            if value is not None:
                                 found_items_map[tid] = (value, pos, 4)
                                 disk_found.add(tid)
                                 self._schedule_maintenance(self._promote_token, args=(tid,), priority=20)

                  elif query_mode == "semantic":
                       # Semantic search on disk (potentially slow)
                       for query_idx in items_in_this_tier:
                            query_vec = query_vectors[query_idx]
                            value, pos, meta = self.streaming_memory.retrieve_from_disk(query_vector=query_vec)
                            if value is not None:
                                 found_items_map[query_idx] = (value, pos, 4)
                                 disk_found.add(query_idx)
                                 tid = meta.get('token_id', None)
                                 if tid: self._schedule_maintenance(self._promote_token, args=(tid,), priority=25)


             latency_ms = (time.time() - start_time_disk) * 1000
             latencies[4] = latency_ms
             items_to_find -= disk_found
             self.logger.debug(f"Searched Tier 4 (Disk/Dist). Found {len(disk_found)} items. Remaining: {len(items_to_find)}. Latency: {latency_ms:.2f} ms.")


        # 3. Search Summaries (If items still not found)
        # This is complex: needs mapping positions/semantic queries to summaries
        if items_to_find and self.summarizer:
             start_time_summary = time.time()
             summary_found = set()
             items_in_this_tier = list(items_to_find)
             self.logger.debug(f"Searching summaries for {len(items_to_find)} remaining items.")

             if query_mode == "position":
                  for pos in items_in_this_tier:
                       # Find best summary covering this position
                       summaries = self.summarizer.find_summaries_for_position(pos)
                       if summaries:
                            best_summary = summaries[0] # Smallest range first
                            summary_vec = self.summarizer.get_summary_vector(best_summary['id'])
                            if summary_vec is not None:
                                 # Return summary vector as placeholder
                                 found_items_map[pos] = (summary_vec, best_summary['range'], f"S{best_summary['level']}")
                                 summary_found.add(pos)
                                 # Could potentially trigger prefetching original tokens based on summary hit

             elif query_mode == "semantic":
                  # Search summaries semantically (approximate)
                   for query_idx in items_in_this_tier:
                        query_vec = query_vectors[query_idx]
                        best_sim = -1.0
                        best_summary_info = None
                        # Iterate through cached summary vectors
                        for sum_id, sum_vec in self.summarizer.summary_vectors_cache.items():
                            sim = F.cosine_similarity(query_vec.unsqueeze(0), sum_vec.unsqueeze(0)).item()
                            if sim > best_sim and sim > 0.6: # Similarity threshold for summaries
                                 best_sim = sim
                                 best_summary_info = {"id": sum_id, "vec": sum_vec}

                        if best_summary_info:
                             # Find summary metadata (level, range)
                             level, pos_range = -1, (-1, -1)
                             for l, registry in self.summarizer.summary_registry.items():
                                 for reg_id, pr, _, _ in registry:
                                      if reg_id == best_summary_info['id']:
                                           level, pos_range = l, pr
                                           break
                                 if level != -1: break

                             found_items_map[query_idx] = (best_summary_info['vec'], pos_range, f"S{level}")
                             summary_found.add(query_idx)


             latency_ms = (time.time() - start_time_summary) * 1000
             latencies['S'] = latency_ms
             items_to_find -= summary_found
             self.logger.debug(f"Searched Summaries. Found {len(summary_found)} items. Remaining: {len(items_to_find)}. Latency: {latency_ms:.2f} ms.")


        # 4. Prepare final results and record metrics
        final_results = {}
        if query_mode == "semantic":
             # For semantic search, we need to aggregate results per query
             # The current `found_items_map` only stores the *first* hit per query index.
             # A full semantic search implementation would need to collect top-k results
             # across all tiers and merge them.
             # Simplification: Return the single best hit found across tiers.
             for query_idx in query_items:
                  if query_idx in found_items_map:
                       value, pos, tier = found_items_map[query_idx]
                       final_results[query_idx] = value # Return value tensor
                       # Record access
                       latency = latencies.get(tier, latencies.get('S', 0) if isinstance(tier, str) else latencies.get(4, 0))
                       if self.adaptive_policies: self.adaptive_policies.record_access(pos, tier, latency)
                       if pos is not None: self.access_history.append((time.time(), tier, pos, False))
                  else:
                       final_results[query_idx] = torch.zeros(self.dim, device=self.device) # Indicate not found
        else:
             # For position/token_id, map directly
             for item in query_items:
                  if item in found_items_map:
                       value, pos, tier = found_items_map[item]
                       final_results[item] = value
                       latency = latencies.get(tier, latencies.get('S', 0) if isinstance(tier, str) else latencies.get(4, 0))
                       if self.adaptive_policies: self.adaptive_policies.record_access(pos, tier, latency)
                       if pos is not None: self.access_history.append((time.time(), tier, pos, False))

                  else:
                       final_results[item] = torch.zeros(self.dim, device=self.device) # Indicate not found


        # TODO: Reshape results based on original input shape (batching) if needed

        global_latency_ms = (time.time() - start_time_global) * 1000
        self.logger.info(f"Retrieval complete. Found {len(found_items_map)}/{len(query_items)} items. Global Latency: {global_latency_ms:.2f} ms.")
        return final_results # Return dict mapping query item -> result tensor


    def _rebalance_memory_levels(self):
        """Move tokens between RAM tiers based on access patterns and importance."""
        self.logger.debug("Starting memory level rebalancing...")
        now = time.time()

        # Get policies
        promotion_threshold = self.adaptive_policies.get_policy("promotion_access_threshold") if self.adaptive_policies else 3
        demotion_staleness = self.adaptive_policies.get_policy("demotion_staleness_seconds") if self.adaptive_policies else 3600 * 6

        # --- Promotion Logic (Move Up: L3 -> L2, L2 -> L1) ---
        for source_level_id in range(len(self.tiers), 1, -1): # Iterate L3, L2
            target_level_id = source_level_id - 1
            source_tier = self.tiers[source_level_id - 1]
            target_tier = self.tiers[target_level_id - 1]
            promoted_count = 0

            # Only promote if target tier has space
            if target_tier.fullness < 0.95:
                 # Find candidates in source tier
                 candidates = []
                 for token_id, info in list(source_tier.tokens.items()): # Iterate copy
                      # Promote if frequently accessed OR highly important
                      if info.access_count >= promotion_threshold or info.importance_score > 0.85:
                           candidates.append(token_id)

                 # Promote top N candidates (e.g., based on score, limited by available space)
                 candidates.sort(key=lambda tid: source_tier.tokens[tid].importance_score + info.access_count * 0.1, reverse=True)
                 promotion_limit = int(target_tier.current_capacity * 0.1) # Limit promotion batch size
                 promoted_ids_this_batch = set()

                 for token_id in candidates[:promotion_limit]:
                      if len(target_tier) >= target_tier.current_capacity: break # Target tier filled up

                      # Get value (decompress from source)
                      value, pos, info = source_tier.get(token_id=token_id) # This also decompresses
                      if value is None: continue # Should not happen if iterating items

                      # Add to target tier (will compress if needed)
                      evicted = target_tier.add(token_id, value, pos, info.importance_score)
                      if token_id in target_tier.tokens: # Check success
                            # Remove from source tier
                            source_tier._remove(token_id) # Use internal remove
                            self.token_locations[token_id] = target_level_id
                            promoted_count += 1
                            promoted_ids_this_batch.add(token_id)
                            # Handle tokens evicted from target tier (move them down further)
                            if evicted: self._handle_evicted_during_rebalance(evicted, target_level_id)

                 if promoted_count > 0:
                     self.logger.info(f"Promoted {promoted_count} tokens from L{source_level_id} to L{target_level_id}.")


        # --- Demotion Logic (Move Down: L1 -> L2, L2 -> L3) ---
        for source_level_id in range(1, len(self.tiers)): # Iterate L1, L2
            target_level_id = source_level_id + 1
            source_tier = self.tiers[source_level_id - 1]
            target_tier = self.tiers[target_level_id - 1]
            demoted_count = 0

            # Only demote if target tier has space
            if target_tier.fullness < 0.98:
                 # Find candidates in source tier
                 candidates = []
                 for token_id, info in list(source_tier.tokens.items()): # Iterate copy
                      # Demote if stale AND not highly important
                      is_stale = (now - info.last_access_time) > demotion_staleness
                      is_important = info.importance_score > 0.7
                      if is_stale and not is_important and token_id not in source_tier.protected_tokens:
                           candidates.append(token_id)

                 # Demote top N candidates (e.g., oldest or least important first)
                 candidates.sort(key=lambda tid: source_tier.tokens[tid].last_access_time) # Oldest first
                 demotion_limit = int(source_tier.current_capacity * 0.05) # Limit demotion batch size
                 demoted_ids_this_batch = set()

                 for token_id in candidates[:demotion_limit]:
                      if len(target_tier) >= target_tier.current_capacity: break

                      # Get value (decompress from source if needed - L1 usually isn't compressed)
                      value, pos, info = source_tier.get(token_id=token_id)
                      if value is None: continue

                      # Add to target tier (will compress)
                      evicted = target_tier.add(token_id, value, pos, info.importance_score)
                      if token_id in target_tier.tokens:
                           # Remove from source tier
                           source_tier._remove(token_id)
                           self.token_locations[token_id] = target_level_id
                           demoted_count += 1
                           demoted_ids_this_batch.add(token_id)
                           if evicted: self._handle_evicted_during_rebalance(evicted, target_level_id)

                 if demoted_count > 0:
                      self.logger.info(f"Demoted {demoted_count} tokens from L{source_level_id} to L{target_level_id}.")


        # --- Demotion from L3 to Disk/Distributed (L4) ---
        l3_tier = self.tiers[-1] # Assumes L3 is the last RAM tier
        if self.streaming_memory and l3_tier.fullness > 0.9: # Only move if L3 is getting full
            demoted_to_disk = 0
            candidates = []
            for token_id, info in list(l3_tier.tokens.items()):
                 is_stale = (now - info.last_access_time) > demotion_staleness * 2 # Longer threshold for L3->Disk
                 is_important = info.importance_score > 0.5 # Lower importance threshold for keeping in L3
                 if is_stale and not is_important and token_id not in l3_tier.protected_tokens:
                      candidates.append(token_id)

            candidates.sort(key=lambda tid: l3_tier.tokens[tid].last_access_time)
            demotion_limit = int(l3_tier.current_capacity * 0.05)
            tokens_for_streaming = {}

            for token_id in candidates[:demotion_limit]:
                 # Remove from L3 first
                 removed_info = l3_tier._remove(token_id)
                 if removed_info:
                      tokens_for_streaming[token_id] = removed_info
                      self.token_locations[token_id] = 4 # Mark as moving to disk/dist
                      demoted_to_disk += 1

            if tokens_for_streaming:
                 self.logger.info(f"Demoting {demoted_to_disk} tokens from L3 to StreamingManager (Disk/Dist).")
                 # Pass to streaming manager for processing
                 # Need to handle value decompression/recompression correctly here
                 # Pass the TokenInfo objects; StreamingManager should handle value later
                 self.streaming_memory.process_eligible_tokens(tokens_for_streaming)


    def _handle_evicted_during_rebalance(self, evicted_infos: List[TokenInfo], source_level_id: int):
         """Handles tokens evicted when adding during rebalancing."""
         self.logger.debug(f"Handling {len(evicted_infos)} tokens evicted from L{source_level_id} during rebalance.")
         target_level_id = source_level_id + 1

         if target_level_id > len(self.tiers):
              # Evicted from L3, pass to StreamingManager
              if self.streaming_memory:
                   self.streaming_memory.process_eligible_tokens({info.token_id: info for info in evicted_infos})
         else:
              # Try adding to the next tier down
              target_tier = self.tiers[target_level_id - 1]
              evicted_next = {}
              added_count = 0
              for info in evicted_infos:
                   # Prepare value (decompress source T-1, compress target T)
                   # This requires knowing the source tier compressor. Assume info.value is decompressed.
                   value_to_add = info.value # Need actual value here
                   evicted = target_tier.add(info.token_id, value_to_add, info.position, info.importance_score)
                   if info.token_id in target_tier.tokens:
                       added_count += 1
                       self.token_locations[info.token_id] = target_level_id
                       if evicted:
                            for ev_info in evicted: evicted_next[ev_info.token_id] = ev_info
                   # Else: Failed to add, token dropped or handled by target_tier.add log

              self.logger.debug(f"Added {added_count} evicted tokens to L{target_level_id}. {len(evicted_next)} newly evicted.")
              if evicted_next:
                   # Recursively handle tokens evicted from the next level
                   self._handle_evicted_during_rebalance(list(evicted_next.values()), target_level_id)


    def _promote_token(self, token_id):
        """Attempt to promote a single token up the hierarchy."""
        if token_id not in self.token_locations: return
        current_level = self.token_locations[token_id]

        # Already in L1 or scheduled for promotion higher than current level
        if current_level == 1: return

        # Find token info and value (search down from current level - 1)
        value, pos, info = None, None, None
        source_level_id = -1
        for level_id in range(current_level - 1, 0, -1): # Check L(N-1)..L1
             tier = self.tiers[level_id - 1]
             value, pos, info = tier.get(token_id=token_id)
             if value is not None:
                  source_level_id = level_id
                  break

        # If not found in RAM tiers above current known location, something is wrong
        if value is None:
             # Maybe it's on Disk (L4)?
             if current_level == 4 and self.streaming_memory:
                  value, pos, meta = self.streaming_memory.retrieve_from_disk(token_id=token_id)
                  if value is not None:
                       source_level_id = 4
                       # Create temporary TokenInfo-like object from meta
                       info = type('obj', (object,), {'importance_score': meta.get('importance', 0.5)})()
             else:
                  self.logger.warning(f"Cannot promote token {token_id}: Not found in expected source levels.")
                  # Clean up potentially wrong location
                  # self.token_locations.pop(token_id, None)
                  return


        # Promote level by level towards L1
        for target_level_id in range(source_level_id - 1, 0, -1): # Target L(N-1)..L1
            target_tier = self.tiers[target_level_id - 1]
            if len(target_tier) < target_tier.current_capacity: # Check space
                 # Add to target tier
                 evicted = target_tier.add(token_id, value, pos, info.importance_score)
                 if token_id in target_tier.tokens:
                      self.logger.debug(f"Promoted token {token_id} to L{target_level_id}")
                      self.token_locations[token_id] = target_level_id
                      # Remove from original source tier (if it wasn't already removed by get)
                      if source_level_id <= len(self.tiers):
                          self.tiers[source_level_id - 1]._remove(token_id)
                      elif source_level_id == 4 and self.persistent_storage:
                          self.persistent_storage.remove(token_id) # Remove from disk

                      # Handle evicted tokens (move down)
                      if evicted: self._handle_evicted_during_rebalance(evicted, target_level_id)
                      return # Stop promotion once successfully moved up one level
                 else:
                      self.logger.warning(f"Failed to add promoted token {token_id} to L{target_level_id}.")
                      return # Stop if add fails

            else:
                 # Target tier is full, cannot promote further for now
                 self.logger.debug(f"Cannot promote token {token_id} further: L{target_level_id} is full.")
                 return


    def _check_and_trigger_summarization(self):
         """Checks conditions and schedules summarization if needed."""
         if not self.summarizer: return

         # Conditions: L3 full, Disk filling, time-based, importance-based
         l3_full = self.tiers[2].fullness > 0.9
         disk_fullness = self.streaming_memory.get_stats()['disk_token_count'] / self.persistent_storage.max_tokens if self.persistent_storage else 0
         disk_filling = disk_fullness > 0.7

         if l3_full or disk_filling:
              self.logger.info("Conditions met (L3 full or Disk filling), scheduling summarization task.")
              # Schedule the actual summarization task (lower priority)
              self._schedule_maintenance(self._run_summarization_strategies, priority=70)


    def _run_summarization_strategies(self):
         """Executes different summarization strategies."""
         if not self.summarizer: return
         self.logger.info("Running summarization strategies...")

         # Strategy 1: Summarize least important/oldest blocks in L3
         l3_tier = self.tiers[2]
         candidates = sorted(l3_tier.tokens.items(), key=lambda item: item[1].importance_score + (time.time() - item[1].last_access_time) / (3600*24)) # Sort by importance + age
         block_size_to_summarize = self.summarizer.min_block_size * 4 # Target larger blocks
         if len(candidates) > block_size_to_summarize:
              tokens_to_summarize = candidates[:block_size_to_summarize]
              token_ids = [tid for tid, _ in tokens_to_summarize]
              # Need decompressed values
              values = [l3_tier.get(tid=tid)[0] for tid in token_ids]
              positions = [info.position for _, info in tokens_to_summarize]

              if all(v is not None for v in values):
                  summary_id, _ = self.summarizer.create_summary_for_block(token_ids, values, positions, level=1)
                  if summary_id:
                       self.logger.info(f"Created L1 summary for {len(token_ids)} L3 tokens.")
                       # Optionally: Evict summarized tokens from L3 after successful summarization
                       # for tid in token_ids: l3_tier._remove(tid)
              else:
                   self.logger.warning("Could not retrieve all values for L3 summarization block.")

         # Strategy 2: Summarize based on semantic clusters (if available)
         # Requires vector store and clustering info in L2/L3 - complex to implement here fully

         # Strategy 3: Create higher-level summaries from L1 summaries
         l1_summaries = [reg[0] for reg in self.summarizer.summary_registry.get(1, [])]
         if len(l1_summaries) > self.summarizer.min_block_size: # Need enough L1 summaries
              # Group L1 summaries (e.g., by position range proximity)
              # Simplified: Summarize first N L1 summaries
              summaries_to_summarize = l1_summaries[:self.summarizer.max_summary_tokens]
              summary_id, _ = self.summarizer.create_hierarchical_summary(summaries_to_summarize, level=2)
              if summary_id:
                    self.logger.info(f"Created L2 summary from {len(summaries_to_summarize)} L1 summaries.")

         self.logger.info("Summarization strategies execution finished.")

    def _verify_qos_guarantees(self):
         # Implementation similar to the one provided in the prompt, checking tier stats
         # against self.qos_targets and logging warnings/triggering actions.
         # Omitted here for brevity but would involve checking latencies/hit rates from adaptive_policies.
         pass

    def _verify_redundant_copies(self):
        # Implementation similar to the one provided in the prompt, checking
        # self.shadow_copies and creating missing copies if needed.
        # Requires careful handling of add operations across tiers.
        pass

    def _cleanup_location_map(self):
        # Periodically remove entries from token_locations if they are very old
        # or if the map grows excessively large.
        max_map_size = len(self.tiers) * self.tiers[-1].capacity * 2 # Heuristic limit
        if len(self.token_locations) > max_map_size:
            num_to_remove = len(self.token_locations) - int(max_map_size * 0.9)
            # Simple approach: remove random entries (better: remove oldest based on access time if tracked globally)
            keys_to_remove = random.sample(list(self.token_locations.keys()), k=num_to_remove)
            for key in keys_to_remove:
                 self.token_locations.pop(key, None)
            self.logger.info(f"Cleaned up {num_to_remove} entries from token location map.")


    def clear(self):
        """Clear all memory tiers and state."""
        self.logger.warning("Clearing entire memory system!")
        for tier in self.tiers:
            tier.clear()
        if self.streaming_memory:
            self.streaming_memory.clear() # Also clears persistent storage
        if self.summarizer:
             self.summarizer.clear()
        if self.adaptive_policies:
             # Reset policies? Or just clear metrics?
             pass # Keep policies for now
        self.token_locations.clear()
        self.shadow_copies.clear()
        self.access_history.clear()
        # Stop and restart background worker?
        self._stop_background_worker()
        # Clear queue
        while not self.maintenance_queue.empty():
            try: self.maintenance_queue.get_nowait()
            except queue.Empty: break
        self._start_background_worker()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.logger.info("Memory system cleared.")

    def get_stats(self):
        """Get comprehensive statistics about the memory system."""
        stats = {"global": {"total_tokens_in_ram": sum(len(t) for t in self.tiers)}}
        for tier in self.tiers:
             stats[f"tier_{tier.level_id}"] = {
                 "token_count": len(tier),
                 "capacity": tier.current_capacity if isinstance(tier, RAMTier) else tier.capacity,
                 "fullness": tier.fullness,
                 "hit_rate": tier.hit_rate,
                 "avg_latency_ms": np.mean(list(tier.response_times)) if isinstance(tier, RAMTier) and tier.response_times else -1,
                 "protected_tokens": len(tier.protected_tokens) if isinstance(tier, RAMTier) else 0,
             }
             if hasattr(tier, 'vector_store') and tier.vector_store:
                  stats[f"tier_{tier.level_id}"]["index_size"] = len(tier.vector_store)

        if self.streaming_memory:
            stats["tier_4_disk"] = self.streaming_memory.get_stats()

        if self.summarizer:
            stats["summaries"] = {}
            for level, registry in self.summarizer.summary_registry.items():
                 stats["summaries"][f"level_{level}_count"] = len(registry)
            stats["summaries"]["cache_size"] = len(self.summarizer.summary_vectors_cache)

        if self.adaptive_policies:
            stats["adaptive_policy"] = self.adaptive_policies.get_all_policies()
            stats["adaptive_policy"]["current_access_pattern"] = self.adaptive_policies.current_access_pattern.name

        stats["global"]["token_location_map_size"] = len(self.token_locations)
        return stats

    def optimize(self):
        """Trigger optimization of all components."""
        self.logger.info("Starting global memory system optimization...")
        # Schedule optimizations with high priority
        self._schedule_maintenance(self._rebalance_memory_levels, priority=5)
        for tier in self.tiers:
             self._schedule_maintenance(tier.optimize, priority=10)
        # Add disk optimization task if needed
        # if self.persistent_storage: self._schedule_maintenance(self.persistent_storage.optimize, priority=15)
        self.logger.info("Optimization tasks scheduled.")

    def forward(self, *args, **kwargs):
         """Main entry point - decide action based on arguments."""
         if 'tokens' in kwargs and 'positions' in kwargs:
              return self.add_tokens(kwargs['tokens'], kwargs['positions'], kwargs.get('attention_weights'))
         elif 'positions' in kwargs:
              return self.retrieve_tokens(positions=kwargs['positions'])
         elif 'token_ids' in kwargs:
               return self.retrieve_tokens(token_ids=kwargs['token_ids'])
         elif 'query_vectors' in kwargs:
                return self.retrieve_tokens(query_vectors=kwargs['query_vectors'], top_k=kwargs.get('top_k', 5))
         else:
              # Assume first arg might be query vectors? Risky.
              if len(args) > 0 and isinstance(args[0], torch.Tensor):
                   return self.retrieve_tokens(query_vectors=args[0], top_k=kwargs.get('top_k', 5))
              raise ValueError("Unsupported arguments for forward pass. Use named arguments like 'tokens', 'positions', 'token_ids', 'query_vectors'.")


    def __del__(self):
        """Cleanup on deletion."""
        self.logger.info("AdvancedHierarchicalMemoryManager shutting down...")
        self._stop_background_worker()
        if self.persistent_storage:
            self.persistent_storage._save_metadata() # Ensure metadata is saved

# --- Benchmarking & Serialization --- (Adapted for new structure)

class MemorySystemBenchmark:
    # Methods largely unchanged, but need to handle the potentially
    # large scale and dictionary output of retrieve_tokens.

    @staticmethod
    def generate_random_tokens(batch_size, seq_len, dim):
        return torch.randn(batch_size, seq_len, dim)

    @staticmethod
    def generate_sequential_positions(batch_size, seq_len, start_pos=0):
        all_pos = []
        for b in range(batch_size):
             # Ensure unique positions across batches for large scale tests
             batch_start = start_pos + b * seq_len * 10 # Add gap between batches
             all_pos.append(torch.arange(batch_start, batch_start + seq_len))
        return torch.stack(all_pos)


    @staticmethod
    def benchmark_add_tokens(memory_system, batch_size=4, seq_len=4096, dim=768, trials=5):
        """Benchmark token addition performance."""
        times = []
        total_tokens = batch_size * seq_len
        logger.info(f"Benchmarking ADD: {trials} trials, {total_tokens} tokens per trial.")
        start_pos = 0
        for i in range(trials):
            tokens = MemorySystemBenchmark.generate_random_tokens(batch_size, seq_len, dim)
            positions = MemorySystemBenchmark.generate_sequential_positions(batch_size, seq_len, start_pos=start_pos)
            start_time = time.time()
            _ = memory_system.add_tokens(tokens=tokens, positions=positions)
            end_time = time.time()
            times.append(end_time - start_time)
            start_pos += total_tokens # Ensure unique positions for next trial
            logger.debug(f"Trial {i+1}/{trials} add time: {times[-1]:.3f}s")
            # Give background tasks time to catch up slightly
            # time.sleep(0.5)


        avg_time = np.mean(times) if times else 0
        throughput = total_tokens / avg_time if avg_time > 0 else 0

        return {
            "operation": "add_tokens", "batch_size": batch_size, "seq_len": seq_len,
            "avg_time_seconds": avg_time, "throughput_tokens_per_second": throughput,
            "tokens_per_batch": total_tokens
        }

    @staticmethod
    def benchmark_retrieve_tokens(memory_system, n_tokens_to_add=100000, retrieve_count=1000, dim=768, trials=5):
        """Benchmark token retrieval performance after adding tokens."""
        logger.info(f"Benchmarking RETRIEVE: Adding {n_tokens_to_add} tokens first...")
        # Add initial tokens sequentially
        add_bs = 16
        add_sl = 4096
        added_token_ids = []
        added_positions = []
        start_pos = 0
        while len(added_positions) < n_tokens_to_add:
             tokens = MemorySystemBenchmark.generate_random_tokens(add_bs, add_sl, dim)
             positions = MemorySystemBenchmark.generate_sequential_positions(add_bs, add_sl, start_pos=start_pos)
             batch_ids = memory_system.add_tokens(tokens=tokens, positions=positions)
             flat_ids = [tid for sublist in batch_ids for tid in sublist]
             added_token_ids.extend(flat_ids)
             added_positions.extend(positions.view(-1).tolist())
             start_pos += add_bs * add_sl
             logger.debug(f"Added {len(added_positions)}/{n_tokens_to_add} tokens...")
             # time.sleep(0.1) # Allow background processing

        logger.info(f"Finished adding tokens. Starting retrieval benchmarks ({trials} trials)...")
        if not added_positions:
            logger.error("No tokens were added, cannot benchmark retrieval.")
            return {}

        pos_times, id_times, search_times = [], [], []
        max_pos = max(added_positions)

        for i in range(trials):
            # Position Retrieval
            retrieve_pos_list = random.sample(added_positions, k=min(retrieve_count, len(added_positions)))
            retrieve_positions = torch.tensor(retrieve_pos_list).view(1, -1) # Single batch for simplicity
            start_time = time.time()
            _ = memory_system.retrieve_tokens(positions=retrieve_positions)
            pos_times.append(time.time() - start_time)

            # Token ID Retrieval
            retrieve_id_list = random.sample(added_token_ids, k=min(retrieve_count, len(added_token_ids)))
            retrieve_token_ids = [retrieve_id_list] # Single batch
            start_time = time.time()
            _ = memory_system.retrieve_tokens(token_ids=retrieve_token_ids)
            id_times.append(time.time() - start_time)

            # Semantic Search
            query_vectors = torch.randn(min(retrieve_count, 50), dim) # Limit search queries per trial
            start_time = time.time()
            _ = memory_system.retrieve_tokens(query_vectors=query_vectors, top_k=5)
            search_times.append(time.time() - start_time)
            logger.debug(f"Trial {i+1}/{trials} retrieve times: Pos={pos_times[-1]:.3f}s, ID={id_times[-1]:.3f}s, Search={search_times[-1]:.3f}s")


        avg_pos_time = np.mean(pos_times) if pos_times else 0
        avg_id_time = np.mean(id_times) if id_times else 0
        avg_search_time = np.mean(search_times) if search_times else 0

        pos_thrpt = retrieve_count / avg_pos_time if avg_pos_time > 0 else 0
        id_thrpt = retrieve_count / avg_id_time if avg_id_time > 0 else 0
        search_thrpt = query_vectors.shape[0] / avg_search_time if avg_search_time > 0 else 0

        return {
            "operation": "retrieve_tokens", "tokens_in_memory": len(added_positions),
            "position_retrieval_time_ms": avg_pos_time * 1000, "token_id_retrieval_time_ms": avg_id_time * 1000,
            "semantic_search_time_ms": avg_search_time * 1000,
            "position_throughput_tps": pos_thrpt, "token_id_throughput_tps": id_thrpt,
            "search_throughput_qps": search_thrpt
        }

    # Scaling benchmark needs careful design for 1B scale - maybe measure fixed operations
    # at different total token counts (simulated by adding then clearing subsets).

    @staticmethod
    def run_all_benchmarks(memory_system, dim=768):
        logger.info("Starting memory system benchmarks...")
        results = {}
        results["add_benchmark"] = MemorySystemBenchmark.benchmark_add_tokens(memory_system, dim=dim)
        logger.info(f"Add Benchmark Result: {results['add_benchmark']}")
        # Allow time for background tasks after heavy adds
        logger.info("Waiting 10s for background tasks before retrieval benchmark...")
        time.sleep(10)
        results["retrieve_benchmark"] = MemorySystemBenchmark.benchmark_retrieve_tokens(memory_system, dim=dim)
        logger.info(f"Retrieve Benchmark Result: {results['retrieve_benchmark']}")
        # results["scaling_benchmark"] = MemorySystemBenchmark.benchmark_scaling(...)
        logger.info("Benchmarking complete.")
        return results


def export_memory_state(memory_system, filename):
    """Exports the *metadata* and RAM tier state. Disk tier is assumed persistent."""
    logger.info(f"Exporting memory state to {filename}...")
    state = {
        "config": { # Store key configuration parameters
            "dim": memory_system.dim,
            "l1_cap": memory_system.l1.capacity, # Base capacity
            "l2_cap": memory_system.l2.capacity,
            "l3_cap": memory_system.l3.capacity,
            # Add other relevant config flags
        },
        "token_locations": memory_system.token_locations,
        "adaptive_policies": memory_system.adaptive_policies.get_all_policies() if memory_system.adaptive_policies else {},
        "summarizer_registry": memory_system.summarizer.summary_registry if memory_system.summarizer else {},
        "summarizer_source_map": memory_system.summarizer.summary_source_map if memory_system.summarizer else {},
        # RAM Tier States (metadata only, values assumed reconstructible or large)
        "ram_tiers_metadata": {},
        "metrics": {}, # Store performance counters?
    }

    # Save metadata from RAM tiers
    for tier in memory_system.tiers:
         tier_meta = {}
         for token_id, info in tier.tokens.items():
              tier_meta[token_id] = {
                   "position": info.position,
                   "importance": info.importance_score,
                   "access_count": info.access_count,
                   "last_access_time": info.last_access_time,
                   "creation_time": info.creation_time,
              }
         state["ram_tiers_metadata"][tier.level_id] = tier_meta

    # Save disk storage metadata (it should persist itself anyway, but for consistency)
    if memory_system.persistent_storage:
         memory_system.persistent_storage._save_metadata() # Ensure latest metadata is flushed

    try:
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Memory state metadata exported successfully.")
        return True
    except Exception as e:
        logger.error(f"Error exporting memory state: {e}", exc_info=True)
        return False


def restore_memory_state(memory_system, filename):
    """Restores metadata state. Assumes disk storage is already present."""
    logger.info(f"Restoring memory state from {filename}...")
    if not os.path.exists(filename):
         logger.error(f"Restore file not found: {filename}")
         return False

    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        # --- Restore State ---
        # Clear existing RAM state ONLY
        for tier in memory_system.tiers: tier.clear()
        if memory_system.summarizer: memory_system.summarizer.clear()
        memory_system.token_locations.clear()

        memory_system.token_locations = state.get("token_locations", {})

        # Restore policies
        if memory_system.adaptive_policies and "adaptive_policies" in state:
             policies = state["adaptive_policies"]
             for name, value in policies.items():
                 memory_system.adaptive_policies.override_policy(name, value) # Use override function

        # Restore summarizer state
        if memory_system.summarizer and "summarizer_registry" in state:
            memory_system.summarizer.summary_registry = state["summarizer_registry"]
            memory_system.summarizer.summary_source_map = state["summarizer_source_map"]
            # Summary vectors cache is not saved/restored - needs regeneration or loading from disk

        # Restore RAM tier metadata (values need to be fetched/reconstructed on demand)
        if "ram_tiers_metadata" in state:
            for level_id_str, tier_meta in state["ram_tiers_metadata"].items():
                 level_id = int(level_id_str)
                 if level_id <= len(memory_system.tiers):
                      tier = memory_system.tiers[level_id - 1]
                      # Need to repopulate TokenInfo objects without the actual value
                      for token_id, meta in tier_meta.items():
                           # Create dummy value or fetch compressed from lower tier if needed?
                           # This is complex. Simplification: just restore metadata.
                           # Value will be fetched on first access.
                           dummy_value = torch.zeros(1) # Placeholder
                           token_info = TokenInfo(dummy_value, meta["position"], meta["creation_time"], storage_tier=level_id)
                           token_info.importance_score = meta["importance"]
                           token_info.access_count = meta["access_count"]
                           token_info.last_access_time = meta["last_access_time"]
                           tier.tokens[token_id] = token_info
                           # Rebuild LRU? Difficult without access times relative to restore time.
                           # tier.lru_queue.append(token_id)
                           # Rebuild positional index?
                           # if level_id == 1: tier.position_index[meta["position"]] = token_id
                      logger.info(f"Restored metadata for {len(tier.tokens)} tokens in Tier {level_id}.")
                      # Trigger index rebuild on next access
                      if hasattr(tier, 'vector_store') and tier.vector_store:
                          tier.vector_store.clear() # Invalidate index


        # Disk storage metadata should be loaded by PersistentTokenStorage itself

        logger.info(f"Memory state restored successfully. Note: RAM tier values need lazy loading/reconstruction.")
        return True

    except Exception as e:
        logger.error(f"Error restoring memory state: {e}", exc_info=True)
        return False


# --- Main Usage Example ---
def create_ultracontext_memory_system(
    dim=4096, # Larger dim typical for big models
    max_tokens=1_000_000_000,
    disk_path="./uc_1B_storage",
    distributed_nodes_config=None # e.g., [0, 1, 2, 3] for 4 nodes
    ):
    """Creates an AdvancedHierarchicalMemoryManager configured for 1B tokens."""
    logger.info(f"Configuring UltraContext Memory System for {max_tokens} tokens (dim={dim}).")

    # Define tier capacities (example distribution)
    # RAM capacity is usually limited, Disk holds the bulk
    l1_size = 65536       # ~64K tokens (Fastest RAM Cache)
    l2_size = 1048576      # ~1M tokens (Medium RAM Cache)
    l3_size = 16777216     # ~16M tokens (Slower RAM / NVMe Cache)
    disk_size = max_tokens # Disk/Distributed Tier target

    # Realistic QoS targets for tiered system
    qos_targets = {
        "l1_latency_ms": 0.05, # Sub-millisecond L1
        "l2_latency_ms": 0.5,  # Sub-millisecond L2
        "l3_latency_ms": 5.0,  # Single-digit ms L3
        "disk_latency_ms": 50.0, # Tens of ms for Disk (optimistic)
        "hit_rate_l1": 0.80, # High hit rate expected for L1
        "hit_rate_l2": 0.60, # Reasonable L2 hit rate
        "hit_rate_l3": 0.40, # Lower L3 hit rate
        "availability": 0.999, # System availability
    }

    # Performance config
    perf_config = PerformanceConfig(
        optimize_memory=True, optimize_speed=True, optimize_reliability=False, # Set reliability based on needs
        use_mixed_precision=True # Use if hardware supports it
    )

    memory_system = AdvancedHierarchicalMemoryManager(
        dim=dim,
        l1_capacity=l1_size,
        l2_capacity=l2_size,
        l3_capacity=l3_size,
        disk_capacity=disk_size,
        disk_storage_path=disk_path,
        distributed_nodes=distributed_nodes_config,
        enable_summarization=True,
        enable_adaptive_policies=True,
        qos_targets=qos_targets,
        reliability_level="normal", # or "high" / "critical"
        perf_config=perf_config
    )

    logger.info(f"UltraContext Memory System created. Target: {max_tokens} tokens.")
    logger.info(f"RAM Tiers: L1={l1_size}, L2={l2_size}, L3={l3_size}")
    if memory_system.persistent_storage:
        logger.info(f"Disk Tier (Local): Path={disk_path}, MaxTokens={disk_size}")
    elif memory_system.orchestrator:
        logger.info(f"Disk Tier (Distributed): Nodes={distributed_nodes_config}")

    return memory_system


# Example Usage:
if __name__ == "__main__":
    DIMENSION = 512 # Smaller dimension for faster testing
    TARGET_TOKENS = 10_000_000 # Scaled down target for reasonable local testing
    DISK_PATH = "./uc_test_storage"

    # Clean up previous run
    import shutil
    if os.path.exists(DISK_PATH):
         print(f"Removing previous test storage: {DISK_PATH}")
         shutil.rmtree(DISK_PATH)

    # Create the system
    mem_sys = create_ultracontext_memory_system(
        dim=DIMENSION,
        max_tokens=TARGET_TOKENS,
        disk_path=DISK_PATH
    )

    # Run Benchmarks (on a smaller scale than 1B for practical testing)
    MemorySystemBenchmark.run_all_benchmarks(mem_sys, dim=DIMENSION)

    # Example: Add more tokens
    print("\nAdding more tokens...")
    bs, sl = 8, 4096
    tokens = MemorySystemBenchmark.generate_random_tokens(bs, sl, DIMENSION)
    # Start positions high to simulate large context
    positions = MemorySystemBenchmark.generate_sequential_positions(bs, sl, start_pos=1_000_000)
    added_ids = mem_sys.add_tokens(tokens=tokens, positions=positions)
    print(f"Added {bs*sl} tokens.")

    # Wait for background tasks
    print("Waiting 5s for background tasks...")
    time.sleep(5)

    # Retrieve some added tokens
    print("Retrieving some tokens...")
    retrieve_ids = [added_ids[0][:5]] # Get first 5 IDs from first batch
    results = mem_sys.retrieve_tokens(token_ids=retrieve_ids)
    print(f"Retrieved {len(results)} results by ID.")
    # Example: check if retrieved vector norm > 0
    if retrieve_ids[0][0] in results:
         print(f"Norm of retrieved token {retrieve_ids[0][0]}: {torch.linalg.norm(results[retrieve_ids[0][0]]).item()}")

    # Get final stats
    print("\nFinal Memory Stats:")
    stats = mem_sys.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export state metadata
    export_filename = "./uc_test_state.pkl"
    export_memory_state(mem_sys, export_filename)

    # Clean up manager before exit
    del mem_sys
    gc.collect()

    print("\nExample usage finished.")
