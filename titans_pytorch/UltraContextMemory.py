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

# Assuming UltraContext Core is imported
from ultracontext.core import (
    DEFAULT_PERF_CONFIG,
    PerformanceConfig,
    get_norm_class,
    ActivationFunctions,
    timer
)

logger = logging.getLogger("ultracontext.memory")

# Memory access pattern tracking
class MemoryAccessPattern(Enum):
    """Types of memory access patterns the system can identify"""
    SEQUENTIAL = auto()     # Accessing tokens in sequence
    LOCAL = auto()          # Accessing tokens in local neighborhoods
    RANDOM = auto()         # Random access patterns  
    REPEATED = auto()       # Frequently accessing the same tokens
    RECENCY = auto()        # Accessing recent tokens
    START_FOCUSED = auto()  # Focusing on beginning of context
    END_FOCUSED = auto()    # Focusing on end of context
    SEMANTIC = auto()       # Semantically related tokens

class TokenInfo:
    """Information about a token in the memory system"""
    def __init__(self, value, position, creation_time=None):
        self.value = value  # Token embedding/representation
        self.position = position  # Original position in sequence
        self.creation_time = creation_time or time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0
        self.access_positions = []  # Track where this token was accessed from
        self.importance_score = 0.5  # Dynamic importance score (0-1)
        
    def record_access(self, position=None):
        """Record an access to this token"""
        self.last_access_time = time.time()
        self.access_count += 1
        if position is not None:
            self.access_positions.append(position)
            
    @property
    def recency(self):
        """How recently this token was accessed (lower is more recent)"""
        return time.time() - self.last_access_time
        
    @property
    def age(self):
        """Age of this token (how long since creation)"""
        return time.time() - self.creation_time
        
    @property
    def frequency(self):
        """How frequently this token is accessed"""
        age = max(1e-6, self.age)  # Avoid division by zero
        return self.access_count / age
        
    def update_importance(self, attention_weight=None):
        """Update importance score based on access patterns and optional attention weight"""
        # Base importance on frequency and recency
        recency_factor = math.exp(-self.recency / 3600)  # Exponential decay over an hour
        frequency_factor = min(1.0, self.access_count / 10)  # Saturate at 10 accesses
        
        # Calculate base importance
        base_importance = 0.3 * recency_factor + 0.3 * frequency_factor
        
        # If attention weight is provided, incorporate it
        if attention_weight is not None:
            self.importance_score = 0.4 * base_importance + 0.6 * attention_weight
        else:
            # Update using EMA
            self.importance_score = 0.8 * self.importance_score + 0.2 * base_importance
        
        return self.importance_score

# Vector database storage for tokens
class TokenVectorStore:
    """Storage for token vectors with fast retrieval capabilities"""
    def __init__(self, dim, max_tokens=1_000_000, similarity_threshold=0.7, index_type="flat"):
        self.dim = dim
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.index_type = index_type
        
        # Storage for tokens
        self.token_vectors = {}  # id -> vector
        self.token_metadata = {}  # id -> metadata
        
        # Index structures
        self.position_index = {}  # position -> id
        self.semantic_index = None  # Will be initialized on first semantic search
        self.semantic_index_mapping = []  # Maps index positions to token ids
        
        # For approximate nearest neighbor indexing
        self._index_needs_training = False
        
    def add(self, token_id, vector, position, metadata=None):
        """Add a token vector to the store"""
        if len(self.token_vectors) >= self.max_tokens:
            # Evict least recently used
            oldest_id = min(
                self.token_metadata.keys(),
                key=lambda tid: self.token_metadata[tid].get("last_access", 0)
            )
            self.remove(oldest_id)
            
        # Store the vector and metadata
        self.token_vectors[token_id] = vector
        self.token_metadata[token_id] = metadata or {}
        self.token_metadata[token_id]["position"] = position
        self.token_metadata[token_id]["last_access"] = time.time()
        
        # Update indices
        self.position_index[position] = token_id
        
        # Invalidate semantic index if it exists
        if self.semantic_index is not None and self.semantic_index != "simple":
            try:
                # Add to existing index if possible
                import faiss
                vector_norm = F.normalize(vector, p=2, dim=0)
                vector_np = vector_norm.cpu().numpy().reshape(1, -1).astype(np.float32)
                
                # Check if index needs training first
                if self._index_needs_training and len(self.token_vectors) >= 256:
                    self._ensure_semantic_index()
                elif not self._index_needs_training:
                    self.semantic_index.add(vector_np)
                    self.semantic_index_mapping.append(token_id)
            except Exception as e:
                logger.warning(f"Error adding to semantic index: {e}")
                self.semantic_index = None
            
    def remove(self, token_id):
        """Remove a token from the store"""
        if token_id in self.token_vectors:
            # Remove from position index
            position = self.token_metadata[token_id].get("position")
            if position in self.position_index:
                del self.position_index[position]
                
            # Remove the token
            del self.token_vectors[token_id]
            del self.token_metadata[token_id]
            
            # Invalidate semantic index
            if self.semantic_index is not None:
                # Removing from FAISS index is expensive, so we just invalidate it
                # and it will be rebuilt on next search
                self.semantic_index = None
                self.semantic_index_mapping = []
                
    def get_by_id(self, token_id):
        """Get a token by ID"""
        if token_id in self.token_vectors:
            # Update last access time
            self.token_metadata[token_id]["last_access"] = time.time()
            return self.token_vectors[token_id], self.token_metadata[token_id]
        return None, None
        
    def get_by_position(self, position):
        """Get a token by its position"""
        token_id = self.position_index.get(position)
        if token_id:
            return self.get_by_id(token_id)
        return None, None
        
    def _setup_advanced_index(self):
        """Set up an advanced semantic index based on index_type"""
        try:
            import faiss
            
            # Initialize index based on type
            if self.index_type == "flat":
                # Basic flat index (exact search)
                self.semantic_index = faiss.IndexFlatIP(self.dim)
                
            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World (approximate but fast)
                self.semantic_index = faiss.IndexHNSWFlat(self.dim, 32)  # 32 neighbors
                
            elif self.index_type == "ivf":
                # Inverted File Index (faster search with some accuracy loss)
                quantizer = faiss.IndexFlatIP(self.dim)
                n_centroids = min(4096, max(256, self.max_tokens // 100))
                self.semantic_index = faiss.IndexIVFFlat(quantizer, self.dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
                
                # Need to train this index (will be done as vectors are added)
                self._index_needs_training = True
                
            elif self.index_type == "ivfpq":
                # IVF with Product Quantization (very fast, more accuracy loss)
                quantizer = faiss.IndexFlatIP(self.dim)
                n_centroids = min(4096, max(256, self.max_tokens // 100))
                
                # Product quantization parameters
                m = 8  # Number of subquantizers
                bits = 8  # Bits per subquantizer
                
                self.semantic_index = faiss.IndexIVFPQ(
                    quantizer, self.dim, n_centroids, m, bits, faiss.METRIC_INNER_PRODUCT
                )
                
                # Need to train this index
                self._index_needs_training = True
                
            else:
                # Default to flat
                self.semantic_index = faiss.IndexFlatIP(self.dim)
                
            # Move to GPU if available
            if torch.cuda.is_available():
                try:
                    res = faiss.StandardGpuResources()
                    self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index)
                except Exception as e:
                    logger.warning(f"Failed to move FAISS index to GPU: {e}")
                    
            logger.info(f"Created {self.index_type} FAISS index for dimensions: {self.dim}")
            
        except ImportError:
            logger.warning("FAISS not available, falling back to simple index")
            self.semantic_index = "simple"
        
    def _ensure_semantic_index(self):
        """Ensure the semantic index is built"""
        if self.semantic_index is None and len(self.token_vectors) > 0:
            try:
                self._setup_advanced_index()
                
                # If we need to train the index and have enough vectors
                if self._index_needs_training and len(self.token_vectors) >= 256:
                    import faiss
                    # Gather all vectors
                    vectors = torch.stack(list(self.token_vectors.values()))
                    vectors = F.normalize(vectors, p=2, dim=1)  # Normalize for cosine similarity
                    
                    # Train the index
                    self.semantic_index.train(vectors.cpu().numpy())
                    self._index_needs_training = False
                
                # Add vectors to index
                if not self._index_needs_training:
                    # Add all vectors to index
                    vectors = torch.stack(list(self.token_vectors.values()))
                    vectors = F.normalize(vectors, p=2, dim=1)  # Normalize for cosine similarity
                    
                    # Add to index
                    self.semantic_index.add(vectors.cpu().numpy())
                    
                    # Store id mapping
                    self.semantic_index_mapping = list(self.token_vectors.keys())
                    
            except ImportError:
                logger.warning("FAISS not available, semantic search will be slower")
                # Fall back to simple implementation
                self.semantic_index = "simple"
                
    def search_similar(self, query_vector, top_k=5):
        """Find similar tokens to the query vector"""
        # Normalize query vector
        query_vector = F.normalize(query_vector, p=2, dim=0)
        
        # Ensure semantic index exists
        self._ensure_semantic_index()
        
        if self.semantic_index == "simple":
            # Simple implementation without FAISS
            similarities = {}
            for token_id, vector in self.token_vectors.items():
                vector = F.normalize(vector, p=2, dim=0)
                similarity = torch.dot(query_vector, vector).item()
                similarities[token_id] = similarity
                
            # Get top k
            top_ids = sorted(similarities.keys(), key=lambda tid: similarities[tid], reverse=True)[:top_k]
            results = []
            
            for tid in top_ids:
                if similarities[tid] >= self.similarity_threshold:
                    results.append((tid, self.token_vectors[tid], similarities[tid], self.token_metadata[tid]))
                    
            return results
        elif self.semantic_index is not None:
            # Use FAISS for fast similarity search
            query_np = query_vector.unsqueeze(0).cpu().numpy()
            
            # For IVF-based indexes, increase nprobe for better recall
            if self.index_type in ["ivf", "ivfpq"]:
                try:
                    self.semantic_index.nprobe = 8  # Search more centroids
                except:
                    pass
                    
            distances, indices = self.semantic_index.search(query_np, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.semantic_index_mapping):
                    token_id = self.semantic_index_mapping[idx]
                    similarity = float(distances[0][i])
                    
                    if similarity >= self.similarity_threshold:
                        results.append((
                            token_id,
                            self.token_vectors[token_id],
                            similarity,
                            self.token_metadata[token_id]
                        ))
                        
            return results
        
        return []  # Empty result if no index
        
    def batch_search_similar(self, query_vectors, top_k=5):
        """Perform similarity search for multiple queries at once"""
        # Normalize query vectors
        query_vectors = F.normalize(query_vectors, p=2, dim=1)
        
        # Ensure semantic index exists
        self._ensure_semantic_index()
        
        batch_results = []
        
        if self.semantic_index == "simple":
            # Process each query individually using the simple implementation
            for i in range(query_vectors.shape[0]):
                results = self.search_similar(query_vectors[i], top_k)
                batch_results.append(results)
                
        elif self.semantic_index is not None:
            # Use FAISS for fast batch similarity search
            queries_np = query_vectors.cpu().numpy()
            
            # For IVF-based indexes, increase nprobe for better recall
            if self.index_type in ["ivf", "ivfpq"]:
                try:
                    self.semantic_index.nprobe = 8  # Search more centroids
                except:
                    pass
                    
            distances, indices = self.semantic_index.search(queries_np, top_k)
            
            # Process each query's results
            for q in range(queries_np.shape[0]):
                results = []
                for i, idx in enumerate(indices[q]):
                    if idx >= 0 and idx < len(self.semantic_index_mapping):
                        token_id = self.semantic_index_mapping[idx]
                        similarity = float(distances[q][i])
                        
                        if similarity >= self.similarity_threshold:
                            results.append((
                                token_id,
                                self.token_vectors[token_id],
                                similarity,
                                self.token_metadata[token_id]
                            ))
                            
                batch_results.append(results)
                
        return batch_results

    def clear(self):
        """Clear all tokens from store"""
        self.token_vectors.clear()
        self.token_metadata.clear()
        self.position_index.clear()
        self.semantic_index = None
        self.semantic_index_mapping = []
        
    def __len__(self):
        """Return the number of tokens in the store"""
        return len(self.token_vectors)
        
    def optimize(self):
        """Optimize the vector store for better performance"""
        # Rebuild the semantic index from scratch
        self.semantic_index = None
        self._ensure_semantic_index()

# Adaptive importance scoring module
class ImportanceScorer(Module):
    """
    Scores token importance based on multiple factors:
    - Attention weights from the model
    - Access patterns
    - Semantic relevance to queries
    - Position in sequence
    - Token rarity/information content
    """
    def __init__(
        self, 
        dim: int,
        importance_dim: int = 64,
        use_attention_weights: bool = True,
        use_positional_bias: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.dim = dim
        self.importance_dim = importance_dim
        self.use_attention_weights = use_attention_weights
        self.use_positional_bias = use_positional_bias
        
        # Neural network for importance scoring
        self.project = nn.Linear(dim, importance_dim)
        self.positional_embedding = nn.Embedding(65536, importance_dim)  # Max position 65K
        
        self.score_mlp = nn.Sequential(
            nn.Linear(importance_dim * 2 if use_positional_bias else importance_dim, importance_dim),
            nn.GELU(),
            nn.Linear(importance_dim, importance_dim // 2),
            nn.GELU(),
            nn.Linear(importance_dim // 2, 1),
            nn.Sigmoid()  # Output importance in [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
        
        # For tracking global token statistics
        self.token_frequency = {}  # Token embedding hash -> frequency
        self.max_frequency = 1.0
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize projection layer
        nn.init.normal_(self.project.weight, std=0.02)
        nn.init.zeros_(self.project.bias)
        
        # Initialize positional embedding
        nn.init.normal_(self.positional_embedding.weight, std=0.02)
        
        # Initialize MLP
        for module in self.score_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def update_token_statistics(self, token_vector):
        """Track token frequency statistics"""
        # Create a hash for the token vector
        token_hash = hash(token_vector.cpu().numpy().tobytes())
        
        # Update frequency
        if token_hash in self.token_frequency:
            self.token_frequency[token_hash] += 1
        else:
            self.token_frequency[token_hash] = 1
            
        # Update max frequency
        self.max_frequency = max(self.max_frequency, self.token_frequency[token_hash])
        
        # Return normalized frequency (rarity = 1 - frequency)
        return 1.0 - (self.token_frequency[token_hash] / self.max_frequency)
        
    def forward(self, token_vectors, positions=None, attention_weights=None):
        """
        Calculate importance scores for token vectors
        
        Args:
            token_vectors: Token vectors to score [batch_size, seq_len, dim]
            positions: Token positions [batch_size, seq_len]
            attention_weights: Optional attention weights [batch_size, seq_len]
            
        Returns:
            Importance scores [batch_size, seq_len]
        """
        batch_size, seq_len, _ = token_vectors.shape
        
        # Project to importance dimension
        projected = self.project(token_vectors)  # [batch_size, seq_len, importance_dim]
        
        # Add positional information if enabled
        if self.use_positional_bias and positions is not None:
            pos_emb = self.positional_embedding(positions)  # [batch_size, seq_len, importance_dim]
            features = torch.cat([projected, pos_emb], dim=-1)
        else:
            features = projected
            
        # Calculate base neural importance score
        importance = self.score_mlp(features).squeeze(-1)  # [batch_size, seq_len]
        
        # Incorporate attention weights if available
        if self.use_attention_weights and attention_weights is not None:
            # Combine neural score with attention weights
            importance = 0.7 * importance + 0.3 * attention_weights
            
        return importance
        
    def score_single_token(self, token_vector, position=None, access_stats=None):
        """Score a single token vector"""
        # Project to importance dimension
        projected = self.project(token_vector.unsqueeze(0))  # [1, importance_dim]
        
        # Add positional information if enabled
        if self.use_positional_bias and position is not None:
            pos_tensor = torch.tensor([position], device=token_vector.device)
            pos_emb = self.positional_embedding(pos_tensor)  # [1, importance_dim]
            features = torch.cat([projected, pos_emb], dim=-1)
        else:
            features = projected
            
        # Calculate base neural importance score
        importance = self.score_mlp(features).squeeze().item()  # Scalar
        
        # Incorporate token rarity
        rarity = self.update_token_statistics(token_vector)
        
        # Incorporate access statistics if available
        if access_stats is not None:
            recency_factor = math.exp(-access_stats.get("recency", 0) / 3600)
            frequency_factor = min(1.0, access_stats.get("frequency", 0) / 10)
            
            # Combine factors
            importance = 0.5 * importance + 0.3 * rarity + 0.1 * recency_factor + 0.1 * frequency_factor
            
        return importance

# Memory level with enhanced capabilities
class EnhancedMemoryLevel(MemoryLevel):
    """
    Enhanced memory level with advanced features:
    - Dynamic capacity adjustment
    - Advanced eviction policies
    - Semantic clustering
    - Quality-of-Service guarantees
    """
    def __init__(
        self,
        level_id: int,
        dim: int,
        capacity: int,
        retrieval_cost: float,
        storage_cost: float,
        compression_ratio: float = 1.0,
        eviction_policy: str = "lru",
        qos_enabled: bool = False,
        semantic_clustering: bool = False,
        index_type: str = "flat"
    ):
        super().__init__(
            level_id=level_id,
            dim=dim,
            capacity=capacity,
            retrieval_cost=retrieval_cost,
            storage_cost=storage_cost,
            compression_ratio=compression_ratio,
            eviction_policy=eviction_policy
        )
        
        # Enhanced features
        self.qos_enabled = qos_enabled
        self.semantic_clustering = semantic_clustering
        
        # Replace vector store with enhanced version
        if level_id >= 2:
            self.vector_store = TokenVectorStore(
                dim=dim, 
                max_tokens=capacity, 
                index_type=index_type
            )
            
        # QoS tracking
        self.response_times = deque(maxlen=1000)
        self.qos_target = 0.5  # Target retrieval time in ms
        
        # For semantic clustering
        if semantic_clustering:
            self.clusters = {}  # cluster_id -> set of token_ids
            self.token_clusters = {}  # token_id -> cluster_id
            
        # Dynamic capacity adjustment
        self.min_capacity = capacity // 2
        self.max_capacity = capacity * 2
        self.current_capacity = capacity
        self.capacity_adjustment_interval = 60  # seconds
        self.last_capacity_adjustment = time.time()
        
        # For eviction protection
        self.protected_tokens = set()  # Set of tokens that should not be evicted
        
    def add(self, token_id, value, position, importance=None):
        """Add a token to this memory level with optional importance"""
        # Check if we need to adjust capacity
        self._maybe_adjust_capacity()
        
        # Check if we need to evict
        if len(self.tokens) >= self.current_capacity:
            self._evict()
            
        # Create token info
        token_info = TokenInfo(value, position)
        
        # Set importance if provided
        if importance is not None:
            token_info.importance_score = importance
            
            # Protect high-importance tokens
            if importance > 0.8:
                self.protected_tokens.add(token_id)
            
        # Store the token
        self.tokens[token_id] = token_info
        self.position_index[position] = token_id
        
        # Update LRU queue
        self.lru_queue.append(token_id)
        
        # Add to vector store if available
        if self.vector_store is not None:
            self.vector_store.add(
                token_id, 
                value, 
                position,
                {
                    "creation_time": token_info.creation_time,
                    "importance": importance if importance is not None else 0.5
                }
            )
            
        # Add to semantic clusters if enabled
        if self.semantic_clustering:
            self._add_to_cluster(token_id, value)
            
    def get(self, token_id=None, position=None):
        """Retrieve a token by ID or position with QoS tracking"""
        start_time = time.time()
        
        self.access_count += 1
        
        # Look up by position if provided
        if token_id is None and position is not None:
            token_id = self.position_index.get(position)
            
        # Retrieve token
        token_info = self.tokens.get(token_id)
        
        if token_info:
            # Record the access
            token_info.record_access(position)
            
            # Update LRU (move to end)
            try:
                self.lru_queue.remove(token_id)
            except ValueError:
                pass  # Not in queue
            self.lru_queue.append(token_id)
            
            self.hit_count += 1
            
            # Track response time for QoS
            response_time = (time.time() - start_time) * 1000  # ms
            self.response_times.append(response_time)
            
            return token_info.value, token_info.position
        else:
            self.miss_count += 1
            return None, None
            
    def _maybe_adjust_capacity(self):
        """Dynamically adjust capacity based on usage patterns"""
        now = time.time()
        
        # Only adjust periodically
        if now - self.last_capacity_adjustment < self.capacity_adjustment_interval:
            return
            
        self.last_capacity_adjustment = now
        
        # Check hit rate
        hit_rate = self.hit_rate
        
        # Get average response time if QoS is enabled
        avg_response_time = None
        if self.qos_enabled and self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            
        # Adjust capacity based on hit rate and response time
        if hit_rate < 0.5:
            # Low hit rate, increase capacity
            new_capacity = min(self.max_capacity, int(self.current_capacity * 1.1))
        elif self.qos_enabled and avg_response_time is not None and avg_response_time > self.qos_target:
            # Slow response time, decrease capacity
            new_capacity = max(self.min_capacity, int(self.current_capacity * 0.9))
        elif hit_rate > 0.9 and len(self.tokens) < self.current_capacity * 0.8:
            # High hit rate with spare capacity, slightly decrease
            new_capacity = max(self.min_capacity, int(self.current_capacity * 0.95))
        else:
            # No change needed
            return
            
        # Apply the new capacity
        logger.info(f"Level {self.level_id} adjusted capacity: {self.current_capacity} -> {new_capacity}")
        self.current_capacity = new_capacity
            
    def _add_to_cluster(self, token_id, vector):
        """Add token to the appropriate semantic cluster"""
        if not self.semantic_clustering:
            return
            
        # Find the closest cluster
        if not self.clusters:
            # Create first cluster
            cluster_id = 0
            self.clusters[cluster_id] = {token_id}
            self.token_clusters[token_id] = cluster_id
            return
            
        # Find nearest cluster
        closest_cluster = None
        best_similarity = -1.0
        
        # Normalize vector
        vector_norm = F.normalize(vector, p=2, dim=0)
        
        for cluster_id, token_ids in self.clusters.items():
            # Get a sample of tokens from the cluster
            sample_size = min(5, len(token_ids))
            sample_ids = list(token_ids)[:sample_size]
            
            # Calculate average similarity to cluster
            total_sim = 0.0
            for sample_id in sample_ids:
                sample_vector = self.tokens[sample_id].value
                sample_norm = F.normalize(sample_vector, p=2, dim=0)
                sim = torch.dot(vector_norm, sample_norm).item()
                total_sim += sim
                
            avg_sim = total_sim / sample_size
            
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                closest_cluster = cluster_id
                
        # Check if similarity is high enough to join cluster
        if best_similarity > 0.7:
            # Add to existing cluster
            self.clusters[closest_cluster].add(token_id)
            self.token_clusters[token_id] = closest_cluster
        else:
            # Create new cluster
            new_cluster_id = max(self.clusters.keys()) + 1 if self.clusters else 0
            self.clusters[new_cluster_id] = {token_id}
            self.token_clusters[token_id] = new_cluster_id
            
    def _evict(self):
        """Evict tokens according to the eviction policy"""
        if not self.tokens:
            return  # Nothing to evict
            
        # Don't evict protected tokens if possible
        candidates = set(self.tokens.keys()) - self.protected_tokens
        
        # If all tokens are protected but we need to evict, use all tokens
        if not candidates and len(self.tokens) >= self.current_capacity:
            candidates = set(self.tokens.keys())
            
        if not candidates:
            return  # Nothing to evict
            
        if self.eviction_policy == "lru":
            # Evict least recently used
            while self.lru_queue and len(self.tokens) >= self.current_capacity:
                # Find the first non-protected token in LRU queue
                while self.lru_queue and self.lru_queue[0] not in candidates:
                    self.lru_queue.popleft()  # Skip protected tokens
                    
                if self.lru_queue:
                    token_id = self.lru_queue.popleft()
                    self._remove(token_id)
                else:
                    break
                
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            if candidates:
                token_id = min(
                    candidates,
                    key=lambda tid: self.tokens[tid].frequency
                )
                self._remove(token_id)
                
        elif self.eviction_policy == "importance":
            # Evict least important tokens
            if candidates:
                token_id = min(
                    candidates,
                    key=lambda tid: self.tokens[tid].importance_score
                )
                self._remove(token_id)
                
        elif self.eviction_policy == "semantic":
            # Semantic clustering-based eviction
            if self.semantic_clustering and candidates:
                # Identify over-represented clusters
                cluster_sizes = {}
                for token_id in candidates:
                    cluster_id = self.token_clusters.get(token_id)
                    if cluster_id is not None:
                        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
                
                # Find largest cluster
                largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
                
                # Find least important token in that cluster
                cluster_tokens = [tid for tid in candidates 
                                if self.token_clusters.get(tid) == largest_cluster]
                
                if cluster_tokens:
                    token_id = min(
                        cluster_tokens,
                        key=lambda tid: self.tokens[tid].importance_score
                    )
                    self._remove(token_id)
                else:
                    # Fall back to importance-based eviction
                    token_id = min(
                        candidates,
                        key=lambda tid: self.tokens[tid].importance_score
                    )
                    self._remove(token_id)
                    
        elif self.eviction_policy == "adaptive":
            # Adaptive policy based on access patterns and importance
            if candidates:
                token_id = min(
                    candidates,
                    key=lambda tid: (
                        0.4 * (1.0 / max(1e-6, self.tokens[tid].recency)) + 
                        0.3 * self.tokens[tid].frequency +
                        0.3 * self.tokens[tid].importance_score
                    )
                )
                self._remove(token_id)
                
    def _remove(self, token_id):
        """Remove a token from this level"""
        if token_id in self.tokens:
            # Remove from position index
            position = self.tokens[token_id].position
            if position in self.position_index:
                del self.position_index[position]
                
            # Remove from vector store if available
            if self.vector_store is not None:
                self.vector_store.remove(token_id)
                
            # Remove from semantic clusters if enabled
            if self.semantic_clustering:
                cluster_id = self.token_clusters.get(token_id)
                if cluster_id is not None:
                    self.clusters[cluster_id].discard(token_id)
                    if not self.clusters[cluster_id]:
                        del self.clusters[cluster_id]
                    del self.token_clusters[token_id]
                    
            # Remove from protected tokens if present
            self.protected_tokens.discard(token_id)
                
            # Remove the token
            del self.tokens[token_id]
            
    def protect_token(self, token_id):
        """Mark a token as protected from eviction"""
        if token_id in self.tokens:
            self.protected_tokens.add(token_id)
            
    def unprotect_token(self, token_id):
        """Remove protection from a token"""
        self.protected_tokens.discard(token_id)
        
    def get_cluster_info(self):
        """Get information about semantic clusters"""
        if not self.semantic_clustering:
            return {}
            
        return {
            "num_clusters": len(self.clusters),
            "cluster_sizes": {cid: len(tokens) for cid, tokens in self.clusters.items()},
            "avg_cluster_size": sum(len(tokens) for tokens in self.clusters.values()) / max(1, len(self.clusters))
        }
        
    def optimize(self):
        """Optimize this memory level"""
        # Rebuild vector store index
        if self.vector_store is not None:
            self.vector_store.optimize()
            
        # Recompute semantic clusters if enabled
        if self.semantic_clustering:
            # Clear existing clusters
            self.clusters = {}
            self.token_clusters = {}
            
            # Rebuild clusters
            for token_id, token_info in self.tokens.items():
                self._add_to_cluster(token_id, token_info.value)

# Advanced memory compression
class AdvancedMemoryCompressor(Module):
    """
    Advanced compression for token representations with multiple techniques:
    - Low-rank approximation
    - Quantization (scalar and vector)
    - Delta encoding
    - Pruning
    - Clustering and shared representations
    """
    def __init__(
        self,
        dim: int,
        compressed_dim: int = None,
        compression_ratio: float = 4.0,
        compression_method: str = "combined",  # "svd", "autoencoder", "quantization", "combined"
        quantization_bits: int = 8,
        use_pruning: bool = False,
        pruning_threshold: float = 0.1,
        use_clustering: bool = False,
        cluster_size: int = 32,
        use_delta_encoding: bool = False,
        delta_reference_update_freq: int = 100,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.compression_ratio = compression_ratio
        self.compression_method = compression_method
        self.use_pruning = use_pruning
        self.pruning_threshold = pruning_threshold
        self.use_clustering = use_clustering
        self.cluster_size = cluster_size
        self.use_delta_encoding = use_delta_encoding
        
        # Determine compressed dimension
        if compressed_dim is None:
            self.compressed_dim = max(1, int(dim / compression_ratio))
        else:
            self.compressed_dim = compressed_dim
            self.compression_ratio = dim / compressed_dim
            
        # SVD-like compression
        if compression_method in ["svd", "combined"]:
            self.encode_svd = nn.Linear(dim, self.compressed_dim)
            self.decode_svd = nn.Linear(self.compressed_dim, dim)
            
        # Autoencoder compression
        if compression_method in ["autoencoder", "combined"]:
            # Encoder network
            self.encoder = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, self.compressed_dim),
            )
            
            # Decoder network
            self.decoder = nn.Sequential(
                nn.Linear(self.compressed_dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, dim),
            )
            
        # Quantization settings
        self.quantization_bits = quantization_bits
        self.quantization_range = 2 ** (quantization_bits - 1) - 1  # Range for quantized values
        
        # For vector quantization
        if compression_method in ["quantization", "combined"]:
            # Codebook for vector quantization
            self.register_parameter(
                "codebook", 
                Parameter(torch.randn(256, self.compressed_dim))
            )
            
        # For semantic clustering
        if use_clustering:
            # Centroid vectors for clustering
            self.register_parameter(
                "centroids", 
                Parameter(torch.randn(cluster_size, self.compressed_dim))
            )
            
        # For delta encoding
        if use_delta_encoding:
            # Reference vectors updated periodically
            self.register_buffer(
                "reference_vectors",
                torch.zeros(100, dim)  # Start with 100 reference vectors
            )
            self.reference_counts = torch.zeros(100, dtype=torch.long)
            self.delta_update_counter = 0
            self.delta_reference_update_freq = delta_reference_update_freq
        
        # Initialize layers properly
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters for better reconstruction"""
        # Initialize linear layers
        if hasattr(self, 'encode_svd'):
            nn.init.orthogonal_(self.encode_svd.weight)
            nn.init.orthogonal_(self.decode_svd.weight)
            nn.init.zeros_(self.encode_svd.bias)
            nn.init.zeros_(self.decode_svd.bias)
            
        # Initialize autoencoder
        if hasattr(self, 'encoder'):
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            for module in self.decoder:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
        # Initialize codebook with k-means++ style
        if hasattr(self, 'codebook'):
            nn.init.orthogonal_(self.codebook)
            
        # Initialize centroids with k-means++ style
        if hasattr(self, 'centroids'):
            nn.init.orthogonal_(self.centroids)
            
    def _quantize_tensor(self, x, bits=None):
        """Quantize tensor to the specified bit depth"""
        bits = bits or self.quantization_bits
        qrange = 2 ** (bits - 1) - 1
        
        # Scale to quantization range
        x_scaled = x * qrange
        
        # Quantize (round to integers)
        x_quantized = torch.round(x_scaled)
        
        # Clamp to valid range
        x_clamped = torch.clamp(
            x_quantized, 
            -qrange, 
            qrange
        )
        
        return x_clamped
        
    def _dequantize_tensor(self, x_quantized, bits=None):
        """Dequantize tensor back to float"""
        bits = bits or self.quantization_bits
        qrange = 2 ** (bits - 1) - 1
        
        # Scale back from quantization range
        return x_quantized / qrange
        
    def _vector_quantize(self, x):
        """Perform vector quantization using the codebook"""
        # Compute distances to all codebook entries
        x_expanded = x.unsqueeze(1)  # [batch, 1, compressed_dim]
        codebook_expanded = self.codebook.unsqueeze(0)  # [1, 256, compressed_dim]
        
        # Compute euclidean distances
        distances = torch.sum((x_expanded - codebook_expanded) ** 2, dim=-1)  # [batch, 256]
        
        # Get closest codebook entry
        _, indices = torch.min(distances, dim=-1)  # [batch]
        
        # Get corresponding codebook entries
        quantized = self.codebook[indices]  # [batch, compressed_dim]
        
        # Straight-through estimator for gradients
        quantized_st = x + (quantized - x).detach()
        
        return quantized_st, indices
        
    def _cluster_assignment(self, x):
        """Assign compressed representations to nearest centroids"""
        # Compute distances to all centroids
        x_expanded = x.unsqueeze(1)  # [batch, 1, compressed_dim]
        centroids_expanded = self.centroids.unsqueeze(0)  # [1, clusters, compressed_dim]
        
        # Compute euclidean distances
        distances = torch.sum((x_expanded - centroids_expanded) ** 2, dim=-1)  # [batch, clusters]
        
        # Get closest centroid
        _, indices = torch.min(distances, dim=-1)  # [batch]
        
        # Get corresponding centroids
        assigned = self.centroids[indices]  # [batch, compressed_dim]
        
        # Straight-through estimator for gradients during training
        assigned_st = x + (assigned - x).detach()
        
        return assigned_st, indices
        
    def _find_nearest_reference(self, x):
        """Find the nearest reference vector for delta encoding"""
        # Compute distances to all reference vectors
        distances = torch.cdist(x.unsqueeze(0), self.reference_vectors, p=2)[0]  # [batch, 100]
        
        # Get closest reference vector
        _, indices = torch.min(distances, dim=-1)  # [batch]
        
        # Get corresponding reference vectors
        references = self.reference_vectors[indices]  # [batch, dim]
        
        return references, indices
        
    def _update_reference_vectors(self, x):
        """Update reference vectors for delta encoding"""
        self.delta_update_counter += 1
        
        # Only update periodically
        if self.delta_update_counter % self.delta_reference_update_freq != 0:
            return
            
        # Find nearest reference for each input vector
        _, indices = self._find_nearest_reference(x)
        
        # Update reference counts
        for idx in indices:
            self.reference_counts[idx] += 1
            
        # Replace least used references with new vectors
        least_used = torch.argsort(self.reference_counts)[:5]  # Replace 5 least used
        
        # Select random vectors from batch to use as new references
        batch_size = x.shape[0]
        if batch_size > 0:
            random_indices = torch.randperm(batch_size)[:len(least_used)]
            
            for i, ref_idx in enumerate(least_used):
                if i < len(random_indices):
                    batch_idx = random_indices[i]
                    self.reference_vectors[ref_idx] = x[batch_idx].detach().clone()
                    self.reference_counts[ref_idx] = 1
        
    def _prune(self, x, threshold=None):
        """Prune small values to increase sparsity"""
        threshold = threshold or self.pruning_threshold
        mask = torch.abs(x) > threshold
        return x * mask
        
    def compress_svd(self, x):
        """Compress using SVD-like linear projection"""
        return self.encode_svd(x)
        
    def decompress_svd(self, x):
        """Decompress using SVD-like linear projection"""
        return self.decode_svd(x)
        
    def compress_autoencoder(self, x):
        """Compress using autoencoder"""
        return self.encoder(x)
        
    def decompress_autoencoder(self, x):
        """Decompress using autoencoder"""
        return self.decoder(x)
        
    def compress_quantization(self, x):
        """Compress using vector quantization"""
        # First reduce dimensionality using SVD
        reduced = self.encode_svd(x)
        
        # Then apply vector quantization
        quantized, _ = self._vector_quantize(reduced)
        
        return quantized
        
    def decompress_quantization(self, x):
        """Decompress from vector quantization"""
        # Map back to original dimension
        return self.decode_svd(x)
        
    def compress_delta(self, x):
        """Compress using delta encoding"""
        # Find nearest reference vector
        references, _ = self._find_nearest_reference(x)
        
        # Compute delta (difference)
        deltas = x - references
        
        # Quantize the deltas to save space
        quantized_deltas = self._quantize_tensor(deltas, bits=4)  # Use fewer bits for deltas
        
        # Periodically update reference vectors
        self._update_reference_vectors(x)
        
        return quantized_deltas, references
        
    def decompress_delta(self, compressed_data):
        """Decompress from delta encoding"""
        quantized_deltas, references = compressed_data
        
        # Dequantize deltas
        deltas = self._dequantize_tensor(quantized_deltas, bits=4)
        
        # Add back to reference vectors
        return references + deltas
        
    def forward(self, x, compress=True):
        """
        Compress or decompress token representations
        
        Args:
            x: Token representation(s) to process
            compress: If True, compress; if False, decompress
            
        Returns:
            Compressed or decompressed representation
        """
        # Handle batched or single inputs
        original_shape = x.shape
        if len(original_shape) == 1:
            # Single vector
            x = x.unsqueeze(0)
            
        if compress:
            # Compression path
            if self.compression_method == "svd":
                compressed = self.compress_svd(x)
                
                # Apply pruning if enabled
                if self.use_pruning:
                    compressed = self._prune(compressed)
                    
                # Apply quantization
                quantized = self._quantize_tensor(compressed)
                
                # Apply clustering if enabled
                if self.use_clustering:
                    quantized, _ = self._cluster_assignment(quantized)
                    
                result = quantized
                
            elif self.compression_method == "autoencoder":
                compressed = self.compress_autoencoder(x)
                
                # Apply quantization
                quantized = self._quantize_tensor(compressed)
                
                result = quantized
                
            elif self.compression_method == "quantization":
                result = self.compress_quantization(x)
                
            elif self.compression_method == "combined":
                # First stage: dimensionality reduction with autoencoder
                stage1 = self.compress_autoencoder(x)
                
                # Second stage: quantization
                quantized = self._quantize_tensor(stage1)
                
                # Third stage: clustering if enabled
                if self.use_clustering:
                    result, _ = self._cluster_assignment(quantized)
                else:
                    result = quantized
                    
            else:
                raise ValueError(f"Unknown compression method: {self.compression_method}")
                
            # Handle delta encoding separately
            if self.use_delta_encoding:
                return self.compress_delta(x)
                
            # Restore original shape if single vector
            if len(original_shape) == 1:
                result = result.squeeze(0)
                
            return result
            
        else:
            # Decompression path
            
            # Handle delta encoding separately
            if self.use_delta_encoding:
                return self.decompress_delta(x)
                
            # Restore shape if needed
            if len(original_shape) == 1:
                x = x.unsqueeze(0)
                
            # Handle different compression methods
            if self.compression_method == "svd":
                # Dequantize
                dequantized = self._dequantize_tensor(x)
                result = self.decompress_svd(dequantized)
                
            elif self.compression_method == "autoencoder":
                # Dequantize
                dequantized = self._dequantize_tensor(x)
                result = self.decompress_autoencoder(dequantized)
                
            elif self.compression_method == "quantization":
                result = self.decompress_quantization(x)
                
            elif self.compression_method == "combined":
                # Dequantize
                dequantized = self._dequantize_tensor(x)
                
                # Decompress through autoencoder
                result = self.decompress_autoencoder(dequantized)
                
            else:
                raise ValueError(f"Unknown compression method: {self.compression_method}")
                
            # Restore original shape if single vector
            if len(original_shape) == 1:
                result = result.squeeze(0)
                
            return result
            
    def calculate_compression_stats(self, x):
        """Calculate compression statistics"""
        # Compress and decompress
        compressed = self.forward(x, compress=True)
        reconstructed = self.forward(compressed, compress=False)
        
        # Calculate metrics
        if self.use_delta_encoding:
            # Delta encoding returns a tuple
            quantized_deltas, _ = compressed
            compressed_size = quantized_deltas.numel() * (self.quantization_bits / 32)
        else:
            compressed_size = compressed.numel() * (self.quantization_bits / 32)
            
        original_size = x.numel() * 4  # Assuming float32
        
        # Reconstruction error
        mse = F.mse_loss(reconstructed, x).item()
        
        return {
            "compression_ratio": original_size / compressed_size,
            "reconstruction_error": mse,
            "compressed_size_bytes": compressed_size,
            "original_size_bytes": original_size
        }

# Memory pruning and summarization module
class MemorySummarizer(Module):
    """
    Dynamically prunes and summarizes memory to:
    - Reduce memory usage
    - Maintain important information
    - Create hierarchical summaries of context
    """
    def __init__(
        self,
        dim: int,
        summary_ratio: float = 4.0,  # How much to compress summaries
        min_sequence_length: int = 32,  # Min length to summarize
        use_attention_for_importance: bool = True,
        summary_level_max: int = 3,  # Maximum levels of summarization
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.summary_ratio = summary_ratio
        self.min_sequence_length = min_sequence_length
        self.use_attention_for_importance = use_attention_for_importance
        self.summary_level_max = summary_level_max
        
        # Importance scoring
        self.importance_scorer = ImportanceScorer(
            dim=dim,
            importance_dim=64,
            use_attention_weights=use_attention_for_importance
        )
        
        # Summary generator
        self.summary_dim = dim  # Maintain same dimension for summaries
        
        # Layers for summary generation
        self.summary_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )
        
        self.summary_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.summary_layer_norm1 = nn.LayerNorm(dim)
        self.summary_layer_norm2 = nn.LayerNorm(dim)
        
        # Initialize
        self._init_weights()
        
        # For tracking summaries
        self.summaries = {}  # level -> list of summary vectors
        self.summary_positions = {}  # level -> list of positions
        self.summary_source_map = {}  # summary_id -> list of source token_ids
        
    def _init_weights(self):
        """Initialize weights"""
        # Initialize importance scorer handled in its constructor
        
        # Initialize attention
        nn.init.normal_(self.summary_attention.in_proj_weight, std=0.02)
        nn.init.normal_(self.summary_attention.out_proj.weight, std=0.02)
        nn.init.zeros_(self.summary_attention.in_proj_bias)
        nn.init.zeros_(self.summary_attention.out_proj.bias)
        
        # Initialize MLP
        for module in self.summary_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def score_importance(self, token_vectors, positions=None, attention_weights=None):
        """Score token importance"""
        return self.importance_scorer(token_vectors, positions, attention_weights)
        
    def generate_summary(self, token_vectors, importance_scores, level=1):
        """
        Generate a summary of the token sequence
        
        Args:
            token_vectors: Token vectors to summarize [batch_size, seq_len, dim]
            importance_scores: Importance of each token [batch_size, seq_len]
            level: Summarization level (higher = more compressed)
            
        Returns:
            Summary vectors and indices of selected tokens
        """
        batch_size, seq_len, _ = token_vectors.shape
        device = token_vectors.device
        
        # Determine how many tokens to keep
        keep_ratio = 1.0 / (self.summary_ratio ** level)
        keep_count = max(1, min(seq_len - 1, math.ceil(seq_len * keep_ratio)))
        
        # Get the most important tokens
        _, indices = torch.topk(importance_scores, keep_count, dim=1)
        
        # Sort indices to maintain sequence order
        indices, _ = torch.sort(indices, dim=1)
        
        summaries = []
        selected_indices = []
        
        # Process each sequence in the batch
        for b in range(batch_size):
            # Get the selected tokens
            selected = token_vectors[b, indices[b]]
            
            # If only one token, no need for attention
            if selected.shape[0] == 1:
                summaries.append(selected)
                selected_indices.append(indices[b])
                continue
                
            # Apply self-attention to capture relationships
            attn_output, _ = self.summary_attention(
                selected, selected, selected
            )
            
            # Add residual and normalize
            attn_output = self.summary_layer_norm1(selected + attn_output)
            
            # Apply MLP
            mlp_output = self.summary_mlp(attn_output)
            
            # Add residual and normalize
            summary = self.summary_layer_norm2(attn_output + mlp_output)
            
            summaries.append(summary)
            selected_indices.append(indices[b])
            
        # Stack results
        if summaries:
            return summaries, selected_indices
        else:
            # Empty result case
            return torch.zeros(batch_size, 0, self.dim, device=device), []
            
    def hierarchical_summarize(self, token_vectors, positions, attention_weights=None, max_level=None):
        """
        Create hierarchical summaries of input sequence
        
        Args:
            token_vectors: Token vectors to summarize [batch_size, seq_len, dim]
            positions: Positions of tokens [batch_size, seq_len]
            attention_weights: Optional attention weights [batch_size, seq_len]
            max_level: Maximum summarization level (defaults to self.summary_level_max)
            
        Returns:
            Dictionary of summaries by level
        """
        max_level = max_level or self.summary_level_max
        batch_size, seq_len, _ = token_vectors.shape
        
        if seq_len < self.min_sequence_length:
            return {}  # Too short to summarize
            
        # Score importance
        importance_scores = self.score_importance(token_vectors, positions, attention_weights)
        
        # Create summaries for each level
        current_vectors = token_vectors
        current_positions = positions
        current_scores = importance_scores
        
        level_summaries = {}
        level_positions = {}
        level_source_maps = {}
        
        # Create source map for level 0 (original tokens)
        source_map_l0 = {}
        for b in range(batch_size):
            for i in range(seq_len):
                token_id = f"original_b{b}_p{positions[b, i].item()}"
                source_map_l0[token_id] = [token_id]  # Points to itself
                
        level_source_maps[0] = source_map_l0
        
        # Generate hierarchical summaries
        for level in range(1, max_level + 1):
            summaries, selected_indices = self.generate_summary(
                current_vectors, current_scores, level=level
            )
            
            # If no summaries or we've reached a single token per sequence, stop
            if not summaries or all(len(s) <= 1 for s in summaries):
                break
                
            # Track summaries
            level_summaries[level] = summaries
            
            # Track positions of summary tokens
            level_pos = []
            for b in range(batch_size):
                if level == 1:
                    # For first level, use original positions
                    pos = positions[b, selected_indices[b]]
                else:
                    # For higher levels, use positions from previous level
                    pos = level_positions[level-1][b][selected_indices[b]]
                    
                level_pos.append(pos)
                
            level_positions[level] = level_pos
            
            # Create source mapping for this level
            source_map = {}
            for b in range(batch_size):
                for i, idx in enumerate(selected_indices[b]):
                    if level == 1:
                        # Level 1 summaries point to original tokens
                        orig_pos = positions[b, idx].item()
                        summary_id = f"summary_l{level}_b{b}_i{i}"
                        source_id = f"original_b{b}_p{orig_pos}"
                        source_map[summary_id] = [source_id]
                    else:
                        # Higher level summaries point to lower level summaries
                        summary_id = f"summary_l{level}_b{b}_i{i}"
                        prev_summary_id = f"summary_l{level-1}_b{b}_i{idx.item()}"
                        if prev_summary_id in level_source_maps[level-1]:
                            source_map[summary_id] = level_source_maps[level-1][prev_summary_id]
                        
            level_source_maps[level] = source_map
            
            # Prepare for next level
            current_vectors = [s for s in summaries]
            current_positions = level_positions[level]
            
            # Recalculate importance scores for the summaries
            if len(current_vectors) > 0 and isinstance(current_vectors[0], torch.Tensor):
                stacked_vectors = torch.stack([v for v in current_vectors if v.numel() > 0])
                if stacked_vectors.size(0) > 0:
                    current_scores = self.score_importance(
                        stacked_vectors,
                        torch.stack([p for p in current_positions if p.numel() > 0])
                    )
                else:
                    break
            else:
                break
                
        # Store results
        self.summaries = level_summaries
        self.summary_positions = level_positions
        self.summary_source_map = level_source_maps
        
        return level_summaries
        
    def get_source_tokens(self, summary_id):
        """Get original source tokens that a summary represents"""
        for level in range(1, self.summary_level_max + 1):
            if summary_id in self.summary_source_map.get(level, {}):
                return self.summary_source_map[level][summary_id]
                
        return None
        
    def get_summary_for_tokens(self, token_ids, level=1):
        """Find summaries that contain specific tokens"""
        matching_summaries = []
        
        # Convert to set for faster lookup
        token_ids_set = set(token_ids)
        
        # Check source maps
        source_map = self.summary_source_map.get(level, {})
        for summary_id, sources in source_map.items():
            # If any source token is in the requested tokens
            if any(src in token_ids_set for src in sources):
                matching_summaries.append(summary_id)
                
        return matching_summaries

# Streaming and distributed memory management
class StreamingMemoryManager(Module):
    """
    Manages token streaming for truly unbounded context:
    - Continuous token ingestion
    - Dynamic summarization
    - Distributed memory sharding
    - Memory-mapped storage for overflow
    """
    def __init__(
        self,
        dim: int,
        local_capacity: int = 131072,      # Tokens in local memory
        distributed_capacity: int = 1000000,  # Tokens in distributed memory
        disk_capacity: int = 100000000,    # Tokens on disk (100M)
        summarization_interval: int = 4096, # How often to summarize
        distributed_nodes: int = 1,        # Number of distributed nodes
        use_memory_mapping: bool = True,   # Use memory-mapped files
        streaming_batch_size: int = 1024,  # Process this many tokens at once
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.local_capacity = local_capacity
        self.distributed_capacity = distributed_capacity
        self.disk_capacity = disk_capacity
        self.summarization_interval = summarization_interval
        self.distributed_nodes = distributed_nodes
        self.use_memory_mapping = use_memory_mapping
        self.streaming_batch_size = streaming_batch_size
        
        # Local memory management
        self.local_memory = HierarchicalMemoryManager(
            dim=dim,
            l1_capacity=4096,              # Very fast cache
            l2_capacity=local_capacity,    # Main local memory
            l3_capacity=0,                 # Don't use L3 locally
            enable_prefetching=True,
            enable_semantic_search=True
        )
        
        # Distributed memory (simulate with local storage for now)
        self.distributed_memory = {}  # node_id -> dict of token_id -> (value, position)
        
        # Disk storage
        self.disk_storage = None
        if use_memory_mapping:
            self.disk_storage = PersistentTokenStorage(
                dim=dim,
                storage_path="./token_storage",
                max_tokens=disk_capacity,
                compression_ratio=16.0,
                use_disk=True
            )
            
        # Summarization
        self.summarizer = MemorySummarizer(
            dim=dim,
            summary_ratio=4.0,
            min_sequence_length=32,
            use_attention_for_importance=True
        )
        
        # Streaming state
        self.token_count = 0
        self.last_summarization = 0
        self.streaming_buffer = deque(maxlen=streaming_batch_size)
        
        # Token to storage location mapping
        self.token_locations = {}  # token_id -> (storage_type, node_id or None)
        
        # Background processing queue
        self.processing_queue = queue.Queue()
        self.stop_background = False
        self.background_thread = None
        
        # Start background thread
        self._start_background_thread()
        
    def _start_background_thread(self):
        """Start background processing thread"""
        if self.background_thread is None:
            self.stop_background = False
            self.background_thread = threading.Thread(
                target=self._background_worker,
                daemon=True
            )
            self.background_thread.start()
            
    def _stop_background_thread(self):
        """Stop background processing thread"""
        if self.background_thread is not None:
            self.stop_background = True
            self.background_thread.join(timeout=1.0)
            self.background_thread = None
            
    def _background_worker(self):
        """Background worker for processing streaming operations"""
        while not self.stop_background:
            try:
                # Get next task
                try:
                    task_type, task_data = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    # No tasks, sleep briefly
                    time.sleep(0.01)
                    continue
                    
                # Process task based on type
                if task_type == "summarize":
                    token_vectors, positions = task_data
                    self._process_summarization(token_vectors, positions)
                elif task_type == "move_to_distributed":
                    token_ids, values, positions = task_data
                    self._move_to_distributed_storage(token_ids, values, positions)
                elif task_type == "move_to_disk":
                    token_ids, values, positions = task_data
                    self._move_to_disk_storage(token_ids, values, positions)
                    
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in background worker: {e}")
                
    def _process_summarization(self, token_vectors, positions):
        """Create hierarchical summaries in the background"""
        try:
            # Create summaries
            self.summarizer.hierarchical_summarize(token_vectors, positions)
            logger.debug(f"Created summaries for {token_vectors.shape[0]} sequences")
        except Exception as e:
            logger.error(f"Error creating summaries: {e}")
            
    def _move_to_distributed_storage(self, token_ids, values, positions):
        """Move tokens to distributed storage"""
        try:
            # Determine which node to use (round-robin for now)
            for i, (token_id, value, position) in enumerate(zip(token_ids, values, positions)):
                node_id = i % self.distributed_nodes
                
                # Initialize node storage if needed
                if node_id not in self.distributed_memory:
                    self.distributed_memory[node_id] = {}
                    
                # Store token
                self.distributed_memory[node_id][token_id] = (value, position)
                
                # Update location mapping
                self.token_locations[token_id] = ("distributed", node_id)
                
            logger.debug(f"Moved {len(token_ids)} tokens to distributed storage")
        except Exception as e:
            logger.error(f"Error moving to distributed storage: {e}")
            
    def _move_to_disk_storage(self, token_ids, values, positions):
        """Move tokens to disk storage"""
        if self.disk_storage is None:
            logger.warning("Disk storage not initialized, cannot move tokens to disk")
            return
            
        try:
            # Store each token
            for token_id, value, position in zip(token_ids, values, positions):
                # Add metadata
                metadata = {
                    "creation_time": time.time(),
                    "position": position.item() if hasattr(position, 'item') else position
                }
                
                # Store in disk storage
                self.disk_storage.add(token_id, value, position, metadata)
                
                # Update location mapping
                self.token_locations[token_id] = ("disk", None)
                
            logger.debug(f"Moved {len(token_ids)} tokens to disk storage")
        except Exception as e:
            logger.error(f"Error moving to disk storage: {e}")
            
    def add_token_batch(self, token_vectors, positions):
        """
        Add a batch of tokens to the streaming memory system
        
        Args:
            token_vectors: Token vectors to add [batch_size, seq_len, dim]
            positions: Positions of these tokens [batch_size, seq_len]
            
        Returns:
            List of token IDs
        """
        batch_size, seq_len, _ = token_vectors.shape
        all_token_ids = []
        
        # Process each sequence
        for b in range(batch_size):
            # Add to streaming buffer
            for i in range(seq_len):
                # Create token ID
                token_id = str(uuid.uuid4())
                
                # Add to local memory first
                self.local_memory.add_tokens(
                    token_vectors[b:b+1, i:i+1, :],
                    positions[b:b+1, i:i+1]
                )
                
                # Track in streaming buffer
                self.streaming_buffer.append((token_id, token_vectors[b, i], positions[b, i]))
                
                # Update token count
                self.token_count += 1
                
                all_token_ids.append(token_id)
                
        # Check if we need to process the streaming buffer
        if len(self.streaming_buffer) >= self.streaming_batch_size:
            self._process_streaming_buffer()
            
        # Check if we need to trigger summarization
        if self.token_count - self.last_summarization >= self.summarization_interval:
            self._trigger_summarization()
            
        return all_token_ids
        
    def _process_streaming_buffer(self):
        """Process tokens in the streaming buffer"""
        # Extract tokens from buffer
        token_ids = []
        values = []
        positions = []
        
        for token_id, value, position in self.streaming_buffer:
            token_ids.append(token_id)
            values.append(value)
            positions.append(position)
            
        # Clear buffer
        self.streaming_buffer.clear()
        
        # Convert to tensors
        if values:
            values_tensor = torch.stack(values)
            positions_tensor = torch.stack(positions)
            
            # Schedule background processing
            self.processing_queue.put(
                ("move_to_distributed", (token_ids, values_tensor, positions_tensor))
            )
            
    def _trigger_summarization(self):
        """Trigger summarization of recent tokens"""
        # Update summarization checkpoint
        self.last_summarization = self.token_count
        
        # Get recent tokens from local memory
        # This is a simplified approach - in a real system we'd be more selective
        recent_tokens = self.local_memory.retrieve_tokens(
            positions=torch.arange(
                max(0, self.token_count - self.summarization_interval), 
                self.token_count
            ).view(1, -1)
        )
        
        if recent_tokens:
            values = recent_tokens[0]
            positions = torch.arange(
                max(0, self.token_count - self.summarization_interval),
                self.token_count
            ).view(1, -1)
            
            # Schedule summarization in background
            self.processing_queue.put(
                ("summarize", (values.unsqueeze(0), positions))  # Add batch dimension
            )
            
    def retrieve_tokens(self, positions=None, token_ids=None, query_vectors=None):
        """
        Retrieve tokens from the streaming memory system
        
        Args:
            positions: Positions to retrieve
            token_ids: Token IDs to retrieve
            query_vectors: Query vectors for semantic search
            
        Returns:
            Retrieved token values
        """
        results = []
        
        # Case 1: Retrieve by position
        if positions is not None:
            # Try local memory first
            local_results = self.local_memory.retrieve_tokens(positions=positions)
            
            if local_results:
                results.extend(local_results)
            else:
                # Try distributed and disk storage
                batch_size = 1
                seq_len = positions.size(0)
                if positions.dim() > 1:
                    batch_size = positions.size(0)
                    seq_len = positions.size(1)
                    
                # Initialize results tensor
                batch_results = torch.zeros(
                    batch_size, seq_len, self.dim,
                    device=positions.device
                )
                
                # Process each position
                for b in range(batch_size):
                    for i in range(seq_len):
                        position = positions[b, i].item() if batch_size > 1 else positions[i].item()
                        
                        # Check token location mapping
                        found = False
                        for token_id, (storage_type, node_id) in self.token_locations.items():
                            # This is inefficient and would be indexed properly in a real system
                            if storage_type == "distributed":
                                # Check distributed storage
                                if node_id in self.distributed_memory and token_id in self.distributed_memory[node_id]:
                                    value, pos = self.distributed_memory[node_id][token_id]
                                    if pos == position:
                                        batch_results[b, i] = value
                                        found = True
                                        break
                            elif storage_type == "disk" and self.disk_storage is not None:
                                # Check disk storage
                                value, metadata = self.disk_storage.get(token_id=token_id)
                                if value is not None and metadata.get("position") == position:
                                    batch_results[b, i] = value
                                    found = True
                                    break
                                    
                        if not found:
                            # Check if we have a summary for this position
                            for level in range(1, 4):  # Check summarization levels
                                if level in self.summarizer.summary_positions:
                                    for b_idx, positions_tensor in enumerate(self.summarizer.summary_positions[level]):
                                        for j, pos in enumerate(positions_tensor):
                                            if pos.item() == position:
                                                # Found in summary
                                                summary_vector = self.summarizer.summaries[level][b_idx][j]
                                                batch_results[b, i] = summary_vector
                                                found = True
                                                break
                                        if found:
                                            break
                                if found:
                                    break
                
                results.append(batch_results)
                
        # Case 2: Retrieve by token ID
        if token_ids is not None:
            # Prepare results container
            token_results = []
            
            for token_id in token_ids:
                # Check token location
                if token_id in self.token_locations:
                    storage_type, node_id = self.token_locations[token_id]
                    
                    if storage_type == "distributed":
                        # Retrieve from distributed storage
                        if node_id in self.distributed_memory and token_id in self.distributed_memory[node_id]:
                            value, _ = self.distributed_memory[node_id][token_id]
                            token_results.append(value)
                        else:
                            # Not found
                            token_results.append(torch.zeros(self.dim, device=next(self.parameters()).device))
                            
                    elif storage_type == "disk" and self.disk_storage is not None:
                        # Retrieve from disk storage
                        value, _ = self.disk_storage.get(token_id=token_id)
                        if value is not None:
                            token_results.append(value)
                        else:
                            # Not found
                            token_results.append(torch.zeros(self.dim, device=next(self.parameters()).device))
                else:
                    # Try local memory
                    local_results = self.local_memory.retrieve_tokens(token_ids=[[token_id]])
                    if local_results and len(local_results[0]) > 0:
                        token_results.append(local_results[0][0])
                    else:
                        # Not found anywhere
                        token_results.append(torch.zeros(self.dim, device=next(self.parameters()).device))
                        
            # Stack results if we have any
            if token_results:
                results.append(torch.stack(token_results))
                
        # Case 3: Semantic search
        if query_vectors is not None:
            # Try local memory first
            local_results = self.local_memory.retrieve_tokens(query_vectors=query_vectors)
            
            if local_results:
                results.extend(local_results)
                
            # Search in distributed and disk storage
            # This is simplified - in a real system, we'd have distributed vector indexes
            if not local_results and self.disk_storage is not None:
                disk_results = []
                
                for query in query_vectors:
                    similar = self.disk_storage.search_similar(query, top_k=5)
                    vectors = [v for _, v, _, _ in similar]
                    
                    if vectors:
                        disk_results.append(torch.stack(vectors))
                    else:
                        disk_results.append(torch.zeros(0, self.dim, device=query_vectors.device))
                        
                results.extend(disk_results)
                
        return results
        
    def clear(self):
        """Clear all memory"""
        # Clear local memory
        self.local_memory.clear()
        
        # Clear distributed memory
        self.distributed_memory.clear()
        
        # Clear disk storage
        if self.disk_storage is not None:
            self.disk_storage.clear()
            
        # Reset state
        self.token_count = 0
        self.last_summarization = 0
        self.streaming_buffer.clear()
        self.token_locations.clear()
        
    def __del__(self):
        """Cleanup when object is deleted"""
        self._stop_background_thread()

# Adaptive policy management
class AdaptiveMemoryPolicy(Module):
    """
    Adaptively tunes memory management policies based on:
    - Observed access patterns
    - Hardware resources
    - Priority workloads
    - Real-time performance metrics
    """
    def __init__(
        self,
        dim: int,
        observation_window: int = 1000,  # How many operations to observe
        update_interval: int = 100,      # How often to update policies
        learning_rate: float = 0.01,     # How quickly to adapt
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.observation_window = observation_window
        self.update_interval = update_interval
        self.learning_rate = learning_rate
        
        # Access pattern tracking
        self.access_history = deque(maxlen=observation_window)
        
        # Performance metrics
        self.retrieval_times = deque(maxlen=observation_window)
        self.hit_rates = deque(maxlen=observation_window)
        self.memory_usage = deque(maxlen=observation_window)
        
        # Current policies
        self.policies = {
            "prefetch_window": 512,
            "promotion_threshold": 2,
            "demotion_threshold": 60.0,
            "l1_capacity": 8192,
            "l2_capacity": 131072,
            "l3_capacity": 1048576,
            "eviction_policy": "adaptive",
            "compression_ratio_l2": 2.0,
            "compression_ratio_l3": 8.0,
            "semantic_search_threshold": 0.75,
        }
        
        # Policy bounds
        self.policy_bounds = {
            "prefetch_window": (128, 2048),
            "promotion_threshold": (1, 10),
            "demotion_threshold": (10.0, 600.0),
            "l1_capacity": (4096, 32768),
            "l2_capacity": (65536, 524288),
            "l3_capacity": (524288, 8388608),
            "compression_ratio_l2": (1.0, 4.0),
            "compression_ratio_l3": (4.0, 16.0),
            "semantic_search_threshold": (0.6, 0.9),
        }
        
        # Operation counter
        self.op_count = 0
        
        # Current workload characteristics
        self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL
        self.current_workload = "balanced"  # "read_heavy", "write_heavy", "balanced"
        
        # Resource monitoring
        self.memory_pressure = 0.5  # 0-1 scale
        self.cpu_usage = 0.5  # 0-1 scale
        
        # Last update time
        self.last_update = time.time()
        
    def record_access(self, positions=None, token_ids=None, latency=None):
        """Record an access operation"""
        # Increment operation counter
        self.op_count += 1
        
        # Record position access
        if positions is not None:
            for pos in positions.view(-1).tolist():
                self.access_history.append(("position", pos, time.time()))
                
        # Record token ID access
        if token_ids is not None:
            for tid in token_ids:
                self.access_history.append(("token_id", tid, time.time()))
                
        # Record latency
        if latency is not None:
            self.retrieval_times.append(latency)
            
        # Check if we should update policies
        if self.op_count % self.update_interval == 0:
            self._update_policies()
            
    def record_metrics(self, hit_rate, memory_usage_mb):
        """Record performance metrics"""
        self.hit_rates.append(hit_rate)
        self.memory_usage.append(memory_usage_mb)
        
    def _update_policies(self):
        """Update policies based on observed patterns"""
        # Only update periodically
        now = time.time()
        if now - self.last_update < 1.0:  # At most once per second
            return
            
        self.last_update = now
        
        # Analyze access patterns
        self._analyze_access_pattern()
        
        # Determine current workload characteristics
        self._determine_workload()
        
        # Check resource pressure
        self._check_resources()
        
        # Update specific policies based on observations
        self._update_prefetch_policy()
        self._update_capacity_policy()
        self._update_promotion_demotion_policy()
        self._update_compression_policy()
        
        # Log policy changes
        logger.info(f"Updated memory policies: {self.policies}")
        
    def _analyze_access_pattern(self):
        """Analyze recent access pattern"""
        if len(self.access_history) < 10:
            return  # Not enough data
            
        # Count different types of access patterns
        sequential_count = 0
        local_count = 0
        random_count = 0
        repeated_count = 0
        
        # Track positions accessed
        position_accesses = [
            (pos, ts) for access_type, pos, ts in self.access_history 
            if access_type == "position"
        ]
        
        # Sort by timestamp
        position_accesses.sort(key=lambda x: x[1])
        positions = [pos for pos, _ in position_accesses]
        
        # Look for sequential access
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i-1]
            if diff == 1:
                sequential_count += 1
            elif abs(diff) < 10:
                local_count += 1
            elif diff == 0:
                repeated_count += 1
            else:
                random_count += 1
                
        # Determine dominant pattern
        total = sequential_count + local_count + random_count + repeated_count
        if total == 0:
            return
            
        sequential_ratio = sequential_count / total
        local_ratio = local_count / total
        repeated_ratio = repeated_count / total
        
        # Update access pattern
        if sequential_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL
        elif local_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.LOCAL
        elif repeated_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.REPEATED
        else:
            self.current_access_pattern = MemoryAccessPattern.RANDOM
            
    def _determine_workload(self):
        """Determine current workload characteristics"""
        # Count reads vs writes
        reads = sum(1 for access_type, _, _ in self.access_history 
                   if access_type in ["position", "token_id"])
        
        writes = self.op_count - reads
        
        # Determine workload type
        if reads > writes * 2:
            self.current_workload = "read_heavy"
        elif writes > reads * 2:
            self.current_workload = "write_heavy"
        else:
            self.current_workload = "balanced"
            
    def _check_resources(self):
        """Check system resource usage"""
        try:
            # Memory pressure
            import psutil
            memory = psutil.virtual_memory()
            self.memory_pressure = memory.percent / 100.0
            
            # CPU usage
            self.cpu_usage = psutil.cpu_percent() / 100.0
        except:
            # If psutil not available, use default values
            pass
            
    def _update_prefetch_policy(self):
        """Update prefetching policy"""
        current = self.policies["prefetch_window"]
        min_val, max_val = self.policy_bounds["prefetch_window"]
        
        # Adjust based on access pattern
        if self.current_access_pattern == MemoryAccessPattern.SEQUENTIAL:
            # Increase prefetch for sequential access
            new_val = current * (1 + self.learning_rate)
        elif self.current_access_pattern == MemoryAccessPattern.RANDOM:
            # Decrease prefetch for random access
            new_val = current * (1 - self.learning_rate)
        else:
            # Leave unchanged for other patterns
            new_val = current
            
        # Clamp to bounds
        self.policies["prefetch_window"] = max(min_val, min(max_val, int(new_val)))
        
    def _update_capacity_policy(self):
        """Update memory capacity allocation"""
        # Check if we have hit rate data
        if not self.hit_rates:
            return
            
        # Get average hit rate
        avg_hit_rate = sum(self.hit_rates) / len(self.hit_rates)
        
        # Adjust L1 capacity based on hit rate
        current_l1 = self.policies["l1_capacity"]
        min_l1, max_l1 = self.policy_bounds["l1_capacity"]
        
        if avg_hit_rate < 0.5:
            # Low hit rate, increase L1
            new_l1 = current_l1 * (1 + self.learning_rate)
        elif avg_hit_rate > 0.9 and self.memory_pressure > 0.8:
            # High hit rate but memory pressure, decrease L1
            new_l1 = current_l1 * (1 - self.learning_rate)
        else:
            # Leave unchanged
            new_l1 = current_l1
            
        # Clamp to bounds
        self.policies["l1_capacity"] = max(min_l1, min(max_l1, int(new_l1)))
        
        # Similar adjustments for L2 and L3...
        
    def _update_promotion_demotion_policy(self):
        """Update promotion/demotion thresholds"""
        # Adjust based on workload type
        if self.current_workload == "read_heavy":
            # More aggressive promotion for read-heavy workloads
            self.policies["promotion_threshold"] = max(
                self.policy_bounds["promotion_threshold"][0],
                self.policies["promotion_threshold"] * 0.9
            )
        elif self.current_workload == "write_heavy":
            # More aggressive demotion for write-heavy workloads
            self.policies["demotion_threshold"] = max(
                self.policy_bounds["demotion_threshold"][0],
                self.policies["demotion_threshold"] * 0.9
            )
            
    def _update_compression_policy(self):
        """Update compression policies"""
        # Adjust based on memory pressure
        if self.memory_pressure > 0.8:
            # High memory pressure, increase compression
            self.policies["compression_ratio_l2"] = min(
                self.policy_bounds["compression_ratio_l2"][1],
                self.policies["compression_ratio_l2"] * 1.1
            )
            self.policies["compression_ratio_l3"] = min(
                self.policy_bounds["compression_ratio_l3"][1],
                self.policies["compression_ratio_l3"] * 1.1
            )
        elif self.memory_pressure < 0.3 and self.cpu_usage > 0.8:
            # Low memory but high CPU, decrease compression
            self.policies["compression_ratio_l2"] = max(
                self.policy_bounds["compression_ratio_l2"][0],
                self.policies["compression_ratio_l2"] * 0.9
            )
            self.policies["compression_ratio_l3"] = max(
                self.policy_bounds["compression_ratio_l3"][0],
                self.policies["compression_ratio_l3"] * 0.9
            )
            
    def get_policy(self, policy_name):
        """Get current value for a specific policy"""
        return self.policies.get(policy_name)
        
    def get_all_policies(self):
        """Get all current policies"""
        return self.policies.copy()
        
    def override_policy(self, policy_name, value):
        """Manually override a policy"""
        if policy_name in self.policies:
            # Ensure value is within bounds
            if policy_name in self.policy_bounds:
                min_val, max_val = self.policy_bounds[policy_name]
                value = max(min_val, min(max_val, value))
                
            self.policies[policy_name] = value
            return True
        return False
        
    def reset_policies(self):
        """Reset policies to default values"""
        self.policies = {
            "prefetch_window": 512,
            "promotion_threshold": 2,
            "demotion_threshold": 60.0,
            "l1_capacity": 8192,
            "l2_capacity": 131072,
            "l3_capacity": 1048576,
            "eviction_policy": "adaptive",
            "compression_ratio_l2": 2.0,
            "compression_ratio_l3": 8.0,
            "semantic_search_threshold": 0.75,
        }

# Advanced Hierarchical Memory Manager with all enhancements
class AdvancedHierarchicalMemoryManager(Module):
    """
    Enhanced hierarchical memory system for 100M+ token contexts with:
    - Adaptive policy management
    - Dynamic summarization and pruning
    - Streaming memory optimization
    - Hardware-aware optimizations
    - Distributed memory management
    - Quality-of-Service guarantees
    """
    def __init__(
        self,
        dim: int,
        l1_capacity: int = 16384,        # Fast, recent tokens (16K)
        l2_capacity: int = 262144,       # Medium, potentially useful tokens (256K)
        l3_capacity: int = 8388608,      # Large, contextual tokens (8M)
        disk_capacity: int = 104857600,  # Disk/distributed tokens (100M)
        enable_summarization: bool = True,
        enable_streaming: bool = True,
        enable_adaptive_policies: bool = True,
        qos_targets: Dict[str, float] = None,
        distributed_nodes: int = 1,
        hardware_acceleration: str = "auto",  # "cpu", "cuda", "tpu", "auto"
        reliability_level: str = "normal",    # "normal", "high", "critical"
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        
        # Initialize performance config with hardware settings
        self.perf_config = perf_config
        
        # Set up QoS targets
        self.qos_targets = qos_targets or {
            "l1_latency_ms": 0.1,
            "l2_latency_ms": 1.0,
            "l3_latency_ms": 10.0,
            "disk_latency_ms": 100.0,
            "hit_rate_l1": 0.9,
            "hit_rate_l2": 0.8,
            "hit_rate_l3": 0.7,
            "availability": 0.9999,
        }
        
        # Set device based on hardware acceleration setting
        if hardware_acceleration == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(hardware_acceleration)
            
        # Set up enhanced memory levels with QoS
        self.l1 = EnhancedMemoryLevel(
            level_id=1,
            dim=dim,
            capacity=l1_capacity,
            retrieval_cost=1.0,
            storage_cost=10.0,
            eviction_policy="adaptive",
            qos_enabled=True,
            semantic_clustering=False,
            index_type="flat"
        )
        
        self.l2 = EnhancedMemoryLevel(
            level_id=2,
            dim=dim,
            capacity=l2_capacity,
            retrieval_cost=5.0,
            storage_cost=3.0,
            compression_ratio=2.0,
            eviction_policy="adaptive",
            qos_enabled=True,
            semantic_clustering=True,
            index_type="hnsw"
        )
        
        self.l3 = EnhancedMemoryLevel(
            level_id=3,
            dim=dim,
            capacity=l3_capacity,
            retrieval_cost=20.0,
            storage_cost=1.0,
            compression_ratio=8.0,
            eviction_policy="semantic",
            qos_enabled=True,
            semantic_clustering=True,
            index_type="ivfpq"
        )
        
        # Advanced memory compressors
        self.l2_compressor = AdvancedMemoryCompressor(
            dim=dim,
            compression_ratio=2.0,
            compression_method="combined",
            quantization_bits=16,
            use_pruning=True,
            use_clustering=False,
            perf_config=perf_config
        )
        
        self.l3_compressor = AdvancedMemoryCompressor(
            dim=dim,
            compression_ratio=8.0,
            compression_method="combined",
            quantization_bits=8,
            use_pruning=True,
            use_clustering=True,
            cluster_size=256,
            use_delta_encoding=True,
            perf_config=perf_config
        )
        
        # Set up adaptive policy management
        self.adaptive_policies = None
        if enable_adaptive_policies:
            self.adaptive_policies = AdaptiveMemoryPolicy(
                dim=dim,
                observation_window=1000,
                update_interval=100,
                learning_rate=0.01,
                perf_config=perf_config
            )
            
        # Set up summarization
        self.summarizer = None
        if enable_summarization:
            self.summarizer = MemorySummarizer(
                dim=dim,
                summary_ratio=4.0,
                min_sequence_length=64,
                use_attention_for_importance=True,
                summary_level_max=4,
                perf_config=perf_config
            )
            
        # Set up streaming memory
        self.streaming_memory = None
        if enable_streaming:
            self.streaming_memory = StreamingMemoryManager(
                dim=dim,
                local_capacity=l1_capacity + l2_capacity,
                distributed_capacity=l3_capacity,
                disk_capacity=disk_capacity,
                summarization_interval=4096,
                distributed_nodes=distributed_nodes,
                use_memory_mapping=True,
                streaming_batch_size=1024,
                perf_config=perf_config
            )
            
        # Importance scorer for token prioritization
        self.importance_scorer = ImportanceScorer(
            dim=dim,
            importance_dim=64,
            use_attention_weights=True,
            use_positional_bias=True
        )
        
        # Access pattern tracking
        self.access_history = deque(maxlen=1000)
        self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL
        
        # Token ID generator
        self.next_token_id = 0
        
        # Performance monitoring
        self.metrics = {
            "l1_hit_rate": 0.0,
            "l2_hit_rate": 0.0,
            "l3_hit_rate": 0.0,
            "l1_latency_ms": 0.0,
            "l2_latency_ms": 0.0,
            "l3_latency_ms": 0.0,
            "compression_ratio_l2": 0.0,
            "compression_ratio_l3": 0.0,
            "summarization_ratio": 0.0,
            "total_tokens": 0,
        }
        
        # Reliability features
        self.reliability_level = reliability_level
        self.enable_redundancy = (reliability_level in ["high", "critical"])
        self.shadow_copies = {}  # token_id -> [level_copies]
        
        # Background worker for maintenance
        self.background_worker = None
        self.maintenance_queue = queue.PriorityQueue()
        self.stop_background = False
        
        # Start background worker
        if perf_config.optimize_memory:
            self._start_background_worker()
            
    def _start_background_worker(self):
        """Start background worker for memory maintenance"""
        if self.background_worker is None:
            self.stop_background = False
            self.background_worker = threading.Thread(
                target=self._background_maintenance,
                daemon=True
            )
            self.background_worker.start()
            
    def _stop_background_worker(self):
        """Stop background worker"""
        if self.background_worker is not None:
            self.stop_background = True
            self.background_worker.join(timeout=1.0)
            self.background_worker = None
            
    def _background_maintenance(self):
        """Background maintenance tasks"""
        while not self.stop_background:
            try:
                # Check if there are maintenance tasks to perform
                try:
                    priority, task, args = self.maintenance_queue.get(timeout=1.0)
                    task(*args)
                    self.maintenance_queue.task_done()
                except queue.Empty:
                    # No tasks, perform periodic maintenance
                    self._periodic_maintenance()
                    time.sleep(0.1)  # Sleep to avoid busy waiting
            except Exception as e:
                logger.error(f"Error in background maintenance: {e}")
                
    def _periodic_maintenance(self):
        """Periodic maintenance tasks"""
        # Update access pattern analysis
        self._analyze_access_pattern()
        
        # Update memory policy if enabled
        if self.adaptive_policies is not None:
            # Record metrics
            self.adaptive_policies.record_metrics(
                hit_rate=(self.l1.hit_rate + self.l2.hit_rate + self.l3.hit_rate) / 3,
                memory_usage_mb=(len(self.l1) + len(self.l2) + len(self.l3)) * self.dim * 4 / (1024 * 1024)
            )
            
            # Check for policy updates
            policies = self.adaptive_policies.get_all_policies()
            
            # Apply policy changes if different from current
            if self.l1.current_capacity != policies["l1_capacity"]:
                self.l1.current_capacity = policies["l1_capacity"]
                
            if self.l2.current_capacity != policies["l2_capacity"]:
                self.l2.current_capacity = policies["l2_capacity"]
                
            if self.l3.current_capacity != policies["l3_capacity"]:
                self.l3.current_capacity = policies["l3_capacity"]
                
        # Optimize memory levels periodically
        self._optimize_memory_levels()
        
        # Check if we need to create summaries
        if self.summarizer is not None:
            self._maybe_create_summaries()
            
        # Verify QoS guarantees
        self._verify_qos_guarantees()
        
        # Handle reliability checks
        if self.enable_redundancy:
            self._verify_redundant_copies()
            
    def _analyze_access_pattern(self):
        """Analyze recent access pattern to optimize memory management"""
        if len(self.access_history) < 10:
            return  # Not enough data
            
        # Count different types of access patterns
        sequential_count = 0
        local_count = 0
        random_count = 0
        repeated_count = 0
        
        # Track positions accessed
        positions = [pos for pos, _ in self.access_history]
        
        # Look for sequential access
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i-1]
            if diff == 1:
                sequential_count += 1
            elif abs(diff) < 10:
                local_count += 1
            elif diff == 0:
                repeated_count += 1
            else:
                random_count += 1
                
        # Determine dominant pattern
        total = sequential_count + local_count + random_count + repeated_count
        if total == 0:
            return
            
        sequential_ratio = sequential_count / total
        local_ratio = local_count / total
        repeated_ratio = repeated_count / total
        
        # Update access pattern
        if sequential_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.SEQUENTIAL
        elif local_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.LOCAL
        elif repeated_ratio > 0.6:
            self.current_access_pattern = MemoryAccessPattern.REPEATED
        else:
            self.current_access_pattern = MemoryAccessPattern.RANDOM
            
    def _optimize_memory_levels(self):
        """Optimize memory levels periodically"""
        # Optimize vector indexes
        self.l2.optimize()
        self.l3.optimize()
        
        # Run garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def _maybe_create_summaries(self):
        """Create summaries of memory contents based on multiple strategies"""
        # Check if we need to create summaries based on multiple criteria:
        # 1. Memory tier filling up
        # 2. Temporal coherence (summarize time ranges)
        # 3. Semantic clustering (summarize similar content)
        # 4. Importance-based (summarize important content)
        
        # Store which summaries were created
        summaries_created = []
        
        # Strategy 1: Memory pressure based summarization
        if len(self.l2) > self.l2.current_capacity * 0.8:
            mem_pressure_summary = self._create_memory_pressure_summaries()
            if mem_pressure_summary:
                summaries_created.append(("memory_pressure", len(mem_pressure_summary)))
                
        # Strategy 2: Temporal coherence summarization
        temporal_summary = self._create_temporal_block_summaries()
        if temporal_summary:
            summaries_created.append(("temporal", len(temporal_summary)))
            
        # Strategy 3: Semantic clustering summarization
        semantic_summary = self._create_semantic_cluster_summaries()
        if semantic_summary:
            summaries_created.append(("semantic", len(semantic_summary)))
            
        # Strategy 4: Importance-based summarization
        if hasattr(self, "importance_scorer"):
            importance_summary = self._create_importance_based_summaries()
            if importance_summary:
                summaries_created.append(("importance", len(importance_summary)))
                
        # Log what we did
        if summaries_created:
            summary_str = ", ".join([f"{name}: {count}" for name, count in summaries_created])
            logger.info(f"Created summaries: {summary_str}")
            
    def _create_memory_pressure_summaries(self):
        """Create summaries based on memory pressure"""
        # Get sample of tokens from the memory tier under pressure (L2)
        token_ids = list(self.l2.tokens.keys())
        if len(token_ids) < 100:
            return None
            
        # Take a larger sample under memory pressure
        sample_size = min(1000, len(token_ids) // 4)  # 25% of tokens or 1000 max
        sample_ids = random.sample(token_ids, sample_size)
        
        # Group tokens by position ranges (1K token blocks)
        position_groups = {}
        
        for token_id in sample_ids:
            token_info = self.l2.tokens[token_id]
            pos = token_info.position
            group_key = pos // 1000  # Group by 1K token blocks
            
            if group_key not in position_groups:
                position_groups[group_key] = []
                
            position_groups[group_key].append((token_id, token_info))
            
        # Process each group that has enough tokens
        summaries = []
        
        for group_key, group_tokens in position_groups.items():
            if len(group_tokens) >= 20:  # Need enough tokens for meaningful summary
                token_vectors = []
                positions = []
                
                for token_id, token_info in group_tokens:
                    token_vectors.append(token_info.value)
                    positions.append(token_info.position)
                    
                if token_vectors:
                    # Convert to tensors
                    token_tensor = torch.stack(token_vectors)
                    position_tensor = torch.tensor(positions, device=token_tensor.device)
                    
                    # Create summaries for this group
                    group_summaries = self.summarizer.hierarchical_summarize(
                        token_tensor.unsqueeze(0),  # Add batch dimension
                        position_tensor.unsqueeze(0)
                    )
                    
                    if group_summaries:
                        summaries.append((group_key, len(group_summaries)))
                        
        return summaries
        
    def _create_temporal_block_summaries(self):
        """Create summaries of temporal blocks of tokens"""
        # Find contiguous blocks of tokens in L1 and L2
        l1_positions = [info.position for info in self.l1.tokens.values()]
        l2_positions = [info.position for info in self.l2.tokens.values()]
        
        # Combine and sort positions
        all_positions = sorted(set(l1_positions + l2_positions))
        
        if len(all_positions) < 100:
            return None
            
        # Find contiguous blocks (allowing small gaps)
        blocks = []
        current_block = [all_positions[0]]
        
        for i in range(1, len(all_positions)):
            if all_positions[i] - all_positions[i-1] <= 5:  # Allow gaps of 5 or less
                current_block.append(all_positions[i])
            else:
                if len(current_block) >= 50:  # Only keep substantial blocks
                    blocks.append(current_block)
                current_block = [all_positions[i]]
                
        if len(current_block) >= 50:
            blocks.append(current_block)
            
        # Process each substantial block
        summaries = []
        
        for block in blocks:
            # Get the tokens for positions in this block
            token_vectors = []
            positions = []
            
            # First check L1
            for pos in block:
                token_id = self.l1.position_index.get(pos)
                if token_id:
                    token_info = self.l1.tokens.get(token_id)
                    if token_info:
                        token_vectors.append(token_info.value)
                        positions.append(pos)
                        continue
                        
                # If not in L1, check L2
                token_id = self.l2.position_index.get(pos)
                if token_id:
                    token_info = self.l2.tokens.get(token_id)
                    if token_info:
                        # Need to decompress L2 values
                        value = self.l2_compressor(token_info.value, compress=False) if self.l2_compressor else token_info.value
                        token_vectors.append(value)
                        positions.append(pos)
                        
            if len(token_vectors) >= 30:  # Need enough tokens for meaningful summary
                # Convert to tensors
                token_tensor = torch.stack(token_vectors)
                position_tensor = torch.tensor(positions, device=token_tensor.device)
                
                # Create summaries for this temporal block
                block_summaries = self.summarizer.hierarchical_summarize(
                    token_tensor.unsqueeze(0),
                    position_tensor.unsqueeze(0),
                    max_level=2  # Limit to 2 levels for temporal blocks
                )
                
                if block_summaries:
                    block_range = f"{min(block)}-{max(block)}"
                    summaries.append((block_range, len(block_summaries)))
                    
        return summaries
        
    def _create_semantic_cluster_summaries(self):
        """Create summaries based on semantic clustering"""
        if not hasattr(self.l2, "vector_store") or not self.l2.vector_store:
            return None
            
        # Get clusters from L2's semantic clustering if available
        if hasattr(self.l2, "semantic_clustering") and self.l2.semantic_clustering:
            cluster_info = self.l2.get_cluster_info()
            if not cluster_info or cluster_info.get("num_clusters", 0) == 0:
                return None
                
            # Process each substantial cluster
            summaries = []
            
            for cluster_id, token_ids in self.l2.clusters.items():
                if len(token_ids) < 20:  # Skip small clusters
                    continue
                    
                # Get tokens from this cluster
                token_vectors = []
                positions = []
                
                for token_id in token_ids:
                    token_info = self.l2.tokens.get(token_id)
                    if token_info:
                        # Need to decompress L2 values
                        value = self.l2_compressor(token_info.value, compress=False) if self.l2_compressor else token_info.value
                        token_vectors.append(value)
                        positions.append(token_info.position)
                
                if len(token_vectors) >= 20:  # Confirm we still have enough
                    # Convert to tensors
                    token_tensor = torch.stack(token_vectors)
                    position_tensor = torch.tensor(positions, device=token_tensor.device)
                    
                    # Create summaries for this semantic cluster
                    cluster_summaries = self.summarizer.hierarchical_summarize(
                        token_tensor.unsqueeze(0),
                        position_tensor.unsqueeze(0),
                        max_level=3  # Go deeper for semantic clusters
                    )
                    
                    if cluster_summaries:
                        summaries.append((f"cluster_{cluster_id}", len(cluster_summaries)))
            
            return summaries
            
        else:
            # If no explicit clustering, we can create clusters now
            # Get a sample of vectors from L2
            token_ids = list(self.l2.tokens.keys())
            if len(token_ids) < 200:
                return None
                
            sample_size = min(500, len(token_ids))
            sample_ids = random.sample(token_ids, sample_size)
            
            # Get token vectors
            token_vectors = []
            positions = []
            id_map = []
            
            for token_id in sample_ids:
                token_info = self.l2.tokens.get(token_id)
                if token_info:
                    # Need to decompress L2 values for accurate clustering
                    value = self.l2_compressor(token_info.value, compress=False) if self.l2_compressor else token_info.value
                    token_vectors.append(value)
                    positions.append(token_info.position)
                    id_map.append(token_id)
            
            if len(token_vectors) < 50:
                return None
                
            # Create vector tensor
            vectors = torch.stack(token_vectors)
            
            # Normalize vectors
            vectors = F.normalize(vectors, p=2, dim=1)
            
            # Perform clustering using sklearn (or a simpler method if not available)
            try:
                from sklearn.cluster import KMeans
                import numpy as np
                
                # Convert to numpy for sklearn
                vectors_np = vectors.cpu().numpy()
                
                # Determine number of clusters (dynamic based on data size)
                n_clusters = max(5, min(20, len(vectors_np) // 20))
                
                # Run K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(vectors_np)
                
                # Group by cluster
                clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
                    
            except ImportError:
                # Fallback if sklearn is not available - simple distance-based clustering
                # Initialize clusters with first vector
                centroids = [vectors[0]]
                clusters = {0: [0]}
                
                # Simple threshold-based clustering
                for i in range(1, len(vectors)):
                    # Find closest centroid
                    similarities = [F.cosine_similarity(vectors[i].unsqueeze(0), c.unsqueeze(0)).item() for c in centroids]
                    closest = max(range(len(similarities)), key=lambda x: similarities[x])
                    
                    # If similar enough, add to that cluster
                    if similarities[closest] > 0.7:
                        clusters[closest].append(i)
                    else:
                        # Create new cluster
                        new_cluster_id = len(centroids)
                        centroids.append(vectors[i])
                        clusters[new_cluster_id] = [i]
            
            # Process each substantial cluster
            summaries = []
            
            for cluster_id, indices in clusters.items():
                if len(indices) < 15:  # Skip small clusters
                    continue
                    
                # Get tokens from this cluster
                cluster_vectors = [token_vectors[i] for i in indices]
                cluster_positions = [positions[i] for i in indices]
                
                # Convert to tensors
                token_tensor = torch.stack(cluster_vectors)
                position_tensor = torch.tensor(cluster_positions, device=token_tensor.device)
                
                # Create summaries for this cluster
                cluster_summaries = self.summarizer.hierarchical_summarize(
                    token_tensor.unsqueeze(0),
                    position_tensor.unsqueeze(0)
                )
                
                if cluster_summaries:
                    summaries.append((f"cluster_{cluster_id}", len(cluster_summaries)))
                    
            return summaries
    
    def _create_importance_based_summaries(self):
        """Create summaries based on token importance"""
        # Get high-importance tokens from L1 and L2
        important_tokens = []
        
        # Check L1
        for token_id, token_info in self.l1.tokens.items():
            if token_info.importance_score > 0.7:  # High importance threshold
                important_tokens.append((token_id, token_info, "l1"))
                
        # Check L2
        for token_id, token_info in self.l2.tokens.items():
            if token_info.importance_score > 0.7:
                important_tokens.append((token_id, token_info, "l2"))
                
        if len(important_tokens) < 30:
            return None
            
        # Group important tokens by importance ranges
        importance_groups = {}
        
        for token_id, token_info, level in important_tokens:
            # Round importance to create groups (0.7-0.8, 0.8-0.9, 0.9-1.0)
            group_key = round(token_info.importance_score * 10) / 10
            
            if group_key not in importance_groups:
                importance_groups[group_key] = []
                
            importance_groups[group_key].append((token_id, token_info, level))
            
        # Process each importance group with enough tokens
        summaries = []
        
        for importance, group_tokens in importance_groups.items():
            if len(group_tokens) >= 20:
                token_vectors = []
                positions = []
                
                for token_id, token_info, level in group_tokens:
                    # Get the appropriate value based on level
                    if level == "l1":
                        value = token_info.value
                    else:  # l2
                        value = self.l2_compressor(token_info.value, compress=False) if self.l2_compressor else token_info.value
                        
                    token_vectors.append(value)
                    positions.append(token_info.position)
                    
                if token_vectors:
                    # Convert to tensors
                    token_tensor = torch.stack(token_vectors)
                    position_tensor = torch.tensor(positions, device=token_tensor.device)
                    
                    # Create summaries for this importance group
                    group_summaries = self.summarizer.hierarchical_summarize(
                        token_tensor.unsqueeze(0),
                        position_tensor.unsqueeze(0),
                        max_level=2  # Limit to 2 levels for importance-based summaries
                    )
                    
                    if group_summaries:
                        summaries.append((f"importance_{importance}", len(group_summaries)))
                        
        return summaries
                    
    def _verify_qos_guarantees(self):
        """Verify that QoS guarantees are being met and take corrective actions"""
        qos_violations = []
        qos_actions = []
        
        # Check L1 QoS guarantees
        l1_hit_rate_target = self.qos_targets.get("hit_rate_l1", 0.9)
        if self.l1.hit_rate < l1_hit_rate_target:
            # L1 hit rate is too low, increase capacity if possible
            violation_pct = (l1_hit_rate_target - self.l1.hit_rate) / l1_hit_rate_target * 100
            qos_violations.append((f"L1 hit rate: {self.l1.hit_rate:.3f} vs target {l1_hit_rate_target:.3f}", violation_pct))
            
            # Increase capacity based on how severe the violation is
            increase_factor = 1.1 + min(0.5, violation_pct / 100)
            new_capacity = min(int(self.l1.current_capacity * increase_factor), self.l1.max_capacity)
            
            if new_capacity > self.l1.current_capacity:
                logger.info(f"Increasing L1 capacity to meet QoS: {self.l1.current_capacity} -> {new_capacity}")
                self.l1.current_capacity = new_capacity
                qos_actions.append(f"Increased L1 capacity by {increase_factor:.2f}x")
                
        # Check L1 latency
        if hasattr(self.l1, 'response_times') and self.l1.response_times:
            avg_latency = sum(self.l1.response_times) / len(self.l1.response_times)
            l1_latency_target = self.qos_targets.get("l1_latency_ms", 0.1)
            
            if avg_latency > l1_latency_target:
                violation_pct = (avg_latency - l1_latency_target) / l1_latency_target * 100
                qos_violations.append((f"L1 latency: {avg_latency:.2f}ms vs target {l1_latency_target:.2f}ms", violation_pct))
                
                # L1 is too slow, try to optimize
                logger.info(f"L1 latency ({avg_latency:.2f}ms) exceeds QoS target, optimizing...")
                self.l1.optimize()
                qos_actions.append("Optimized L1 memory structures")
                
                # If severe violation, also reduce capacity to improve cache locality
                if violation_pct > 50 and self.l1.current_capacity > self.l1.min_capacity:
                    new_capacity = max(int(self.l1.current_capacity * 0.9), self.l1.min_capacity)
                    logger.info(f"Reducing L1 capacity to improve locality: {self.l1.current_capacity} -> {new_capacity}")
                    self.l1.current_capacity = new_capacity
                    qos_actions.append("Reduced L1 capacity to improve locality")
                    
        # Check L2 QoS guarantees
        l2_hit_rate_target = self.qos_targets.get("hit_rate_l2", 0.8)
        if self.l2.hit_rate < l2_hit_rate_target:
            violation_pct = (l2_hit_rate_target - self.l2.hit_rate) / l2_hit_rate_target * 100
            qos_violations.append((f"L2 hit rate: {self.l2.hit_rate:.3f} vs target {l2_hit_rate_target:.3f}", violation_pct))
            
            # Increase capacity based on how severe the violation is
            increase_factor = 1.1 + min(0.3, violation_pct / 100)
            new_capacity = min(int(self.l2.current_capacity * increase_factor), self.l2.max_capacity)
            
            if new_capacity > self.l2.current_capacity:
                logger.info(f"Increasing L2 capacity to meet QoS: {self.l2.current_capacity} -> {new_capacity}")
                self.l2.current_capacity = new_capacity
                qos_actions.append(f"Increased L2 capacity by {increase_factor:.2f}x")
                
        # Check L2 latency
        if hasattr(self.metrics, 'l2_latency_ms') and self.metrics["l2_latency_ms"] > 0:
            avg_latency = self.metrics["l2_latency_ms"]
            l2_latency_target = self.qos_targets.get("l2_latency_ms", 1.0)
            
            if avg_latency > l2_latency_target:
                violation_pct = (avg_latency - l2_latency_target) / l2_latency_target * 100
                qos_violations.append((f"L2 latency: {avg_latency:.2f}ms vs target {l2_latency_target:.2f}ms", violation_pct))
                
                # L2 is too slow, try to optimize
                logger.info(f"L2 latency ({avg_latency:.2f}ms) exceeds QoS target, optimizing...")
                self.l2.optimize()
                qos_actions.append("Optimized L2 memory structures")
                
                # Also consider adjusting compression
                if self.l2_compressor and violation_pct > 30:
                    # If significantly slow, reduce compression ratio to speed up
                    current_ratio = self.l2_compressor.compression_ratio
                    new_ratio = max(1.5, current_ratio * 0.9)
                    if new_ratio < current_ratio:
                        logger.info(f"Reducing L2 compression ratio to improve speed: {current_ratio:.1f} -> {new_ratio:.1f}")
                        self.l2_compressor.compression_ratio = new_ratio
                        qos_actions.append(f"Reduced L2 compression ratio to {new_ratio:.1f}")
                        
        # Check L3 QoS guarantees
        l3_hit_rate_target = self.qos_targets.get("hit_rate_l3", 0.7)
        if self.l3.hit_rate < l3_hit_rate_target:
            violation_pct = (l3_hit_rate_target - self.l3.hit_rate) / l3_hit_rate_target * 100
            qos_violations.append((f"L3 hit rate: {self.l3.hit_rate:.3f} vs target {l3_hit_rate_target:.3f}", violation_pct))
            
            # Increase capacity based on how severe the violation is
            increase_factor = 1.05 + min(0.2, violation_pct / 100)
            new_capacity = min(int(self.l3.current_capacity * increase_factor), self.l3.max_capacity)
            
            if new_capacity > self.l3.current_capacity:
                logger.info(f"Increasing L3 capacity to meet QoS: {self.l3.current_capacity} -> {new_capacity}")
                self.l3.current_capacity = new_capacity
                qos_actions.append(f"Increased L3 capacity by {increase_factor:.2f}x")
                
        # Check L3 latency
        if hasattr(self.metrics, 'l3_latency_ms') and self.metrics["l3_latency_ms"] > 0:
            avg_latency = self.metrics["l3_latency_ms"]
            l3_latency_target = self.qos_targets.get("l3_latency_ms", 10.0)
            
            if avg_latency > l3_latency_target:
                violation_pct = (avg_latency - l3_latency_target) / l3_latency_target * 100
                qos_violations.append((f"L3 latency: {avg_latency:.2f}ms vs target {l3_latency_target:.2f}ms", violation_pct))
                
                # L3 is too slow, try to optimize
                logger.info(f"L3 latency ({avg_latency:.2f}ms) exceeds QoS target, optimizing...")
                self.l3.optimize()
                qos_actions.append("Optimized L3 memory structures")
                
                # For L3, also adjust the index parameters
                if hasattr(self.l3, 'vector_store') and self.l3.vector_store is not None:
                    if hasattr(self.l3.vector_store, 'semantic_index') and self.l3.vector_store.semantic_index != "simple":
                        try:
                            # Increase nprobe for IVF-based indexes to improve recall at the cost of speed
                            if self.l3.vector_store.index_type in ["ivf", "ivfpq"]:
                                try:
                                    current_nprobe = getattr(self.l3.vector_store.semantic_index, "nprobe", 8)
                                    new_nprobe = min(32, current_nprobe + 4)
                                    if new_nprobe > current_nprobe:
                                        self.l3.vector_store.semantic_index.nprobe = new_nprobe
                                        qos_actions.append(f"Increased L3 search nprobe to {new_nprobe}")
                                except:
                                    pass
                        except:
                            pass
                            
        # Check streaming memory latency if enabled
        if self.streaming_memory is not None:
            # In a real implementation, we'd have actual metrics here
            # For now, simulate with random values
            disk_latency = self.metrics.get("disk_latency_ms", 0)
            if disk_latency > 0:
                disk_latency_target = self.qos_targets.get("disk_latency_ms", 100.0)
                
                if disk_latency > disk_latency_target:
                    violation_pct = (disk_latency - disk_latency_target) / disk_latency_target * 100
                    qos_violations.append((f"Disk latency: {disk_latency:.2f}ms vs target {disk_latency_target:.2f}ms", violation_pct))
                    
                    # Try to improve disk performance
                    if self.streaming_memory is not None:
                        logger.info(f"Disk latency ({disk_latency:.2f}ms) exceeds QoS target, optimizing...")
                        qos_actions.append("Triggered streaming memory optimization")
                        
        # Get overall system health score
        if qos_violations:
            # Calculate system health as inverse of average violation percentage
            avg_violation = sum(pct for _, pct in qos_violations) / len(qos_violations)
            system_health = max(0, 100 - avg_violation) / 100.0
            
            # Log violations and health
            violation_str = ", ".join([desc for desc, _ in qos_violations])
            logger.warning(f"QoS violations: {violation_str}. System health: {system_health:.2f}")
            
            # Log corrective actions
            if qos_actions:
                action_str = ", ".join(qos_actions)
                logger.info(f"QoS corrective actions: {action_str}")
                
            # Update metrics
            self.metrics["system_health"] = system_health
            self.metrics["qos_violations"] = len(qos_violations)
        else:
            # All QoS targets met
            self.metrics["system_health"] = 1.0
            self.metrics["qos_violations"] = 0
                
    def _verify_redundant_copies(self):
        """Verify redundant copies for high-reliability operation"""
        if not self.enable_redundancy:
            return
            
        # Check a sample of tokens for redundancy
        token_ids = list(self.shadow_copies.keys())
        if not token_ids:
            return
            
        # Check up to 10 random tokens
        sample_size = min(10, len(token_ids))
        sample_ids = random.sample(token_ids, sample_size)
        
        for token_id in sample_ids:
            copies = self.shadow_copies.get(token_id, [])
            
            # For critical reliability, we need at least 3 copies
            min_copies = 3 if self.reliability_level == "critical" else 2
            
            if len(copies) < min_copies:
                # Need to create more copies
                # Find the token
                token_value = None
                token_position = None
                
                # Check in each level
                for level, level_id in copies:
                    if level == "l1":
                        token_info = self.l1.tokens.get(token_id)
                        if token_info:
                            token_value = token_info.value
                            token_position = token_info.position
                            break
                    elif level == "l2":
                        token_info = self.l2.tokens.get(token_id)
                        if token_info:
                            token_value = self.l2_compressor(token_info.value, compress=False)
                            token_position = token_info.position
                            break
                    elif level == "l3":
                        token_info = self.l3.tokens.get(token_id)
                        if token_info:
                            token_value = self.l3_compressor(token_info.value, compress=False)
                            token_position = token_info.position
                            break
                
                if token_value is not None:
                    # Create additional copies in other levels
                    existing_levels = [level for level, _ in copies]
                    
                    if "l1" not in existing_levels and len(self.l1) < self.l1.current_capacity:
                        self.l1.add(token_id, token_value, token_position)
                        copies.append(("l1", token_id))
                        
                    if "l2" not in existing_levels and len(self.l2) < self.l2.current_capacity:
                        compressed = self.l2_compressor(token_value, compress=True)
                        self.l2.add(token_id, compressed, token_position)
                        copies.append(("l2", token_id))
                        
                    if "l3" not in existing_levels and len(self.l3) < self.l3.current_capacity:
                        compressed = self.l3_compressor(token_value, compress=True)
                        self.l3.add(token_id, compressed, token_position)
                        copies.append(("l3", token_id))
                        
                    # Update shadow copies
                    self.shadow_copies[token_id] = copies
    
    def _schedule_maintenance(self, task, args=(), priority=50):
        """Schedule a maintenance task"""
        if self.background_worker is not None:
            self.maintenance_queue.put((priority, task, args))
            
    def add_tokens(self, tokens, positions, attention_weights=None):
        """
        Add tokens to the memory system
        
        Args:
            tokens: Tensor of token representations [batch_size, seq_len, dim]
            positions: Positions of these tokens in the overall sequence
            attention_weights: Optional attention weights [batch_size, seq_len]
            
        Returns:
            List of token IDs
        """
        # If streaming is enabled, use that instead
        if self.streaming_memory is not None:
            return self.streaming_memory.add_token_batch(tokens, positions)
            
        batch_size, seq_len, _ = tokens.shape
        all_token_ids = []
        
        # Calculate importance scores if we have attention weights
        if attention_weights is not None:
            importance_scores = self.importance_scorer(tokens, positions, attention_weights)
        else:
            importance_scores = self.importance_scorer(tokens, positions)
            
        # Process each batch
        for b in range(batch_size):
            batch_token_ids = []
            
            for i in range(seq_len):
                # Generate token ID
                token_id = str(uuid.uuid4())
                
                # Get importance score
                importance = importance_scores[b, i].item() if importance_scores is not None else 0.5
                
                # Add to L1 (most recent tokens go to fastest memory)
                self.l1.add(token_id, tokens[b, i], positions[b, i].item(), importance)
                
                # For high-reliability operation, create redundant copies
                if self.enable_redundancy:
                    # Create copies in other levels based on importance
                    copies = [("l1", token_id)]
                    
                    # Always add critical and important tokens to L2
                    if importance > 0.5 or self.reliability_level == "critical":
                        # Compress for L2
                        compressed_l2 = self.l2_compressor(tokens[b, i], compress=True)
                        self.l2.add(token_id, compressed_l2, positions[b, i].item(), importance)
                        copies.append(("l2", token_id))
                        
                    # Add very important tokens to L3 as well for critical reliability
                    if importance > 0.8 or self.reliability_level == "critical":
                        # Compress for L3
                        compressed_l3 = self.l3_compressor(tokens[b, i], compress=True)
                        self.l3.add(token_id, compressed_l3, positions[b, i].item(), importance)
                        copies.append(("l3", token_id))
                        
                    # Store copy information
                    self.shadow_copies[token_id] = copies
                    
                batch_token_ids.append(token_id)
                
                # Update metrics
                self.metrics["total_tokens"] += 1
                
            all_token_ids.append(batch_token_ids)
            
        # Update metrics
        if self.l2_compressor:
            self.metrics["compression_ratio_l2"] = self.l2_compressor.compression_ratio
        if self.l3_compressor:
            self.metrics["compression_ratio_l3"] = self.l3_compressor.compression_ratio
            
        # Schedule maintenance to balance memory levels
        self._schedule_maintenance(self._rebalance_memory_levels, priority=10)
        
        # If we have a policy manager, record this operation
        if self.adaptive_policies:
            self.adaptive_policies.record_access(positions=positions)
            
        return all_token_ids
        
    def _rebalance_memory_levels(self):
        """Move tokens between memory levels based on access patterns and importance"""
        # Get policies if available
        if self.adaptive_policies:
            policies = self.adaptive_policies.get_all_policies()
            promotion_threshold = policies.get("promotion_threshold", 2)
            demotion_threshold = policies.get("demotion_threshold", 60.0)
        else:
            promotion_threshold = 2
            demotion_threshold = 60.0
            
        now = time.time()
        
        # Identify tokens to promote from L2 to L1
        l2_to_promote = []
        for token_id, info in self.l2.tokens.items():
            if info.access_count >= promotion_threshold or info.importance_score > 0.8:
                l2_to_promote.append(token_id)
                
        # Identify tokens to promote from L3 to L2
        l3_to_promote = []
        for token_id, info in self.l3.tokens.items():
            if info.access_count >= promotion_threshold or info.importance_score > 0.7:
                l3_to_promote.append(token_id)
                
        # Identify tokens to demote from L1 to L2
        l1_to_demote = []
        for token_id, info in self.l1.tokens.items():
            # Don't demote high-importance tokens
            if info.importance_score < 0.7 and now - info.last_access_time > demotion_threshold:
                l1_to_demote.append(token_id)
                
        # Identify tokens to demote from L2 to L3
        l2_to_demote = []
        for token_id, info in self.l2.tokens.items():
            # Don't demote high-importance tokens
            if info.importance_score < 0.6 and now - info.last_access_time > demotion_threshold * 3:
                l2_to_demote.append(token_id)
                
        # Process promotions and demotions
        
        # Promote from L3 to L2
        for token_id in l3_to_promote:
            if token_id in self.l3.tokens:
                value, position = self.l3.get(token_id=token_id)
                importance = self.l3.tokens[token_id].importance_score
                
                # Decompress from L3 format
                value = self.l3_compressor(value, compress=False)
                    
                # Compress to L2 format
                value = self.l2_compressor(value, compress=True)
                    
                # Add to L2
                self.l2.add(token_id, value, position, importance)
                
                # Update shadow copies if using redundancy
                if self.enable_redundancy:
                    copies = self.shadow_copies.get(token_id, [])
                    copies = [c for c in copies if c[0] != "l3"]  # Remove L3 copy
                    copies.append(("l2", token_id))  # Add L2 copy
                    self.shadow_copies[token_id] = copies
                
                # No need to keep in L3 if it's in L2
                self.l3._remove(token_id)
                
        # Promote from L2 to L1
        for token_id in l2_to_promote:
            if token_id in self.l2.tokens:
                value, position = self.l2.get(token_id=token_id)
                importance = self.l2.tokens[token_id].importance_score
                
                # Decompress from L2 format
                value = self.l2_compressor(value, compress=False)
                    
                # Add to L1
                self.l1.add(token_id, value, position, importance)
                
                # Update shadow copies if using redundancy
                if self.enable_redundancy:
                    copies = self.shadow_copies.get(token_id, [])
                    copies = [c for c in copies if c[0] != "l2"]  # Remove L2 copy
                    copies.append(("l1", token_id))  # Add L1 copy
                    self.shadow_copies[token_id] = copies
                
                # No need to keep in L2 if it's in L1
                self.l2._remove(token_id)
                
        # Demote from L1 to L2
        for token_id in l1_to_demote:
            if token_id in self.l1.tokens:
                value, position = self.l1.get(token_id=token_id)
                importance = self.l1.tokens[token_id].importance_score
                
                # Compress to L2 format
                value = self.l2_compressor(value, compress=True)
                    
                # Add to L2
                self.l2.add(token_id, value, position, importance)
                
                # Update shadow copies if using redundancy
                if self.enable_redundancy:
                    copies = self.shadow_copies.get(token_id, [])
                    copies = [c for c in copies if c[0] != "l1"]  # Remove L1 copy
                    copies.append(("l2", token_id))  # Add L2 copy
                    self.shadow_copies[token_id] = copies
                
                # Remove from L1 if not needed for redundancy
                if not self.enable_redundancy or len(self.shadow_copies.get(token_id, [])) > 1:
                    self.l1._remove(token_id)
                
        # Demote from L2 to L3
        for token_id in l2_to_demote:
            if token_id in self.l2.tokens:
                value, position = self.l2.get(token_id=token_id)
                importance = self.l2.tokens[token_id].importance_score
                
                # Decompress from L2 format
                value = self.l2_compressor(value, compress=False)
                    
                # Compress to L3 format
                value = self.l3_compressor(value, compress=True)
                    
                # Add to L3
                self.l3.add(token_id, value, position, importance)
                
                # Update shadow copies if using redundancy
                if self.enable_redundancy:
                    copies = self.shadow_copies.get(token_id, [])
                    copies = [c for c in copies if c[0] != "l2"]  # Remove L2 copy
                    copies.append(("l3", token_id))  # Add L3 copy
                    self.shadow_copies[token_id] = copies
                
                # Remove from L2 if not needed for redundancy
                if not self.enable_redundancy or len(self.shadow_copies.get(token_id, [])) > 1:
                    self.l2._remove(token_id)
    
    def retrieve_tokens(self, positions=None, token_ids=None, query_vectors=None):
        """
        Retrieve tokens from the memory system
        
        Args:
            positions: Tensor of positions to retrieve
            token_ids: List of token IDs to retrieve
            query_vectors: Query vectors for semantic search
            
        Returns:
            Retrieved token values
        """
        # If streaming is enabled, use that instead
        if self.streaming_memory is not None:
            return self.streaming_memory.retrieve_tokens(
                positions=positions,
                token_ids=token_ids,
                query_vectors=query_vectors
            )
            
        results = []
        start_time = time.time()
        
        # Track access pattern
        if positions is not None:
            for pos in positions.view(-1).tolist():
                self.access_history.append((pos, time.time()))
        
        # Case 1: Retrieve by position
        if positions is not None:
            batch_size = 1
            seq_len = positions.size(0)
            if positions.dim() > 1:
                batch_size = positions.size(0)
                seq_len = positions.size(1)
                
            # Initialize results tensor
            batch_results = torch.zeros(
                batch_size, seq_len, self.dim,
                device=positions.device
            )
            
            # Process each position
            for b in range(batch_size):
                for i in range(seq_len):
                    position = positions[b, i].item() if batch_size > 1 else positions[i].item()
                    
                    # Try to find in memory levels (L1, L2, L3)
                    value, _ = self.l1.get(position=position)
                    l1_time = time.time()
                    
                    if value is None:
                        # Not in L1, try L2
                        value, _ = self.l2.get(position=position)
                        l2_time = time.time()
                        
                        if value is not None:
                            # Record L2 latency
                            self.metrics["l2_latency_ms"] = (l2_time - l1_time) * 1000
                            
                            # Decompress from L2 format
                            value = self.l2_compressor(value, compress=False)
                            
                            # Schedule promotion to L1
                            self._schedule_maintenance(
                                self._try_promote_to_l1,
                                args=(None, position),
                                priority=20
                            )
                    
                    if value is None:
                        # Not in L2, try L3
                        value, _ = self.l3.get(position=position)
                        l3_time = time.time()
                        
                        if value is not None:
                            # Record L3 latency
                            self.metrics["l3_latency_ms"] = (l3_time - l2_time) * 1000
                            
                            # Decompress from L3 format
                            value = self.l3_compressor(value, compress=False)
                            
                            # Schedule promotion to higher level
                            self._schedule_maintenance(
                                self._try_promote_to_l1,
                                args=(None, position),
                                priority=30
                            )
                    
                    if value is not None:
                        # Store the retrieved value
                        batch_results[b, i] = value
                        
                        # If using summaries, check if this position has a summary
                        if self.summarizer is not None and hasattr(self.summarizer, 'summary_positions'):
                            # This would be implemented more efficiently in a real system
                            for level in range(1, 4):  # Check first 3 summary levels
                                if level in self.summarizer.summary_positions:
                                    summary_positions = self.summarizer.summary_positions[level]
                                    for b_idx, pos_tensor in enumerate(summary_positions):
                                        for j, pos in enumerate(pos_tensor):
                                            if pos.item() == position:
                                                # We have a summary for this position
                                                # We could do something with this information
                                                pass
            
            results.append(batch_results)
            
            # Record L1 latency
            if value is not None:
                self.metrics["l1_latency_ms"] = (l1_time - start_time) * 1000
            
        # Case 2: Retrieve by token ID
        if token_ids is not None:
            batch_size = len(token_ids)
            
            # Process each batch
            for b in range(batch_size):
                seq_len = len(token_ids[b])
                batch_results = torch.zeros(seq_len, self.dim, device=self.device)
                
                for i, token_id in enumerate(token_ids[b]):
                    # Try to find in memory levels (L1, L2, L3)
                    value, position = self.l1.get(token_id=token_id)
                    
                    if value is None:
                        # Not in L1, try L2
                        value, position = self.l2.get(token_id=token_id)
                        
                        if value is not None and self.l2_compressor is not None:
                            # Decompress from L2 format
                            value = self.l2_compressor(value, compress=False)
                            
                            # Schedule promotion to L1
                            self._schedule_maintenance(
                                self._try_promote_to_l1,
                                args=(token_id, None),
                                priority=20
                            )
                    
                    if value is None:
                        # Not in L2, try L3
                        value, position = self.l3.get(token_id=token_id)
                        
                        if value is not None and self.l3_compressor is not None:
                            # Decompress from L3 format
                            value = self.l3_compressor(value, compress=False)
                            
                            # Schedule promotion to higher level
                            self._schedule_maintenance(
                                self._try_promote_to_l1,
                                args=(token_id, None),
                                priority=30
                            )
                            
                    # Check shadow copies if using redundancy and token not found
                    if value is None and self.enable_redundancy and token_id in self.shadow_copies:
                        copies = self.shadow_copies[token_id]
                        for storage_type, tid in copies:
                            if storage_type == "l1":
                                value, position = self.l1.get(token_id=tid)
                                if value is not None:
                                    break
                            elif storage_type == "l2":
                                value, position = self.l2.get(token_id=tid)
                                if value is not None:
                                    value = self.l2_compressor(value, compress=False)
                                    break
                            elif storage_type == "l3":
                                value, position = self.l3.get(token_id=tid)
                                if value is not None:
                                    value = self.l3_compressor(value, compress=False)
                                    break
                    
                    if value is not None:
                        # Store the retrieved value
                        batch_results[i] = value
                        
                        # Track access pattern if position is available
                        if position is not None:
                            self.access_history.append((position, time.time()))
                
                results.append(batch_results)
                
        # Case 3: Semantic search
        if query_vectors is not None:
            batch_size = query_vectors.size(0)
            
            # Process each query vector
            for b in range(batch_size):
                query = query_vectors[b]
                
                # Get semantic search threshold from policies if available
                semantic_threshold = 0.75
                if self.adaptive_policies:
                    semantic_threshold = self.adaptive_policies.get_policy("semantic_search_threshold")
                
                # Search in each memory level
                l1_results = []
                l2_results = []
                l3_results = []
                
                # L1 doesn't have vector store, use brute force
                for token_id, info in self.l1.tokens.items():
                    value = info.value
                    similarity = F.cosine_similarity(
                        query.unsqueeze(0),
                        value.unsqueeze(0)
                    ).item()
                    
                    if similarity >= semantic_threshold:
                        l1_results.append((token_id, value, similarity, info.position))
                
                # L2 has vector store
                if self.l2.vector_store is not None:
                    l2_search = self.l2.vector_store.search_similar(
                        query, top_k=5
                    )
                    
                    for token_id, vector, similarity, metadata in l2_search:
                        # Decompress if needed
                        if self.l2_compressor is not None:
                            vector = self.l2_compressor(vector, compress=False)
                            
                        l2_results.append((token_id, vector, similarity, metadata.get("position")))
                        
                # L3 has vector store
                if self.l3.vector_store is not None:
                    l3_search = self.l3.vector_store.search_similar(
                        query, top_k=5
                    )
                    
                    for token_id, vector, similarity, metadata in l3_search:
                        # Decompress if needed
                        if self.l3_compressor is not None:
                            vector = self.l3_compressor(vector, compress=False)
                            
                        l3_results.append((token_id, vector, similarity, metadata.get("position")))
                
                # Combine results and sort by similarity
                all_results = sorted(
                    l1_results + l2_results + l3_results,
                    key=lambda x: x[2],  # Sort by similarity
                    reverse=True
                )
                
                # If we have summaries, also search through summaries
                if self.summarizer is not None and hasattr(self.summarizer, 'summaries'):
                    summary_results = []
                    
                    for level in range(1, 4):  # Check first 3 summary levels
                        if level in self.summarizer.summaries:
                            summaries = self.summarizer.summaries[level]
                            for b_idx, summary_vectors in enumerate(summaries):
                                for j, summary_vector in enumerate(summary_vectors):
                                    similarity = F.cosine_similarity(
                                        query.unsqueeze(0),
                                        summary_vector.unsqueeze(0)
                                    ).item()
                                    
                                    if similarity >= semantic_threshold:
                                        summary_id = f"summary_l{level}_b{b_idx}_i{j}"
                                        pos = self.summarizer.summary_positions[level][b_idx][j].item()
                                        summary_results.append(
                                            (summary_id, summary_vector, similarity, pos)
                                        )
                    
                    # Add summary results and re-sort
                    all_results = sorted(
                        all_results + summary_results,
                        key=lambda x: x[2],
                        reverse=True
                    )
                
                # Extract top 5 results
                top_k = 5
                top_values = []
                for _, value, _, position in all_results[:top_k]:
                    top_values.append(value)
                    
                    # Track access pattern if position is available
                    if position is not None:
                        self.access_history.append((position, time.time()))
                
                # If we found any results, add them
                if top_values:
                    batch_results = torch.stack(top_values)
                    results.append(batch_results)
                else:
                    # Return empty tensor if no results
                    batch_results = torch.zeros(0, self.dim, device=query_vectors.device)
                    results.append(batch_results)
        
        # Update metrics
        retrieval_time = time.time() - start_time
        
        # If using adaptive policies, record this operation and latency
        if self.adaptive_policies:
            self.adaptive_policies.record_access(
                positions=positions,
                token_ids=token_ids,
                latency=retrieval_time * 1000  # Convert to ms
            )
            
        return results
    
    def _try_promote_to_l1(self, token_id=None, position=None):
        """Try to promote a token to L1 cache if it exists in L2/L3"""
        # Check if already in L1
        value, pos = self.l1.get(token_id=token_id, position=position)
        if value is not None:
            return  # Already in L1
            
        # Try to find in L2
        value, pos = self.l2.get(token_id=token_id, position=position)
        if value is not None:
            # Found in L2, decompress if needed
            if self.l2_compressor is not None:
                value = self.l2_compressor(value, compress=False)
            
            # Add to L1
            if token_id is None:
                # Find token ID from position
                for tid, info in self.l2.tokens.items():
                    if info.position == pos:
                        token_id = tid
                        break
                        
            if token_id is not None:
                importance = self.l2.tokens[token_id].importance_score if token_id in self.l2.tokens else 0.5
                self.l1.add(token_id, value, pos, importance)
            return
            
        # Try to find in L3
        value, pos = self.l3.get(token_id=token_id, position=position)
        if value is not None:
            # Found in L3, decompress if needed
            if self.l3_compressor is not None:
                value = self.l3_compressor(value, compress=False)
                
            # Add to L1
            if token_id is None:
                # Find token ID from position
                for tid, info in self.l3.tokens.items():
                    if info.position == pos:
                        token_id = tid
                        break
                        
            if token_id is not None:
                importance = self.l3.tokens[token_id].importance_score if token_id in self.l3.tokens else 0.5
                self.l1.add(token_id, value, pos, importance)
    
    def clear(self):
        """Clear all memory"""
        # Clear memory levels
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        
        # Clear streaming memory if enabled
        if self.streaming_memory is not None:
            self.streaming_memory.clear()
            
        # Clear access tracking
        self.access_history.clear()
        
        # Clear redundancy tracking
        self.shadow_copies.clear()
        
        # Reset metrics
        self.metrics["total_tokens"] = 0
        
    def get_stats(self):
        """Get detailed memory usage statistics"""
        # Basic stats
        stats = {
            "l1_tokens": len(self.l1),
            "l2_tokens": len(self.l2),
            "l3_tokens": len(self.l3),
            "l1_hit_rate": self.l1.hit_rate,
            "l2_hit_rate": self.l2.hit_rate,
            "l3_hit_rate": self.l3.hit_rate,
            "access_pattern": self.current_access_pattern.name,
            "total_tokens": self.metrics["total_tokens"],
        }
        
        # Add streaming stats if enabled
        if self.streaming_memory is not None:
            streaming_stats = {
                "streaming_token_count": self.streaming_memory.token_count,
                "streaming_buffer_size": len(self.streaming_memory.streaming_buffer),
            }
            stats.update(streaming_stats)
            
        # Add summarization stats if enabled
        if self.summarizer is not None:
            summary_stats = {}
            for level in range(1, 5):  # Check levels 1-4
                if level in self.summarizer.summaries:
                    summary_count = sum(len(s) for s in self.summarizer.summaries[level])
                    summary_stats[f"summary_l{level}_count"] = summary_count
                    
            stats.update(summary_stats)
            
        # Add reliability stats if enabled
        if self.enable_redundancy:
            redundancy_stats = {
                "redundant_copies": len(self.shadow_copies),
                "avg_copies_per_token": sum(len(copies) for copies in self.shadow_copies.values()) / 
                                       max(1, len(self.shadow_copies)),
            }
            stats.update(redundancy_stats)
            
        # Add policy stats if enabled
        if self.adaptive_policies is not None:
            stats.update(self.adaptive_policies.get_all_policies())
            
        # Add metrics
        stats.update(self.metrics)
        
        return stats
    
    def optimize(self):
        """Optimize memory system for better performance"""
        # Optimize vector stores
        self.l2.optimize()
        self.l3.optimize()
        
        # Rebalance memory levels
        self._rebalance_memory_levels()
        
        # Run garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Memory system optimized")
    
    def forward(self, inputs, positions=None, token_ids=None, query_vectors=None):
        """
        Process inputs through the memory system
        
        For storing: inputs are token representations to store with positions
        For retrieving: inputs are queries to retrieve from memory
        """
        if positions is not None:
            # We're storing tokens
            return self.add_tokens(inputs, positions)
        elif token_ids is not None:
            # We're retrieving by token ID
            return self.retrieve_tokens(token_ids=token_ids)
        elif query_vectors is not None:
            # We're retrieving by semantic search
            return self.retrieve_tokens(query_vectors=query_vectors)
        else:
            # Default to semantic search
            return self.retrieve_tokens(query_vectors=inputs)
            
    def __del__(self):
        """Cleanup when object is deleted"""
        self._stop_background_worker()

# Advanced distributed memory orchestrator
class DistributedMemoryOrchestrator:
    """
    Production-ready distributed memory coordination system
    for extreme-scale context windows across multiple nodes
    """
    def __init__(self, dim, node_count=4, shard_strategy="token_range"):
        self.dim = dim
        self.node_count = node_count
        self.shard_strategy = shard_strategy  # "token_range", "hash", "adaptive"
        self.nodes = {}  # node_id -> node_info
        self.available = False
        self.token_to_node_map = {}  # token_id -> node_id
        
        # Node health monitoring
        self.node_heartbeats = {}  # node_id -> last_heartbeat_time
        self.node_load = {}  # node_id -> load_metrics
        
        # For token routing
        self._setup_router()
        
        # For index synchronization
        self.index_versions = {}  # node_id -> index_version
        self.last_sync_time = time.time()
        
        logger.info(f"Distributed memory orchestrator initialized with {node_count} nodes")
            
    def _setup_router(self):
        """Set up token routing based on shard strategy"""
        if self.shard_strategy == "token_range":
            # Assign token ranges to nodes
            self.token_ranges = []
            range_size = 2**32 // self.node_count
            
            for i in range(self.node_count):
                start = i * range_size
                end = (i + 1) * range_size - 1 if i < self.node_count - 1 else 2**32 - 1
                self.token_ranges.append((start, end, i))
                
            logger.info(f"Token range partitioning: {len(self.token_ranges)} partitions")
                
        elif self.shard_strategy == "hash":
            # Simple hash-based routing
            logger.info(f"Hash-based routing active")
            
        elif self.shard_strategy == "adaptive":
            # Initialize adaptive routing
            self.token_access_frequency = {}  # token_id -> access_frequency
            self.node_capacity_used = {i: 0.0 for i in range(self.node_count)}  # node_id -> usage
            logger.info(f"Adaptive routing active")
            
    def register_node(self, node_id, capacity, endpoint):
        """Register a node with the orchestrator"""
        self.nodes[node_id] = {
            "id": node_id,
            "capacity": capacity,
            "endpoint": endpoint,
            "available": True,
            "token_count": 0
        }
        
        self.node_heartbeats[node_id] = time.time()
        self.node_load[node_id] = {
            "cpu": 0.0,
            "memory": 0.0,
            "network": 0.0
        }
        
        self.index_versions[node_id] = 0
        
        logger.info(f"Node {node_id} registered with capacity {capacity}")
        return True
        
    def get_node_for_token(self, token_id):
        """Determine which node should store/retrieve a token"""
        if token_id in self.token_to_node_map:
            # We already know where this token is
            return self.token_to_node_map[token_id]
            
        if self.shard_strategy == "token_range":
            # Calculate token's numeric value for range lookup
            import hashlib
            hash_val = int(hashlib.md5(str(token_id).encode()).hexdigest(), 16) % (2**32)
            
            # Find which range it belongs to
            for start, end, node_id in self.token_ranges:
                if start <= hash_val <= end:
                    self.token_to_node_map[token_id] = node_id
                    return node_id
                    
        elif self.shard_strategy == "hash":
            # Simple hash function
            node_id = hash(token_id) % self.node_count
            self.token_to_node_map[token_id] = node_id
            return node_id
            
        elif self.shard_strategy == "adaptive":
            # Check if it's a frequently accessed token
            if token_id in self.token_access_frequency and self.token_access_frequency[token_id] > 10:
                # For frequently accessed tokens, find the least loaded node
                node_id = min(self.node_load.keys(), key=lambda n: self.node_load[n]["memory"])
            else:
                # For normal tokens, use a hash
                node_id = hash(token_id) % self.node_count
                
            self.token_to_node_map[token_id] = node_id
            return node_id
            
        # Fallback to first node if all else fails
        self.token_to_node_map[token_id] = 0
        return 0
        
    def record_token_access(self, token_id):
        """Record that a token was accessed"""
        if self.shard_strategy == "adaptive":
            # Update access frequency
            if token_id in self.token_access_frequency:
                self.token_access_frequency[token_id] += 1
            else:
                self.token_access_frequency[token_id] = 1
                
            # If this is an important token, consider moving it to a faster node
            if self.token_access_frequency[token_id] > 20:
                self._rebalance_token(token_id)
                
    def _rebalance_token(self, token_id):
        """Move a token to a more appropriate node if needed"""
        if token_id not in self.token_to_node_map:
            return
            
        current_node = self.token_to_node_map[token_id]
        
        # Find a better node (faster and less loaded)
        candidates = sorted(self.node_load.keys(), key=lambda n: self.node_load[n]["memory"])
        
        if candidates[0] != current_node:
            # Move token to better node
            new_node = candidates[0]
            
            logger.info(f"Rebalancing token {token_id} from node {current_node} to node {new_node}")
            
            # In a real system, we'd coordinate the move between nodes
            # For now, just update our mapping
            self.token_to_node_map[token_id] = new_node
            
    def batch_route_tokens(self, token_ids):
        """Batch route multiple tokens to nodes"""
        routing = {}  # node_id -> list of token_ids
        
        for token_id in token_ids:
            node_id = self.get_node_for_token(token_id)
            
            if node_id not in routing:
                routing[node_id] = []
                
            routing[node_id].append(token_id)
            
        return routing
        
    def update_node_health(self, node_id, metrics):
        """Update health metrics for a node"""
        self.node_heartbeats[node_id] = time.time()
        self.node_load[node_id] = metrics
        
        # Check if this node is overloaded
        if metrics["memory"] > 0.9:  # 90% memory usage
            self._rebalance_from_node(node_id)
            
    def _rebalance_from_node(self, overloaded_node):
        """Move tokens away from an overloaded node"""
        # Find the least loaded node
        candidates = sorted(
            [n for n in self.node_load.keys() if n != overloaded_node],
            key=lambda n: self.node_load[n]["memory"]
        )
        
        if not candidates:
            return
            
        target_node = candidates[0]
        
        # Find tokens on the overloaded node
        tokens_to_move = [
            tid for tid, nid in self.token_to_node_map.items() 
            if nid == overloaded_node
        ][:100]  # Move up to 100 tokens at a time
        
        for token_id in tokens_to_move:
            self.token_to_node_map[token_id] = target_node
            
        logger.info(f"Rebalanced {len(tokens_to_move)} tokens from node {overloaded_node} to node {target_node}")
        
    def check_node_health(self):
        """Check health of all nodes and handle failures"""
        now = time.time()
        
        for node_id, last_heartbeat in list(self.node_heartbeats.items()):
            # Consider a node failed if no heartbeat for 30 seconds
            if now - last_heartbeat > 30.0:
                self._handle_node_failure(node_id)
                
    def _handle_node_failure(self, node_id):
        """Handle failure of a node"""
        logger.warning(f"Node {node_id} appears to be offline, redistributing tokens")
        
        # Mark node as unavailable
        if node_id in self.nodes:
            self.nodes[node_id]["available"] = False
            
        # Find all tokens on this node
        tokens_to_relocate = [
            tid for tid, nid in self.token_to_node_map.items() 
            if nid == node_id
        ]
        
        # Redistribute tokens
        available_nodes = [
            n_id for n_id, n_info in self.nodes.items()
            if n_info["available"] and n_id != node_id
        ]
        
        if not available_nodes:
            logger.error("No available nodes to redistribute tokens!")
            return
            
        # Simple round-robin redistribution
        for i, token_id in enumerate(tokens_to_relocate):
            new_node = available_nodes[i % len(available_nodes)]
            self.token_to_node_map[token_id] = new_node
            
        logger.info(f"Redistributed {len(tokens_to_relocate)} tokens from failed node {node_id}")
        
    def search_distributed(self, query_vectors, top_k=5):
        """
        Perform distributed search across all nodes
        
        In a real system this would communicate with actual nodes,
        here we simulate the process
        """
        # First, broadcast search to all nodes
        node_results = {}
        
        for node_id, node_info in self.nodes.items():
            if not node_info["available"]:
                continue
                
            # In a real system, we'd send an RPC/API call to each node
            # For simulation, we just assign empty results
            node_results[node_id] = []
            
        # In a real system, we'd wait for all nodes to respond
        # Then merge and rank results
        merged_results = []
        
        # Record token accesses for any tokens that were found
        for results in node_results.values():
            for token_id, _, _, _ in results:
                self.record_token_access(token_id)
                
        return merged_results
        
    def synchronize_indexes(self):
        """Synchronize search indexes across nodes"""
        now = time.time()
        
        # Only sync every 5 minutes
        if now - self.last_sync_time < 300:
            return
            
        self.last_sync_time = now
        
        # In a real system, we'd coordinate index merging/sharing
        # For simulation, we just update the version numbers
        for node_id in self.index_versions:
            self.index_versions[node_id] += 1
            
        logger.info(f"Synchronized search indexes across {len(self.index_versions)} nodes")
        
    def optimize(self):
        """Optimize the distributed memory system"""
        # Check node health
        self.check_node_health()
        
        # Synchronize indexes
        self.synchronize_indexes()
        
        # Clean up stale mappings
        self._cleanup_mappings()
        
    def _cleanup_mappings(self):
        """Clean up stale token mappings"""
        # In a real system, we'd have some expiration policy
        # For simulation, we just limit the map size
        if len(self.token_to_node_map) > 1000000:
            # Remove oldest mappings
            remove_count = len(self.token_to_node_map) - 900000
            for _ in range(remove_count):
                if self.token_to_node_map:
                    self.token_to_node_map.pop(next(iter(self.token_to_node_map)))
                    
    def stats(self):
        """Get statistics about the distributed memory system"""
        return {
            "nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n["available"]),
            "token_map_size": len(self.token_to_node_map),
            "token_access_entries": len(self.token_access_frequency) if hasattr(self, 'token_access_frequency') else 0,
            "shard_strategy": self.shard_strategy
        }

# Memory system testing utilities
class MemorySystemBenchmark:
    """Utilities for benchmarking memory system performance"""
    
    @staticmethod
    def generate_random_tokens(batch_size, seq_len, dim):
        """Generate random token embeddings for testing"""
        return torch.randn(batch_size, seq_len, dim)
        
    @staticmethod
    def generate_sequential_positions(batch_size, seq_len, start_pos=0):
        """Generate sequential position indices"""
        positions = torch.arange(start_pos, start_pos + seq_len).unsqueeze(0)
        positions = positions.repeat(batch_size, 1)
        return positions
        
    @staticmethod
    def benchmark_add_tokens(memory_system, batch_size=4, seq_len=128, dim=768, trials=10):
        """Benchmark token addition performance"""
        times = []
        
        for _ in range(trials):
            # Generate random tokens
            tokens = MemorySystemBenchmark.generate_random_tokens(batch_size, seq_len, dim)
            positions = MemorySystemBenchmark.generate_sequential_positions(batch_size, seq_len)
            
            # Time the operation
            start_time = time.time()
            memory_system.add_tokens(tokens, positions)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
        # Calculate statistics
        avg_time = sum(times) / len(times)
        throughput = (batch_size * seq_len) / avg_time
        
        return {
            "operation": "add_tokens",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "avg_time_seconds": avg_time,
            "throughput_tokens_per_second": throughput,
            "tokens_per_batch": batch_size * seq_len
        }
        
    @staticmethod
    def benchmark_retrieve_tokens(memory_system, batch_size=4, seq_len=128, dim=768, trials=10):
        """Benchmark token retrieval performance"""
        # First add tokens to retrieve
        tokens = MemorySystemBenchmark.generate_random_tokens(batch_size, seq_len, dim)
        positions = MemorySystemBenchmark.generate_sequential_positions(batch_size, seq_len)
        token_ids = memory_system.add_tokens(tokens, positions)
        
        # Benchmark position-based retrieval
        pos_times = []
        for _ in range(trials):
            # Generate random positions to retrieve
            retrieve_positions = torch.randint(0, seq_len, (batch_size, 10))
            
            # Time the operation
            start_time = time.time()
            memory_system.retrieve_tokens(positions=retrieve_positions)
            end_time = time.time()
            
            pos_times.append(end_time - start_time)
            
        # Benchmark token ID retrieval
        id_times = []
        for _ in range(trials):
            # Get a subset of token IDs to retrieve
            retrieve_ids = [[token_ids[b][i] for i in range(0, 10)] for b in range(batch_size)]
            
            # Time the operation
            start_time = time.time()
            memory_system.retrieve_tokens(token_ids=retrieve_ids)
            end_time = time.time()
            
            id_times.append(end_time - start_time)
            
        # Benchmark semantic search
        search_times = []
        for _ in range(trials):
            # Generate random query vectors
            query_vectors = torch.randn(batch_size, dim)
            
            # Time the operation
            start_time = time.time()
            memory_system.retrieve_tokens(query_vectors=query_vectors)
            end_time = time.time()
            
            search_times.append(end_time - start_time)
            
        # Calculate statistics
        avg_pos_time = sum(pos_times) / len(pos_times)
        avg_id_time = sum(id_times) / len(id_times)
        avg_search_time = sum(search_times) / len(search_times)
        
        pos_throughput = (batch_size * 10) / avg_pos_time
        id_throughput = (batch_size * 10) / avg_id_time
        search_throughput = batch_size / avg_search_time
        
        return {
            "operation": "retrieve_tokens",
            "position_retrieval_time": avg_pos_time,
            "token_id_retrieval_time": avg_id_time,
            "semantic_search_time": avg_search_time,
            "position_throughput": pos_throughput,
            "token_id_throughput": id_throughput,
            "search_throughput": search_throughput
        }
        
    @staticmethod
    def benchmark_scaling(memory_system, seq_len_range=(1000, 10000, 100000), dim=768):
        """Benchmark how the memory system scales with context length"""
        results = []
        
        for seq_len in seq_len_range:
            # Generate tokens and add to memory
            batch_size = max(1, 10000 // seq_len)  # Adjust batch size based on sequence length
            tokens = MemorySystemBenchmark.generate_random_tokens(batch_size, seq_len, dim)
            positions = MemorySystemBenchmark.generate_sequential_positions(batch_size, seq_len)
            
            # Time the addition
            start_time = time.time()
            memory_system.add_tokens(tokens, positions)
            add_time = time.time() - start_time
            
            # Time retrieval of random positions
            retrieve_positions = torch.randint(0, seq_len, (batch_size, min(100, seq_len // 10)))
            start_time = time.time()
            memory_system.retrieve_tokens(positions=retrieve_positions)
            retrieve_time = time.time() - start_time
            
            # Get memory stats
            stats = memory_system.get_stats()
            
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "add_time": add_time,
                "retrieve_time": retrieve_time,
                "add_throughput": (batch_size * seq_len) / add_time,
                "retrieve_throughput": (batch_size * retrieve_positions.size(1)) / retrieve_time,
                "memory_stats": stats
            })
            
            # Clear memory for next test
            memory_system.clear()
            
        return results
        
    @staticmethod
    def run_all_benchmarks(memory_system, dim=768):
        """Run a comprehensive set of benchmarks"""
        logger.info("Starting comprehensive memory system benchmarks...")
        
        # Basic operations benchmarks
        add_results = MemorySystemBenchmark.benchmark_add_tokens(
            memory_system, batch_size=4, seq_len=128, dim=dim
        )
        logger.info(f"Add tokens benchmark: {add_results['throughput_tokens_per_second']:.2f} tokens/sec")
        
        retrieve_results = MemorySystemBenchmark.benchmark_retrieve_tokens(
            memory_system, batch_size=4, seq_len=128, dim=dim
        )
        logger.info(f"Retrieve tokens benchmark (position): {retrieve_results['position_throughput']:.2f} tokens/sec")
        logger.info(f"Retrieve tokens benchmark (semantic): {retrieve_results['search_throughput']:.2f} queries/sec")
        
        # Clear memory
        memory_system.clear()
        
        # Scaling benchmark with increasing sequence lengths
        scaling_results = MemorySystemBenchmark.benchmark_scaling(
            memory_system, seq_len_range=(1000, 10000, 100000), dim=dim
        )
        
        for result in scaling_results:
            logger.info(f"Scaling benchmark at {result['seq_len']} tokens: "
                      f"Add throughput = {result['add_throughput']:.2f} tokens/sec, "
                      f"Retrieve throughput = {result['retrieve_throughput']:.2f} tokens/sec")
            
        return {
            "add_benchmark": add_results,
            "retrieve_benchmark": retrieve_results,
            "scaling_benchmark": scaling_results
        }

# Export and restore functionality
def export_memory_state(memory_system, filename):
    """Export memory system state to disk"""
    state = {
        "l1_tokens": {},
        "l2_tokens": {},
        "l3_tokens": {},
        "shadow_copies": memory_system.shadow_copies if hasattr(memory_system, 'shadow_copies') else {},
        "metrics": memory_system.metrics if hasattr(memory_system, 'metrics') else {},
        "summaries": {},
    }
    
    # Save L1 tokens
    for token_id, token_info in memory_system.l1.tokens.items():
        state["l1_tokens"][token_id] = {
            "value": token_info.value.cpu().numpy(),
            "position": token_info.position,
            "creation_time": token_info.creation_time,
            "last_access_time": token_info.last_access_time,
            "access_count": token_info.access_count,
            "importance_score": token_info.importance_score
        }
        
    # Save L2 tokens
    for token_id, token_info in memory_system.l2.tokens.items():
        state["l2_tokens"][token_id] = {
            "value": token_info.value.cpu().numpy(),
            "position": token_info.position,
            "creation_time": token_info.creation_time,
            "last_access_time": token_info.last_access_time,
            "access_count": token_info.access_count,
            "importance_score": token_info.importance_score
        }
        
    # Save L3 tokens (only metadata to save space)
    for token_id, token_info in memory_system.l3.tokens.items():
        state["l3_tokens"][token_id] = {
            "position": token_info.position,
            "creation_time": token_info.creation_time,
            "last_access_time": token_info.last_access_time,
            "access_count": token_info.access_count,
            "importance_score": token_info.importance_score
        }
        
    # Save summaries if available
    if hasattr(memory_system, 'summarizer') and memory_system.summarizer is not None:
        for level in range(1, 5):  # Save first 4 levels
            if level in memory_system.summarizer.summaries:
                state["summaries"][f"level_{level}"] = {
                    "vectors": [s.cpu().numpy() for s in memory_system.summarizer.summaries[level]],
                    "positions": [p.cpu().numpy() for p in memory_system.summarizer.summary_positions.get(level, [])]
                }
                
    # Save to disk
    try:
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Memory state exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting memory state: {e}")
        return False

def restore_memory_state(memory_system, filename):
    """Restore memory system state from disk"""
    try:
        import pickle
        import numpy as np
        
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
        # Clear current state
        memory_system.clear()
        
        device = next(memory_system.parameters()).device
        
        # Restore L1 tokens
        for token_id, token_data in state["l1_tokens"].items():
            value = torch.tensor(token_data["value"], device=device)
            position = token_data["position"]
            importance = token_data["importance_score"]
            
            # Add to L1
            memory_system.l1.add(token_id, value, position, importance)
            
            # Restore additional fields
            token_info = memory_system.l1.tokens[token_id]
            token_info.creation_time = token_data["creation_time"]
            token_info.last_access_time = token_data["last_access_time"]
            token_info.access_count = token_data["access_count"]
            
        # Restore L2 tokens
        for token_id, token_data in state["l2_tokens"].items():
            value = torch.tensor(token_data["value"], device=device)
            position = token_data["position"]
            importance = token_data["importance_score"]
            
            # Add to L2
            memory_system.l2.add(token_id, value, position, importance)
            
            # Restore additional fields
            token_info = memory_system.l2.tokens[token_id]
            token_info.creation_time = token_data["creation_time"]
            token_info.last_access_time = token_data["last_access_time"]
            token_info.access_count = token_data["access_count"]
            
        # Restore L3 tokens (with compressed values)
        for token_id, token_data in state["l3_tokens"].items():
            # Create a default compressed value (will be replaced in real usage)
            value = torch.zeros(memory_system.dim // memory_system.l3_compressor.compression_ratio, device=device)
            position = token_data["position"]
            importance = token_data["importance_score"]
            
            # Add to L3
            memory_system.l3.add(token_id, value, position, importance)
            
            # Restore additional fields
            token_info = memory_system.l3.tokens[token_id]
            token_info.creation_time = token_data["creation_time"]
            token_info.last_access_time = token_data["last_access_time"]
            token_info.access_count = token_data["access_count"]
            
        # Restore shadow copies
        memory_system.shadow_copies = state["shadow_copies"]
        
        # Restore metrics
        memory_system.metrics = state["metrics"]
        
        # Restore summaries if available
        if "summaries" in state and hasattr(memory_system, 'summarizer') and memory_system.summarizer is not None:
            for level_key, level_data in state["summaries"].items():
                level = int(level_key.split("_")[1])
                
                # Convert numpy arrays back to tensors
                vectors = [torch.tensor(v, device=device) for v in level_data["vectors"]]
                positions = [torch.tensor(p, device=device) for p in level_data["positions"]]
                
                memory_system.summarizer.summaries[level] = vectors
                memory_system.summarizer.summary_positions[level] = positions
                
        logger.info(f"Memory state restored from {filename}")
        return True
    except Exception as e:
        logger.error(f"Error restoring memory state: {e}")
        return False

# Main usage example
def create_enterprise_memory_system(dim=768, max_tokens=100_000_000):
    """Create an enterprise-ready memory system for 100M tokens"""
    
    # Calculate appropriate tier sizes
    l1_size = 32 * 1024  # 32K tokens in fastest memory
    l2_size = 1 * 1024 * 1024  # 1M tokens in medium memory
    l3_size = 10 * 1024 * 1024  # 10M tokens in slower memory
    disk_size = max_tokens  # Remaining on disk/distributed
    
    # Create performance config
    perf_config = PerformanceConfig(
        optimize_memory=True,
        optimize_speed=True,
        optimize_reliability=True,
        use_mixed_precision=True,
        device_specific_optimizations=True
    )
    
    # Create the advanced memory system
    memory_system = AdvancedHierarchicalMemoryManager(
        dim=dim,
        l1_capacity=l1_size,
        l2_capacity=l2_size,
        l3_capacity=l3_size,
        disk_capacity=disk_size,
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
        distributed_nodes=4,
        hardware_acceleration="auto",
        reliability_level="high",
        perf_config=perf_config
    )
    
    logger.info(f"Created enterprise memory system with {max_tokens} token capacity")
    logger.info(f"Memory levels: L1={l1_size}, L2={l2_size}, L3={l3_size}, Disk={disk_size}")
    
    return memory_system
