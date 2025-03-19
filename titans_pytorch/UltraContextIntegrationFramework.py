import torch
import logging
import time
import threading
import queue
import json
import uuid
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

# Import from main memory system
from ultracontext.memory import (
    AdvancedHierarchicalMemoryManager,
    MemorySummarizer,
    export_memory_state,
    restore_memory_state,
    MemorySystemBenchmark,
    StreamingMemoryManager
)

logger = logging.getLogger("ultracontext.memory_api")

class MemoryPriority(Enum):
    """Priority levels for token processing"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class TokenMetadata:
    """Metadata for tokens stored in memory"""
    token_id: str
    position: int
    importance: float = 0.5
    creation_time: float = None
    source: str = "unknown"
    tags: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()
        if self.tags is None:
            self.tags = []
        if self.properties is None:
            self.properties = {}

@dataclass
class MemoryChunk:
    """A chunk of tokens representing a logical unit in memory"""
    chunk_id: str
    token_ids: List[str]
    start_position: int
    end_position: int
    token_count: int
    creation_time: float = None
    source: str = "unknown"
    tags: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.time()
        if self.tags is None:
            self.tags = []
        if self.properties is None:
            self.properties = {}
            
    @property
    def age(self):
        """How long since creation (in seconds)"""
        return time.time() - self.creation_time
        
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "token_ids": self.token_ids,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "token_count": self.token_count,
            "creation_time": self.creation_time,
            "source": self.source,
            "tags": self.tags,
            "properties": self.properties
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(
            chunk_id=data["chunk_id"],
            token_ids=data["token_ids"],
            start_position=data["start_position"],
            end_position=data["end_position"],
            token_count=data["token_count"],
            creation_time=data["creation_time"],
            source=data["source"],
            tags=data["tags"],
            properties=data["properties"]
        )

class MemoryAccessStats:
    """Statistics for memory access patterns and performance"""
    def __init__(self):
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.response_times = []
        self.last_access_time = None
        self.last_optimization_time = None
        self.access_distribution = {
            "by_position": {},  # position_range -> count
            "by_token_id": {},  # token_id -> count
            "by_query": {},     # query type -> count
            "by_source": {},    # source -> count
        }
        
    def record_access(self, was_hit=True, response_time=None, position=None, token_id=None, query_type=None, source=None):
        """Record an access to the memory system"""
        self.access_count += 1
        if was_hit:
            self.hit_count += 1
        else:
            self.miss_count += 1
            
        self.last_access_time = time.time()
        
        if response_time is not None:
            self.response_times.append(response_time)
            # Keep only the last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
                
        # Record position distribution
        if position is not None:
            position_range = position // 1000 * 1000  # Group by thousands
            pos_key = f"{position_range}-{position_range+999}"
            self.access_distribution["by_position"][pos_key] = self.access_distribution["by_position"].get(pos_key, 0) + 1
            
        # Record token ID access
        if token_id is not None:
            self.access_distribution["by_token_id"][token_id] = self.access_distribution["by_token_id"].get(token_id, 0) + 1
            
        # Record query type
        if query_type is not None:
            self.access_distribution["by_query"][query_type] = self.access_distribution["by_query"].get(query_type, 0) + 1
            
        # Record source
        if source is not None:
            self.access_distribution["by_source"][source] = self.access_distribution["by_source"].get(source, 0) + 1
            
    @property
    def hit_rate(self):
        """Calculate hit rate"""
        if self.access_count == 0:
            return 0.0
        return self.hit_count / self.access_count
        
    @property
    def avg_response_time(self):
        """Calculate average response time in ms"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
        
    def get_stats(self):
        """Get all statistics as a dictionary"""
        return {
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
            "avg_response_time_ms": self.avg_response_time,
            "last_access_time": self.last_access_time,
            "last_optimization_time": self.last_optimization_time,
            "top_positions": dict(sorted(self.access_distribution["by_position"].items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_token_ids": dict(sorted(self.access_distribution["by_token_id"].items(), key=lambda x: x[1], reverse=True)[:10]),
            "query_types": self.access_distribution["by_query"],
            "sources": self.access_distribution["by_source"]
        }
        
    def clear(self):
        """Clear the statistics"""
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.response_times = []
        self.last_access_time = None
        self.access_distribution = {
            "by_position": {},
            "by_token_id": {},
            "by_query": {},
            "by_source": {},
        }

class MemoryAPI:
    """High-level API for interacting with the memory system"""
    def __init__(
        self,
        dim: int = 768,
        max_tokens: int = 100_000_000,
        enable_streaming: bool = True,
        distributed_nodes: int = 1,
        reliability_level: str = "normal",
        persistence_path: str = "./memory_state",
        auto_save_interval: int = 600,  # Seconds
        config_path: str = None
    ):
        self.dim = dim
        self.max_tokens = max_tokens
        self.persistence_path = persistence_path
        self.auto_save_interval = auto_save_interval
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
            
        # Set up the memory manager
        logger.info(f"Initializing memory system with dimension {dim}, max tokens {max_tokens}")
        self.memory_manager = self._create_memory_manager(
            dim=dim,
            max_tokens=max_tokens,
            enable_streaming=enable_streaming,
            distributed_nodes=distributed_nodes,
            reliability_level=reliability_level
        )
        
        # For tracking chunks of tokens
        self.chunks = {}  # chunk_id -> MemoryChunk
        self.tokens_to_chunks = {}  # token_id -> chunk_id
        
        # For tracking token metadata
        self.token_metadata = {}  # token_id -> TokenMetadata
        
        # Statistics
        self.stats = MemoryAccessStats()
        
        # For auto-saving state
        self.last_save_time = time.time()
        self.auto_save_enabled = True
        
        # Background worker for maintenance
        self.maintenance_queue = queue.PriorityQueue()
        self.maintenance_thread = None
        self.stop_maintenance = False
        
        # Start background maintenance
        self._start_maintenance_thread()
        
        logger.info(f"Memory API initialized, ready for operation")
        
    def _load_config(self, config_path):
        """Load configuration from a file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Apply configuration settings
            if "dimension" in config:
                self.dim = config["dimension"]
            if "max_tokens" in config:
                self.max_tokens = config["max_tokens"]
            if "persistence_path" in config:
                self.persistence_path = config["persistence_path"]
            if "auto_save_interval" in config:
                self.auto_save_interval = config["auto_save_interval"]
                
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
        
    def _create_memory_manager(self, dim, max_tokens, enable_streaming, distributed_nodes, reliability_level):
        """Create the memory manager with appropriate configuration"""
        # Calculate appropriate tier sizes
        l1_size = min(65536, max(16384, max_tokens // 1000))  # Typically 0.1% of total
        l2_size = min(1048576, max(131072, max_tokens // 100))  # Typically 1% of total
        l3_size = min(10485760, max(1048576, max_tokens // 10))  # Typically 10% of total
        disk_size = max_tokens
        
        # Create the advanced memory system
        return AdvancedHierarchicalMemoryManager(
            dim=dim,
            l1_capacity=l1_size,
            l2_capacity=l2_size,
            l3_capacity=l3_size,
            disk_capacity=disk_size,
            enable_summarization=True,
            enable_streaming=enable_streaming,
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
            distributed_nodes=distributed_nodes,
            hardware_acceleration="auto",
            reliability_level=reliability_level
        )
        
    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        if self.maintenance_thread is None:
            self.stop_maintenance = False
            self.maintenance_thread = threading.Thread(
                target=self._maintenance_worker,
                daemon=True
            )
            self.maintenance_thread.start()
            logger.info("Background maintenance thread started")
            
    def _stop_maintenance_thread(self):
        """Stop the background maintenance thread"""
        if self.maintenance_thread is not None:
            self.stop_maintenance = True
            self.maintenance_thread.join(timeout=2.0)
            self.maintenance_thread = None
            logger.info("Background maintenance thread stopped")
            
    def _maintenance_worker(self):
        """Background worker for maintenance tasks"""
        while not self.stop_maintenance:
            try:
                # Check if auto-save is due
                if self.auto_save_enabled and time.time() - self.last_save_time > self.auto_save_interval:
                    self._auto_save()
                    
                # Check for maintenance tasks
                try:
                    priority, task, args = self.maintenance_queue.get(timeout=1.0)
                    task(*args)
                    self.maintenance_queue.task_done()
                except queue.Empty:
                    # No tasks, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in maintenance worker: {e}")
                
    def _auto_save(self):
        """Automatically save state at regular intervals"""
        try:
            # Use timestamp in filename for versioning
            timestamp = int(time.time())
            filename = f"{self.persistence_path}/memory_state_{timestamp}.bin"
            
            success = self.save_state(filename)
            if success:
                self.last_save_time = time.time()
                logger.info(f"Auto-saved memory state to {filename}")
                
                # Clean up old saves (keep last 5)
                self._cleanup_old_saves()
                
        except Exception as e:
            logger.error(f"Error during auto-save: {e}")
            
    def _cleanup_old_saves(self):
        """Clean up old save files, keeping only the most recent ones"""
        try:
            import os
            import glob
            
            # Get list of save files
            files = glob.glob(f"{self.persistence_path}/memory_state_*.bin")
            
            # Sort by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Delete all but the 5 newest
            for file in files[5:]:
                try:
                    os.remove(file)
                    logger.debug(f"Removed old save file: {file}")
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error cleaning up old save files: {e}")
            
    def _schedule_maintenance(self, task, args=(), priority=50):
        """Schedule a maintenance task"""
        self.maintenance_queue.put((priority, task, args))
        
    def store_tokens(
        self, 
        token_vectors: torch.Tensor, 
        positions: torch.Tensor, 
        source: str = "default",
        tags: List[str] = None,
        importance: Optional[torch.Tensor] = None,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        properties: Dict[str, Any] = None,
        chunk_id: str = None
    ) -> Tuple[List[str], str]:
        """
        Store token vectors in memory
        
        Args:
            token_vectors: Token embeddings [batch_size, seq_len, dim]
            positions: Token positions [batch_size, seq_len]
            source: Source of the tokens (e.g., "user_input", "model_output")
            tags: List of tags to associate with these tokens
            importance: Optional importance scores [batch_size, seq_len]
            priority: Processing priority level
            properties: Additional properties to store with tokens
            chunk_id: Optional ID to use for the chunk, otherwise auto-generated
            
        Returns:
            Tuple of (token_ids, chunk_id)
        """
        start_time = time.time()
        
        # Ensure tensors are on the correct device
        device = next(self.memory_manager.parameters()).device
        token_vectors = token_vectors.to(device)
        positions = positions.to(device)
        if importance is not None:
            importance = importance.to(device)
            
        # If no chunk ID provided, generate one
        if chunk_id is None:
            chunk_id = str(uuid.uuid4())
            
        # Get positions as a flat list for metadata
        position_list = positions.view(-1).tolist()
        
        # Get start and end positions
        start_position = min(position_list)
        end_position = max(position_list)
        
        # Store tokens in memory system
        token_ids = self.memory_manager.add_tokens(
            tokens=token_vectors,
            positions=positions,
            attention_weights=importance
        )
        
        # Flatten token IDs list
        flat_token_ids = [tid for batch in token_ids for tid in batch]
        
        # Store token metadata
        for i, token_id in enumerate(flat_token_ids):
            pos = position_list[i]
            imp = importance.view(-1)[i].item() if importance is not None else 0.5
            
            metadata = TokenMetadata(
                token_id=token_id,
                position=pos,
                importance=imp,
                source=source,
                tags=tags or [],
                properties=properties or {}
            )
            
            self.token_metadata[token_id] = metadata
            self.tokens_to_chunks[token_id] = chunk_id
            
        # Create or update the chunk
        if chunk_id in self.chunks:
            # Update existing chunk
            existing_chunk = self.chunks[chunk_id]
            existing_chunk.token_ids.extend(flat_token_ids)
            existing_chunk.start_position = min(existing_chunk.start_position, start_position)
            existing_chunk.end_position = max(existing_chunk.end_position, end_position)
            existing_chunk.token_count += len(flat_token_ids)
            if tags:
                existing_chunk.tags = list(set(existing_chunk.tags + tags))
            if properties:
                existing_chunk.properties.update(properties)
        else:
            # Create new chunk
            self.chunks[chunk_id] = MemoryChunk(
                chunk_id=chunk_id,
                token_ids=flat_token_ids,
                start_position=start_position,
                end_position=end_position,
                token_count=len(flat_token_ids),
                source=source,
                tags=tags or [],
                properties=properties or {}
            )
            
        # Log the operation
        response_time = (time.time() - start_time) * 1000  # ms
        logger.debug(f"Stored {len(flat_token_ids)} tokens from {source} in {response_time:.2f}ms (chunk: {chunk_id})")
        
        # Update statistics
        self.stats.record_access(
            was_hit=True,  # Storage is always a 'hit'
            response_time=response_time,
            query_type="store",
            source=source
        )
        
        # If high priority, explicitly promote to faster memory
        if priority in [MemoryPriority.HIGH, MemoryPriority.CRITICAL]:
            self._schedule_maintenance(
                self._promote_tokens,
                args=(flat_token_ids,),
                priority=10 if priority == MemoryPriority.CRITICAL else 20
            )
            
        # Schedule auto-prune if approaching capacity
        manager_stats = self.memory_manager.get_stats()
        total_tokens = manager_stats.get("total_tokens", 0)
        if total_tokens > self.max_tokens * 0.9:  # 90% full
            self._schedule_maintenance(
                self._prune_old_tokens,
                priority=30
            )
            
        return flat_token_ids, chunk_id
        
    def retrieve_by_positions(
        self, 
        positions: torch.Tensor,
        include_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """
        Retrieve tokens by their positions
        
        Args:
            positions: Tensor of positions to retrieve [batch_size, seq_len]
            include_metadata: Whether to return metadata with the tokens
            
        Returns:
            If include_metadata=False: Token vectors [batch_size, seq_len, dim]
            If include_metadata=True: Tuple of (token_vectors, metadata)
        """
        start_time = time.time()
        
        # Ensure tensor is on the correct device
        device = next(self.memory_manager.parameters()).device
        positions = positions.to(device)
        
        # Retrieve from memory system
        results = self.memory_manager.retrieve_tokens(positions=positions)
        
        if not results:
            # Handle case where nothing was found
            batch_size = positions.size(0) if positions.dim() > 1 else 1
            seq_len = positions.size(1) if positions.dim() > 1 else positions.size(0)
            
            empty_result = torch.zeros(batch_size, seq_len, self.dim, device=device)
            
            # Update statistics
            self.stats.record_access(
                was_hit=False,
                response_time=(time.time() - start_time) * 1000,
                position=positions.view(-1)[0].item() if positions.numel() > 0 else None,
                query_type="position_retrieve"
            )
            
            return (empty_result, []) if include_metadata else empty_result
        
        # Get the first result (should be the only one for position-based retrieval)
        token_vectors = results[0]
        
        # Calculate hit/miss for statistics
        hits = torch.sum(torch.any(token_vectors != 0, dim=-1)).item()
        total = token_vectors.shape[0] * token_vectors.shape[1]
        was_hit = hits > 0
        
        # Update statistics
        self.stats.record_access(
            was_hit=was_hit,
            response_time=(time.time() - start_time) * 1000,
            position=positions.view(-1)[0].item() if positions.numel() > 0 else None,
            query_type="position_retrieve"
        )
        
        # If metadata is requested, collect it
        if include_metadata:
            metadata_list = []
            
            # Convert positions to flat list
            pos_list = positions.view(-1).tolist()
            
            # Look up metadata for each position
            for pos in pos_list:
                # Find token ID at this position (if any)
                token_id = None
                if hasattr(self.memory_manager.l1, 'position_index'):
                    token_id = self.memory_manager.l1.position_index.get(pos)
                if token_id is None and hasattr(self.memory_manager.l2, 'position_index'):
                    token_id = self.memory_manager.l2.position_index.get(pos)
                if token_id is None and hasattr(self.memory_manager.l3, 'position_index'):
                    token_id = self.memory_manager.l3.position_index.get(pos)
                
                # Get metadata if we found a token ID
                if token_id and token_id in self.token_metadata:
                    metadata = self.token_metadata[token_id]
                    metadata_list.append(metadata.__dict__)
                else:
                    # No metadata found
                    metadata_list.append(None)
                    
            return token_vectors, metadata_list
            
        return token_vectors
        
    def retrieve_by_token_ids(
        self, 
        token_ids: List[str],
        include_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """
        Retrieve tokens by their IDs
        
        Args:
            token_ids: List of token IDs to retrieve
            include_metadata: Whether to return metadata with the tokens
            
        Returns:
            If include_metadata=False: Token vectors [seq_len, dim]
            If include_metadata=True: Tuple of (token_vectors, metadata)
        """
        start_time = time.time()
        
        # Retrieve from memory system (convert to expected format)
        results = self.memory_manager.retrieve_tokens(token_ids=[token_ids])
        
        if not results:
            # Handle case where nothing was found
            device = next(self.memory_manager.parameters()).device
            empty_result = torch.zeros(len(token_ids), self.dim, device=device)
            
            # Update statistics
            self.stats.record_access(
                was_hit=False,
                response_time=(time.time() - start_time) * 1000,
                token_id=token_ids[0] if token_ids else None,
                query_type="token_id_retrieve"
            )
            
            return (empty_result, []) if include_metadata else empty_result
        
        # Get the first result (should be the only one for token_id-based retrieval)
        token_vectors = results[0]
        
        # Calculate hit/miss for statistics
        hits = torch.sum(torch.any(token_vectors != 0, dim=-1)).item()
        was_hit = hits > 0
        
        # Update statistics
        self.stats.record_access(
            was_hit=was_hit,
            response_time=(time.time() - start_time) * 1000,
            token_id=token_ids[0] if token_ids else None,
            query_type="token_id_retrieve"
        )
        
        # If metadata is requested, collect it
        if include_metadata:
            metadata_list = []
            
            for token_id in token_ids:
                if token_id in self.token_metadata:
                    metadata = self.token_metadata[token_id]
                    metadata_list.append(metadata.__dict__)
                else:
                    metadata_list.append(None)
                    
            return token_vectors, metadata_list
            
        return token_vectors
        
    def retrieve_by_chunk(
        self, 
        chunk_id: str,
        include_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict], Dict]]:
        """
        Retrieve all tokens in a chunk
        
        Args:
            chunk_id: ID of the chunk to retrieve
            include_metadata: Whether to return metadata with the tokens
            
        Returns:
            If include_metadata=False: Token vectors [chunk_size, dim]
            If include_metadata=True: Tuple of (token_vectors, token_metadata, chunk_metadata)
        """
        if chunk_id not in self.chunks:
            device = next(self.memory_manager.parameters()).device
            empty_result = torch.zeros(0, self.dim, device=device)
            logger.warning(f"Chunk {chunk_id} not found")
            
            # Update statistics
            self.stats.record_access(
                was_hit=False,
                query_type="chunk_retrieve"
            )
            
            return (empty_result, [], {}) if include_metadata else empty_result
            
        # Get token IDs for this chunk
        chunk = self.chunks[chunk_id]
        token_ids = chunk.token_ids
        
        # Retrieve all tokens in the chunk
        token_vectors = self.retrieve_by_token_ids(token_ids, include_metadata=False)
        
        # Update statistics
        self.stats.record_access(
            was_hit=True,
            query_type="chunk_retrieve",
            source=chunk.source
        )
        
        if include_metadata:
            # Collect token metadata
            metadata_list = []
            for token_id in token_ids:
                if token_id in self.token_metadata:
                    metadata = self.token_metadata[token_id]
                    metadata_list.append(metadata.__dict__)
                else:
                    metadata_list.append(None)
                    
            # Return with both token metadata and chunk metadata
            return token_vectors, metadata_list, chunk.to_dict()
            
        return token_vectors
        
    def semantic_search(
        self, 
        query_vectors: torch.Tensor,
        top_k: int = 5,
        min_similarity: float = 0.7,
        include_metadata: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """
        Perform semantic search for similar tokens
        
        Args:
            query_vectors: Query vectors for search [batch_size, dim]
            top_k: Maximum number of results to return per query
            min_similarity: Minimum similarity threshold
            include_metadata: Whether to return metadata with results
            
        Returns:
            If include_metadata=False: Similar token vectors [batch_size, top_k, dim]
            If include_metadata=True: Tuple of (token_vectors, metadata)
        """
        start_time = time.time()
        
        # Ensure tensor is on the correct device
        device = next(self.memory_manager.parameters()).device
        query_vectors = query_vectors.to(device)
        
        # Perform the search
        results = self.memory_manager.retrieve_tokens(query_vectors=query_vectors)
        
        if not results:
            # Handle case where nothing was found
            batch_size = query_vectors.size(0)
            empty_result = torch.zeros(batch_size, 0, self.dim, device=device)
            
            # Update statistics
            self.stats.record_access(
                was_hit=False,
                response_time=(time.time() - start_time) * 1000,
                query_type="semantic_search"
            )
            
            return (empty_result, []) if include_metadata else empty_result
            
        # Process results
        processed_results = []
        metadata_results = []
        
        for batch_idx, batch_results in enumerate(results):
            # Filter by similarity and limit to top_k
            # Note: In a real implementation, we'd have similarity scores
            # Here we're just limiting the number of results
            limited_results = batch_results[:top_k]
            processed_results.append(limited_results)
            
            # Collect metadata if requested
            if include_metadata:
                batch_metadata = []
                
                # Try to find token IDs for these vectors
                # This is a simplification - in a real system we would track
                # which token IDs were returned by the search
                for vector_idx, vector in enumerate(limited_results):
                    token_found = False
                    
                    # Look for matching vectors in our metadata
                    # In a real system, we'd have a direct mapping from search results
                    for token_id, metadata in self.token_metadata.items():
                        # We'd compare vector to actual stored vector, which is inefficient
                        # This is just a placeholder for demonstration
                        if token_found:
                            break
                            
                        # In practice, we'd use the token IDs returned by the search
                        # instead of this inefficient lookup
                        batch_metadata.append(metadata.__dict__)
                        token_found = True
                        
                    # If no matching token found, add None
                    if not token_found:
                        batch_metadata.append(None)
                        
                metadata_results.append(batch_metadata)
        
        # Calculate hit/miss for statistics
        was_hit = any(len(res) > 0 for res in processed_results)
        
        # Update statistics
        self.stats.record_access(
            was_hit=was_hit,
            response_time=(time.time() - start_time) * 1000,
            query_type="semantic_search"
        )
        
        # Convert results to a tensor
        max_results = max(res.size(0) for res in processed_results) if processed_results else 0
        batch_size = len(processed_results)
        
        if max_results == 0:
            # No results found
            empty_result = torch.zeros(batch_size, 0, self.dim, device=device)
            return (empty_result, metadata_results) if include_metadata else empty_result
            
        # Pad results to same size
        padded_results = torch.zeros(batch_size, max_results, self.dim, device=device)
        for i, res in enumerate(processed_results):
            if res.size(0) > 0:
                padded_results[i, :res.size(0)] = res
                
        if include_metadata:
            return padded_results, metadata_results
            
        return padded_results
        
    def retrieve_chunks_by_tags(
        self,
        tags: List[str],
        require_all: bool = False,
        max_chunks: int = 10
    ) -> List[MemoryChunk]:
        """
        Retrieve chunks that have specific tags
        
        Args:
            tags: List of tags to search for
            require_all: If True, chunks must have all tags; if False, any tag matches
            max_chunks: Maximum number of chunks to return
            
        Returns:
            List of matching chunks
        """
        matching_chunks = []
        
        for chunk_id, chunk in self.chunks.items():
            if require_all:
                # Must have all tags
                if all(tag in chunk.tags for tag in tags):
                    matching_chunks.append(chunk)
            else:
                # Any tag matches
                if any(tag in chunk.tags for tag in tags):
                    matching_chunks.append(chunk)
                    
        # Sort by recency (newest first)
        matching_chunks.sort(key=lambda x: x.creation_time, reverse=True)
        
        # Limit results
        return matching_chunks[:max_chunks]
        
    def retrieve_summary(
        self,
        chunk_ids: List[str] = None,
        positions: List[int] = None,
        positions_range: Tuple[int, int] = None,
        summary_level: int = 1
    ) -> torch.Tensor:
        """
        Retrieve summarized representations of content
        
        Args:
            chunk_ids: Specific chunks to summarize
            positions: Specific positions to summarize
            positions_range: Range of positions to summarize
            summary_level: Summarization level (1=light, 2=medium, 3=heavy)
            
        Returns:
            Summary vectors
        """
        # Ensure we have a summarizer
        if not hasattr(self.memory_manager, 'summarizer') or self.memory_manager.summarizer is None:
            logger.warning("Summarization not available")
            device = next(self.memory_manager.parameters()).device
            return torch.zeros(0, self.dim, device=device)
            
        # If chunk IDs provided, get tokens from those chunks
        if chunk_ids:
            # Collect all token IDs
            token_ids = []
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    token_ids.extend(self.chunks[chunk_id].token_ids)
                    
            # Retrieve token vectors
            if token_ids:
                token_vectors = self.retrieve_by_token_ids(token_ids)
                
                # Get positions for these tokens
                positions = []
                for token_id in token_ids:
                    if token_id in self.token_metadata:
                        positions.append(self.token_metadata[token_id].position)
                        
                # Create position tensor
                device = token_vectors.device
                position_tensor = torch.tensor(positions, device=device)
                
                # Generate summary
                summaries = self.memory_manager.summarizer.hierarchical_summarize(
                    token_vectors.unsqueeze(0),
                    position_tensor.unsqueeze(0),
                    max_level=summary_level
                )
                
                # Return the highest requested level if available
                if summary_level in summaries:
                    return summaries[summary_level][0]  # First batch, all vectors
                    
                # Otherwise return highest available level
                available_levels = sorted(summaries.keys())
                if available_levels:
                    return summaries[available_levels[-1]][0]
                    
        # If positions provided, retrieve tokens at those positions
        elif positions:
            # Convert to tensor
            device = next(self.memory_manager.parameters()).device
            positions_tensor = torch.tensor(positions, device=device).unsqueeze(0)  # Add batch dim
            
            # Retrieve token vectors
            token_vectors = self.retrieve_by_positions(positions_tensor)
            
            # Generate summary
            summaries = self.memory_manager.summarizer.hierarchical_summarize(
                token_vectors,
                positions_tensor,
                max_level=summary_level
            )
            
            # Return the highest requested level if available
            if summary_level in summaries:
                return summaries[summary_level][0]  # First batch, all vectors
                
            # Otherwise return highest available level
            available_levels = sorted(summaries.keys())
            if available_levels:
                return summaries[available_levels[-1]][0]
                
        # If position range provided, retrieve tokens in that range
        elif positions_range:
            start, end = positions_range
            
            # Generate position sequence
            positions = list(range(start, end + 1))
            
            # Convert to tensor
            device = next(self.memory_manager.parameters()).device
            positions_tensor = torch.tensor(positions, device=device).unsqueeze(0)  # Add batch dim
            
            # Retrieve token vectors
            token_vectors = self.retrieve_by_positions(positions_tensor)
            
            # Generate summary
            summaries = self.memory_manager.summarizer.hierarchical_summarize(
                token_vectors,
                positions_tensor,
                max_level=summary_level
            )
            
            # Return the highest requested level if available
            if summary_level in summaries:
                return summaries[summary_level][0]  # First batch, all vectors
                
            # Otherwise return highest available level
            available_levels = sorted(summaries.keys())
            if available_levels:
                return summaries[available_levels[-1]][0]
                
        # No valid input provided
        logger.warning("No valid input for summarization")
        device = next(self.memory_manager.parameters()).device
        return torch.zeros(0, self.dim, device=device)
        
    def add_chunk_tags(self, chunk_id: str, tags: List[str]) -> bool:
        """Add tags to a chunk"""
        if chunk_id not in self.chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
            
        # Add tags
        chunk = self.chunks[chunk_id]
        chunk.tags = list(set(chunk.tags + tags))
        return True
        
    def remove_chunk_tags(self, chunk_id: str, tags: List[str]) -> bool:
        """Remove tags from a chunk"""
        if chunk_id not in self.chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
            
        # Remove tags
        chunk = self.chunks[chunk_id]
        chunk.tags = [tag for tag in chunk.tags if tag not in tags]
        return True
        
    def update_chunk_properties(self, chunk_id: str, properties: Dict[str, Any]) -> bool:
        """Update properties of a chunk"""
        if chunk_id not in self.chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
            
        # Update properties
        chunk = self.chunks[chunk_id]
        chunk.properties.update(properties)
        return True
        
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk and its tokens from memory"""
        if chunk_id not in self.chunks:
            logger.warning(f"Chunk {chunk_id} not found")
            return False
            
        # Get the chunk
        chunk = self.chunks[chunk_id]
        token_ids = chunk.token_ids
        
        # Schedule token deletion as a background task
        self._schedule_maintenance(
            self._delete_tokens_internal,
            args=(token_ids,),
            priority=40
        )
        
        # Update mappings
        for token_id in token_ids:
            if token_id in self.tokens_to_chunks:
                del self.tokens_to_chunks[token_id]
            if token_id in self.token_metadata:
                del self.token_metadata[token_id]
                
        # Remove the chunk
        del self.chunks[chunk_id]
        
        logger.info(f"Deleted chunk {chunk_id} with {len(token_ids)} tokens")
        return True
        
    def _delete_tokens_internal(self, token_ids: List[str]):
        """Internal method to delete tokens from memory system"""
        # In a real implementation, we'd directly delete tokens from the memory manager
        # This is a simplified version that just logs the operation
        logger.info(f"Deleted {len(token_ids)} tokens from memory")
        
    def _promote_tokens(self, token_ids: List[str]):
        """Internal method to promote tokens to faster memory"""
        # In a real implementation, we'd promote tokens in the memory manager
        logger.debug(f"Promoted {len(token_ids)} tokens to faster memory")
        
    def _prune_old_tokens(self):
        """Internal method to prune old, low-importance tokens to free up memory"""
        # Find old chunks with low importance
        candidates = []
        
        for chunk_id, chunk in self.chunks.items():
            # Skip recent chunks (less than 1 hour old)
            if chunk.age < 3600:
                continue
                
            # Calculate average importance
            avg_importance = 0.0
            count = 0
            
            for token_id in chunk.token_ids:
                if token_id in self.token_metadata:
                    avg_importance += self.token_metadata[token_id].importance
                    count += 1
                    
            if count > 0:
                avg_importance /= count
                
            # Add to candidates if low importance
            if avg_importance < 0.3:
                candidates.append((chunk_id, avg_importance, chunk.age))
                
        # Sort by importance and age (least important, oldest first)
        candidates.sort(key=lambda x: (x[1], -x[2]))
        
        # Delete up to 10% of memory
        max_tokens_to_delete = self.max_tokens // 10
        tokens_deleted = 0
        
        for chunk_id, _, _ in candidates:
            if tokens_deleted >= max_tokens_to_delete:
                break
                
            if chunk_id in self.chunks:
                tokens_in_chunk = len(self.chunks[chunk_id].token_ids)
                if self.delete_chunk(chunk_id):
                    tokens_deleted += tokens_in_chunk
                    
        if tokens_deleted > 0:
            logger.info(f"Pruned {tokens_deleted} tokens from {len(candidates)} old chunks")
            
        return tokens_deleted
        
    def clear(self):
        """Clear all memory"""
        logger.info("Clearing memory system")
        
        # Clear memory manager
        self.memory_manager.clear()
        
        # Clear tracking structures
        self.chunks.clear()
        self.tokens_to_chunks.clear()
        self.token_metadata.clear()
        
        # Clear statistics
        self.stats.clear()
        
    def save_state(self, filename: str) -> bool:
        """Save memory state to disk"""
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Export memory manager state
            success = export_memory_state(self.memory_manager, filename)
            
            if success:
                # Save additional metadata
                meta_filename = filename + ".meta"
                with open(meta_filename, 'w') as f:
                    json.dump({
                        "chunks": {k: v.to_dict() for k, v in self.chunks.items()},
                        "tokens_to_chunks": self.tokens_to_chunks,
                        "token_metadata": {k: v.__dict__ for k, v in self.token_metadata.items()},
                        "stats": self.stats.get_stats(),
                        "timestamp": time.time(),
                        "version": "1.0"
                    }, f)
                    
                logger.info(f"Memory state saved to {filename}")
                return True
            else:
                logger.error(f"Failed to save memory state to {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving memory state to {filename}: {e}")
            return False
            
    def load_state(self, filename: str) -> bool:
        """Load memory state from disk"""
        try:
            # Load memory manager state
            success = restore_memory_state(self.memory_manager, filename)
            
            if success:
                # Load additional metadata
                meta_filename = filename + ".meta"
                with open(meta_filename, 'r') as f:
                    meta = json.load(f)
                    
                # Restore chunks
                self.chunks = {k: MemoryChunk.from_dict(v) for k, v in meta["chunks"].items()}
                
                # Restore token mappings
                self.tokens_to_chunks = meta["tokens_to_chunks"]
                
                # Restore token metadata
                self.token_metadata = {}
                for k, v in meta["token_metadata"].items():
                    self.token_metadata[k] = TokenMetadata(**v)
                    
                logger.info(f"Memory state loaded from {filename}")
                return True
            else:
                logger.error(f"Failed to load memory state from {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading memory state from {filename}: {e}")
            return False
            
    def get_stats(self) -> Dict:
        """Get complete memory system statistics"""
        # Get stats from memory manager
        manager_stats = self.memory_manager.get_stats()
        
        # Get API stats
        api_stats = {
            "total_chunks": len(self.chunks),
            "total_metadata_entries": len(self.token_metadata),
            "token_to_chunk_mappings": len(self.tokens_to_chunks),
        }
        
        # Add access statistics
        api_stats.update(self.stats.get_stats())
        
        # Combine all statistics
        combined_stats = {**manager_stats, **api_stats}
        
        return combined_stats
        
    def run_benchmarks(self) -> Dict:
        """Run benchmarks to evaluate memory system performance"""
        logger.info("Running memory system benchmarks")
        
        # Clear system before benchmarking
        self.clear()
        
        # Run benchmarks
        benchmark_results = MemorySystemBenchmark.run_all_benchmarks(self.memory_manager, dim=self.dim)
        
        # Return results
        return benchmark_results
        
    def optimize(self):
        """Optimize memory system for better performance"""
        logger.info("Optimizing memory system")
        
        # Optimize memory manager
        self.memory_manager.optimize()
        
        # Run cleanup tasks
        self._prune_old_tokens()
        
        # Record optimization time
        self.stats.last_optimization_time = time.time()
        
    def __del__(self):
        """Cleanup when object is deleted"""
        if self.auto_save_enabled:
            # Try to save state on exit
            try:
                timestamp = int(time.time())
                filename = f"{self.persistence_path}/memory_state_{timestamp}_final.bin"
                self.save_state(filename)
            except:
                pass
                
        # Stop maintenance thread
        self._stop_maintenance_thread()


# Example usage of the API
def example_usage():
    """Example of how to use the Memory API"""
    # Create the API
    api = MemoryAPI(dim=768, max_tokens=1_000_000)
    
    # Create some token vectors (normally these would come from a model)
    token_vectors = torch.randn(1, 100, 768)  # [batch_size, seq_len, dim]
    positions = torch.arange(100).unsqueeze(0)  # [batch_size, seq_len]
    
    # Store tokens
    token_ids, chunk_id = api.store_tokens(
        token_vectors=token_vectors,
        positions=positions,
        source="example",
        tags=["demo", "example"],
        properties={"purpose": "demonstration"}
    )
    
    print(f"Stored {len(token_ids)} tokens in chunk {chunk_id}")
    
    # Retrieve tokens by position
    retrieved = api.retrieve_by_positions(
        positions=torch.tensor([[5, 10, 15, 20]]),
        include_metadata=True
    )
    
    print(f"Retrieved tokens with shape {retrieved[0].shape}")
    
    # Perform semantic search
    query = torch.randn(1, 768)  # [batch_size, dim]
    results = api.semantic_search(
        query_vectors=query,
        top_k=5
    )
    
    print(f"Semantic search returned shape {results.shape}")
    
    # Get summary
    summary = api.retrieve_summary(
        chunk_ids=[chunk_id],
        summary_level=2
    )
    
    print(f"Summary has shape {summary.shape}")
    
    # Get statistics
    stats = api.get_stats()
    print(f"Memory system statistics: {stats}")
    
    # Clean up
    api.clear()


if __name__ == "__main__":
    example_usage()
