import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
from typing import Optional, List, Dict, Tuple, Any, Union, Callable
import math
import logging
from dataclasses import dataclass, field
import time
from functools import partial
import gc
import heapq
import numpy as np
from collections import defaultdict, Counter

# Assuming UltraContext Core is imported
from ultracontext.core import (
    DEFAULT_PERF_CONFIG,
    PerformanceConfig,
    get_norm_class,
    ActivationFunctions,
    timer,
    HierarchicalAttention,
    UltraMemoryMLP,
    AdvancedResidualBlock
)

# Assuming Memory System is imported
from ultracontext.memory import (
    HierarchicalMemoryManager,
    MemoryCompressor,
    MemoryAccessPattern
)

logger = logging.getLogger("ultracontext.processing")

# Automatic Contextual Compression System
class ContextualCompressor(Module):
    """
    Contextually-aware token compression for ultra-long contexts
    
    Features:
    - Content-based token importance estimation
    - Dynamic compression rates based on token importance
    - Multiple compression strategies (pruning, merging, summarizing)
    - Preserves crucial information while reducing context size
    """
    def __init__(
        self,
        dim: int,
        target_compression_ratio: float = 4.0,
        min_tokens_before_compression: int = 1024,
        strategies: List[str] = ["prune", "merge", "summarize"],
        use_global_importance: bool = True,
        importance_threshold: float = 0.3,
        token_budget: Optional[int] = None,
        adaptive_compression: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.target_compression_ratio = target_compression_ratio
        self.min_tokens_before_compression = min_tokens_before_compression
        self.strategies = strategies
        self.use_global_importance = use_global_importance
        self.importance_threshold = importance_threshold
        self.token_budget = token_budget
        self.adaptive_compression = adaptive_compression
        
        # Token importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Content classifier to determine best compression strategy
        self.content_classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, len(strategies))
        )
        
        # Token merging network (for combining similar tokens)
        self.token_merger = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Token summarization network (for creating summary tokens)
        if "summarize" in strategies:
            self.summarizer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        
        # Memory for tracking compressed token statistics
        self.register_buffer('compression_stats', torch.zeros(3))  # [pruned, merged, summarized]
        
    def _score_token_importance(self, x):
        """
        Score each token's importance
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            
        Returns:
            Importance scores [batch_size, seq_len, 1]
        """
        # Get raw importance scores
        importance = self.importance_scorer(x)
        
        # Optionally apply global normalization
        if self.use_global_importance:
            # Normalize across the entire sequence
            batch_size = x.size(0)
            for b in range(batch_size):
                # Min-max normalize within each sequence
                min_val = importance[b].min()
                max_val = importance[b].max()
                if max_val > min_val:  # Avoid division by zero
                    importance[b] = (importance[b] - min_val) / (max_val - min_val)
        
        return importance
        
    def _classify_content(self, x):
        """
        Classify content to determine best compression strategy
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            
        Returns:
            Strategy classification logits [batch_size, seq_len, num_strategies]
        """
        return self.content_classifier(x)
        
    def _prune_tokens(self, x, importance, target_ratio):
        """
        Prune less important tokens
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            importance: Token importance scores [batch_size, seq_len, 1]
            target_ratio: Target ratio of tokens to keep
            
        Returns:
            Pruned embeddings and indices of kept tokens
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Determine number of tokens to keep
        num_keep = max(1, int(seq_len * target_ratio))
        
        # Prepare results
        all_pruned_embeddings = []
        all_kept_indices = []
        
        for b in range(batch_size):
            # Get importance for this sequence
            seq_importance = importance[b, :, 0]
            
            # Select top-k indices by importance
            _, indices = torch.topk(seq_importance, num_keep, dim=0)
            indices, _ = torch.sort(indices)  # Sort to maintain sequence order
            
            # Gather the selected embeddings
            pruned_embeddings = x[b:b+1, indices, :]
            
            all_pruned_embeddings.append(pruned_embeddings)
            all_kept_indices.append(indices)
            
        # Stack results
        pruned_embeddings = torch.cat(all_pruned_embeddings, dim=0)
        kept_indices = torch.stack(all_kept_indices)
        
        # Update stats
        self.compression_stats[0] += seq_len - num_keep
        
        return pruned_embeddings, kept_indices
        
    def _merge_similar_tokens(self, x, importance, target_ratio):
        """
        Merge similar tokens based on embedding similarity
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            importance: Token importance scores [batch_size, seq_len, 1]
            target_ratio: Target ratio of tokens to keep
            
        Returns:
            Merged embeddings and merge mapping
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Determine target number of tokens after merging
        target_len = max(1, int(seq_len * target_ratio))
        num_to_merge = seq_len - target_len
        
        if num_to_merge <= 0:
            # No merging needed
            return x, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare results
        all_merged_embeddings = []
        all_merge_mappings = []
        
        for b in range(batch_size):
            seq_x = x[b]
            seq_importance = importance[b, :, 0]
            
            # Compute pairwise similarities
            normalized_x = F.normalize(seq_x, p=2, dim=1)
            similarities = torch.matmul(normalized_x, normalized_x.transpose(0, 1))
            
            # Set diagonal to -1 to avoid self-matches
            similarities.fill_diagonal_(-1)
            
            # Find pairs to merge (most similar pairs with lowest importance)
            pairs_to_merge = []
            merged_indices = set()
            
            # Create importance-weighted similarity
            combined_score = similarities.clone()
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        # Weight by inverse importance (higher importance = less likely to merge)
                        imp_i = max(0.1, seq_importance[i].item())  # Avoid division by zero
                        imp_j = max(0.1, seq_importance[j].item())
                        
                        # Discount similarity for important tokens
                        combined_score[i, j] = similarities[i, j] / (imp_i * imp_j)
            
            # Find top pairs to merge
            while len(pairs_to_merge) < num_to_merge and combined_score.max() > 0:
                # Find most similar pair
                flat_idx = combined_score.argmax().item()
                i = flat_idx // seq_len
                j = flat_idx % seq_len
                
                if i not in merged_indices and j not in merged_indices:
                    pairs_to_merge.append((i, j))
                    merged_indices.add(i)
                    merged_indices.add(j)
                
                # Mark this pair as processed
                combined_score[i, j] = -1
            
            # Create new sequence with merged tokens
            merge_mapping = torch.arange(seq_len, device=device)
            merged_embeddings = []
            next_idx = 0
            
            # Process pairs to merge
            for i, j in pairs_to_merge:
                # Merge the pair using the token merger
                pair_input = torch.cat([seq_x[i], seq_x[j]]).unsqueeze(0)
                merged_token = self.token_merger(pair_input).squeeze(0)
                merged_embeddings.append(merged_token)
                
                # Update merge mapping
                merge_mapping[i] = seq_len + next_idx
                merge_mapping[j] = seq_len + next_idx
                next_idx += 1
            
            # Add unmerged tokens
            unmerged_indices = list(set(range(seq_len)) - merged_indices)
            for i in unmerged_indices:
                merged_embeddings.append(seq_x[i])
                merge_mapping[i] = seq_len + next_idx
                next_idx += 1
            
            # Stack merged embeddings
            merged_embeddings = torch.stack(merged_embeddings)
            
            all_merged_embeddings.append(merged_embeddings.unsqueeze(0))
            all_merge_mappings.append(merge_mapping.unsqueeze(0))
            
        # Stack results
        merged_embeddings = torch.cat(all_merged_embeddings, dim=0)
        merge_mappings = torch.cat(all_merge_mappings, dim=0)
        
        # Update stats
        self.compression_stats[1] += num_to_merge
        
        return merged_embeddings, merge_mappings
        
    def _summarize_regions(self, x, importance, target_ratio):
        """
        Create summary tokens for regions of the sequence
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            importance: Token importance scores [batch_size, seq_len, 1]
            target_ratio: Target ratio of tokens to keep
            
        Returns:
            Compressed sequence with summary tokens
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Determine target length after summarization
        target_len = max(1, int(seq_len * target_ratio))
        
        # If already short enough, no need to summarize
        if seq_len <= target_len:
            return x, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Determine which tokens to keep and which to summarize
        all_summarized = []
        all_mappings = []
        
        for b in range(batch_size):
            seq_x = x[b]
            seq_importance = importance[b, :, 0]
            
            # Sort tokens by importance
            _, indices = torch.sort(seq_importance, descending=True)
            
            # Keep the most important tokens directly
            num_keep_direct = target_len // 2
            keep_indices = indices[:num_keep_direct]
            summarize_indices = indices[num_keep_direct:]
            
            # Sort indices to maintain sequence order
            keep_indices = keep_indices.sort()[0]
            
            # Create chunks for summarization
            remaining_slots = target_len - num_keep_direct
            if len(summarize_indices) <= remaining_slots:
                # No need to create summaries, just keep more tokens
                keep_indices = indices[:target_len]
                keep_indices = keep_indices.sort()[0]
                summarized = seq_x[keep_indices].unsqueeze(0)
                mapping = torch.ones(seq_len, dtype=torch.long, device=device) * -1
                mapping[keep_indices] = torch.arange(len(keep_indices), device=device)
                
                all_summarized.append(summarized)
                all_mappings.append(mapping.unsqueeze(0))
                continue
            
            # Determine chunk size for summarization
            chunk_size = max(2, len(summarize_indices) // remaining_slots)
            
            # Create chunks
            chunks = []
            chunk_mappings = []
            
            for i in range(0, len(summarize_indices), chunk_size):
                chunk_indices = summarize_indices[i:i+chunk_size]
                if len(chunk_indices) > 0:
                    chunks.append(seq_x[chunk_indices])
                    chunk_mappings.append(chunk_indices)
            
            # Summarize each chunk
            summary_tokens = []
            
            for chunk in chunks[:remaining_slots]:  # Limit to remaining slots
                # Apply summarizer
                chunk_expanded = chunk.unsqueeze(0)  # Add batch dim
                summarized_chunk = self.summarizer(chunk_expanded)
                
                # Average pooling to create a single token
                summary_token = summarized_chunk.mean(dim=1)
                summary_tokens.append(summary_token)
            
            # Combine kept tokens and summary tokens
            kept_tokens = seq_x[keep_indices].unsqueeze(0)
            summary_tokens = torch.cat(summary_tokens, dim=0).unsqueeze(0)
            
            summarized = torch.cat([
                kept_tokens, 
                summary_tokens[:, :remaining_slots]
            ], dim=1)
            
            # Create mapping from original to compressed indices
            mapping = torch.ones(seq_len, dtype=torch.long, device=device) * -1
            mapping[keep_indices] = torch.arange(len(keep_indices), device=device)
            
            # Map summarized regions to summary token indices
            for i, chunk_indices in enumerate(chunk_mappings[:remaining_slots]):
                summary_idx = num_keep_direct + i
                mapping[chunk_indices] = summary_idx
            
            all_summarized.append(summarized)
            all_mappings.append(mapping.unsqueeze(0))
            
        # Stack results
        summarized = torch.cat(all_summarized, dim=0)
        mappings = torch.cat(all_mappings, dim=0)
        
        # Update stats
        self.compression_stats[2] += seq_len - target_len
        
        return summarized, mappings
        
    def _adaptive_compression(self, x, importance):
        """
        Apply different compression strategies based on content properties
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            importance: Token importance scores [batch_size, seq_len, 1]
            
        Returns:
            Compressed sequence and mapping information
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Classify content to determine best strategy for each token
        strategy_logits = self._classify_content(x)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        
        # Get best strategy for each token
        best_strategy = strategy_probs.argmax(dim=-1)
        
        # Organize tokens by strategy
        results = {}
        token_maps = {}
        counts = {}
        
        for b in range(batch_size):
            token_groups = defaultdict(list)
            token_indices = defaultdict(list)
            
            for i in range(seq_len):
                strategy_idx = best_strategy[b, i].item()
                if strategy_idx < len(self.strategies):
                    strategy = self.strategies[strategy_idx]
                    token_groups[strategy].append(x[b, i])
                    token_indices[strategy].append(i)
            
            results[b] = token_groups
            token_maps[b] = token_indices
            counts[b] = {strategy: len(indices) for strategy, indices in token_indices.items()}
        
        # Determine compression ratios for each strategy
        target_len = max(1, int(seq_len / self.target_compression_ratio))
        
        # Apply compression by strategy
        all_compressed = []
        all_mappings = []
        
        for b in range(batch_size):
            compressed_parts = []
            orig_to_compressed = torch.ones(seq_len, dtype=torch.long, device=device) * -1
            next_idx = 0
            
            for strategy_idx, strategy in enumerate(self.strategies):
                if strategy not in results[b]:
                    continue
                    
                tokens = results[b][strategy]
                indices = token_maps[b][strategy]
                
                if not tokens:
                    continue
                    
                # Skip compression for very small groups
                if len(tokens) <= 3:
                    group_x = torch.stack(tokens).unsqueeze(0)
                    for i, orig_idx in enumerate(indices):
                        compressed_parts.append(group_x[0, i])
                        orig_to_compressed[orig_idx] = next_idx + i
                    next_idx += len(tokens)
                    continue
                
                # Create embedding tensor for this group
                group_x = torch.stack(tokens).unsqueeze(0)
                group_importance = importance[b:b+1, indices, :]
                
                # Determine appropriate ratio for this strategy
                total_to_keep = target_len - next_idx
                if total_to_keep <= 0:
                    # Already reached target length
                    break
                    
                if strategy == "prune":
                    # More aggressive for pruning
                    local_ratio = min(0.5, total_to_keep / len(tokens))
                    pruned_x, kept_indices = self._prune_tokens(
                        group_x, group_importance, local_ratio
                    )
                    
                    # Update mapping
                    for i, local_idx in enumerate(kept_indices[0]):
                        orig_idx = indices[local_idx]
                        compressed_parts.append(pruned_x[0, i])
                        orig_to_compressed[orig_idx] = next_idx + i
                        
                    next_idx += pruned_x.size(1)
                    
                elif strategy == "merge":
                    # Less aggressive for merging
                    local_ratio = min(0.7, total_to_keep / len(tokens))
                    merged_x, merge_mapping = self._merge_similar_tokens(
                        group_x, group_importance, local_ratio
                    )
                    
                    # Update tokens
                    for i in range(merged_x.size(1)):
                        compressed_parts.append(merged_x[0, i])
                    
                    # Update mapping (more complex for merging)
                    for i, local_idx in enumerate(indices):
                        merged_idx = merge_mapping[0, i].item()
                        if merged_idx >= group_x.size(1):
                            # This is a merged token
                            new_idx = next_idx + (merged_idx - group_x.size(1))
                            orig_to_compressed[local_idx] = new_idx
                            
                    next_idx += merged_x.size(1)
                    
                elif strategy == "summarize":
                    # Medium aggression for summarization
                    local_ratio = min(0.6, total_to_keep / len(tokens))
                    summarized_x, summary_mapping = self._summarize_regions(
                        group_x, group_importance, local_ratio
                    )
                    
                    # Update tokens
                    for i in range(summarized_x.size(1)):
                        compressed_parts.append(summarized_x[0, i])
                    
                    # Update mapping
                    for i, orig_idx in enumerate(indices):
                        summary_idx = summary_mapping[0, i].item()
                        if summary_idx >= 0:
                            orig_to_compressed[orig_idx] = next_idx + summary_idx
                            
                    next_idx += summarized_x.size(1)
            
            # Stack all compressed parts
            if compressed_parts:
                compressed_sequence = torch.stack(compressed_parts).unsqueeze(0)
                all_compressed.append(compressed_sequence)
                all_mappings.append(orig_to_compressed.unsqueeze(0))
            else:
                # Fallback: keep first tokens up to target length
                keep_len = min(seq_len, target_len)
                compressed_sequence = x[b:b+1, :keep_len]
                all_compressed.append(compressed_sequence)
                
                mapping = torch.ones(seq_len, dtype=torch.long, device=device) * -1
                mapping[:keep_len] = torch.arange(keep_len, device=device)
                all_mappings.append(mapping.unsqueeze(0))
        
        # Pad sequences to same length
        max_len = max(seq.size(1) for seq in all_compressed)
        padded_compressed = []
        
        for seq in all_compressed:
            pad_len = max_len - seq.size(1)
            if pad_len > 0:
                padding = torch.zeros(1, pad_len, self.dim, device=device)
                padded_compressed.append(torch.cat([seq, padding], dim=1))
            else:
                padded_compressed.append(seq)
                
        compressed_result = torch.cat(padded_compressed, dim=0)
        mapping_result = torch.cat(all_mappings, dim=0)
        
        return compressed_result, mapping_result
        
    def forward(self, x, compress=True):
        """
        Apply contextual compression to the input sequence
        
        Args:
            x: Token embeddings [batch_size, seq_len, dim]
            compress: Whether to compress or not
            
        Returns:
            Compressed sequence and mapping information
        """
        batch_size, seq_len, _ = x.shape
        
        # Skip compression if sequence is too short
        if not compress or seq_len <= self.min_tokens_before_compression:
            # Return identity mapping
            mapping = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            return x, mapping
            
        # Calculate target length based on budget or ratio
        if self.token_budget is not None:
            target_len = min(seq_len, self.token_budget)
            actual_ratio = seq_len / max(1, target_len)
        else:
            actual_ratio = self.target_compression_ratio
            
        # Score token importance
        importance = self._score_token_importance(x)
        
        # Apply compression based on the strategy
        if self.adaptive_compression:
            # Use adaptive multi-strategy compression
            return self._adaptive_compression(x, importance)
        elif "prune" in self.strategies:
            # Use token pruning
            return self._prune_tokens(x, importance, 1.0 / actual_ratio)
        elif "merge" in self.strategies:
            # Use token merging
            return self._merge_similar_tokens(x, importance, 1.0 / actual_ratio)
        elif "summarize" in self.strategies:
            # Use token summarization
            return self._summarize_regions(x, importance, 1.0 / actual_ratio)
        else:
            # Fallback: simple pruning
            return self._prune_tokens(x, importance, 1.0 / actual_ratio)
    
    def get_stats(self):
        """Get statistics about compression operations"""
        total = self.compression_stats.sum().item()
        if total == 0:
            return {"pruned": 0, "merged": 0, "summarized": 0}
            
        return {
            "pruned": self.compression_stats[0].item() / total,
            "merged": self.compression_stats[1].item() / total,
            "summarized": self.compression_stats[2].item() / total
        }

# Retrieval-Augmented Context Processing
class RetrievalAugmentedProcessor(Module):
    """
    Enhances context with retrieval from external knowledge
    
    Features:
    - Query generation from current context
    - Retrieval from large token stores
    - Integration of retrieved information
    - Attention-based weighting of retrieved content
    """
    def __init__(
        self,
        dim: int,
        memory_manager = None,  # HierarchicalMemoryManager instance
        num_retrievers: int = 3,  # Multiple retrievers for diversity
        max_retrieved_tokens: int = 256,
        retrieval_threshold: float = 0.7,
        query_generator_layers: int = 2,
        fusion_type: str = "attention",  # "attention", "concat", "gating"
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.memory_manager = memory_manager
        self.max_retrieved_tokens = max_retrieved_tokens
        self.retrieval_threshold = retrieval_threshold
        self.fusion_type = fusion_type
        
        # Query generators (multiple for diversity)
        self.query_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            for _ in range(num_retrievers)
        ])
        
        # Query poolers to create query vectors
        self.query_poolers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )
            for _ in range(num_retrievers)
        ])
        
        # Query transformation network
        query_generator = []
        for _ in range(query_generator_layers):
            query_generator.append(nn.Linear(dim, dim))
            query_generator.append(nn.GELU())
        self.query_transform = nn.Sequential(*query_generator)
        
        # Fusion mechanism
        if fusion_type == "attention":
            # Attention-based fusion
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(dim)
            
        elif fusion_type == "gating":
            # Gated fusion
            self.fusion_gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
            self.fusion_transform = nn.Linear(dim * 2, dim)
            
        # Content relevance scorer
        self.relevance_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and hasattr(torch, "compile"):
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
            
    def _generate_queries(self, x):
        """
        Generate queries from the current context
        
        Args:
            x: Context embeddings [batch_size, seq_len, dim]
            
        Returns:
            Query vectors for retrieval [num_retrievers, batch_size, dim]
        """
        batch_size = x.shape[0]
        queries = []
        
        for generator, pooler in zip(self.query_generators, self.query_poolers):
            # Generate query features
            query_features = generator(x)
            
            # Pool to create a single query vector per sequence
            # Using attention-weighted pooling
            attention_weights = torch.softmax(
                torch.matmul(query_features, query_features.transpose(-2, -1)) / math.sqrt(self.dim),
                dim=-1
            )
            pooled = torch.matmul(attention_weights, query_features).mean(dim=1)
            
            # Transform pooled vector
            query = pooler(pooled)
            queries.append(query)
            
        return torch.stack(queries)
        
    def _retrieve_relevant_tokens(self, queries):
        """
        Retrieve relevant tokens from memory
        
        Args:
            queries: Query vectors [num_retrievers, batch_size, dim]
            
        Returns:
            Retrieved tokens and relevance scores
        """
        if self.memory_manager is None:
            # No memory manager, return empty results
            return None, None
            
        # Flatten batch dimension for retrieval
        num_retrievers, batch_size, _ = queries.shape
        flat_queries = queries.view(num_retrievers * batch_size, -1)
        
        # Retrieve from memory
        retrieved_tokens = self.memory_manager.retrieve_tokens(query_vectors=flat_queries)
        
        # Process retrieved tokens
        all_tokens = []
        all_relevance = []
        
        for i, tokens in enumerate(retrieved_tokens):
            # Skip if no tokens retrieved
            if tokens.size(0) == 0:
                continue
                
            # Calculate query index
            retriever_idx = i // batch_size
            batch_idx = i % batch_size
            
            # Limit number of tokens
            num_tokens = min(tokens.size(0), self.max_retrieved_tokens)
            tokens = tokens[:num_tokens]
            
            # Calculate relevance score
            query = queries[retriever_idx, batch_idx].unsqueeze(0).expand(num_tokens, -1)
            token_query_pairs = torch.cat([tokens, query], dim=1)
            relevance = self.relevance_scorer(token_query_pairs).squeeze(-1)
            
            # Filter by relevance threshold
            mask = relevance >= self.retrieval_threshold
            if mask.sum() > 0:
                filtered_tokens = tokens[mask]
                filtered_relevance = relevance[mask]
                
                all_tokens.append((batch_idx, filtered_tokens))
                all_relevance.append((batch_idx, filtered_relevance))
                
        return all_tokens, all_relevance
        
    def _fuse_retrieved_content(self, x, retrieved_tokens, relevance_scores):
        """
        Fuse retrieved content with original context
        
        Args:
            x: Original context [batch_size, seq_len, dim]
            retrieved_tokens: List of (batch_idx, tokens) pairs
            relevance_scores: List of (batch_idx, scores) pairs
            
        Returns:
            Enhanced context with retrieved information
        """
        if not retrieved_tokens:
            return x
            
        batch_size, seq_len, _ = x.shape
        enhanced_x = x.clone()
        
        # Organize retrieved tokens by batch
        batch_retrieved = [[] for _ in range(batch_size)]
        batch_relevance = [[] for _ in range(batch_size)]
        
        for batch_idx, tokens in retrieved_tokens:
            batch_retrieved[batch_idx].append(tokens)
            
        for batch_idx, scores in relevance_scores:
            batch_relevance[batch_idx].append(scores)
            
        # Apply fusion for each batch
        for b in range(batch_size):
            if not batch_retrieved[b]:
                continue
                
            # Concatenate all retrieved tokens for this batch
            retrieved = torch.cat(batch_retrieved[b], dim=0)
            relevance = torch.cat(batch_relevance[b], dim=0)
            
            # Sort by relevance and limit
            num_to_use = min(retrieved.size(0), self.max_retrieved_tokens)
            if num_to_use < retrieved.size(0):
                _, indices = torch.topk(relevance, num_to_use)
                retrieved = retrieved[indices]
                relevance = relevance[indices]
            
            if self.fusion_type == "attention":
                # Attention-based fusion
                orig = enhanced_x[b].unsqueeze(0)  # [1, seq_len, dim]
                ret = retrieved.unsqueeze(0)  # [1, num_retrieved, dim]
                
                # Use retrieved as keys/values, original as queries
                fused, _ = self.fusion_attention(orig, ret, ret)
                
                # Residual connection
                enhanced_x[b] = self.fusion_norm(orig.squeeze(0) + fused.squeeze(0))
                
            elif self.fusion_type == "concat":
                # Simple concatenation (limited by max sequence length)
                max_retrieved = min(self.max_retrieved_tokens, seq_len // 2)
                if retrieved.size(0) > max_retrieved:
                    retrieved = retrieved[:max_retrieved]
                    
                # Replace second half of sequence with retrieved tokens
                split_point = seq_len - retrieved.size(0)
                enhanced_x[b, split_point:split_point+retrieved.size(0)] = retrieved
                
            elif self.fusion_type == "gating":
                # Gated fusion at each position
                orig = enhanced_x[b]  # [seq_len, dim]
                
                # Create retrieved context representation via attention
                retrieved_context = retrieved.mean(dim=0)  # [dim]
                
                # Expand to match sequence length
                retrieved_context = retrieved_context.unsqueeze(0).expand(seq_len, -1)  # [seq_len, dim]
                
                # Concatenate original and retrieved
                concat = torch.cat([orig, retrieved_context], dim=1)  # [seq_len, dim*2]
                
                # Calculate gates
                gates = self.fusion_gate(concat)
                
                # Apply gated transformation
                transformed = self.fusion_transform(concat)
                
                # Combine with original using gates
                enhanced_x[b] = orig * (1 - gates) + transformed * gates
                
        return enhanced_x
        
    def forward(self, x, use_retrieval=True):
        """
        Process context with retrieval augmentation
        
        Args:
            x: Context embeddings [batch_size, seq_len, dim]
            use_retrieval: Whether to use retrieval augmentation
            
        Returns:
            Enhanced context embeddings
        """
        if not use_retrieval or self.memory_manager is None:
            return x
            
        # Generate queries
        queries = self._generate_queries(x)
        
        # Retrieve relevant tokens
        retrieved_tokens, relevance_scores = self._retrieve_relevant_tokens(queries)
        
        # Fuse retrieved content
        enhanced_x = self._fuse_retrieved_content(x, retrieved_tokens, relevance_scores)
        
        return enhanced_x

# Ultradense Segment Tree for O(log n) access to any subsegment of a sequence
class SegmentTree:
    """
    Efficient data structure for retrieving summaries of any subsequence
    
    Features:
    - O(log n) access to any subsegment of the sequence
    - Multiple reduction operations (sum, max, mean, etc.)
    - Lazy propagation for efficient updates
    """
    def __init__(self, capacity, dim, reduction="sum"):
        """
        Initialize a segment tree to efficiently query ranges
        
        Args:
            capacity: Maximum sequence length
            dim: Embedding dimension
            reduction: Operation to use ("sum", "max", "mean", etc.)
        """
        # Round up capacity to next power of 2
        self.base_capacity = capacity
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
            
        self.dim = dim
        self.reduction = reduction
        
        # Initialize tree storage
        self.tree = [torch.zeros(dim) for _ in range(2 * self.capacity)]
        
        # Set reduction operation
        if reduction == "sum":
            self.reduce_fn = lambda a, b: a + b
        elif reduction == "max":
            self.reduce_fn = lambda a, b: torch.max(a, b)
        elif reduction == "mean":
            # Special handling for mean
            self.counts = [0 for _ in range(2 * self.capacity)]
            self.reduce_fn = self._mean_reduce
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
            
    def _mean_reduce(self, a, b, a_idx, b_idx):
        """Special reducer for mean operation"""
        a_count = self.counts[a_idx]
        b_count = self.counts[b_idx]
        total_count = a_count + b_count
        
        if total_count == 0:
            return torch.zeros_like(a)
            
        # Weighted average
        return (a * a_count + b * b_count) / total_count
        
    def update(self, idx, value):
        """
        Update a single position in the tree
        
        Args:
            idx: Position to update (0-indexed)
            value: New value
        """
        if idx < 0 or idx >= self.base_capacity:
            raise ValueError(f"Index {idx} out of bounds")
            
        # Update starting at leaf node
        pos = idx + self.capacity
        self.tree[pos] = value
        
        if self.reduction == "mean":
            self.counts[pos] = 1
            
        # Update parent nodes
        while pos > 1:
            parent = pos // 2
            left = parent * 2
            right = parent * 2 + 1
            
            if self.reduction == "mean":
                self.tree[parent] = self._mean_reduce(
                    self.tree[left], self.tree[right], left, right
                )
                self.counts[parent] = self.counts[left] + self.counts[right]
            else:
                self.tree[parent] = self.reduce_fn(self.tree[left], self.tree[right])
                
            pos = parent
            
    def query(self, start, end):
        """
        Query a range in the tree
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            
        Returns:
            Reduced value for the range [start, end)
        """
        if start < 0 or end > self.base_capacity or start >= end:
            raise ValueError(f"Invalid range: [{start}, {end})")
            
        # Convert to 0-indexed
        start += self.capacity
        end += self.capacity
        
        result = None
        count = 0
        
        while start < end:
            if start % 2 == 1:
                # If start is odd (right child), include it
                if result is None:
                    result = self.tree[start]
                    if self.reduction == "mean":
                        count = self.counts[start]
                else:
                    if self.reduction == "mean":
                        result = (result * count + self.tree[start] * self.counts[start]) / (count + self.counts[start])
                        count += self.counts[start]
                    else:
                        result = self.reduce_fn(result, self.tree[start])
                start += 1
                
            if end % 2 == 1:
                # If end is odd, include the element before it
                end -= 1
                if result is None:
                    result = self.tree[end]
                    if self.reduction == "mean":
                        count = self.counts[end]
                else:
                    if self.reduction == "mean":
                        result = (result * count + self.tree[end] * self.counts[end]) / (count + self.counts[end])
                        count += self.counts[end]
                    else:
                        result = self.reduce_fn(result, self.tree[end])
                        
            # Move to parent nodes
            start //= 2
            end //= 2
            
        if result is None:
            # Empty range
            return torch.zeros(self.dim)
            
        return result
        
    def build_from_sequence(self, sequence):
        """
        Build the segment tree from a complete sequence
        
        Args:
            sequence: Tensor of shape [seq_len, dim]
        """
        seq_len, dim = sequence.shape
        
        if seq_len > self.base_capacity:
            raise ValueError(f"Sequence length {seq_len} exceeds capacity {self.base_capacity}")
            
        # Fill leaf nodes
        for i in range(seq_len):
            self.tree[i + self.capacity] = sequence[i]
            if self.reduction == "mean":
                self.counts[i + self.capacity] = 1
                
        # Fill remaining leaf nodes with zeros
        for i in range(seq_len, self.capacity):
            self.tree[i + self.capacity] = torch.zeros(dim, device=sequence.device)
            if self.reduction == "mean":
                self.counts[i + self.capacity] = 0
                
        # Build internal nodes
        for i in range(self.capacity - 1, 0, -1):
            left = i * 2
            right = i * 2 + 1
            
            if self.reduction == "mean":
                self.tree[i] = self._mean_reduce(
                    self.tree[left], self.tree[right], left, right
                )
                self.counts[i] = self.counts[left] + self.counts[right]
            else:
                self.tree[i] = self.reduce_fn(self.tree[left], self.tree[right])

# Hierarchical Sequence Processor for extremely long contexts
class HierarchicalProcessingModule(Module):
    """
    Process extremely long sequences by splitting into hierarchical chunks
    
    Features:
    - Multi-level processing for different context ranges
    - Local and global attention mechanisms
    - Efficient cross-chunk information flow
    - Token routing between levels
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        num_levels: int = 3,
        chunk_sizes: List[int] = [128, 512, 2048],
        max_seq_len: int = 100_000,
        summary_ratio: float = 0.1,
        use_retrieval: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        self.chunk_sizes = chunk_sizes
        self.max_seq_len = max_seq_len
        self.summary_ratio = summary_ratio
        self.use_retrieval = use_retrieval
        
        # Ensure we have enough levels
        if len(chunk_sizes) < num_levels:
            chunk_sizes.extend([chunk_sizes[-1] * 2] * (num_levels - len(chunk_sizes)))
        
        # Level-specific processors
        self.level_processors = nn.ModuleList()
        
        for level in range(num_levels):
            # Create level-specific processing block
            inner_dim = num_heads * head_dim
            
            # Create attention module
            attn = HierarchicalAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=chunk_sizes[level],
                global_tokens=max(16, int(chunk_sizes[level] * 0.05)),
                causal=True,
                perf_config=perf_config
            )
            
            # Create MLP
            mlp = UltraMemoryMLP(
                dim=dim,
                expansion_factor=4.0,
                activation=perf_config.activation_function,
                dropout=0.1,
                factorized=True,
                perf_config=perf_config
            )
            
            # Create residual blocks
            attn_block = AdvancedResidualBlock(
                dim=dim,
                layer=attn,
                pre_norm=True,
                dropout=0.1,
                adaptive_scaling=True,
                perf_config=perf_config
            )
            
            mlp_block = AdvancedResidualBlock(
                dim=dim,
                layer=mlp,
                pre_norm=True,
                dropout=0.1,
                adaptive_scaling=True,
                perf_config=perf_config
            )
            
            # Sequential processor for this level
            level_processor = nn.Sequential(attn_block, mlp_block)
            self.level_processors.append(level_processor)
            
        # Cross-level attention for information flow between levels
        self.cross_level_attention = nn.ModuleList()
        
        for level in range(num_levels - 1):
            cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.cross_level_attention.append(cross_attn)
            
        # Level-specific summarizers to create summaries for higher levels
        self.summarizers = nn.ModuleList()
        
        for level in range(num_levels - 1):
            summarizer = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            self.summarizers.append(summarizer)
            
        # Normalization layers
        norm_class = get_norm_class(perf_config.normalization)
        self.pre_norms = nn.ModuleList([
            norm_class(dim, perf_config=perf_config)
            for _ in range(num_levels)
        ])
        self.post_norms = nn.ModuleList([
            norm_class(dim, perf_config=perf_config)
            for _ in range(num_levels)
        ])
        
        # Token importance predictors for routing
        self.importance_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(num_levels - 1)
        ])
        
        # Optional retrieval components
        if use_retrieval:
            self.retrievers = nn.ModuleList([
                RetrievalAugmentedProcessor(
                    dim=dim,
                    max_retrieved_tokens=min(256, chunk_sizes[level] // 2),
                    fusion_type="attention",
                    perf_config=perf_config
                )
                for level in range(num_levels)
            ])
        
        # Apply torch.compile if requested
        if perf_config.use_torch_compile and hasattr(torch, "compile"):
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )
            
    def _create_level_chunks(self, x, level):
        """
        Split sequence into chunks for processing at a specific level
        
        Args:
            x: Input sequence [batch_size, seq_len, dim]
            level: Hierarchy level to process
            
        Returns:
            Chunked sequence and chunk metadata
        """
        batch_size, seq_len, _ = x.shape
        chunk_size = self.chunk_sizes[level]
        
        # Calculate number of chunks
        num_chunks = math.ceil(seq_len / chunk_size)
        
        # Pad sequence to multiple of chunk size
        padded_len = num_chunks * chunk_size
        if padded_len > seq_len:
            padding = torch.zeros(batch_size, padded_len - seq_len, self.dim, device=x.device)
            padded_x = torch.cat([x, padding], dim=1)
        else:
            padded_x = x
            
        # Reshape into chunks
        chunked_x = padded_x.view(batch_size, num_chunks, chunk_size, self.dim)
        
        # Create chunk metadata
        chunk_meta = {
            "original_len": seq_len,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "padded_len": padded_len
        }
        
        return chunked_x, chunk_meta
        
    def _create_chunk_summaries(self, chunked_x, level):
        """
        Create summaries of chunks for higher levels
        
        Args:
            chunked_x: Chunked sequence [batch_size, num_chunks, chunk_size, dim]
            level: Current level
            
        Returns:
            Chunk summaries for the next level
        """
        batch_size, num_chunks, chunk_size, _ = chunked_x.shape
        
        # Calculate number of summary tokens per chunk
        summary_tokens = max(1, int(chunk_size * self.summary_ratio))
        
        # Reshape for importance prediction
        flat_chunks = chunked_x.view(batch_size * num_chunks, chunk_size, self.dim)
        
        # Predict token importance
        importance = self.importance_predictors[level](flat_chunks)
        
        # Select most important tokens per chunk
        _, indices = torch.topk(importance.squeeze(-1), summary_tokens, dim=1)
        indices, _ = torch.sort(indices, dim=1)  # Sort to maintain sequence order
        
        # Gather important tokens
        batch_indices = torch.arange(batch_size * num_chunks, device=chunked_x.device).unsqueeze(-1)
        important_tokens = flat_chunks[batch_indices, indices]
        
        # Apply summarizer to create fixed-size summaries
        summaries = self.summarizers[level](important_tokens)
        
        # Reshape back to batch structure
        summaries = summaries.view(batch_size, num_chunks, summary_tokens, self.dim)
        
        return summaries
        
    def _process_level(self, chunked_x, level):
        """
        Process a level in the hierarchy
        
        Args:
            chunked_x: Chunked sequence [batch_size, num_chunks, chunk_size, dim]
            level: Current level
            
        Returns:
            Processed chunks
        """
        batch_size, num_chunks, chunk_size, _ = chunked_x.shape
        
        # Reshape for processing
        flat_chunks = chunked_x.view(batch_size * num_chunks, chunk_size, self.dim)
        
        # Apply retrieval augmentation if enabled
        if self.use_retrieval:
            flat_chunks = self.retrievers[level](flat_chunks)
            
        # Process with level-specific processor
        processed_chunks = self.level_processors[level](flat_chunks)
        
        # Reshape back to chunk structure
        processed_chunks = processed_chunks.view(batch_size, num_chunks, chunk_size, self.dim)
        
        return processed_chunks
        
    def _apply_cross_level_attention(self, lower_chunks, higher_summaries, level):
        """
        Apply cross-level attention to flow information between levels
        
        Args:
            lower_chunks: Processed chunks at current level
            higher_summaries: Summaries from higher level
            level: Current level
            
        Returns:
            Enhanced chunks with cross-level information
        """
        batch_size, num_chunks, chunk_size, _ = lower_chunks.shape
        higher_batch, higher_chunks, summary_size, _ = higher_summaries.shape
        
        # Ensure compatible batch sizes
        assert batch_size == higher_batch, "Batch size mismatch between levels"
        
        # Calculate chunk mapping (which higher-level chunk influences which lower chunks)
        chunk_ratio = num_chunks / higher_chunks
        
        # Process each chunk
        enhanced_chunks = []
        
        for b in range(batch_size):
            batch_enhanced = []
            
            for c in range(num_chunks):
                # Find corresponding higher-level chunk
                higher_idx = min(higher_chunks - 1, int(c / chunk_ratio))
                
                # Get current chunk and higher summary
                chunk = lower_chunks[b, c]  # [chunk_size, dim]
                summary = higher_summaries[b, higher_idx]  # [summary_size, dim]
                
                # Apply cross-attention (chunk as query, summary as key/value)
                enhanced, _ = self.cross_level_attention[level](
                    chunk.unsqueeze(0),  # [1, chunk_size, dim]
                    summary.unsqueeze(0),  # [1, summary_size, dim]
                    summary.unsqueeze(0)   # [1, summary_size, dim]
                )
                
                batch_enhanced.append(enhanced.squeeze(0))
                
            enhanced_chunks.append(torch.stack(batch_enhanced))
            
        # Stack into tensor
        enhanced = torch.stack(enhanced_chunks)
        
        return enhanced
        
    def _merge_chunks(self, processed_chunks, original_len):
        """
        Merge processed chunks back into a sequence
        
        Args:
            processed_chunks: Processed chunks [batch_size, num_chunks, chunk_size, dim]
            original_len: Original sequence length
            
        Returns:
            Merged sequence [batch_size, seq_len, dim]
        """
        batch_size, num_chunks, chunk_size, _ = processed_chunks.shape
        
        # Reshape to flat sequence
        flat_seq = processed_chunks.view(batch_size, num_chunks * chunk_size, self.dim)
        
        # Truncate to original length
        merged = flat_seq[:, :original_len, :]
        
        return merged
        
    def forward(self, x):
        """
        Process a sequence using hierarchical processing
        
        Args:
            x: Input sequence [batch_size, seq_len, dim]
            
        Returns:
            Processed sequence [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Skip hierarchical processing for short sequences
        if seq_len <= self.chunk_sizes[0]:
            return self.level_processors[0](x)
            
        # Process from bottom (smallest chunks) to top (largest chunks)
        level_outputs = []
        level_summaries = []
        
        # Bottom level processing
        chunked_x, chunk_meta = self._create_level_chunks(x, 0)
        processed_chunks = self._process_level(chunked_x, 0)
        level_outputs.append(processed_chunks)
        
        # Create summaries for next level
        if self.num_levels > 1:
            summaries = self._create_chunk_summaries(processed_chunks, 0)
            level_summaries.append(summaries)
            
            # Process middle levels
            for level in range(1, self.num_levels - 1):
                # Flatten summaries from previous level
                prev_summaries = level_summaries[-1]
                batch_size, prev_chunks, summary_size, _ = prev_summaries.shape
                flattened = prev_summaries.view(batch_size, prev_chunks * summary_size, self.dim)
                
                # Create chunks for this level
                chunked_x, chunk_meta = self._create_level_chunks(flattened, level)
                processed_chunks = self._process_level(chunked_x, level)
                level_outputs.append(processed_chunks)
                
                # Create summaries for next level
                summaries = self._create_chunk_summaries(processed_chunks, level)
                level_summaries.append(summaries)
                
            # Top level processing
            level = self.num_levels - 1
            prev_summaries = level_summaries[-1]
            batch_size, prev_chunks, summary_size, _ = prev_summaries.shape
            flattened = prev_summaries.view(batch_size, prev_chunks * summary_size, self.dim)
            
            # Create chunks for top level
            chunked_x, chunk_meta = self._create_level_chunks(flattened, level)
            processed_chunks = self._process_level(chunked_x, level)
            level_outputs.append(processed_chunks)
            
        # Process from top to bottom
        for level in range(self.num_levels - 2, -1, -1):
            # Apply cross-level attention
            lower_chunks = level_outputs[level]
            higher_chunks = level_outputs[level + 1]
            
            # Convert higher chunks to summaries if needed
            if level < self.num_levels - 2:
                # Need to expand higher chunks to match lower level structure
                higher_batch, higher_chunks, higher_size, _ = higher_chunks.shape
                higher_flat = higher_chunks.view(higher_batch, higher_chunks * higher_size, self.dim)
                higher_summaries = level_summaries[level + 1]
            else:
                # Top level already has the right structure
                higher_summaries = self._create_chunk_summaries(higher_chunks, level)
                
            # Enhance with cross-level attention
            enhanced_chunks = self._apply_cross_level_attention(
                lower_chunks, higher_summaries, level
            )
            
            # Update level outputs with enhanced chunks
            level_outputs[level] = enhanced_chunks
            
        # Merge bottom level chunks for final output
        bottom_chunks = level_outputs[0]
        merged = self._merge_chunks(bottom_chunks, seq_len)
        
        return merged

# Token Streaming System for online processing
class TokenStreamProcessor(Module):
    """
    Process token streams efficiently for real-time generation
    
    Features:
    - Efficient handling of streamed tokens
    - Adaptive window management
    - Compressed history representation
    - Low-latency inference optimizations
    """
    def __init__(
        self,
        dim: int,
        memory_manager = None,  # HierarchicalMemoryManager instance
        processor = None,  # UltraContextModule instance
        active_window_size: int = 4096,
        history_compression_ratio: float = 4.0,
        kv_cache_size: int = 100_000,
        prefill_chunk_size: int = 512,
        use_adaptive_windows: bool = True,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
    ):
        super().__init__()
        self.dim = dim
        self.memory_manager = memory_manager
        self.processor = processor
        self.active_window_size = active_window_size
        self.history_compression_ratio = history_compression_ratio
        self.kv_cache_size = kv_cache_size
        self.prefill_chunk_size = prefill_chunk_size
        self.use_adaptive_windows = use_adaptive_windows
        
        # Token windows
        self.active_tokens = None
        self.active_positions = None
        self.history_tokens = None
        self.history_positions = None
        self.current_position = 0
        
        # Window compressor
        self.window_compressor = ContextualCompressor(
            dim=dim,
            target_compression_ratio=history_compression_ratio,
            min_tokens_before_compression=128,
            strategies=["merge", "summarize"],
            adaptive_compression=True,
            perf_config=perf_config
        )
        
        # Token importance predictor for adaptive window sizing
        if use_adaptive_windows:
            self.importance_predictor = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
            
        # Stats tracking
        self.stats = {
            "total_tokens": 0,
            "compressed_tokens": 0,
            "active_window_size": active_window_size,
            "history_size": 0,
            "compression_ratio": 0
        }
        
    def _adjust_window_size(self):
        """Dynamically adjust the active window size based on content"""
        if not self.use_adaptive_windows or self.active_tokens is None:
            return
            
        # Calculate token importance
        importance = self.importance_predictor(self.active_tokens)
        avg_importance = importance.mean().item()
        
        # Adjust window size based on importance
        if avg_importance > 0.7:
            # High importance content, increase window
            self.active_window_size = min(8192, int(self.active_window_size * 1.1))
        elif avg_importance < 0.3:
            # Low importance content, decrease window
            self.active_window_size = max(1024, int(self.active_window_size * 0.9))
            
        # Update stats
        self.stats["active_window_size"] = self.active_window_size
        
    def _compress_history(self):
        """Compress history tokens to save memory"""
        if self.history_tokens is None or self.history_tokens.size(1) < 128:
            return
            
        # Compress history tokens
        compressed_tokens, _ = self.window_compressor(self.history_tokens)
        
        # Update history
        old_size = self.history_tokens.size(1)
        new_size = compressed_tokens.size(1)
        self.history_tokens = compressed_tokens
        
        # Scale positions to match compression
        scale_factor = new_size / old_size
        self.history_positions = torch.round(self.history_positions * scale_factor).long()
        
        # Update stats
        self.stats["compressed_tokens"] += old_size - new_size
        self.stats["history_size"] = new_size
        self.stats["compression_ratio"] = old_size / max(1, new_size)
        
    def _store_in_memory(self, tokens, positions):
        """Store tokens in long-term memory"""
        if self.memory_manager is not None:
            self.memory_manager.add_tokens(tokens, positions)
            
    def _shift_tokens_to_history(self, num_tokens=None):
        """Shift oldest tokens from active window to history"""
        if self.active_tokens is None:
            return
            
        # Determine number of tokens to shift
        if num_tokens is None:
            # Shift half of active window
            num_tokens = self.active_tokens.size(1) // 2
            
        # Ensure we keep some tokens in active window
        num_tokens = min(num_tokens, self.active_tokens.size(1) - 64)
        if num_tokens <= 0:
            return
            
        # Get tokens to shift
        tokens_to_shift = self.active_tokens[:, :num_tokens]
        positions_to_shift = self.active_positions[:, :num_tokens]
        
        # Store in memory manager
        self._store_in_memory(tokens_to_shift, positions_to_shift)
        
        # Update active window
        self.active_tokens = self.active_tokens[:, num_tokens:]
        self.active_positions = self.active_positions[:, num_tokens:]
        
        # Update history
        if self.history_tokens is None:
            self.history_tokens = tokens_to_shift
            self.history_positions = positions_to_shift
        else:
            self.history_tokens = torch.cat([self.history_tokens, tokens_to_shift], dim=1)
            self.history_positions = torch.cat([self.history_positions, positions_to_shift], dim=1)
            
        # Compress history if it gets too large
        if self.history_tokens.size(1) > 1024:
            self._compress_history()
            
    def reset(self):
        """Reset the stream processor state"""
        self.active_tokens = None
        self.active_positions = None
        self.history_tokens = None
        self.history_positions = None
        self.current_position = 0
        
        # Reset stats
        self.stats = {
            "total_tokens": 0,
            "compressed_tokens": 0,
            "active_window_size": self.active_window_size,
            "history_size": 0,
            "compression_ratio": 0
        }
        
    def prefill(self, tokens):
        """
        Process a batch of tokens in the initial prefill phase
        
        Args:
            tokens: Token embeddings [batch_size, seq_len, dim]
            
        Returns:
            Processed tokens
        """
        batch_size, seq_len, _ = tokens.shape
        device = tokens.device
        
        # Process in chunks for memory efficiency
        if seq_len > self.prefill_chunk_size and self.processor is not None:
            processed_chunks = []
            
            for i in range(0, seq_len, self.prefill_chunk_size):
                end = min(i + self.prefill_chunk_size, seq_len)
                chunk = tokens[:, i:end]
                
                # Process chunk
                processed_chunk = self.processor(chunk)
                processed_chunks.append(processed_chunk)
                
            # Combine processed chunks
            processed = torch.cat(processed_chunks, dim=1)
        elif self.processor is not None:
            # Process entire sequence
            processed = self.processor(tokens)
        else:
            # No processor, pass through
            processed = tokens
            
        # Set up active window and positions
        self.active_tokens = processed
        self.active_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        self.current_position = seq_len
        
        # Update stats
        self.stats["total_tokens"] = seq_len
        
        # Handle overflowing active window
        if seq_len > self.active_window_size:
            # Shift excess tokens to history
            self._shift_tokens_to_history(seq_len - self.active_window_size)
            
        return processed[:, -1:]  # Return last token
        
    def forward(self, token):
        """
        Process a single incoming token or small batch of tokens
        
        Args:
            token: New token embedding [batch_size, 1, dim] or [batch_size, num_tokens, dim]
            
        Returns:
            Processed token
        """
        batch_size, num_new, _ = token.shape
        device = token.device
        
        # If first token, initialize active window
        if self.active_tokens is None:
            self.active_tokens = token
            self.active_positions = torch.tensor([[self.current_position]], device=device).expand(batch_size, -1)
            self.current_position += num_new
            
            # Update stats
            self.stats["total_tokens"] = num_new
            
            # Process with processor if available
            if self.processor is not None:
                return self.processor(token)
            return token
            
        # Create position tensor for new token
        new_positions = torch.arange(
            self.current_position, 
            self.current_position + num_new,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Add new token to active window
        self.active_tokens = torch.cat([self.active_tokens, token], dim=1)
        self.active_positions = torch.cat([self.active_positions, new_positions], dim=1)
        self.current_position += num_new
        
        # Update stats
        self.stats["total_tokens"] += num_new
        
        # Check if we need to shift tokens to history
        if self.active_tokens.size(1) > self.active_window_size:
            # Shift excess tokens to history
            self._shift_tokens_to_history(self.active_tokens.size(1) - self.active_window_size)
            
        # Periodically adjust window size
        if self.stats["total_tokens"] % 1000 == 0:
            self._adjust_window_size()
            
        # Process with processor if available
        if self.processor is not None:
            return self.processor(token)
        return token
        
    def get_stats(self):
        """Get current statistics about the stream processor"""
        return self.stats
        
    def get_full_context(self):
        """
        Get the full available context (active + history)
        
        Returns:
            Tokens and positions for full context
        """
        if self.active_tokens is None:
            return None, None
            
        if self.history_tokens is None:
            return self.active_tokens, self.active_positions
            
        # Combine history and active tokens
        full_tokens = torch.cat([self.history_tokens, self.active_tokens], dim=1)
        full_positions = torch.cat([self.history_positions, self.active_positions], dim=1)
        
        return full_tokens, full_positions

# Create a scalable tokenizer that supports arbitrary context lengths
class ScalableTokenizer:
    """
    Tokenizer wrapper that can handle arbitrarily long contexts 
    without running out of memory
    
    Features:
    - Streaming tokenization for long texts
    - Adaptive chunking based on content
    - Efficient batch processing
    - Parallel processing capabilities
    """
    def __init__(
        self,
        base_tokenizer=None,  # Pass your actual tokenizer here
        max_chunk_size: int = 8192,
        memory_efficient: bool = True,
        num_workers: int = 0,
        use_sliding_window: bool = True,
        sliding_window_size: int = 4096,
        sliding_window_overlap: int = 1024,
    ):
        self.base_tokenizer = base_tokenizer
        self.max_chunk_size = max_chunk_size
        self.memory_efficient = memory_efficient
        self.num_workers = num_workers
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.sliding_window_overlap = sliding_window_overlap
        
    def _find_chunk_boundaries(self, text):
        """Find natural boundaries to split text into chunks"""
        # Prefer splitting at paragraph or sentence boundaries
        paragraph_boundaries = [match.start() for match in re.finditer(r'\n\s*\n', text)]
        sentence_boundaries = [match.start() for match in re.finditer(r'[.!?]\s', text)]
        
        # Combine and sort boundaries
        all_boundaries = sorted(set(paragraph_boundaries + sentence_boundaries + [0, len(text)]))
        
        # Create chunks of appropriate size
        chunks = []
        start = 0
        
        for i in range(1, len(all_boundaries)):
            end = all_boundaries[i]
            
            if end - start > self.max_chunk_size:
                # Chunk too large, find intermediate boundary
                intermediate = -1
                
                for boundary in sentence_boundaries:
                    if start < boundary < start + self.max_chunk_size:
                        intermediate = boundary
                        
                if intermediate == -1:
                    # No sentence boundary, split at max size
                    intermediate = start + self.max_chunk_size
                    
                chunks.append((start, intermediate))
                start = intermediate
            
            if end - start > 0:
                chunks.append((start, end))
                start = end
                
        return chunks
        
    def _tokenize_chunk(self, text_chunk):
        """Tokenize a single chunk of text"""
        if self.base_tokenizer is None:
            # Dummy tokenization for testing
            return list(range(len(text_chunk.split())))
            
        return self.base_tokenizer(text_chunk)
        
    def _process_with_sliding_window(self, tokens):
        """Process tokens using sliding window to maintain context"""
        if not self.use_sliding_window:
            return tokens
            
        # Split into overlapping windows
        window_size = self.sliding_window_size
        overlap = self.sliding_window_overlap
        step = window_size - overlap
        
        windows = []
        for i in range(0, len(tokens), step):
            end = min(len(tokens), i + window_size)
            windows.append(tokens[i:end])
            
            if end == len(tokens):
                break
                
        return windows
        
    def tokenize(self, text, return_windows=False):
        """
        Tokenize text of arbitrary length
        
        Args:
            text: Text to tokenize
            return_windows: Whether to return as sliding windows
            
        Returns:
            Tokenized text (either as full sequence or list of windows)
        """
        if len(text) <= self.max_chunk_size and not return_windows:
            # Short text, tokenize directly
            return self._tokenize_chunk(text)
            
        # Find chunk boundaries
        chunks = self._find_chunk_boundaries(text)
        
        # Tokenize chunks
        chunk_tokens = []
        
        if self.num_workers > 0:
            # Parallel processing
            from concurrent.futures import ThreadPoolExecutor
            
            chunk_texts = [text[start:end] for start, end in chunks]
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                chunk_tokens = list(executor.map(self._tokenize_chunk, chunk_texts))
        else:
            # Sequential processing
            for start, end in chunks:
                chunk_text = text[start:end]
                tokens = self._tokenize_chunk(chunk_text)
                chunk_tokens.append(tokens)
                
        # Combine tokens
        all_tokens = []
        for tokens in chunk_tokens:
            all_tokens.extend(tokens)
            
        # Process with sliding window if requested
        if return_windows:
            return self._process_with_sliding_window(all_tokens)
            
        return all_tokens
