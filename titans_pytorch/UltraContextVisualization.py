import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx
from typing import List, Dict, Optional, Union, Tuple, Any
import math
import logging
import json
import os
import io
import base64
from dataclasses import dataclass, field
from IPython.display import HTML, display
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext.visualizer")

# Try to import UltraContext components
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
    logger.warning("UltraContext package not installed. Some visualization features may be limited.")

#######################################
# Token Importance Visualizer
#######################################

class TokenImportanceVisualizer:
    """Visualize token importance in the context window"""
    
    def __init__(
        self, 
        tokens: List[str], 
        importance_scores: Optional[List[float]] = None,
        attention_scores: Optional[List[float]] = None,
        usage_counts: Optional[List[int]] = None,
        positions: Optional[List[int]] = None,
        color_map: str = "viridis"
    ):
        """
        Initialize the token importance visualizer
        
        Args:
            tokens: List of tokens to visualize
            importance_scores: Optional list of importance scores (0-1)
            attention_scores: Optional list of attention scores
            usage_counts: Optional list of usage counts
            positions: Optional list of original positions
            color_map: Matplotlib color map name
        """
        self.tokens = tokens
        self.importance_scores = importance_scores or [0.5] * len(tokens)
        self.attention_scores = attention_scores or [0.0] * len(tokens)
        self.usage_counts = usage_counts or [0] * len(tokens)
        self.positions = positions or list(range(len(tokens)))
        self.color_map = color_map
        
        # Ensure all lists have the same length
        min_len = min(len(tokens), len(self.importance_scores), 
                     len(self.attention_scores), len(self.usage_counts),
                     len(self.positions))
        
        self.tokens = self.tokens[:min_len]
        self.importance_scores = self.importance_scores[:min_len]
        self.attention_scores = self.attention_scores[:min_len]
        self.usage_counts = self.usage_counts[:min_len]
        self.positions = self.positions[:min_len]
        
    def plot_token_importance(self, figsize=(15, 8), max_tokens=100):
        """
        Plot token importance as a colored bar chart
        
        Args:
            figsize: Figure size (width, height)
            max_tokens: Maximum number of tokens to display
            
        Returns:
            Matplotlib figure
        """
        # Limit number of tokens to display
        if len(self.tokens) > max_tokens:
            # Sample tokens evenly
            indices = np.linspace(0, len(self.tokens) - 1, max_tokens, dtype=int)
            tokens = [self.tokens[i] for i in indices]
            importance = [self.importance_scores[i] for i in indices]
            positions = [self.positions[i] for i in indices]
        else:
            tokens = self.tokens
            importance = self.importance_scores
            positions = self.positions
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create colormap
        cmap = plt.get_cmap(self.color_map)
        
        # Plot bars
        bars = ax.bar(range(len(tokens)), importance, color=[cmap(score) for score in importance])
        
        # Add labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        
        # Add position labels above bars
        for i, (pos, imp) in enumerate(zip(positions, importance)):
            ax.text(i, imp + 0.02, str(pos), ha='center', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Importance Score")
        ax.set_title("Token Importance in Context Window")
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm)
        cbar.set_label("Importance")
        
        plt.tight_layout()
        
        return fig
    
    def plot_heatmap(self, figsize=(15, 10), max_tokens=500, text_tokens=True):
        """
        Plot token importance as a heatmap
        
        Args:
            figsize: Figure size (width, height)
            max_tokens: Maximum number of tokens to display
            text_tokens: Whether to display token text in heatmap
            
        Returns:
            Matplotlib figure
        """
        # Limit number of tokens to display
        if len(self.tokens) > max_tokens:
            # Sample tokens evenly
            indices = np.linspace(0, len(self.tokens) - 1, max_tokens, dtype=int)
            tokens = [self.tokens[i] for i in indices]
            importance = [self.importance_scores[i] for i in indices]
            positions = [self.positions[i] for i in indices]
        else:
            tokens = self.tokens
            importance = self.importance_scores
            positions = self.positions
        
        # Create grid layout based on number of tokens
        # Try to make it roughly square
        grid_size = math.ceil(math.sqrt(len(tokens)))
        rows = math.ceil(len(tokens) / grid_size)
        cols = grid_size
        
        # Create data grid
        grid = np.zeros((rows, cols))
        for i, score in enumerate(importance):
            r = i // cols
            c = i % cols
            grid[r, c] = score
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(grid, cmap=self.color_map, aspect='auto')
        
        # Add text with token values
        if text_tokens:
            for i in range(len(tokens)):
                r = i // cols
                c = i % cols
                text_color = "black" if importance[i] < 0.5 else "white"
                ax.text(c, r, tokens[i], ha="center", va="center", color=text_color, fontsize=8)
                
                # Add position as small text in corner
                ax.text(c + 0.35, r - 0.35, str(positions[i]), ha="center", va="center", 
                       color=text_color, fontsize=6, alpha=0.7)
        
        # Add colorbar
        cbar = fig.colorbar(im)
        cbar.set_label("Importance")
        
        # Set title
        ax.set_title("Token Importance Heatmap")
        
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        return fig
    
    def plot_interactive(self, max_tokens=1000):
        """
        Create an interactive plot using Plotly
        
        Args:
            max_tokens: Maximum number of tokens to display
            
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            logger.error("Plotly not installed. Run 'pip install plotly'")
            return None
        
        # Limit number of tokens to display
        if len(self.tokens) > max_tokens:
            # Sample tokens evenly
            indices = np.linspace(0, len(self.tokens) - 1, max_tokens, dtype=int)
            tokens = [self.tokens[i] for i in indices]
            importance = [self.importance_scores[i] for i in indices]
            attention = [self.attention_scores[i] for i in indices]
            usage = [self.usage_counts[i] for i in indices]
            positions = [self.positions[i] for i in indices]
        else:
            tokens = self.tokens
            importance = self.importance_scores
            attention = self.attention_scores
            usage = self.usage_counts
            positions = self.positions
        
        # Create dataframe
        data = {
            "Token": tokens,
            "Position": positions,
            "Importance": importance,
            "Attention": attention,
            "Usage": usage
        }
        
        # Create plot
        fig = px.scatter(
            data,
            x=list(range(len(tokens))),
            y="Importance",
            color="Importance",
            size=[max(1, u) for u in usage],
            hover_data=["Token", "Position", "Attention", "Usage"],
            color_continuous_scale=self.color_map,
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive Token Importance Visualization",
            xaxis_title="Token Index",
            yaxis_title="Importance Score",
            height=600
        )
        
        # Add token labels
        if len(tokens) <= 100:
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(tokens))),
                    ticktext=tokens,
                    tickangle=90
                )
            )
        
        return fig

#######################################
# Context Window Visualizer
#######################################

class ContextWindowVisualizer:
    """Visualize the context window structure and organization"""
    
    def __init__(
        self,
        active_window_size: int,
        sliding_window_size: int,
        tokens_in_memory: Dict[str, int],
        compression_stats: Optional[Dict[str, float]] = None,
        memory_levels: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the context window visualizer
        
        Args:
            active_window_size: Size of active context window
            sliding_window_size: Size of sliding window
            tokens_in_memory: Dictionary of token counts in different memory components
            compression_stats: Optional dictionary of compression statistics
            memory_levels: Optional list of memory level information
        """
        self.active_window_size = active_window_size
        self.sliding_window_size = sliding_window_size
        self.tokens_in_memory = tokens_in_memory
        self.compression_stats = compression_stats or {}
        self.memory_levels = memory_levels or []
        
    def plot_context_structure(self, figsize=(12, 8)):
        """
        Plot the structure of the context window
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate total tokens
        total_tokens = sum(self.tokens_in_memory.values())
        
        # Define colors
        colors = {
            "active": "#3498db",
            "compressed": "#e74c3c",
            "l1": "#2ecc71",
            "l2": "#f39c12",
            "l3": "#9b59b6",
            "persistent": "#34495e"
        }
        
        # Create stacked bar chart for token distribution
        token_types = []
        token_counts = []
        token_colors = []
        
        for key, value in self.tokens_in_memory.items():
            if value > 0:
                token_types.append(key)
                token_counts.append(value)
                token_colors.append(colors.get(key.lower(), "#95a5a6"))
        
        # Plot bar chart
        bars = ax.bar(token_types, token_counts, color=token_colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:,}',
                ha='center',
                va='bottom'
            )
        
        # Set labels and title
        ax.set_xlabel("Memory Component")
        ax.set_ylabel("Token Count")
        ax.set_title("Token Distribution Across Memory Components")
        
        # Add information about window sizes
        info_text = f"Active Window: {self.active_window_size:,} tokens\n"
        info_text += f"Sliding Window: {self.sliding_window_size:,} tokens\n"
        info_text += f"Total Tokens in Memory: {total_tokens:,}"
        
        # Add compression statistics if available
        if self.compression_stats:
            info_text += "\n\nCompression Statistics:\n"
            for key, value in self.compression_stats.items():
                info_text += f"{key}: {value:.2f}\n"
        
        # Add text box with information
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.05, 0.95, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        
        return fig
    
    def plot_memory_hierarchy(self, figsize=(10, 8)):
        """
        Plot the memory hierarchy as a nested structure
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        if not self.memory_levels:
            logger.warning("No memory level information provided")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort memory levels by ID
        memory_levels = sorted(self.memory_levels, key=lambda x: x.get("level_id", 0))
        
        # Define colors for memory levels
        colors = ["#2ecc71", "#f39c12", "#9b59b6", "#34495e"]
        
        # Calculate total height and level heights
        total_height = 8
        level_heights = []
        
        for level in memory_levels:
            capacity = level.get("capacity", 0)
            level_heights.append(math.log2(capacity + 1))
            
        # Normalize heights
        total_level_height = sum(level_heights)
        level_heights = [h / total_level_height * total_height for h in level_heights]
        
        # Draw memory levels as nested rectangles
        y_pos = 0
        for i, (level, height) in enumerate(zip(memory_levels, level_heights)):
            level_id = level.get("level_id", i+1)
            capacity = level.get("capacity", 0)
            tokens = level.get("tokens", 0)
            retrieval_cost = level.get("retrieval_cost", 0)
            storage_cost = level.get("storage_cost", 0)
            
            # Calculate width based on tokens and capacity
            width = 10
            inner_width = width * (tokens / capacity) if capacity > 0 else 0
            
            # Draw outer rectangle (capacity)
            rect_outer = patches.Rectangle(
                (0, y_pos),
                width,
                height,
                linewidth=1,
                edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)],
                alpha=0.3
            )
            ax.add_patch(rect_outer)
            
            # Draw inner rectangle (tokens)
            rect_inner = patches.Rectangle(
                (0, y_pos),
                inner_width,
                height,
                linewidth=1,
                edgecolor=colors[i % len(colors)],
                facecolor=colors[i % len(colors)],
                alpha=0.7
            )
            ax.add_patch(rect_inner)
            
            # Add text
            ax.text(
                width + 0.5,
                y_pos + height/2,
                f"L{level_id}: {tokens:,}/{capacity:,} tokens\n"
                f"Retrieval Cost: {retrieval_cost:.1f}\n"
                f"Storage Cost: {storage_cost:.1f}",
                va='center'
            )
            
            y_pos += height
        
        # Set axis limits
        ax.set_xlim(0, 15)
        ax.set_ylim(0, total_height)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title
        ax.set_title("Memory Hierarchy")
        
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_memory(self):
        """
        Create an interactive plot of the memory hierarchy using Plotly
        
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Run 'pip install plotly'")
            return None
        
        if not self.memory_levels:
            logger.warning("No memory level information provided")
            return None
        
        # Sort memory levels by ID
        memory_levels = sorted(self.memory_levels, key=lambda x: x.get("level_id", 0))
        
        # Define colors for memory levels
        colors = ["#2ecc71", "#f39c12", "#9b59b6", "#34495e"]
        
        # Create figure
        fig = go.Figure()
        
        # Add bars for each memory level
        level_names = []
        capacities = []
        tokens = []
        fill_colors = []
        
        for i, level in enumerate(memory_levels):
            level_id = level.get("level_id", i+1)
            capacity = level.get("capacity", 0)
            token_count = level.get("tokens", 0)
            
            level_names.append(f"L{level_id}")
            capacities.append(capacity)
            tokens.append(token_count)
            fill_colors.append(colors[i % len(colors)])
        
        # Add trace for capacity (outline)
        fig.add_trace(go.Bar(
            x=level_names,
            y=capacities,
            name="Capacity",
            marker_color="rgba(0,0,0,0)",
            marker_line_color=fill_colors,
            marker_line_width=2,
            opacity=0.5,
            hovertemplate="Capacity: %{y:,} tokens<extra></extra>"
        ))
        
        # Add trace for tokens (filled)
        fig.add_trace(go.Bar(
            x=level_names,
            y=tokens,
            name="Tokens",
            marker_color=fill_colors,
            hovertemplate="Tokens: %{y:,}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Memory Hierarchy",
            xaxis_title="Memory Level",
            yaxis_title="Token Count",
            yaxis_type="log",
            barmode="overlay",
            height=500
        )
        
        # Add annotations with additional info
        for i, level in enumerate(memory_levels):
            retrieval_cost = level.get("retrieval_cost", 0)
            storage_cost = level.get("storage_cost", 0)
            hit_rate = level.get("hit_rate", 0)
            
            fig.add_annotation(
                x=i,
                y=capacities[i],
                text=f"Retrieval: {retrieval_cost:.1f}<br>Storage: {storage_cost:.1f}<br>Hit Rate: {hit_rate:.1%}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=fill_colors[i],
                borderwidth=2
            )
        
        return fig

#######################################
# Attention Pattern Visualizer
#######################################

class AttentionPatternVisualizer:
    """Visualize attention patterns in the context window"""
    
    def __init__(
        self,
        query_positions: List[int],
        key_positions: List[int],
        attention_scores: List[List[float]],
        tokens: Optional[List[str]] = None,
        color_map: str = "viridis"
    ):
        """
        Initialize the attention pattern visualizer
        
        Args:
            query_positions: List of query token positions
            key_positions: List of key token positions
            attention_scores: 2D list of attention scores [query_idx][key_idx]
            tokens: Optional list of tokens
            color_map: Matplotlib color map name
        """
        self.query_positions = query_positions
        self.key_positions = key_positions
        self.attention_scores = attention_scores
        self.tokens = tokens
        self.color_map = color_map
        
        # Ensure attention scores is a 2D numpy array
        self.attention_matrix = np.array(attention_scores)
        
    def plot_attention_heatmap(self, figsize=(12, 10), show_scores=False):
        """
        Plot attention scores as a heatmap
        
        Args:
            figsize: Figure size (width, height)
            show_scores: Whether to display attention scores in the heatmap
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(self.attention_matrix, cmap=self.color_map, aspect='auto')
        
        # Set labels
        if self.tokens and len(self.tokens) == len(self.key_positions):
            # Show tokens on y-axis (keys)
            ax.set_yticks(range(len(self.key_positions)))
            ax.set_yticklabels([f"{self.key_positions[i]}: {t}" for i, t in enumerate(self.tokens)])
            
            # If tokens also match query positions, show on x-axis
            if len(self.tokens) == len(self.query_positions):
                ax.set_xticks(range(len(self.query_positions)))
                ax.set_xticklabels([f"{self.query_positions[i]}: {t}" for i, t in enumerate(self.tokens)], 
                                  rotation=90)
            else:
                # Otherwise just show positions
                ax.set_xticks(range(len(self.query_positions)))
                ax.set_xticklabels(self.query_positions, rotation=90)
        else:
            # Show positions only
            ax.set_xticks(range(len(self.query_positions)))
            ax.set_xticklabels(self.query_positions, rotation=90)
            
            ax.set_yticks(range(len(self.key_positions)))
            ax.set_yticklabels(self.key_positions)
        
        # Show attention scores in each cell
        if show_scores:
            for i in range(len(self.query_positions)):
                for j in range(len(self.key_positions)):
                    score = self.attention_matrix[j, i]
                    text_color = "white" if score > 0.5 else "black"
                    ax.text(i, j, f"{score:.2f}", ha="center", va="center", color=text_color, fontsize=8)
        
        # Add colorbar
        cbar = fig.colorbar(im)
        cbar.set_label("Attention Score")
        
        # Set labels and title
        ax.set_xlabel("Query Position")
        ax.set_ylabel("Key Position")
        ax.set_title("Attention Pattern Heatmap")
        
        plt.tight_layout()
        
        return fig
    
    def plot_attention_graph(self, figsize=(10, 10), threshold=0.1, max_edges=100):
        """
        Plot attention as a graph network
        
        Args:
            figsize: Figure size (width, height)
            threshold: Minimum attention score to display an edge
            max_edges: Maximum number of edges to display
            
        Returns:
            Matplotlib figure
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX not installed. Run 'pip install networkx'")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, pos in enumerate(self.query_positions):
            G.add_node(f"Q{pos}", position=pos, type="query")
            
        for i, pos in enumerate(self.key_positions):
            if f"Q{pos}" not in G:
                G.add_node(f"K{pos}", position=pos, type="key")
            
        # Add edges (with attention scores as weights)
        edges = []
        for i, q_pos in enumerate(self.query_positions):
            for j, k_pos in enumerate(self.key_positions):
                score = self.attention_matrix[j, i]
                if score > threshold:
                    # Store as (query, key, score)
                    node_q = f"Q{q_pos}"
                    node_k = f"Q{k_pos}" if f"Q{k_pos}" in G else f"K{k_pos}"
                    edges.append((node_q, node_k, score))
        
        # Sort edges by score and limit
        edges.sort(key=lambda x: x[2], reverse=True)
        edges = edges[:max_edges]
        
        # Add to graph
        for q, k, score in edges:
            G.add_edge(q, k, weight=score)
        
        # Create positions for nodes
        # Queries on left, keys on right
        pos = {}
        q_nodes = [n for n in G.nodes() if n.startswith("Q")]
        k_nodes = [n for n in G.nodes() if n.startswith("K")]
        
        for i, node in enumerate(q_nodes):
            pos[node] = (-1, i)
            
        for i, node in enumerate(k_nodes):
            pos[node] = (1, i)
        
        # Draw the graph
        # Nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=q_nodes,
            node_color="skyblue",
            node_size=300,
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=k_nodes,
            node_color="lightgreen",
            node_size=300,
            alpha=0.8,
            ax=ax
        )
        
        # Node labels
        if self.tokens and len(self.tokens) == len(self.key_positions):
            # Create label mapping
            labels = {}
            for i, pos in enumerate(self.query_positions):
                if i < len(self.tokens):
                    labels[f"Q{pos}"] = f"{pos}: {self.tokens[i]}"
                else:
                    labels[f"Q{pos}"] = f"{pos}"
                    
            for i, pos in enumerate(self.key_positions):
                node = f"K{pos}"
                if node in G.nodes():
                    labels[node] = f"{pos}: {self.tokens[i]}"
            
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        else:
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        # Edges with width based on attention score
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        
        # Get edge colors based on weights
        cmap = plt.get_cmap(self.color_map)
        edge_colors = [cmap(G[u][v]['weight']) for u, v in G.edges()]
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.7,
            edge_color=edge_colors,
            connectionstyle="arc3,rad=0.2",
            ax=ax
        )
        
        # Remove axis
        ax.axis('off')
        
        # Add title
        ax.set_title("Attention Graph")
        
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_attention(self, max_edges=200):
        """
        Create an interactive attention visualization using Plotly
        
        Args:
            max_edges: Maximum number of edges to display
            
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            logger.error("Required libraries not installed. Run 'pip install plotly networkx'")
            return None
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, pos in enumerate(self.query_positions):
            label = f"{pos}"
            if self.tokens and i < len(self.tokens):
                label += f": {self.tokens[i]}"
            G.add_node(f"Q{pos}", position=pos, type="query", label=label)
            
        for i, pos in enumerate(self.key_positions):
            if f"Q{pos}" not in G:
                label = f"{pos}"
                if self.tokens and i < len(self.tokens):
                    label += f": {self.tokens[i]}"
                G.add_node(f"K{pos}", position=pos, type="key", label=label)
            
        # Add edges (with attention scores as weights)
        for i, q_pos in enumerate(self.query_positions):
            for j, k_pos in enumerate(self.key_positions):
                score = self.attention_matrix[j, i]
                if score > 0.01:  # Minimum threshold
                    node_q = f"Q{q_pos}"
                    node_k = f"Q{k_pos}" if f"Q{k_pos}" in G else f"K{k_pos}"
                    G.add_edge(node_q, node_k, weight=score)
        
        # Limit edges by taking top N by weight
        if len(G.edges()) > max_edges:
            # Sort edges by weight and keep only top N
            edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
            edges.sort(key=lambda x: x[2], reverse=True)
            edges = edges[:max_edges]
            
            # Create new graph with only these edges
            G_filtered = nx.DiGraph()
            
            # Add nodes from original graph
            for node, data in G.nodes(data=True):
                G_filtered.add_node(node, **data)
                
            # Add filtered edges
            for u, v, weight in edges:
                G_filtered.add_edge(u, v, weight=weight)
                
            G = G_filtered
        
        # Create positions for visualization (spectral layout often works well)
        pos = nx.spring_layout(G, seed=42)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create edge weight trace (for coloring)
        edge_weights = []
        
        # Add edges to trace
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=[],
                size=10,
                line=dict(width=2)
            )
        )
        
        # Add nodes to trace
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            
            # Add node type info to color
            node_type = G.nodes[node]['type']
            if node_type == 'query':
                node_trace['marker']['color'] += (0,)  # Blue in Viridis
            else:
                node_trace['marker']['color'] += (0.7,)  # Green in Viridis
            
            # Add node text
            node_trace['text'] += (G.nodes[node]['label'],)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Interactive Attention Pattern Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig

#######################################
# Token Movement Visualizer
#######################################

class TokenMovementVisualizer:
    """Visualize token movement between memory levels"""
    
    def __init__(
        self,
        memory_levels: List[Dict[str, Any]],
        token_movements: List[Dict[str, Any]],
        tokens: Optional[List[str]] = None
    ):
        """
        Initialize the token movement visualizer
        
        Args:
            memory_levels: List of memory level information
            token_movements: List of token movement records
            tokens: Optional list of tokens
        """
        self.memory_levels = memory_levels
        self.token_movements = token_movements
        self.tokens = tokens
        
    def plot_movement_sankey(self, figsize=(15, 10)):
        """
        Plot token movements as a Sankey diagram
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX not installed. Run 'pip install networkx'")
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add memory levels as nodes
        for level in self.memory_levels:
            level_id = level.get("level_id", 0)
            capacity = level.get("capacity", 0)
            tokens = level.get("tokens", 0)
            
            G.add_node(f"L{level_id}", capacity=capacity, tokens=tokens)
        
        # Add token movements as edges
        for movement in self.token_movements:
            from_level = movement.get("from_level", "")
            to_level = movement.get("to_level", "")
            count = movement.get("count", 0)
            
            if from_level and to_level and count > 0:
                if G.has_edge(from_level, to_level):
                    G[from_level][to_level]["weight"] += count
                else:
                    G.add_edge(from_level, to_level, weight=count)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create node positions
        # Arrange levels vertically
        pos = {}
        nodes = sorted(G.nodes())
        
        for i, node in enumerate(nodes):
            pos[node] = (0, i)
        
        # Set up Sankey diagram
        try:
            from matplotlib.sankey import Sankey
            
            sankey = Sankey(ax=ax, scale=0.01, offset=0.2, head_angle=120, margin=0.4)
            
            # Add flows
            for i, node in enumerate(nodes):
                # Calculate inflows and outflows
                outflows = [G[node][target]["weight"] for target in G.successors(node)]
                outflow_labels = [target for target in G.successors(node)]
                
                inflows = [G[source][node]["weight"] for source in G.predecessors(node)]
                inflow_labels = [source for source in G.predecessors(node)]
                
                # Skip if no flows
                if not inflows and not outflows:
                    continue
                
                # Combine flows
                flows = [-flow for flow in inflows] + outflows
                labels = inflow_labels + outflow_labels
                
                # Orientations
                orientations = [-1] * len(inflows) + [1] * len(outflows)
                
                # Add to Sankey
                sankey.add(
                    flows=flows,
                    labels=labels,
                    orientations=orientations,
                    pathlengths=[0.25] * len(flows),
                    patchlabel=node
                )
                
            # Finish diagram
            sankey.finish()
            
        except Exception as e:
            logger.error(f"Error creating Sankey diagram: {e}")
            
            # Fallback to standard graph visualization
            nx.draw_networkx(
                G, pos,
                with_labels=True,
                node_color="skyblue",
                node_size=1000,
                alpha=0.8,
                ax=ax
            )
            
            # Add edge labels
            edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
        
        # Set title
        ax.set_title("Token Movement Between Memory Levels")
        
        # Remove axis
        ax.axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def plot_movement_animation(self, figsize=(12, 8), max_frames=100):
        """
        Create an animation of token movements
        
        Args:
            figsize: Figure size (width, height)
            max_frames: Maximum number of frames in animation
            
        Returns:
            Matplotlib animation
        """
        try:
            from matplotlib.animation import FuncAnimation
        except ImportError:
            logger.error("Matplotlib animation support not available")
            return None
        
        # Sort movements by timestamp
        sorted_movements = sorted(
            self.token_movements,
            key=lambda x: x.get("timestamp", 0)
        )
        
        # Limit number of frames
        if len(sorted_movements) > max_frames:
            # Sample evenly
            indices = np.linspace(0, len(sorted_movements) - 1, max_frames, dtype=int)
            sorted_movements = [sorted_movements[i] for i in indices]
        
        # Set up figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up memory level positions
        memory_levels = {}
        for level in self.memory_levels:
            level_id = level.get("level_id", 0)
            level_name = f"L{level_id}"
            capacity = level.get("capacity", 0)
            memory_levels[level_name] = {
                "capacity": capacity,
                "tokens": 0,
                "x": level_id,
                "y": 0
            }
        
        # Function to draw a frame
        def draw_frame(frame_idx):
            ax.clear()
            
            # Get movement for this frame
            if frame_idx < len(sorted_movements):
                movement = sorted_movements[frame_idx]
                from_level = movement.get("from_level", "")
                to_level = movement.get("to_level", "")
                count = movement.get("count", 0)
                token_ids = movement.get("token_ids", [])
                timestamp = movement.get("timestamp", 0)
                
                # Update token counts
                if from_level in memory_levels:
                    memory_levels[from_level]["tokens"] -= count
                if to_level in memory_levels:
                    memory_levels[to_level]["tokens"] += count
                
                # Draw memory levels
                for level_name, level_info in memory_levels.items():
                    capacity = level_info["capacity"]
                    tokens = level_info["tokens"]
                    x = level_info["x"]
                    
                    # Draw level as rectangle
                    width = 0.8
                    height = math.log2(capacity + 1) / 10
                    
                    # Outer rectangle (capacity)
                    rect_outer = patches.Rectangle(
                        (x - width/2, 0),
                        width,
                        height,
                        linewidth=1,
                        edgecolor="blue",
                        facecolor="skyblue",
                        alpha=0.3
                    )
                    ax.add_patch(rect_outer)
                    
                    # Inner rectangle (tokens)
                    if capacity > 0:
                        inner_height = height * (tokens / capacity)
                        rect_inner = patches.Rectangle(
                            (x - width/2, 0),
                            width,
                            inner_height,
                            linewidth=1,
                            edgecolor="blue",
                            facecolor="skyblue",
                            alpha=0.7
                        )
                        ax.add_patch(rect_inner)
                    
                    # Add label
                    ax.text(
                        x,
                        height + 0.01,
                        f"{level_name}\n{tokens:,}/{capacity:,}",
                        ha="center",
                        va="bottom"
                    )
                
                # Draw arrow for movement
                if from_level in memory_levels and to_level in memory_levels:
                    from_x = memory_levels[from_level]["x"]
                    to_x = memory_levels[to_level]["x"]
                    
                    # Calculate y position based on level heights
                    from_height = math.log2(memory_levels[from_level]["capacity"] + 1) / 10
                    to_height = math.log2(memory_levels[to_level]["capacity"] + 1) / 10
                    
                    # Draw arrow
                    ax.annotate(
                        "",
                        xy=(to_x, to_height / 2),  # End point
                        xytext=(from_x, from_height / 2),  # Start point
                        arrowprops=dict(
                            arrowstyle="->",
                            color="red",
                            lw=2,
                            shrinkA=5,
                            shrinkB=5
                        )
                    )
                    
                    # Add movement label
                    mid_x = (from_x + to_x) / 2
                    mid_y = (from_height + to_height) / 4
                    
                    ax.text(
                        mid_x,
                        mid_y + 0.05,
                        f"{count:,} tokens",
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7)
                    )
                
                # Add timestamp
                ax.text(
                    0.02,
                    0.98,
                    f"Time: {timestamp:.2f}s",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7)
                )
                
                # Display token IDs if available and tokens are provided
                if token_ids and self.tokens:
                    token_str = ""
                    for token_id in token_ids[:5]:  # Show first 5 tokens
                        if token_id < len(self.tokens):
                            token_str += f"{self.tokens[token_id]}, "
                    
                    if len(token_ids) > 5:
                        token_str += f"... ({len(token_ids)} total)"
                    
                    ax.text(
                        0.02,
                        0.92,
                        f"Tokens: {token_str}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
                        fontsize=8
                    )
            
            # Set axis limits
            ax.set_xlim(0, len(memory_levels) + 1)
            ax.set_ylim(0, 1)
            
            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Set title
            ax.set_title(f"Token Movement Animation (Frame {frame_idx+1}/{len(sorted_movements)})")
        
        # Create animation
        anim = FuncAnimation(
            fig,
            draw_frame,
            frames=len(sorted_movements),
            interval=200,  # ms between frames
            blit=False
        )
        
        return anim
    
    def plot_interactive_movement(self):
        """
        Create an interactive visualization of token movements using Plotly
        
        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.error("Plotly not installed. Run 'pip install plotly'")
            return None
        
        # Prepare data for Sankey diagram
        sources = []
        targets = []
        values = []
        labels = []
        
        # Create node labels for memory levels
        for level in self.memory_levels:
            level_id = level.get("level_id", 0)
            level_name = f"L{level_id}"
            capacity = level.get("capacity", 0)
            tokens = level.get("tokens", 0)
            
            labels.append(f"{level_name} ({tokens:,}/{capacity:,})")
        
        # Create mapping of level names to indices
        level_to_idx = {f"L{level['level_id']}": i for i, level in enumerate(self.memory_levels)}
        
        # Add token movements as links
        for movement in self.token_movements:
            from_level = movement.get("from_level", "")
            to_level = movement.get("to_level", "")
            count = movement.get("count", 0)
            
            if from_level in level_to_idx and to_level in level_to_idx and count > 0:
                sources.append(level_to_idx[from_level])
                targets.append(level_to_idx[to_level])
                values.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="blue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Token Movement Between Memory Levels",
            font_size=10
        )
        
        return fig

#######################################
# UltraContext Visualization Dashboard
#######################################

class UltraContextDashboard:
    """Interactive dashboard for visualizing UltraContext"""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        token_data: Optional[Dict] = None,
        memory_data: Optional[Dict] = None,
        attention_data: Optional[Dict] = None,
        tokenizer = None
    ):
        """
        Initialize the UltraContext dashboard
        
        Args:
            config: UltraContext configuration
            token_data: Token importance and usage data
            memory_data: Memory hierarchy and movement data
            attention_data: Attention pattern data
            tokenizer: Optional tokenizer for decoding tokens
        """
        self.config = config or {}
        self.token_data = token_data or {}
        self.memory_data = memory_data or {}
        self.attention_data = attention_data or {}
        self.tokenizer = tokenizer
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), "ultracontext_dashboard")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_dashboard(self, output_file="dashboard.html"):
        """
        Generate an HTML dashboard
        
        Args:
            output_file: Output HTML file name
            
        Returns:
            Path to generated HTML file
        """
        # Create HTML content
        html_content = []
        
        # Add header
        html_content.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UltraContext Visualization Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    background-color: #333;
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                .section {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    padding: 20px;
                }
                .viz-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }
                .viz-item {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    padding: 10px;
                    max-width: 100%;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="header">
                <h1>UltraContext Visualization Dashboard</h1>
            </div>
            <div class="container">
        """)
        
        # Add configuration section
        if self.config:
            html_content.append("""
            <div class="section">
                <h2>Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
            """)
            
            for key, value in self.config.items():
                html_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                
            html_content.append("""
                </table>
            </div>
            """)
        
        # Add token importance visualization
        if "tokens" in self.token_data and "importance_scores" in self.token_data:
            try:
                tokens = self.token_data.get("tokens", [])
                importance_scores = self.token_data.get("importance_scores", [])
                attention_scores = self.token_data.get("attention_scores", [])
                usage_counts = self.token_data.get("usage_counts", [])
                positions = self.token_data.get("positions", [])
                
                # Create visualizer
                visualizer = TokenImportanceVisualizer(
                    tokens=tokens,
                    importance_scores=importance_scores,
                    attention_scores=attention_scores,
                    usage_counts=usage_counts,
                    positions=positions
                )
                
                # Generate interactive plot
                fig = visualizer.plot_interactive()
                if fig:
                    plt_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    html_content.append("""
                    <div class="section">
                        <h2>Token Importance</h2>
                        <div class="viz-container">
                            <div class="viz-item" style="width: 100%;">
                    """)
                    
                    html_content.append(plt_html)
                    
                    html_content.append("""
                            </div>
                        </div>
                    </div>
                    """)
            except Exception as e:
                logger.error(f"Error generating token importance visualization: {e}")
        
        # Add memory hierarchy visualization
        if "memory_levels" in self.memory_data:
            try:
                memory_levels = self.memory_data.get("memory_levels", [])
                tokens_in_memory = self.memory_data.get("tokens_in_memory", {})
                compression_stats = self.memory_data.get("compression_stats", {})
                
                # Create visualizer
                visualizer = ContextWindowVisualizer(
                    active_window_size=self.config.get("active_window_size", 4096),
                    sliding_window_size=self.config.get("sliding_window_size", 2048),
                    tokens_in_memory=tokens_in_memory,
                    compression_stats=compression_stats,
                    memory_levels=memory_levels
                )
                
                # Generate interactive plot
                fig = visualizer.plot_interactive_memory()
                if fig:
                    plt_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    html_content.append("""
                    <div class="section">
                        <h2>Memory Hierarchy</h2>
                        <div class="viz-container">
                            <div class="viz-item" style="width: 100%;">
                    """)
                    
                    html_content.append(plt_html)
                    
                    html_content.append("""
                            </div>
                        </div>
                    </div>
                    """)
            except Exception as e:
                logger.error(f"Error generating memory hierarchy visualization: {e}")
        
        # Add token movement visualization
        if "token_movements" in self.memory_data:
            try:
                memory_levels = self.memory_data.get("memory_levels", [])
                token_movements = self.memory_data.get("token_movements", [])
                tokens = self.token_data.get("tokens", [])
                
                # Create visualizer
                visualizer = TokenMovementVisualizer(
                    memory_levels=memory_levels,
                    token_movements=token_movements,
                    tokens=tokens
                )
                
                # Generate interactive plot
                fig = visualizer.plot_interactive_movement()
                if fig:
                    plt_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    html_content.append("""
                    <div class="section">
                        <h2>Token Movement</h2>
                        <div class="viz-container">
                            <div class="viz-item" style="width: 100%;">
                    """)
                    
                    html_content.append(plt_html)
                    
                    html_content.append("""
                            </div>
                        </div>
                    </div>
                    """)
            except Exception as e:
                logger.error(f"Error generating token movement visualization: {e}")
        
        # Add attention pattern visualization
        if "query_positions" in self.attention_data:
            try:
                query_positions = self.attention_data.get("query_positions", [])
                key_positions = self.attention_data.get("key_positions", [])
                attention_scores = self.attention_data.get("attention_scores", [])
                tokens = self.token_data.get("tokens", [])
                
                # Create visualizer
                visualizer = AttentionPatternVisualizer(
                    query_positions=query_positions,
                    key_positions=key_positions,
                    attention_scores=attention_scores,
                    tokens=tokens
                )
                
                # Generate interactive plot
                fig = visualizer.plot_interactive_attention()
                if fig:
                    plt_html = fig.to_html(full_html=False, include_plotlyjs=False)
                    
                    html_content.append("""
                    <div class="section">
                        <h2>Attention Patterns</h2>
                        <div class="viz-container">
                            <div class="viz-item" style="width: 100%;">
                    """)
                    
                    html_content.append(plt_html)
                    
                    html_content.append("""
                            </div>
                        </div>
                    </div>
                    """)
            except Exception as e:
                logger.error(f"Error generating attention pattern visualization: {e}")
        
        # Add footer
        html_content.append("""
            </div>
            <div class="footer" style="text-align: center; padding: 20px; color: #666;">
                <p>UltraContext Visualization Dashboard</p>
            </div>
        </body>
        </html>
        """)
        
        # Write HTML to file
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
            
        logger.info(f"Dashboard generated at {output_path}")
        
        return output_path
        
    def display_notebook(self):
        """
        Display dashboard in a Jupyter notebook
        
        Returns:
            IPython display object
        """
        try:
            from IPython.display import HTML, display
        except ImportError:
            logger.error("IPython not installed. Run in a Jupyter notebook or install IPython.")
            return None
        
        # Generate HTML path
        html_path = self.generate_dashboard(output_file="dashboard_notebook.html")
        
        # Read HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Display in notebook
        return display(HTML(html_content))
    
    def extract_data_from_model(self, model, input_ids=None, tokenizer=None):
        """
        Extract visualization data from a model with UltraContext
        
        Args:
            model: Model with UltraContext
            input_ids: Optional input token IDs
            tokenizer: Optional tokenizer for decoding tokens
            
        Returns:
            Self with extracted data
        """
        # Check if model has UltraContext
        if not hasattr(model, "ultra_context"):
            logger.error("Model does not have UltraContext")
            return self
            
        # Get UltraContext wrapper
        wrapper = model.ultra_context
        
        # Extract configuration
        if hasattr(wrapper, "config"):
            self.config = {
                "max_context_length": wrapper.config.max_context_length,
                "active_window_size": wrapper.config.active_window_size,
                "sliding_window_size": wrapper.config.sliding_window_size,
                "use_hierarchical_memory": wrapper.config.use_hierarchical_memory,
                "memory_compression_ratio": wrapper.config.memory_compression_ratio,
                "use_token_compression": wrapper.config.use_token_compression,
                "compression_ratio": wrapper.config.compression_ratio,
                "integration_mode": wrapper.config.integration_mode,
                "position_encoding": wrapper.config.position_encoding
            }
        
        # Extract token data
        if hasattr(wrapper, "context_state"):
            state = wrapper.context_state
            
            # Get positions
            positions = list(state.token_usage.keys())
            
            # Get attention scores and usage counts
            attention_scores = [state.attention_scores.get(pos, 0.0) for pos in positions]
            usage_counts = [state.token_usage.get(pos, 0) for pos in positions]
            
            # Generate importance scores
            importance_scores = []
            for pos in positions:
                # Combine attention and usage for importance
                attn = state.attention_scores.get(pos, 0.0)
                usage = state.token_usage.get(pos, 0)
                
                # Normalize usage
                max_usage = max(state.token_usage.values()) if state.token_usage else 1
                norm_usage = usage / max_usage if max_usage > 0 else 0
                
                # Weighted combination for importance
                importance = 0.7 * attn + 0.3 * norm_usage
                importance_scores.append(importance)
            
            # Decode tokens if tokenizer is available
            tokens = []
            if tokenizer is not None and input_ids is not None:
                for pos in positions:
                    if 0 <= pos < len(input_ids):
                        token_id = input_ids[pos]
                        token = tokenizer.decode([token_id])
                        tokens.append(token)
                    else:
                        tokens.append(f"[{pos}]")
            else:
                tokens = [f"Token {pos}" for pos in positions]
            
            # Store token data
            self.token_data = {
                "tokens": tokens,
                "positions": positions,
                "importance_scores": importance_scores,
                "attention_scores": attention_scores,
                "usage_counts": usage_counts
            }
        
        # Extract memory data
        if hasattr(wrapper, "memory_manager"):
            memory_manager = wrapper.memory_manager
            
            # Get memory levels
            memory_levels = []
            if hasattr(memory_manager, "l1"):
                memory_levels.append({
                    "level_id": 1,
                    "capacity": memory_manager.l1.capacity,
                    "tokens": len(memory_manager.l1.tokens),
                    "retrieval_cost": memory_manager.l1.retrieval_cost,
                    "storage_cost": memory_manager.l1.storage_cost,
                    "hit_rate": memory_manager.l1.hit_rate
                })
            
            if hasattr(memory_manager, "l2"):
                memory_levels.append({
                    "level_id": 2,
                    "capacity": memory_manager.l2.capacity,
                    "tokens": len(memory_manager.l2.tokens),
                    "retrieval_cost": memory_manager.l2.retrieval_cost,
                    "storage_cost": memory_manager.l2.storage_cost,
                    "hit_rate": memory_manager.l2.hit_rate
                })
                
            if hasattr(memory_manager, "l3"):
                memory_levels.append({
                    "level_id": 3,
                    "capacity": memory_manager.l3.capacity,
                    "tokens": len(memory_manager.l3.tokens),
                    "retrieval_cost": memory_manager.l3.retrieval_cost,
                    "storage_cost": memory_manager.l3.storage_cost,
                    "hit_rate": memory_manager.l3.hit_rate
                })
            
            # Get tokens in memory
            tokens_in_memory = {
                "L1": len(memory_manager.l1.tokens) if hasattr(memory_manager, "l1") else 0,
                "L2": len(memory_manager.l2.tokens) if hasattr(memory_manager, "l2") else 0,
                "L3": len(memory_manager.l3.tokens) if hasattr(memory_manager, "l3") else 0
            }
            
            # Get compression stats if available
            compression_stats = {}
            if hasattr(wrapper, "token_compressor") and hasattr(wrapper.token_compressor, "get_stats"):
                compression_stats = wrapper.token_compressor.get_stats()
            
            # Store memory data
            self.memory_data = {
                "memory_levels": memory_levels,
                "tokens_in_memory": tokens_in_memory,
                "compression_stats": compression_stats,
                "token_movements": []  # Would need to track these during execution
            }
        
        # Extract attention data
        if hasattr(wrapper, "context_state"):
            state = wrapper.context_state
            positions = list(state.token_usage.keys())
            
            # Create attention matrix (not available directly, approximating)
            # Ideally, we would extract this from the model's attention layers
            query_positions = positions
            key_positions = positions
            
            # Initialize with identity matrix as fallback
            attention_scores = np.eye(len(positions))
            
            # Store attention data
            self.attention_data = {
                "query_positions": query_positions,
                "key_positions": key_positions,
                "attention_scores": attention_scores.tolist()
            }
        
        return self

# Main function for CLI usage
def main():
    """Command-line interface for the visualizer"""
    parser = argparse.ArgumentParser(description="UltraContext Visualization Tools")
    parser.add_argument("--config", type=str, help="Path to UltraContext configuration JSON")
    parser.add_argument("--token_data", type=str, help="Path to token data JSON")
    parser.add_argument("--memory_data", type=str, help="Path to memory data JSON")
    parser.add_argument("--attention_data", type=str, help="Path to attention data JSON")
    parser.add_argument("--output", type=str, default="dashboard.html", help="Output HTML file")
    args = parser.parse_args()
    
    # Load data from files
    config = None
    token_data = None
    memory_data = None
    attention_data = None
    
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
            
    if args.token_data:
        with open(args.token_data, "r") as f:
            token_data = json.load(f)
            
    if args.memory_data:
        with open(args.memory_data, "r") as f:
            memory_data = json.load(f)
            
    if args.attention_data:
        with open(args.attention_data, "r") as f:
            attention_data = json.load(f)
    
    # Create dashboard
    dashboard = UltraContextDashboard(
        config=config,
        token_data=token_data,
        memory_data=memory_data,
        attention_data=attention_data
    )
    
    # Generate dashboard
    output_path = dashboard.generate_dashboard(output_file=args.output)
    
    print(f"Dashboard generated at: {output_path}")

if __name__ == "__main__":
    main()
