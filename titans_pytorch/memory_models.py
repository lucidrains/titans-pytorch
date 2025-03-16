import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
from functools import partial
import math

from einops import rearrange, repeat

# Advanced optimizations
TORCH_2_AVAILABLE = torch.__version__ >= "2.0.0"
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
XFORMERS_AVAILABLE = False

# Try to import xformers for efficient attention
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
except ImportError:
    pass

# Performance configuration class
class PerformanceConfig:
    """Centralized configuration for performance optimizations"""
    def __init__(
        self,
        use_mixed_precision: bool = True,
        use_gradient_checkpointing: bool = False,
        use_xformers: bool = XFORMERS_AVAILABLE,
        use_torch_compile: bool = TORCH_COMPILE_AVAILABLE,
        compile_mode: str = "reduce-overhead",  # Options: 'max-autotune', 'reduce-overhead'
        quantization: Optional[str] = None,  # None, 'dynamic', 'static', 'qat'
        optimize_memory: bool = True,
    ):
        self.use_mixed_precision = use_mixed_precision
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE
        self.use_torch_compile = use_torch_compile and TORCH_COMPILE_AVAILABLE
        self.compile_mode = compile_mode
        self.quantization = quantization
        self.optimize_memory = optimize_memory

# Set default performance configuration
DEFAULT_PERF_CONFIG = PerformanceConfig()

# Optimized activation functions
class FusedActivations:
    """Optimized activation functions with fused operations"""
    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)
    
    @staticmethod
    def gelu_new(x):
        """Approximation of GELU that's faster than the exact formula"""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
    @staticmethod
    def quick_gelu(x):
        """Even faster GELU approximation"""
        return x * torch.sigmoid(1.702 * x)
    
    @staticmethod
    def squared_relu(x):
        """Squared ReLU - efficient and works well in transformers"""
        return torch.pow(F.relu(x), 2)

# functions
def l2norm(t, dim = -1, eps = 1e-12):
    """Optimized L2 normalization with numerical stability"""
    return F.normalize(t, p=2, dim=dim, eps=eps)

# norms
class LayerNorm(Module):
    """Enhanced version of the original LayerNorm with memory optimizations"""
    def __init__(
        self,
        dim,
        elementwise_affine: bool = False,
        eps: float = 1e-5,
        bias: bool = True,
        device=None,
        dtype=None,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # Keep original behavior while adding performance enhancements
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Original behavior: use parameter and affine=False
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=eps, **factory_kwargs)
        self.gamma = Parameter(torch.zeros(dim, **factory_kwargs))
        
        # Apply optimizations if mixed precision is enabled
        self.use_mixed_precision = perf_config.use_mixed_precision
        
    def forward(self, x):
        input_dtype = x.dtype
        gamma = self.gamma
        
        # Handle broadcasting as in original implementation
        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')
        
        # Memory-efficient computation with correct type handling
        if self.use_mixed_precision:
            if input_dtype != torch.float32:
                # Compute in fp32 for stability then cast back
                x_float = x.float()
                normalized = self.ln(x_float)
                result = normalized * (gamma + 1.)
                return result.to(input_dtype)
        
        # Standard path - keep original behavior
        return self.ln(x) * (gamma + 1.)

# norm + residual wrapper, as used in original TTT paper
class ResidualNorm(Module):
    """Enhanced version of ResidualNorm with gradient checkpointing and mixed precision support"""
    def __init__(
        self,
        dim,
        model: Module,
        pre_norm: bool = False,
        dropout: float = 0.0,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG
    ):
        super().__init__()
        self.norm = LayerNorm(dim, perf_config=perf_config)
        self.model = model
        self.pre_norm = pre_norm  # Added pre_norm option but defaults to post-norm for compatibility
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.use_checkpointing = perf_config.use_gradient_checkpointing
    
    def _checkpoint_forward(self, fn, *args, **kwargs):
        if self.use_checkpointing and any(p.requires_grad for p in self.parameters()):
            return checkpoint(fn, *args, **kwargs)
        return fn(*args, **kwargs)
        
    def forward(self, x):
        if self.pre_norm:
            # Pre-norm architecture (better gradient flow)
            normalized = self.norm(x)
            out = self._checkpoint_forward(self.model, normalized)
            if self.dropout is not None:
                out = self.dropout(out)
            return x + out
        else:
            # Post-norm architecture (original behavior)
            out = self._checkpoint_forward(self.model, x)
            if self.dropout is not None:
                out = self.dropout(out)
            return self.norm(out) + x

# memory mlp proposed in TTT
class MemoryMLP(Module):
    """Enhanced MemoryMLP with performance optimizations"""
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.,
        dropout: float = 0.0,
        activation = F.gelu,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        bias: bool = False
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)
        
        self.weights = ParameterList([
            Parameter(torch.randn(dim_in, dim_out)) 
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ])
        
        # Enhanced initialization
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        
        # Add regularization support
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = activation
        
        # Apply mixed precision if requested
        self.use_mixed_precision = perf_config.use_mixed_precision
        
        # Use gradient checkpointing for memory efficiency
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Apply torch.compile if available
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )

    def forward(self, x):
        input_dtype = x.dtype
        
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0
            
            if not is_first:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            
            if self.use_checkpointing and self.training:
                x = checkpoint(lambda x, w: x @ w, x, weight)
            else:
                x = x @ weight
        
        # Ensure output matches input dtype for mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x

# memory mlp, but with gated residual + final projection
class GatedResidualMemoryMLP(Module):
    """Enhanced GatedResidualMemoryMLP with performance optimizations"""
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.,
        dropout: float = 0.0,
        activation = F.gelu,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        bias: bool = False
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])
        
        self.final_proj = Parameter(torch.randn(dim, dim))
        
        # Enhanced initialization
        for param in self.parameters():
            nn.init.xavier_uniform_(param)
        
        # Add regularization support
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = activation
        
        # Apply mixed precision if requested
        self.use_mixed_precision = perf_config.use_mixed_precision
        
        # Use gradient checkpointing for memory efficiency
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Apply torch.compile if available
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )

    def forward(self, x):
        input_dtype = x.dtype
        
        for weight1, weight2, to_gates in self.weights:
            res = x
            
            if self.use_checkpointing and self.training:
                hidden = checkpoint(lambda x, w: x @ w, x, weight1)
                hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                branch_out = checkpoint(lambda h, w: h @ w, hidden, weight2)
                gates = checkpoint(lambda bo, r, w: cat((bo, r), dim=-1) @ w, branch_out, res, to_gates)
            else:
                hidden = x @ weight1
                hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)
                branch_out = hidden @ weight2
                gates = cat((branch_out, res), dim=-1) @ to_gates
            
            # Use lerp as in original (equivalent to gates.sigmoid() * branch_out + (1 - gates.sigmoid()) * res)
            x = res.lerp(branch_out, gates.sigmoid())
        
        # Final projection with checkpointing if enabled
        if self.use_checkpointing and self.training:
            x = checkpoint(lambda x, w: x @ w, x, self.final_proj)
        else:
            x = x @ self.final_proj
        
        # Ensure output matches input dtype for mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x

# memory mlp with factorized weights
class FactorizedMemoryMLP(Module):
    """Enhanced FactorizedMemoryMLP with performance optimizations"""
    def __init__(
        self,
        dim,
        depth,
        k = 32,
        dropout: float = 0.0,
        activation = F.gelu,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        bias: bool = False
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])
        
        # Enhanced initialization
        for weight1, weight2 in self.weights:
            bound1 = 1 / math.sqrt(weight1.size(1))
            bound2 = 1 / math.sqrt(weight2.size(1))
            nn.init.uniform_(weight1, -bound1, bound1)
            nn.init.uniform_(weight2, -bound2, bound2)
        
        # Add regularization support
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = activation
        
        # Apply mixed precision if requested
        self.use_mixed_precision = perf_config.use_mixed_precision
        
        # Use gradient checkpointing for memory efficiency
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Fused implementation options
        self.use_fused_ops = hasattr(torch, 'compile') and perf_config.use_torch_compile
        
        # Apply torch.compile if available
        if self.use_fused_ops:
            self._fused_forward = torch.compile(
                self._raw_forward, 
                mode=perf_config.compile_mode
            )

    def _raw_forward(self, x, weight1, weight2, activation=None, dropout=None):
        """Raw implementation of the forward pass for compilation"""
        x = x @ weight1
        if activation is not None:
            x = activation(x)
        if dropout is not None:
            x = dropout(x)
        x = x @ weight2
        return x
    
    def forward(self, x):
        input_dtype = x.dtype
        
        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0
            
            if not is_first:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            
            # Use fused operations if available
            if self.use_fused_ops and not self.training:
                act = None if is_first else self.activation
                x = self._fused_forward(x, weight1, weight2, act, self.dropout)
            else:
                # Standard computation with optional checkpointing
                if self.use_checkpointing and self.training:
                    x = checkpoint(lambda x, w: x @ w, x, weight1)
                    if not is_first:
                        x = self.activation(x)
                        if self.dropout is not None:
                            x = self.dropout(x)
                    x = checkpoint(lambda x, w: x @ w, x, weight2)
                else:
                    x = x @ weight1
                    if not is_first:
                        x = self.activation(x)
                        if self.dropout is not None:
                            x = self.dropout(x)
                    x = x @ weight2
        
        # Ensure output matches input dtype for mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x

# an MLP modelled after the popular swiglu ff in modern transformers
class MemorySwiGluMLP(Module):
    """Enhanced MemorySwiGluMLP with performance optimizations"""
    def __init__(
        self,
        dim,
        depth = 1,
        expansion_factor = 4.,
        dropout: float = 0.0,
        activation = F.gelu,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        bias: bool = False
    ):
        super().__init__()
        
        dim_inner = int(dim * expansion_factor * 2 / 3)
        
        weights = []
        
        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),
                Parameter(torch.randn(dim_inner, dim)),
            ]))
        
        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim, perf_config=perf_config)
        
        # Add activation function and dropout
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Apply mixed precision if requested
        self.use_mixed_precision = perf_config.use_mixed_precision
        
        # Use gradient checkpointing for memory efficiency
        self.use_checkpointing = perf_config.use_gradient_checkpointing
        
        # Better initialization
        for w1, w2 in self.weights:
            # Scale for SwiGLU
            nn.init.xavier_uniform_(w1, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(w2)
        
        # Apply torch.compile if available
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )

    def forward(self, x):
        input_dtype = x.dtype
        
        for w1, w2 in self.weights:
            residual = x
            
            if self.use_checkpointing and self.training:
                proj = checkpoint(lambda x, w: x @ w, x, w1)
                x, gates = proj.chunk(2, dim=-1)
                gates = self.activation(gates)
                x = x * gates
                if self.dropout is not None:
                    x = self.dropout(x)
                x = checkpoint(lambda x, w: x @ w, x, w2)
            else:
                x, gates = (x @ w1).chunk(2, dim=-1)
                gates = self.activation(gates)
                x = x * gates
                if self.dropout is not None:
                    x = self.dropout(x)
                x = x @ w2
            
            x = x + residual
        
        x = self.norm(x)
        
        # Ensure output matches input dtype for mixed precision
        if self.use_mixed_precision:
            x = x.to(input_dtype)
            
        return x

# improvised attention as memory module
class MemoryAttention(Module):
    """Enhanced MemoryAttention with performance optimizations"""
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.,
        causal = True,
        num_heads = 1,
        dropout: float = 0.0,
        perf_config: PerformanceConfig = DEFAULT_PERF_CONFIG,
        bias: bool = False,
        use_flash = True
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_heads = num_heads
        dim_ff_hidden = int(dim * expansion_factor)
        
        # Original weights structure for drop-in compatibility
        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)),       # queries
            Parameter(torch.randn(dim, dim)),       # keys
            Parameter(torch.randn(dim, dim)),       # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])
        
        # Better initialization for attention 
        nn.init.xavier_uniform_(self.weights[0], gain=1.0)  # Q
        nn.init.xavier_uniform_(self.weights[1], gain=1.0)  # K
        nn.init.xavier_uniform_(self.weights[2], gain=1.0)  # V 
        nn.init.xavier_uniform_(self.weights[3], gain=1.0)  # FF1
        nn.init.xavier_uniform_(self.weights[4], gain=1.0)  # FF2
        
        # Performance enhancements
        self.dropout = dropout
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        self.use_xformers = perf_config.use_xformers and XFORMERS_AVAILABLE
        self.use_mixed_precision = perf_config.use_mixed_precision
        
        # Apply torch.compile if available
        if perf_config.use_torch_compile and TORCH_COMPILE_AVAILABLE:
            self.forward = torch.compile(
                self.forward,
                mode=perf_config.compile_mode
            )

    def forward(self, x):
        input_dtype = x.dtype
        wq, wk, wv, ffw1, ffw2 = self.weights
        
        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv
        
        # Use the most efficient attention implementation available
        if self.use_xformers:
            # Reshape for multi-head if num_heads > 1
            if self.num_heads > 1:
                batch_size, seq_len, _ = x.shape
                head_dim = q.size(-1) // self.num_heads
                
                # Reshape q, k, v for multi-head attention
                q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                
                # Use xFormers memory-efficient attention
                q, k, v = map(lambda t: t.contiguous(), (q, k, v))
                attn_out = xformers.ops.memory_efficient_attention(
                    q, k, v,
                    attn_bias=None,
                    p=self.dropout if self.training else 0.0,
                    op=None,  # let xFormers pick the best algorithm
                    scale=self.scale
                )
                
                # Reshape back
                attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
            else:
                # For single head, use xFormers with reshaped tensor
                batch_size, seq_len, _ = x.shape
                q_reshaped = q.view(batch_size, seq_len, 1, -1).transpose(1, 2)
                k_reshaped = k.view(batch_size, seq_len, 1, -1).transpose(1, 2)
                v_reshaped = v.view(batch_size, seq_len, 1, -1).transpose(1, 2)
                
                q_reshaped, k_reshaped, v_reshaped = map(
                    lambda t: t.contiguous(), (q_reshaped, k_reshaped, v_reshaped)
                )
                
                attn_out = xformers.ops.memory_efficient_attention(
                    q_reshaped, k_reshaped, v_reshaped,
                    attn_bias=None,
                    p=self.dropout if self.training else 0.0,
                    op=None,
                    scale=self.scale
                )
                
                attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        elif self.use_flash:
            # Use Flash Attention via PyTorch's SDPA
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
                scale=self.scale
            ).squeeze(1)
        else:
            # Standard attention calculation
            sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if needed
            if self.causal:
                batch_size, seq_len, _ = x.shape
                mask = torch.ones(
                    (seq_len, seq_len), 
                    dtype=torch.bool, 
                    device=x.device
                ).triu_(diagonal=1)
                sim.masked_fill_(mask, -torch.finfo(sim.dtype).max)
            
            # Apply attention
            attn = F.softmax(sim, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn, v)
        
        # Parallel attention + feedforward block as in original
        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2
        
        result = attn_out + ff_out
        
        # Ensure output matches input dtype for mixed precision
        if self.use_mixed_precision:
            result = result.to(input_dtype)
        
        return result

# Factory function to create optimized networks
def create_optimized_memory_network(
    dim,
    depth,
    attention_type='standard',  # 'standard', 'efficient', 'linear'
    mlp_type='standard',        # 'standard', 'factorized', 'gated', 'swiglu'
    perf_config=None
):
    """Create an optimized memory network with the specified configuration"""
    perf_config = perf_config or DEFAULT_PERF_CONFIG
    
    # Create attention module
    if attention_type == 'efficient':
        attention = MemoryAttention(
            dim=dim,
            num_heads=8,
            use_flash=True,
            causal=True,
            perf_config=perf_config
        )
    elif attention_type == 'linear':
        attention = MemoryAttention(
            dim=dim,
            num_heads=1,
            use_flash=False,
            causal=True,
            perf_config=perf_config
        )
    else:
        attention = MemoryAttention(
            dim=dim,
            num_heads=1,
            perf_config=perf_config
        )
    
    # Create MLP module
    if mlp_type == 'factorized':
        mlp = FactorizedMemoryMLP(
            dim=dim,
            depth=depth,
            perf_config=perf_config
        )
    elif mlp_type == 'gated':
        mlp = GatedResidualMemoryMLP(
            dim=dim,
            depth=depth,
            perf_config=perf_config
        )
    elif mlp_type == 'swiglu':
        mlp = MemorySwiGluMLP(
            dim=dim,
            depth=depth,
            perf_config=perf_config
        )
    else:
        mlp = MemoryMLP(
            dim=dim,
            depth=depth,
            perf_config=perf_config
        )
    
    # Wrap with residual connections
    attn_block = ResidualNorm(
        dim=dim,
        model=attention,
        pre_norm=True,
        perf_config=perf_config
    )
    
    mlp_block = ResidualNorm(
        dim=dim,
        model=mlp,
        pre_norm=True,
        perf_config=perf_config
    )
    
    # Create a sequential container
    network = nn.Sequential(attn_block, mlp_block)
    
    return network
