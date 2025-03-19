import torch
import torch.nn as nn
import logging
import re
import copy
import types
import importlib.util
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

logger = logging.getLogger("ultracontext.integration")

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
        ultra_context=None,  # UltraContext instance
        position_encoding=None,  # Optional position encoding module
        perf_config=None
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
        
        # Use provided UltraContext
        self.ultra_context = ultra_context
        
        # Use provided position encoding if available
        self.position_encoding = position_encoding
        
        # Performance config
        from ultracontext.core import DEFAULT_PERF_CONFIG
        self.perf_config = perf_config or DEFAULT_PERF_CONFIG
        
        # If in replacement mode, create our own attention
        if integration_mode == "replacement" or original_attn is None:
            # Define projections
            self.to_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_k = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_v = nn.Linear(dim, num_heads * self.head_dim, bias=False)
            self.to_out = nn.Linear(num_heads * self.head_dim, dim, bias=False)
            
            # Define attention implementation
            if self.perf_config.use_flash_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                self.attn_fn = lambda q, k, v, mask: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout
                )
            else:
                self.attn_fn = self._standard_attention
                self.attn_dropout = nn.Dropout(dropout)
                
        # For streaming in extension mode with very long contexts
        if integration_mode == "extension" and max_context_length > window_size * 10:
            try:
                # Try to import streaming attention
                from ultracontext.attention import StreamingAttention
                
                # Create streaming attention for the extended context
                self.streaming_attn = StreamingAttention(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=self.head_dim,
                    window_size=window_size,
                    max_kv_cache=min(1_000_000, max_context_length),
                    dropout=dropout,
                    causal=True,
                    perf_config=self.perf_config
                )
            except ImportError:
                logger.warning("StreamingAttention not available, falling back to standard attention")
                self.streaming_attn = None
            
        # Apply torch.compile if requested
        if self.perf_config.use_torch_compile and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(
                    self.forward,
                    mode=self.perf_config.compile_mode,
                    fullgraph=False  # State changes are not compatible with fullgraph
                )
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile to attention: {e}")
            
    def _standard_attention(self, q, k, v, mask=None):
        """Standard attention implementation"""
        scale = q.shape[-1] ** -0.5
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            
        # Apply attention
        attn = torch.nn.functional.softmax(scores, dim=-1)
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
        if ultra_context is None or self.ultra_context is None:
            return None
            
        # Extract context embeddings
        context_emb, context_pos = ultra_context
        
        if context_emb is None:
            return None
            
        if hasattr(self, "streaming_attn") and self.streaming_attn is not None:
            # Use streaming attention for very long contexts
            # Combine current input with context
            combined = torch.cat([context_emb, x], dim=1)
            
            # Process with streaming attention
            out = self.streaming_attn(combined)
            
            # Extract only the current input part
            return out[:, -x.size(1):]
        else:
            # Use context processor from UltraContext
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
            extended_attn = self.ultra_context.context_processor(x)
            
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
        if hasattr(self.position_encoding, "apply_rotary_to_query_key") and positions is not None:
            q, k = self.position_encoding.apply_rotary_to_query_key(q, k, positions)
            
        # Apply relative position encoding if available
        rel_pos_bias = None
        if hasattr(self.position_encoding, "get_rel_pos_embeddings") and positions is not None:
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
        # Create hybrid gate if needed
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


class ModelIntegrator:
    """
    Utility for integrating UltraContext with various model architectures
    """
    def __init__(self, ultra_context, config):
        """
        Initialize integrator
        
        Args:
            ultra_context: UltraContext instance
            config: UltraContext configuration
        """
        self.ultra_context = ultra_context
        self.config = config
        
    def integrate_with_model(self, model):
        """
        Integrate UltraContext with a model
        
        Args:
            model: Model to integrate with
            
        Returns:
            Integrated model
        """
        # Detect model type and choose appropriate integration
        if self._is_huggingface_model(model):
            return self._integrate_with_hf_model(model)
        elif self._is_pytorch_model(model):
            return self._integrate_with_pytorch_model(model)
        else:
            logger.warning("Unknown model type, attempting generic integration")
            return self._integrate_with_generic_model(model)
            
    def _integrate_with_hf_model(self, model):
        """Integrate with a Hugging Face Transformers model"""
        logger.info("Integrating with Hugging Face Transformers model")
        
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
        elif "gpt" in model_type.lower() or "opt" in model_type.lower():
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
        self._replace_attention_layers(model, attention_layer_path, position_encoding_path)
        
        # Patch forward method for HF models
        self._patch_hf_forward(model)
        
        return model
        
    def _integrate_with_pytorch_model(self, model):
        """Integrate with a PyTorch model"""
        logger.info("Integrating with PyTorch model")
        
        # Try to detect attention layers
        attention_layer_path = "auto"
        
        # Integrate UltraContext
        self._replace_attention_layers(model, attention_layer_path)
        
        # Patch forward method
        self._patch_pytorch_forward(model)
        
        return model
        
    def _integrate_with_generic_model(self, model):
        """Integrate with a generic model"""
        logger.info("Attempting generic integration")
        
        # Try to find attention layers
        attention_layers = self._find_attention_layers(model)
        
        if not attention_layers:
            logger.warning("No attention layers found, integration may not work properly")
            
        # Replace attention layers
        for name, layer in attention_layers:
            self._replace_single_attention(model, name, layer)
            
        # Patch forward method
        self._patch_generic_forward(model)
        
        return model
        
    def _find_attention_layers(self, model):
        """Find attention layers in model"""
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
        
    def _replace_attention_layers(self, model, attention_layer_path, position_encoding_path=None):
        """Replace attention layers in model"""
        # Find attention layers
        if attention_layer_path == "auto":
            attention_layers = self._find_attention_layers(model)
        else:
            attention_layers = self._get_layers_by_path(model, attention_layer_path)
            
        # Find position encoding if specified
        position_encoding = None
        if position_encoding_path:
            position_encodings = self._get_layers_by_path(model, position_encoding_path)
            if position_encodings:
                position_encoding = position_encodings[0][1]
                
        # Default to UltraContext's position encoding if not found
        if position_encoding is None:
            position_encoding = self.ultra_context.position_encoding
            
        # Replace attention layers
        for name, layer in attention_layers:
            self._replace_single_attention(model, name, layer, position_encoding)
            
        logger.info(f"Replaced {len(attention_layers)} attention layers")
        
        # Add UltraContext to model
        model.ultra_context = self.ultra_context
        
        # Add helper methods
        model.set_ultracontext = self.ultra_context.set_context
        model.get_ultracontext = self.ultra_context.get_context
        model.clear_ultracontext = self.ultra_context.clear_context
        
    def _replace_single_attention(self, model, name, layer, position_encoding=None):
        """Replace a single attention layer"""
        logger.debug(f"Replacing attention layer: {name}")
        
        # Get parent module and attribute name
        if '.' in name:
            parent_name, attr_name = name.rsplit('.', 1)
            parent = self._get_module_by_path(model, parent_name)
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
                dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                original_attn=layer,
                integration_mode=self.config.integration_mode,
                window_size=self.config.sliding_window_size,
                max_context_length=self.config.max_context_length,
                ultra_context=self.ultra_context,
                position_encoding=position_encoding,
                perf_config=self.config.perf_config
            )
            
            # Replace in list
            module_list[idx] = new_attn
        else:
            # Create new attention layer
            new_attn = UltraContextAttention(
                dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                original_attn=layer,
                integration_mode=self.config.integration_mode,
                window_size=self.config.sliding_window_size,
                max_context_length=self.config.max_context_length,
                ultra_context=self.ultra_context,
                position_encoding=position_encoding,
                perf_config=self.config.perf_config
            )
            
            # Replace attribute
            setattr(parent, attr_name, new_attn)
            
    def _get_layers_by_path(self, model, path):
        """Get layers by path specification"""
        if path is None:
            return []
            
        layers = []
        
        # Handle multiple paths separated by commas
        if ',' in path:
            paths = path.split(',')
            for p in paths:
                layers.extend(self._get_layers_by_path(model, p.strip()))
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
            
    def _get_module_by_path(self, model, path):
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
        
    def _patch_hf_forward(self, model):
        """Patch forward method for Hugging Face models"""
        # Store original forward method
        if not hasattr(model, "_original_forward"):
            model._original_forward = model.forward
            
        # Create new forward method
        def ultracontext_forward(self, *args, **kwargs):
            # Extract input embeddings
            input_embeddings = self._extract_input_embeddings(args, kwargs)
            
            if input_embeddings is not None:
                # This is a standard forward pass with input embeddings
                batch_size, seq_len, _ = input_embeddings.shape
                
                # Determine if this is prefill or extension phase
                is_prefill = seq_len > 1
                
                # Process with UltraContext
                if is_prefill:
                    # Update context with new input
                    self.ultra_context.prefill(input_embeddings)
                else:
                    # Extending with new token
                    self.ultra_context.extend(input_embeddings)
                
                # Get UltraContext for attention layers
                ctx_emb, ctx_pos = self.ultra_context.get_context()
                
                # Store context in kwargs for attention layers
                kwargs["ultra_context"] = (ctx_emb, ctx_pos)
            
            # Run original forward
            return self._original_forward(*args, **kwargs)
            
        # Add helper method to extract input embeddings
        def _extract_input_embeddings(self, args, kwargs):
            """Extract input embeddings from arguments"""
            # Check common kwargs for embeddings
            emb_names = ["inputs_embeds", "encoder_outputs", "hidden_states"]
            for name in emb_names:
                if name in kwargs and kwargs[name] is not None:
                    if name == "encoder_outputs" and isinstance(kwargs[name], tuple):
                        # For encoder-decoder models
                        return kwargs[name][0]
                    else:
                        return kwargs[name]
            
            # Check for input_ids and convert to embeddings
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                if hasattr(self, "get_input_embeddings"):
                    return self.get_input_embeddings()(kwargs["input_ids"])
            
            return None
            
        # Attach methods to model
        model.forward = types.MethodType(ultracontext_forward, model)
        model._extract_input_embeddings = types.MethodType(_extract_input_embeddings, model)
        
    def _patch_pytorch_forward(self, model):
        """Patch forward method for PyTorch models"""
        # Similar to HF but with more generic handling
        if not hasattr(model, "_original_forward"):
            model._original_forward = model.forward
            
        def ultracontext_forward(self, *args, **kwargs):
            # Try to extract input
            x = None
            
            # Check kwargs first
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor) and value.dim() == 3:
                    x = value
                    break
                    
            # If not found in kwargs, check positional args
            if x is None and args:
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dim() == 3:
                        x = arg
                        break
            
            # Process with UltraContext if we found input
            if x is not None:
                batch_size, seq_len, _ = x.shape
                
                # Check if this is prefill or extension
                is_prefill = seq_len > 1
                
                # Process with UltraContext
                if is_prefill:
                    self.ultra_context.prefill(x)
                else:
                    self.ultra_context.extend(x)
                
                # Get context
                ctx_emb, ctx_pos = self.ultra_context.get_context()
                
                # Add to kwargs
                kwargs["ultra_context"] = (ctx_emb, ctx_pos)
            
            # Run original forward
            return self._original_forward(*args, **kwargs)
            
        # Attach method to model
        model.forward = types.MethodType(ultracontext_forward, model)
        
    def _patch_generic_forward(self, model):
        """Patch forward method for generic models"""
        # This is the most basic patching that just adds ultra_context to kwargs
        if not hasattr(model, "_original_forward"):
            model._original_forward = model.forward
            
        def ultracontext_forward(self, *args, **kwargs):
            # Always include ultra_context in kwargs
            ctx_emb, ctx_pos = self.ultra_context.get_context()
            kwargs["ultra_context"] = (ctx_emb, ctx_pos)
            
            # Run original forward
            return self._original_forward(*args, **kwargs)
            
        # Attach method to model
        model.forward = types.MethodType(ultracontext_forward, model)
        
    def _is_huggingface_model(self, model):
        """Check if model is from Hugging Face"""
        # Check for transformers module
        transformers_available = importlib.util.find_spec("transformers") is not None
        
        if not transformers_available:
            return False
            
        # Check for common Hugging Face model attributes
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            return True
            
        # Check model class name
        model_class = model.__class__.__name__
        hf_prefixes = ["GPT", "Llama", "T5", "Bert", "Falcon", "Mistral", "RoBERTa", "Transformer"]
        
        return any(prefix in model_class for prefix in hf_prefixes)
        
    def _is_pytorch_model(self, model):
        """Check if model is a PyTorch model"""
        # All models should be PyTorch models, but this checks for specific PyTorch structure
        return isinstance(model, nn.Module) and not self._is_huggingface_model(model)


# Specific Hugging Face integration
def integrate_with_hf_transformers(model, config, preprocessor=None):
    """
    Integrate UltraContext with Hugging Face transformers
    
    This function provides specialized integration for HF models,
    including tokenization and generation integration.
    
    Args:
        model: Hugging Face model
        config: UltraContext configuration
        preprocessor: Optional preprocessor for inputs
        
    Returns:
        Integrated model and optionally modified tokenizer
    """
    from ultracontext import UltraContext
    
    # Create UltraContext
    ultra_ctx = UltraContext(model, config)
    
    # Integrate with model
    model = ultra_ctx.integrate()
    
    # Try to get tokenizer and modify if needed
    if preprocessor and hasattr(preprocessor, "model_max_length"):
        # Likely a tokenizer
        logger.info("Modifying tokenizer for extended context")
        
        # Save original max length
        if not hasattr(preprocessor, "_original_max_length"):
            preprocessor._original_max_length = preprocessor.model_max_length
            
        # Set new max length
        preprocessor.model_max_length = config.max_context_length
        
    # Patch generation method if it exists
    if hasattr(model, "generate"):
        logger.info("Patching generation method")
        
        # Store original generate method
        if not hasattr(model, "_original_generate"):
            model._original_generate = model.generate
            
        # Create patched generate method
        def ultracontext_generate(self, *args, **kwargs):
            """Patched generate method with UltraContext integration"""
            # Process input_ids if provided
            if "input_ids" in kwargs and kwargs["input_ids"] is not None:
                input_ids = kwargs["input_ids"]
                
                # Get embeddings
                if hasattr(self, "get_input_embeddings"):
                    embeddings = self.get_input_embeddings()(input_ids)
                    
                    # Process with UltraContext (always as prefill)
                    self.ultra_context.prefill(embeddings)
                    
                    # Get context for attention layers
                    ctx_emb, ctx_pos = self.ultra_context.get_context()
                    
                    # Store in kwargs
                    if "ultra_context" not in kwargs:
                        kwargs["ultra_context"] = (ctx_emb, ctx_pos)
            
            # Call original generate method
            return self._original_generate(*args, **kwargs)
            
        # Attach method to model
        model.generate = types.MethodType(ultracontext_generate, model)
        
    return model, preprocessor
