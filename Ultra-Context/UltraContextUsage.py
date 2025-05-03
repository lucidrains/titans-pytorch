import torch
import torch.nn as nn
import time
import os
import argparse
from typing import List, Dict, Optional, Union
import logging
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

# Import UltraContext components
try:
    from ultracontext.core import DEFAULT_PERF_CONFIG, PerformanceConfig
    from ultracontext.integration import (
        UltraContextConfig,
        UltraContextAPI,
        efficient_inference_mode
    )
except ImportError:
    print("UltraContext package not installed. Using local imports.")
    # Fallback to local imports
    from integration import (
        UltraContextConfig,
        UltraContextAPI,
        DEFAULT_PERF_CONFIG,
        PerformanceConfig,
        efficient_inference_mode
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext.examples")

#######################################
# Example 1: Simple PyTorch Integration
#######################################

class SimpleMHA(nn.Module):
    """Simple Multi-Head Attention model for demonstration"""
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, key_padding_mask=mask)
        x = x + attn_out
        
        # FFN with residual connection
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x

def run_simple_example():
    """Run a simple example with a PyTorch model"""
    logger.info("Running Simple PyTorch Integration Example")
    
    # Create a simple model
    model = SimpleMHA(dim=512, num_heads=8)
    
    # Create UltraContext configuration
    config = UltraContextConfig(
        dim=512,
        num_heads=8,
        head_dim=64,
        max_context_length=1_000_000,
        active_window_size=4096,
        sliding_window_size=2048,
        use_hierarchical_memory=True,
        use_token_compression=True,
        integration_mode="extension"
    )
    
    # Integrate UltraContext with the model
    model_with_ultra = UltraContextAPI.integrate(model, config)
    
    # Generate some fake embeddings (batch_size=1, seq_len=2000, dim=512)
    embeddings = torch.randn(1, 2000, 512)
    
    # Process with UltraContext
    with efficient_inference_mode():
        output = UltraContextAPI.process(
            model_with_ultra,
            embeddings,
            is_prefill=True
        )
    
    # Display context size information
    context_size = model_with_ultra.ultra_context.get_context_size()
    logger.info(f"Context size: {context_size}")
    
    # Display output shape
    logger.info(f"Output shape: {output.shape}")
    
    # Now process an extension token
    extension_token = torch.randn(1, 1, 512)
    
    with efficient_inference_mode():
        ext_output = UltraContextAPI.process(
            model_with_ultra,
            extension_token,
            is_prefill=False
        )
    
    # Display output shape
    logger.info(f"Extension output shape: {ext_output.shape}")
    
    # Clear context
    UltraContextAPI.clear_context(model_with_ultra)
    
    logger.info("Simple PyTorch Integration Example Completed")
    
    return model_with_ultra

#########################################
# Example 2: Hugging Face Model Integration
#########################################

def run_huggingface_example(model_name="gpt2", max_length=2048):
    """Run an example with a Hugging Face model"""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error("Transformers library not installed. Run 'pip install transformers'")
        return None
    
    logger.info(f"Running Hugging Face Integration Example with {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Prepare long input text
    text = "This is a very long input text. " * 500  # Repeated to create long text
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=False, max_length=max_length)
    
    # Create UltraContext configuration
    # Auto-detect model dimension (different for each model)
    if hasattr(model.config, "hidden_size"):
        dim = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        dim = model.config.d_model
    else:
        dim = 768  # Default
    
    logger.info(f"Detected model dimension: {dim}")
    
    config = UltraContextConfig(
        dim=dim,
        num_heads=model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12,
        head_dim=dim // (model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12),
        max_context_length=100_000,
        active_window_size=8192,
        sliding_window_size=4096,
        use_hierarchical_memory=True,
        use_token_compression=True,
        integration_mode="extension"
    )
    
    # Integrate UltraContext with the model
    model_with_ultra = UltraContextAPI.integrate(model, config)
    
    # Run the model
    with efficient_inference_mode():
        # Get embeddings directly from the model's embedding layer
        if hasattr(model, "get_input_embeddings"):
            embeddings = model.get_input_embeddings()(inputs.input_ids)
        else:
            # Fallback for models without direct embedding access
            embeddings = model.embeddings.word_embeddings(inputs.input_ids)
        
        # Process with UltraContext
        outputs = UltraContextAPI.process(
            model_with_ultra,
            embeddings,
            is_prefill=True,
            attention_mask=inputs.attention_mask
        )
    
    # Display context size information
    context_size = model_with_ultra.ultra_context.get_context_size()
    logger.info(f"Context size: {context_size}")
    
    # Display output shape
    logger.info(f"Output shape: {outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else outputs.shape}")
    
    # Clear context
    UltraContextAPI.clear_context(model_with_ultra)
    
    logger.info("Hugging Face Integration Example Completed")
    
    return model_with_ultra

#########################################
# Example 3: Long Document Processing
#########################################

def run_long_document_example(document_path=None, model_name="gpt2"):
    """Process a very long document with UltraContext"""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.error("Transformers library not installed. Run 'pip install transformers'")
        return None
    
    logger.info("Running Long Document Processing Example")
    
    # Create or load a long document
    if document_path and os.path.exists(document_path):
        with open(document_path, 'r', encoding='utf-8') as f:
            document = f.read()
    else:
        # Generate a synthetic long document (about 500K characters)
        paragraphs = []
        for i in range(1000):
            paragraph = f"Paragraph {i}. This is a sample paragraph for our ultra-long context demonstration. "
            paragraph += "It contains multiple sentences with various lengths and structures. "
            paragraph += f"We're testing if UltraContext can handle extremely long documents efficiently. "
            paragraph += f"This is sentence {i*4+3} in our test document. "
            paragraphs.append(paragraph)
        
        document = "\n\n".join(paragraphs)
        
        # Save document for reference
        with open("long_document_example.txt", 'w', encoding='utf-8') as f:
            f.write(document)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Get model dimension
    if hasattr(model.config, "hidden_size"):
        dim = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        dim = model.config.d_model
    else:
        dim = 768  # Default
    
    # Tokenize document in chunks
    tokens = tokenizer.encode(document)
    logger.info(f"Document length: {len(document)} characters, {len(tokens)} tokens")
    
    # Process in chunks of 4K tokens
    chunk_size = 4096
    token_chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # Create UltraContext configuration for 1M tokens
    config = UltraContextConfig(
        dim=dim,
        num_heads=model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12,
        head_dim=dim // (model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12),
        max_context_length=1_000_000,
        active_window_size=16384,
        sliding_window_size=8192,
        use_hierarchical_memory=True,
        use_token_compression=True,
        integration_mode="extension"
    )
    
    # Integrate UltraContext with the model
    model_with_ultra = UltraContextAPI.integrate(model, config)
    
    # Process each chunk
    with efficient_inference_mode():
        # Process first chunk as prefill
        input_ids = torch.tensor([token_chunks[0]], device=model.device)
        embeddings = model.get_input_embeddings()(input_ids)
        
        start_time = time.time()
        outputs = UltraContextAPI.process(
            model_with_ultra,
            embeddings,
            is_prefill=True
        )
        first_chunk_time = time.time() - start_time
        
        logger.info(f"First chunk processed in {first_chunk_time:.2f} seconds")
        
        # Process remaining chunks as extensions
        cumulative_tokens = len(token_chunks[0])
        chunk_times = [first_chunk_time]
        
        for i, chunk in enumerate(token_chunks[1:], start=1):
            # Process in smaller pieces (100 tokens at a time)
            for j in range(0, len(chunk), 100):
                sub_chunk = chunk[j:j+100]
                sub_input_ids = torch.tensor([sub_chunk], device=model.device)
                sub_embeddings = model.get_input_embeddings()(sub_input_ids)
                
                start_time = time.time()
                outputs = UltraContextAPI.process(
                    model_with_ultra,
                    sub_embeddings,
                    is_prefill=False
                )
                chunk_time = time.time() - start_time
                
                cumulative_tokens += len(sub_chunk)
                
            chunk_times.append(chunk_time)
            
            # Log progress
            context_size = model_with_ultra.ultra_context.get_context_size()
            logger.info(f"Chunk {i+1}/{len(token_chunks)} processed. "
                       f"Total tokens: {cumulative_tokens}. "
                       f"Context size: {context_size}")
    
    # Plot processing time per chunk
    plt.figure(figsize=(10, 6))
    plt.plot(chunk_times)
    plt.title("Processing Time per Chunk")
    plt.xlabel("Chunk Number")
    plt.ylabel("Time (seconds)")
    plt.savefig("long_document_processing_time.png")
    
    # Clear context
    UltraContextAPI.clear_context(model_with_ultra)
    
    logger.info("Long Document Processing Example Completed")
    
    return model_with_ultra

#########################################
# Example 4: Multi-Document QA with 100M Context
#########################################

def run_qa_example(documents_dir=None):
    """Run QA on multiple documents using UltraContext"""
    try:
        from transformers import AutoModel, AutoTokenizer
        import numpy as np
    except ImportError:
        logger.error("Required libraries not installed. Run 'pip install transformers numpy'")
        return None
    
    logger.info("Running Multi-Document QA Example")
    
    # Create or load documents
    documents = []
    
    if documents_dir and os.path.exists(documents_dir):
        # Load all txt files from directory
        for filename in os.listdir(documents_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(documents_dir, filename), 'r', encoding='utf-8') as f:
                    documents.append({
                        "title": filename,
                        "content": f.read()
                    })
    else:
        # Generate synthetic documents
        for i in range(100):
            # Create document with some facts
            content = f"Document {i} about various topics.\n\n"
            
            # Add some facts that we can query later
            if i % 10 == 0:
                content += f"The capital of Country{i} is City{i}.\n"
            if i % 7 == 0:
                content += f"The population of Region{i} is {i * 1000000} people.\n"
            if i % 5 == 0:
                content += f"The CEO of Company{i} is Person{i}.\n"
                
            # Add filler content
            content += "This document contains various information that might be useful for answering questions.\n"
            content += "It includes data about countries, regions, companies, and people.\n" * 20
            
            documents.append({
                "title": f"Document_{i}.txt",
                "content": content
            })
            
            # Save document for reference
            os.makedirs("qa_documents", exist_ok=True)
            with open(f"qa_documents/document_{i}.txt", 'w', encoding='utf-8') as f:
                f.write(content)
    
    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small model for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Get model dimension
    if hasattr(model.config, "hidden_size"):
        dim = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        dim = model.config.d_model
    else:
        dim = 384  # Default for this model
    
    # Create UltraContext configuration
    config = UltraContextConfig(
        dim=dim,
        num_heads=model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12,
        head_dim=dim // (model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12),
        max_context_length=100_000_000,  # 100M tokens
        active_window_size=16384,
        sliding_window_size=8192,
        use_hierarchical_memory=True,
        memory_compression_ratio=16.0,
        use_token_compression=True,
        compression_ratio=8.0,
        compression_strategies=["prune", "merge", "summarize"],
        integration_mode="extension",
        use_retrieval_augmentation=True
    )
    
    # Integrate UltraContext with the model
    model_with_ultra = UltraContextAPI.integrate(model, config)
    
    # Process all documents
    with efficient_inference_mode():
        total_tokens = 0
        
        for doc_idx, doc in enumerate(documents):
            # Tokenize document
            tokens = tokenizer.encode(doc["content"])
            total_tokens += len(tokens)
            
            # Process in chunks of 4K tokens
            chunk_size = 4096
            token_chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
            
            # Process first chunk as prefill
            if doc_idx == 0:
                input_ids = torch.tensor([token_chunks[0]], device=model.device)
                embeddings = model.get_input_embeddings()(input_ids)
                
                outputs = UltraContextAPI.process(
                    model_with_ultra,
                    embeddings,
                    is_prefill=True
                )
                
                # Process remaining chunks of first document
                for chunk in token_chunks[1:]:
                    input_ids = torch.tensor([chunk], device=model.device)
                    embeddings = model.get_input_embeddings()(input_ids)
                    
                    outputs = UltraContextAPI.process(
                        model_with_ultra,
                        embeddings,
                        is_prefill=False
                    )
            else:
                # For subsequent documents, process all chunks as extensions
                for chunk in token_chunks:
                    input_ids = torch.tensor([chunk], device=model.device)
                    embeddings = model.get_input_embeddings()(input_ids)
                    
                    outputs = UltraContextAPI.process(
                        model_with_ultra,
                        embeddings,
                        is_prefill=False
                    )
            
            # Log progress
            context_size = model_with_ultra.ultra_context.get_context_size()
            logger.info(f"Document {doc_idx+1}/{len(documents)} processed. "
                       f"Total tokens: {total_tokens}. "
                       f"Context size: {context_size}")
    
    # Define some test questions
    questions = [
        "What is the capital of Country0?",
        "Who is the CEO of Company5?",
        "What is the population of Region7?",
        "What is the capital of Country90?"
    ]
    
    # Process questions
    with efficient_inference_mode():
        for question in questions:
            # Encode question
            question_tokens = tokenizer.encode(question)
            input_ids = torch.tensor([question_tokens], device=model.device)
            embeddings = model.get_input_embeddings()(input_ids)
            
            # Process question with UltraContext
            outputs = UltraContextAPI.process(
                model_with_ultra,
                embeddings,
                is_prefill=False
            )
            
            # In a real system, we would now decode the output and display the answer
            # Here we just show that we processed the question
            logger.info(f"Processed question: {question}")
    
    # Clear context
    UltraContextAPI.clear_context(model_with_ultra)
    
    logger.info("Multi-Document QA Example Completed")
    
    return model_with_ultra

#########################################
# Example 5: Streaming Generation with UltraContext
#########################################

def run_streaming_example(prompt="Tell me about artificial intelligence"):
    """Run streaming token generation example"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("Transformers library not installed. Run 'pip install transformers'")
        return None
    
    logger.info("Running Streaming Generation Example")
    
    # Load model and tokenizer
    model_name = "gpt2"  # Using a small model for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get model dimension
    if hasattr(model.config, "hidden_size"):
        dim = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        dim = model.config.d_model
    else:
        dim = 768  # Default
    
    # Create UltraContext configuration
    config = UltraContextConfig(
        dim=dim,
        num_heads=model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12,
        head_dim=dim // (model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else 12),
        max_context_length=50_000,
        active_window_size=4096,
        sliding_window_size=2048,
        use_hierarchical_memory=True,
        use_token_compression=True,
        integration_mode="extension"
    )
    
    # Integrate UltraContext with the model
    model_with_ultra = UltraContextAPI.integrate(model, config)
    
    # Process prompt
    input_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_tokens], device=model.device)
    
    with efficient_inference_mode():
        # Get embeddings
        embeddings = model.get_input_embeddings()(input_ids)
        
        # Process prompt with UltraContext
        outputs = UltraContextAPI.process(
            model_with_ultra,
            embeddings,
            is_prefill=True
        )
        
        # Get logits for next token prediction
        next_token_logits = outputs.logits[:, -1, :]
        
        # Simulate streaming generation of 100 tokens
        generated_tokens = []
        
        for _ in range(100):
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            
            # Process next token
            token_ids = torch.tensor([[next_token]], device=model.device)
            token_embedding = model.get_input_embeddings()(token_ids)
            
            # Extend context with new token
            outputs = UltraContextAPI.process(
                model_with_ultra,
                token_embedding,
                is_prefill=False
            )
            
            # Get logits for next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            
            # Log progress every 10 tokens
            if len(generated_tokens) % 10 == 0:
                generated_text = tokenizer.decode(generated_tokens)
                logger.info(f"Generated {len(generated_tokens)} tokens: {generated_text[:50]}...")
    
    # Decode all generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    logger.info(f"Final generated text: {generated_text}")
    
    # Clear context
    UltraContextAPI.clear_context(model_with_ultra)
    
    logger.info("Streaming Generation Example Completed")
    
    return model_with_ultra

def main():
    """Run all examples"""
    parser = argparse.ArgumentParser(description="UltraContext Examples")
    parser.add_argument("--example", type=str, default="all", 
                       choices=["simple", "huggingface", "longdoc", "qa", "streaming", "all"],
                       help="Example to run")
    parser.add_argument("--doc_path", type=str, default=None, 
                       help="Path to document for long document example")
    parser.add_argument("--docs_dir", type=str, default=None,
                       help="Directory containing documents for QA example")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Model name for Hugging Face examples")
    args = parser.parse_args()
    
    if args.example == "simple" or args.example == "all":
        run_simple_example()
        
    if args.example == "huggingface" or args.example == "all":
        run_huggingface_example(model_name=args.model)
        
    if args.example == "longdoc" or args.example == "all":
        run_long_document_example(document_path=args.doc_path, model_name=args.model)
        
    if args.example == "qa" or args.example == "all":
        run_qa_example(documents_dir=args.docs_dir)
        
    if args.example == "streaming" or args.example == "all":
        run_streaming_example()

if __name__ == "__main__":
    main()
