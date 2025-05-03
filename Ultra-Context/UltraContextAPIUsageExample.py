import torch
import logging
import time
import argparse
from pathlib import Path
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ultracontext.example")

# Import UltraContext
from ultracontext import UltraContext, UltraContextConfig

def load_model_and_tokenizer(model_name):
    """Load Hugging Face model and tokenizer"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except ImportError:
        logger.error("Failed to import transformers. Install with: pip install transformers")
        raise

def create_ultracontext_config(model, max_context_length=100_000, mode="extension"):
    """Create UltraContext configuration"""
    # Detect model dimension
    config = UltraContextConfig.from_model(
        model,
        max_context_length=max_context_length,
        integration_mode=mode,
        # Set smaller window sizes for demo
        active_window_size=min(4096, max_context_length // 2),
        sliding_window_size=min(2048, max_context_length // 4),
    )
    
    return config

def integrate_model(model, tokenizer, config):
    """Integrate model with UltraContext"""
    # Create UltraContext
    ultra_ctx = UltraContext(model, config)
    
    # Optimize UltraContext components
    ultra_ctx.optimize()
    
    # Integrate with model
    model = ultra_ctx.integrate()
    
    # Update tokenizer
    if hasattr(tokenizer, "model_max_length"):
        # Store original value
        tokenizer._original_max_length = tokenizer.model_max_length
        # Update to new context length
        tokenizer.model_max_length = config.max_context_length
    
    return model, tokenizer

def generate_with_long_context(model, tokenizer, prompt, context=None, max_new_tokens=100):
    """Generate text with long context"""
    # Combine context and prompt if provided
    if context:
        full_prompt = context + "\n\n" + prompt
    else:
        full_prompt = prompt
    
    # Tokenize
    logger.info(f"Tokenizing input (length: {len(full_prompt)})")
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    logger.info(f"Input tokens: {input_ids.shape[1]}")
    
    # Generate
    start_time = time.time()
    logger.info("Generating...")
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    generation_time = time.time() - start_time
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(full_prompt):]
    
    logger.info(f"Generation completed in {generation_time:.2f}s")
    
    # If the model has UltraContext, get stats
    if hasattr(model, "ultra_context"):
        context_size = model.ultra_context.get_context_size()
        logger.info(f"Context stats: {context_size}")
    
    return response

def load_context_from_file(file_path):
    """Load context from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def benchmark_performance(model, tokenizer, ultra_ctx, context_lengths=[1024, 4096, 16384, 65536]):
    """Benchmark performance with different context lengths"""
    results = {}
    
    # Create synthetic context
    synth_text = "This is a test sentence for benchmarking long context performance. " * 500
    
    for length in context_lengths:
        if length > tokenizer.model_max_length:
            logger.info(f"Skipping length {length} (exceeds max length {tokenizer.model_max_length})")
            continue
            
        # Create context of appropriate length
        ctx = synth_text[:length * 4]  # Approximate characters to get desired token length
        prompt = "Summarize the above text."
        
        # Clear existing context
        if hasattr(model, "ultra_context"):
            model.ultra_context.clear_context()
        
        # Tokenize
        tokens = tokenizer(ctx + prompt, return_tensors="pt").input_ids.to(model.device)
        actual_length = tokens.shape[1]
        
        if actual_length < length // 2 or actual_length > length * 2:
            # Adjust context if token length is too far off
            scale_factor = length / max(1, actual_length)
            ctx = synth_text[:int(len(ctx) * scale_factor)]
            tokens = tokenizer(ctx + prompt, return_tensors="pt").input_ids.to(model.device)
            actual_length = tokens.shape[1]
        
        logger.info(f"Testing context length: {actual_length} tokens")
        
        # Measure prefill time
        start_time = time.time()
        with torch.inference_mode():
            _ = model(tokens)
        prefill_time = time.time() - start_time
        
        # Measure generation time (fixed 20 tokens)
        start_time = time.time()
        with torch.inference_mode():
            _ = model.generate(tokens, max_new_tokens=20)
        generation_time = time.time() - start_time
        
        # Get context stats
        if hasattr(model, "ultra_context"):
            context_stats = model.ultra_context.get_context_size()
        else:
            context_stats = {}
        
        # Store results
        results[actual_length] = {
            "prefill_time": prefill_time,
            "prefill_tokens_per_second": actual_length / prefill_time,
            "generation_time": generation_time,
            "generation_tokens_per_second": 20 / (generation_time - prefill_time) if generation_time > prefill_time else None,
            "context_stats": context_stats
        }
        
        logger.info(f"Results for {actual_length} tokens:")
        logger.info(f"  Prefill: {prefill_time:.2f}s ({results[actual_length]['prefill_tokens_per_second']:.2f} tokens/s)")
        logger.info(f"  Generation: {generation_time:.2f}s")
        
    return results

def save_benchmark_results(results, output_path):
    """Save benchmark results to file"""
    import json
    
    # Convert to serializable format
    serializable_results = {}
    for length, data in results.items():
        serializable_results[str(length)] = {
            k: v if not isinstance(v, torch.Tensor) else v.tolist() 
            for k, v in data.items()
        }
    
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="UltraContext example")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Model name or path")
    parser.add_argument("--context", type=str, help="Path to context file")
    parser.add_argument("--prompt", type=str, default="Summarize the above text in a few sentences.", help="Prompt to use")
    parser.add_argument("--max-context", type=int, default=100000, help="Maximum context length")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--mode", type=str, default="extension", choices=["extension", "replacement", "hybrid"], help="Integration mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file for generation")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Create UltraContext configuration
    config = create_ultracontext_config(model, args.max_context, args.mode)
    
    # Integrate model
    logger.info("Integrating model with UltraContext")
    model, tokenizer = integrate_model(model, tokenizer, config)
    
    if args.benchmark:
        # Run benchmarks
        logger.info("Running benchmarks")
        results = benchmark_performance(
            model, 
            tokenizer, 
            model.ultra_context,
            [1024, 4096, 16384, min(65536, args.max_context)]
        )
        
        # Save benchmark results
        save_benchmark_results(results, "benchmark_results.json")
    else:
        # Load context if provided
        context = None
        if args.context:
            logger.info(f"Loading context from {args.context}")
            context = load_context_from_file(args.context)
        
        # Generate
        response = generate_with_long_context(
            model,
            tokenizer,
            args.prompt,
            context,
            args.max_new_tokens
        )
        
        # Print response
        print("\nGenerated response:")
        print("=" * 40)
        print(response)
        print("=" * 40)
        
        # Save output
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response)
        logger.info(f"Response saved to {args.output}")
    
    # Save UltraContext state if desired
    if hasattr(model, "ultra_context"):
        model.ultra_context.save_state()

if __name__ == "__main__":
    main()
