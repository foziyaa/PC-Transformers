#!/usr/bin/env python3
"""
Example: FLOP Analysis for PC-Transformer

This example demonstrates how to analyze the computational efficiency
of your PC-Transformer model using FLOPs.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import time
import psutil
import os
from model_architecture.pc_t_model import PCTransformer
from predictive_coding.config import GPTConfig
from utils.model_utils import load_tokenizer
from utils.flop_counter import (
    analyze_model_efficiency,
    count_flops_manual,
    format_flops,
    benchmark_inference_speed
)

def main():
    print("PC-Transformer FLOP Analysis Example")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    
    # Create a small model for demonstration
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size= 256, 
        peak_learning_rate= 2e-5,
        warmup_steps= 217,
        n_embed=64,
        dropout= 0.24684719512514441,
        local_learning_rate= 0.0,
        T= 1,
        is_holding_error = True,
        num_heads=1,
        n_blocks=1,
        num_epochs= 1,
        update_bias= True,
        use_lateral = True,
        internal_energy_fn_name="mse",
        output_energy_fn_name="kld",
        eos_token_id=tokenizer.eos_token_id,
        combined_internal_weight=0.3,
        combined_output_weight=0.7,
        use_flash_attention=True  
    )
    
    # Create model
    model = PCTransformer(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {config.n_embed}")
    print(f"  Number of blocks: {config.n_blocks}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Block size: {config.block_size}")
    print(f"  PC iterations (T): {config.T}")
    print(f"  Device: {device}")
    print()
    
    # Analyze efficiency for different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    print("FLOP Analysis for Different Batch Sizes:")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        input_shape = (batch_size, config.block_size)
        flop_results = count_flops_manual(model, input_shape)
        
        print(f"Batch size {batch_size:2d}:")
        print(f"  Total FLOPs: {format_flops(flop_results['total_flops']):>10s}")
        print(f"  FLOPs/sample: {format_flops(flop_results['flops_per_sample']):>10s}")
        
        # Show breakdown
        breakdown = flop_results['breakdown']
        total_flops = flop_results['total_flops']
        print(f"  Breakdown:")
        for component, flops in breakdown.items():
            percentage = (flops / total_flops) * 100
            print(f"    {component:20s}: {format_flops(flops):>8s} ({percentage:4.1f}%)")
        print()
    
    # Compare different T values (PC iterations)
    print("Impact of Predictive Coding Iterations (T):")
    print("-" * 50)
    
    T_values = [1, 3, 5, 10, 15]
    base_flops = None
    
    for T in T_values:
        # Create config with different T
        temp_config = GPTConfig(**{**config.__dict__, 'T': T})
        temp_model = PCTransformer(temp_config)
        
        flop_results = count_flops_manual(temp_model, (1, config.block_size))
        flops_per_sample = flop_results['flops_per_sample']
        
        if base_flops is None:
            base_flops = flops_per_sample
        
        ratio = flops_per_sample / base_flops
        pc_flops = flop_results['breakdown']['predictive_coding']
        pc_percentage = (pc_flops / flop_results['total_flops']) * 100
        
        print(f"T={T:2d}: {format_flops(flops_per_sample):>10s} ({ratio:4.2f}x, PC: {pc_percentage:4.1f}%)")
    
    print()
    
    # Comprehensive analysis
    print("Comprehensive Model Analysis:")
    analyze_model_efficiency(model, (8, config.block_size), device)
    
    # Benchmark inference speed
    print("Inference Speed Benchmark:")
    print("-" * 30)
    
    try:
        benchmark_results = benchmark_inference_speed(
            model, (4, config.block_size), device, num_runs=20
        )
        
        print(f"Mean inference time: {benchmark_results['mean_time']*1000:.2f} ms")
        print(f"Throughput: {benchmark_results['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"Throughput: {benchmark_results['throughput_tokens_per_sec']:.1f} tokens/sec")
        
        # Calculate computational throughput
        flop_results = count_flops_manual(model, (4, config.block_size))
        flops_per_second = flop_results["total_flops"] / benchmark_results['mean_time']
        print(f"Computational throughput: {format_flops(flops_per_second)} FLOP/s")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    
    # Tips for optimization
    print("\nOptimization Tips:")
    print("- Reduce T (PC iterations) for faster inference")
    print("- Use smaller embedding dimensions for memory efficiency")
    print("- Consider flash attention for longer sequences")
    print("- Batch processing improves FLOP efficiency")

if __name__ == "__main__":
    main()
