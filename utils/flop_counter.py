import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("fvcore not available. Install with: pip install fvcore")

def count_flops_fvcore(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> Dict[str, Any]:
    """
    Count FLOPs using fvcore library.
    
    Args:
        model: The model to analyze
        input_shape: Input tensor shape (batch_size, seq_len)
        device: Device to run on
        
    Returns:
        Dictionary with FLOP statistics
    """
    if not FVCORE_AVAILABLE:
        return {"error": "fvcore not available"}
    
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    batch_size, seq_len = input_shape
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Count FLOPs
    with torch.no_grad():
        flop_dict, _ = flop_count(
            model, 
            (input_ids, target_ids),
            supported_ops={
                "aten::add": lambda inputs, outputs: torch.numel(inputs[0]),
                "aten::addmm": lambda inputs, outputs: torch.numel(inputs[1]) * inputs[2].shape[-1],
                "aten::bmm": lambda inputs, outputs: inputs[0].numel() * inputs[1].shape[-1],
                "aten::mm": lambda inputs, outputs: inputs[0].numel() * inputs[1].shape[-1],
                "aten::mul": lambda inputs, outputs: torch.numel(outputs[0]),
                "aten::linear": lambda inputs, outputs: inputs[0].numel() * inputs[1].shape[-1],
            }
        )
    
    total_flops = sum(flop_dict.values())
    
    return {
        "total_flops": total_flops,
        "flops_per_sample": total_flops / batch_size,
        "detailed_flops": flop_dict,
        "model_params": sum(p.numel() for p in model.parameters()),
        "input_shape": input_shape
    }

def count_flops_manual(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Manual FLOP counting for transformer-like models.
    
    Args:
        model: The model to analyze
        input_shape: Input tensor shape (batch_size, seq_len)
        
    Returns:
        Dictionary with FLOP estimates
    """
    batch_size, seq_len = input_shape
    config = model.config
    
    # Embedding FLOPs (lookup operations, minimal)
    embedding_flops = 0
    
    # Transformer block FLOPs
    block_flops = 0
    for _ in range(config.n_blocks):
        # Self-attention FLOPs
        # Q, K, V projections: 3 * (seq_len * batch_size * n_embed * n_embed)
        qkv_flops = 3 * seq_len * batch_size * config.n_embed * config.n_embed
        
        # Attention scores: (batch_size * num_heads * seq_len * seq_len * head_dim)
        head_dim = config.n_embed // config.num_heads
        attn_scores_flops = batch_size * config.num_heads * seq_len * seq_len * head_dim
        
        # Attention output: (batch_size * num_heads * seq_len * seq_len * head_dim)
        attn_output_flops = batch_size * config.num_heads * seq_len * seq_len * head_dim
        
        # Output projection: (seq_len * batch_size * n_embed * n_embed)
        out_proj_flops = seq_len * batch_size * config.n_embed * config.n_embed
        
        # MLP FLOPs (assuming 4x expansion)
        mlp_hidden_size = config.n_embed * 4
        mlp_flops = 2 * seq_len * batch_size * config.n_embed * mlp_hidden_size
        
        # Layer norm FLOPs (minimal, ~2 ops per element)
        ln_flops = 2 * 2 * seq_len * batch_size * config.n_embed  # 2 layer norms per block
        
        block_flops += qkv_flops + attn_scores_flops + attn_output_flops + out_proj_flops + mlp_flops + ln_flops
    
    # Output projection FLOPs
    output_flops = seq_len * batch_size * config.n_embed * config.vocab_size
    
    # PC-specific FLOPs (predictive coding iterations)
    pc_flops = 0
    if hasattr(config, 'T') and config.T > 1:
        # Estimate additional FLOPs for PC iterations
        # This is a rough estimate - actual PC FLOPs depend on implementation
        pc_iterations = config.T - 1  # Additional iterations beyond standard forward pass
        pc_flops = pc_iterations * block_flops * 0.5  # Assume PC adds ~50% overhead per iteration
    
    total_flops = embedding_flops + block_flops + output_flops + pc_flops
    
    return {
        "total_flops": total_flops,
        "flops_per_sample": total_flops / batch_size,
        "breakdown": {
            "embedding": embedding_flops,
            "transformer_blocks": block_flops,
            "output_projection": output_flops,
            "predictive_coding": pc_flops
        },
        "model_params": sum(p.numel() for p in model.parameters()),
        "input_shape": input_shape,
        "pc_iterations": getattr(config, 'T', 1)
    }

def format_flops(flops: float) -> str:
    """Format FLOP count in human-readable format."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f}T"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f}G"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f}M"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f}K"
    else:
        return f"{flops:.0f}"

def analyze_model_efficiency(model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu") -> None:
    """
    Comprehensive model efficiency analysis.
    
    Args:
        model: The model to analyze
        input_shape: Input tensor shape (batch_size, seq_len)
        device: Device to run on
    """
    print("=" * 60)
    print("MODEL EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print()
    
    # FLOP analysis
    print(f"FLOP Analysis (input shape: {input_shape}):")
    
    # Try fvcore first
    if FVCORE_AVAILABLE:
        try:
            fvcore_results = count_flops_fvcore(model, input_shape, device)
            if "error" not in fvcore_results:
                print(f"  FVCore Total FLOPs: {format_flops(fvcore_results['total_flops'])}")
                print(f"  FVCore FLOPs/sample: {format_flops(fvcore_results['flops_per_sample'])}")
        except Exception as e:
            print(f"  FVCore failed: {e}")
    
    # Manual estimation
    manual_results = count_flops_manual(model, input_shape)
    print(f"  Manual Total FLOPs: {format_flops(manual_results['total_flops'])}")
    print(f"  Manual FLOPs/sample: {format_flops(manual_results['flops_per_sample'])}")
    
    if 'breakdown' in manual_results:
        print(f"  Breakdown:")
        for component, flops in manual_results['breakdown'].items():
            print(f"    {component}: {format_flops(flops)} ({flops/manual_results['total_flops']*100:.1f}%)")
    
    # Efficiency metrics
    flops_per_param = manual_results['total_flops'] / total_params
    print(f"\nEfficiency Metrics:")
    print(f"  FLOPs per parameter: {flops_per_param:.2f}")
    print(f"  PC iterations (T): {manual_results.get('pc_iterations', 1)}")
    
    print("=" * 60)

def benchmark_inference_speed(model: nn.Module, input_shape: Tuple[int, ...], 
                            device: str = "cpu", num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark inference speed.
    
    Args:
        model: The model to benchmark
        input_shape: Input tensor shape
        device: Device to run on
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    batch_size, seq_len = input_shape
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, target_ids)
    
    # Benchmark
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_ids, target_ids)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "mean_time": times.mean(),
        "std_time": times.std(),
        "min_time": times.min(),
        "max_time": times.max(),
        "throughput_samples_per_sec": batch_size / times.mean(),
        "throughput_tokens_per_sec": (batch_size * seq_len) / times.mean()
    }
