import torch
import time
from src.timesfm.holographic_transformer import HTAttention

def test_ht_hw_zs_integration():
    print(">>> HW-ZS Integration Audit: HTAttention Resonant Shunting")
    
    embed_dim = 128
    projection_dim = 64
    seq_len = 100
    batch_size = 2
    
    # Initialize Attention with HW-ZS
    attn = HTAttention(embed_dim, projection_dim=projection_dim, seed=42)
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"  [Audit] Running Ground Pass (Standard Associative)...")
    start = time.perf_counter()
    out_ground = attn(x, mode='ground')
    end = time.perf_counter()
    print(f"    - Latency: {(end-start)*1000:.4f}ms")
    
    print(f"  [Audit] Running Dream Pass (HW-ZS Shunting Active)...")
    start = time.perf_counter()
    out_dream = attn(x, mode='dream')
    end = time.perf_counter()
    print(f"    - Latency: {(end-start)*1000:.4f}ms")
    
    # Check for non-degeneracy
    variance = torch.var(out_dream).item()
    print(f"  [Metric] Output Variance (Dream): {variance:.6f}")
    
    # Check divergence (Dream should bias the Ground)
    divergence = torch.mean(torch.abs(out_dream - out_ground)).item()
    print(f"  [Metric] Mean Semantic Shift: {divergence:.6f}")
    
    if variance > 0.01 and divergence > 1e-6:
        print(">>> INTEGRATION PASS: HW-ZS Attention is functional and resonant.")
    else:
        print(">>> INTEGRATION FAIL: Attention result is degenerate or static.")

if __name__ == "__main__":
    test_ht_hw_zs_integration()
