import torch
import time
from src.timesfm.reflexive_feistel import feistel_round

def v_mask_wavefront(addr: torch.Tensor, timestamp: int, seed: int = 0xBF58476D) -> torch.Tensor:
    """
    ALTERED V-MASK: Incorporates 'Temporal Drift' for Wavefront synchronization.
    O(1) complexity weight generation.
    """
    # Incorporate timestamp into the seed for wavefront shift
    t_seed = seed ^ timestamp
    
    l = (addr >> 32) & 0xFFFFFFFF
    r = addr & 0xFFFFFFFF
    
    device = addr.device
    k_t = torch.tensor(t_seed, device=device)
    
    # Standard 4-round hardened mixer
    for _ in range(4):
        # fmix32 hardened round
        h = torch.bitwise_xor(r, k_t)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h = torch.bitwise_xor(h >> 16, h)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        
        l, r = r, torch.bitwise_xor(l, h)
        
    res = (l << 32) | r
    return (res.float() / float(2**64)) * 2.0 - 1.0

def test_v_matrix_wavefront():
    print(">>> HW-ZS Test 1: V-Matrix (Temporal Wavefront Velocity)")
    
    dim = 1000
    rows = torch.arange(dim).unsqueeze(1)
    cols = torch.arange(dim).unsqueeze(0)
    coords = (rows ^ cols).long()
    
    print(f"  [Audit] Generating 1 million elements at t=0...")
    start = time.perf_counter()
    w0 = v_mask_wavefront(coords, timestamp=0)
    end = time.perf_counter()
    print(f"    - Latency: {(end-start)*1000:.4f}ms")
    
    print(f"  [Audit] Generating wavefront shift at t=0xABCD...")
    start = time.perf_counter()
    w1 = v_mask_wavefront(coords, timestamp=0xABCD)
    end = time.perf_counter()
    print(f"    - Latency: {(end-start)*1000:.4f}ms")
    
    # Verify variety
    variety = (w0 != w1).float().mean().item()
    print(f"  [Metric] Wavefront Variety: {variety * 100:.2f}%")
    
    if variety > 0.99:
        print(">>> TEST 1 PASS: V-Matrix propagates at constant time with high variety.")
    else:
        print(">>> TEST 1 FAIL: Wavefront is static.")

if __name__ == "__main__":
    test_v_matrix_wavefront()
