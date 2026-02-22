import torch
import numpy as np
from src.timesfm.timesfm_x_core import HypervectorEngine

def audit_eco_sampling():
    print(">>> Technical Audit: Sparse-HDC Eco-Sampling (Energy Priority)")
    
    engine = HypervectorEngine()
    dim = engine.DIMENSION
    
    # 1. Create a signal with clear "energy" peaks
    # We use a signal that maps to a few high-seeded HVs
    signal = torch.zeros(10)
    signal[0] = 1.0 # High energy point
    signal[5] = -0.5 # Medium energy point
    
    print("  [Audit] Generating raw signal-to-HV manifold...")
    raw_hv = engine.signal_to_hv(signal)
    
    # 2. Apply Eco-Sampling (50%)
    print("  [Audit] Applying 50% Eco-Sampling...")
    sparse_hv = engine.get_sparse_hv(raw_hv, sparsity=0.5)
    
    n_active = torch.count_nonzero(sparse_hv).item()
    print(f"  [Result] Active Dimensions: {n_active} / 1024")
    
    # 3. Informational Integrity
    # Is the sparse HV still correlated with the raw one?
    cos_sim = torch.nn.functional.cosine_similarity(torch.sign(raw_hv).unsqueeze(0), sparse_hv.unsqueeze(0)).item()
    print(f"  [Result] Information Retention (Cosine Sim): {cos_sim:.6f}")
    
    # Check if the sparsity is truly magnitude-based
    # We'll compare with a random mask simulation
    indices = torch.nonzero(sparse_hv).flatten()
    avg_magnitude = torch.abs(raw_hv[indices]).mean().item()
    global_magnitude = torch.abs(raw_hv).mean().item()
    
    print(f"  [Result] Selected Avg Magnitude: {avg_magnitude:.4f}")
    print(f"  [Result] Global Avg Magnitude:   {global_magnitude:.4f}")
    
    if avg_magnitude > global_magnitude:
        print(">>> AUDIT PASS: Eco-Sampling prioritizes high-energy topological dimensions.")
    else:
        print(">>> AUDIT FAIL: Sampling is unweighted (Shallow).")

if __name__ == "__main__":
    audit_eco_sampling()
