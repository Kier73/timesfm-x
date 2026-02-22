import torch
import numpy as np
from src.timesfm.reflexive_feistel import v_mask_generative

def audit_feistel_persistence():
    print(">>> Technical Audit: Feistel Structural Persistence (2 vs 4 Rounds)")
    
    # Large manifold [1024 x 1024]
    dim = 1024
    rows = torch.arange(dim).unsqueeze(1)
    cols = torch.arange(dim).unsqueeze(0)
    seed = 42
    coords = (seed ^ rows ^ cols).long()
    
    print(f"  [Audit] Projecting {dim}x{dim} manifolds...")
    w2 = v_mask_generative(coords, rounds=2)
    w4 = v_mask_generative(coords, rounds=4)
    
    # 1. Cosine Similarity
    # Flatten for global similarity
    w2_f = w2.flatten()
    w4_f = w4.flatten()
    
    cos_sim = torch.nn.functional.cosine_similarity(w2_f.unsqueeze(0), w4_f.unsqueeze(0)).item()
    print(f"  [Result] Global Cosine Similarity: {cos_sim:.6f}")
    
    # 2. Bitwise Sign Agreement (HDC preserve)
    sign_agree = (torch.sign(w2) == torch.sign(w4)).float().mean().item()
    print(f"  [Result] Bitwise Sign Agreement:  {sign_agree * 100:.2f}%")
    
    # 3. Local Correlation (Topology check)
    # Check if gradients/slopes match roughly (structural check)
    grad_w2 = w2[1:, :] - w2[:-1, :]
    grad_w4 = w4[1:, :] - w4[:-1, :]
    grad_sim = torch.nn.functional.cosine_similarity(grad_w2.flatten().unsqueeze(0), grad_w4.flatten().unsqueeze(0)).item()
    print(f"  [Result] Gradient Correlation:    {grad_sim:.6f}")
    
    if cos_sim > 0.5:
        print(">>> AUDIT PASS: Holographic Manifold persists across rounds.")
    else:
        print(">>> AUDIT FAIL: Rounds are too divergent for Reflexive Dreaming.")

if __name__ == "__main__":
    audit_feistel_persistence()
