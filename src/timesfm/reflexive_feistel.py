import torch
from typing import Tuple

def feistel_round(l: torch.Tensor, r: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rigorous Feistel round with high-entropy mixing (~Murmur3)."""
    # 1. Non-linear expansion
    h = torch.bitwise_xor(r, key)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = torch.bitwise_xor(h >> 16, h)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = torch.bitwise_xor(h >> 13, h)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = torch.bitwise_xor(h >> 16, h)
    
    # 2. Feistel Swap
    return r, torch.bitwise_xor(l, h)

def v_mask_generative(addr: torch.Tensor, rounds: int = 4, seed: int = 0xBF58476D) -> torch.Tensor:
    """
    Generative Feistel Hash for holographic weight generation.
    Supports k=2 (Dream) and k=4 (Ground) evaluation.
    """
    l = (addr >> 32) & 0xFFFFFFFF
    r = addr & 0xFFFFFFFF
    
    device = addr.device
    k_t = torch.tensor(seed, device=device)
    
    for _ in range(rounds):
        l, r = feistel_round(l, r, k_t)
        
    # Combine back to 64-bit
    res = (l << 32) | r
    # Normalize to [-1, 1] for weights
    return (res.float() / float(2**64)) * 2.0 - 1.0

def get_feistel_residual(addr: torch.Tensor, seed: int = 0xBF58476D) -> torch.Tensor:
    """Calculates the diffusion source (Delta) between 2 and 4 rounds."""
    w2 = v_mask_generative(addr, rounds=2, seed=seed)
    w4 = v_mask_generative(addr, rounds=4, seed=seed)
    return w4 - w2
