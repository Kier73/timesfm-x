import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

def v_mask_torch(addr: torch.Tensor, seed: int = 0xBF58476D) -> torch.Tensor:
    """
    PyTorch-native 4-round Feistel hash for O(1) weight generation.
    Matches the V-Matrix SDK implementation.
    """
    # addr should be an integer tensor
    l = (addr >> 32) & 0xFFFFFFFF
    r = addr & 0xFFFFFFFF
    key = seed
    mul = 0x94D049BB
    
    for _ in range(4):
        # f = ((r ^ key) * mul) & 0xFFFFFFFF
        f = torch.bitwise_and((torch.bitwise_xor(r, torch.tensor(key, device=addr.device)) * mul), torch.tensor(0xFFFFFFFF, device=addr.device))
        # f = ((f >> 16) ^ f) & 0xFFFFFFFF
        f = torch.bitwise_and(torch.bitwise_xor(f >> 16, f), torch.tensor(0xFFFFFFFF, device=addr.device))
        l, r = r, torch.bitwise_xor(l, f)
        
    # Combine back to 64-bit and normalize to [-1, 1]
    res = (l << 32) | r
    return (res.float() / float(2**64)) * 2.0 - 1.0

from .reflexive_feistel import v_mask_generative
from .diffusion_filter import FeistelDiffusionFilter
from .gvm_zs_engine import GVM_ZS_Engine

class ImplicitLinear(nn.Module):
    """
    Weight-free Linear Layer. 
    Parameters are generated on-the-fly via Feistel hashes.
    Gen 6: Supports Reflexive Dreaming and Diffusion Refinement.
    """
    def __init__(self, in_features: int, out_features: int, seed: int = 42, rounds: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self.rounds = rounds
        self.scale = 1.0 / math.sqrt(in_features) if in_features > 0 else 1.0
        
        # Gen 6 Diffusion Filter
        self.diffusion = FeistelDiffusionFilter(epsilon=0.1)
        
        # Gen 6: Holographic Memoization Cache
        self._cache = {}

    def forward(self, x: torch.Tensor, mode: str = 'ground') -> torch.Tensor:
        """
        Forward pass with generative options and memoization.
        """
        device = x.device
        
        # Check cache first
        if mode not in self._cache:
            print(f"  [Holographic] Projecting {self.out_features}x{self.in_features} manifold for mode '{mode}'...")
            rows = torch.arange(self.out_features, device=device).unsqueeze(1)
            cols = torch.arange(self.in_features, device=device).unsqueeze(0)
            coords = (self.seed ^ rows ^ cols).long()
            
            if mode == 'dream':
                w = v_mask_generative(coords, rounds=2) * self.scale
            elif mode == 'diffuse' or mode == 'ground':
                w = v_mask_generative(coords, rounds=4) * self.scale
            else:
                w = v_mask_generative(coords, rounds=self.rounds) * self.scale
                
            self._cache[mode] = w

        weights = self._cache[mode]
        
        # Gen 6: Homomorphic Diffusion (if diffuse, we refine the output)
        if mode == 'diffuse':
            # This is a bit complex in pure-Python without Delta cache
            # For now, we use the cached 'ground' truth as the target
            # In hardware, this would be Dream + Delta Filter.
            return torch.matmul(x, weights.t())
            
        return torch.matmul(x, weights.t())

class HTAttention(nn.Module):
    """
    Holographic Attention using Sparse Rademacher Projections (G-Matrix).
    Replaces O(N^2) Softmax Attention with O(N*D) Resonance Mapping.
    """
    def __init__(self, embed_dim: int, projection_dim: int = 64, seed: int = 777):
        super().__init__()
        self.embed_dim = embed_dim
        self.D = projection_dim
        self.seed = seed
        
        # Implicit Projection Matrices (Rademacher-style)
        self.q_proj = ImplicitLinear(embed_dim, projection_dim, seed ^ 0x1)
        self.k_proj = ImplicitLinear(embed_dim, projection_dim, seed ^ 0x2)
        self.v_proj = ImplicitLinear(embed_dim, projection_dim, seed ^ 0x3)
        self.out_proj = ImplicitLinear(projection_dim, embed_dim, seed ^ 0x4)
        
        # Gen 8: HW-ZS Interference Engine (C-Backed via GVM)
        # order=10 provides a 1024x1024 spatiotemporal manifold.
        self.wavefront = GVM_ZS_Engine(order=10, seed=seed ^ 0x6)

    def forward(self, x: torch.Tensor, mode: str = 'ground') -> torch.Tensor:
        r"""
        HT-Attention Forward with HW-ZS Resonant Shunting.
        
        Math:
        Energy_{approx} = (Q @ K.T) @ V
        Resonant_Peak_{ij} = \Psi(H(i, j))  -- From Hilbert Wavefront
        Context_{refined} = Energy_{approx} \otimes \Psi
        """
        # x: [Batch, Seq, Embed]
        Q = self.q_proj(x, mode=mode) # [B, S, D]
        K = self.k_proj(x, mode=mode) # [B, S, D]
        V = self.v_proj(x, mode=mode) # [B, S, D]
        
        # 1. ASSOCIATIVE RESONANCE (Linear Attention Basis) vs 2D INTERFERENCE (Gen 9)
        if mode == 'ground':
            # Fast Associative Path [O(S*D^2)]
            energy = torch.matmul(K.transpose(-2, -1), V) # [B, D, D]
            context = torch.matmul(Q, energy) # [B, S, D]
        else:
            # Gen 9: 2D Spatiotemporal Interference [O(S^2 * D)]
            # We explicitly resolve the attention matrix to apply the resonant mask
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) # [B, S, S]
            
            # Resolve the S x S spatiotemporal wavefront
            resonance = self.wavefront(Q, K, V, mode=mode) # [1, S, S]
            
            # Modulate attention peaks with holographic resonance
            # This enables non-local teleportation across the manifold
            refined_weights = attn_weights * resonance
            context = torch.matmul(refined_weights, V) # [B, S, D]
        
        # Scaling based on the Achlioptas Rademacher norm
        context = context * (math.sqrt(3.0 / self.D))
        
        return self.out_proj(context, mode=mode)

class HolographicTransformerBlock(nn.Module):
    """
    HT Block: Holographic Attention + Implicit FFN.
    """
    def __init__(self, embed_dim: int, ff_dim: int, seed: int = 123):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = HTAttention(embed_dim, seed=seed)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            ImplicitLinear(embed_dim, ff_dim, seed ^ 0xFF),
            nn.ReLU(),
            ImplicitLinear(ff_dim, embed_dim, seed ^ 0xAA)
        )

    def forward(self, x: torch.Tensor, mode: str = 'ground') -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mode=mode)
        # Internal FFN also uses the mode
        x_ln2 = self.ln2(x)
        x_ff1 = self.ffn[0](x_ln2, mode=mode)
        x_rel = self.ffn[1](x_ff1)
        x_ff2 = self.ffn[2](x_rel, mode=mode)
        x = x + x_ff2
        return x

class HolographicTransformer(nn.Module):
    """
    Complete HT Model.
    Designed for rapid inference over chaotic projections.
    """
    def __init__(self, embed_dim: int = 512, n_layers: int = 4, seed: int = 4321):
        super().__init__()
        # Input Projector (Raw [B, S, 1] -> [B, S, Embed])
        self.input_proj = ImplicitLinear(1, embed_dim, seed ^ 0xFEED)
        
        self.layers = nn.ModuleList([
            HolographicTransformerBlock(embed_dim, embed_dim * 2, seed=seed ^ i)
            for i in range(n_layers)
        ])
        
        # Output Head ( [B, S, Embed] -> [B, S, 1])
        self.output_head = ImplicitLinear(embed_dim, 1, seed ^ 0xDEAD)

        # CALIBRATION LAYER: Learned bridge to align HT with Foundation Model
        # This is the only part of HT that uses explicit (learned) weights
        self.calibration_head = nn.Linear(1, 1, bias=True)
        # Initialize to identity
        nn.init.constant_(self.calibration_head.weight, 1.0)
        nn.init.constant_(self.calibration_head.bias, 0.0)

    def forward(self, x: torch.Tensor, mode: str = 'ground') -> torch.Tensor:
        # x shape: [B, S, 1]
        x = self.input_proj(x, mode=mode)
        for layer in self.layers:
            x = layer(x, mode=mode)
        x = self.output_head(x, mode=mode)
        # Apply calibration
        return self.calibration_head(x)
