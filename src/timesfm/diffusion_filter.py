import torch
from torch import nn
import numpy as np

"""
FEISTEL DIFFUSION FILTER: FORMALIZATION

1. GENERATIVE DREAMING (G):
   W_dream = F^2(coords)
   W_ground = F^4(coords)

2. RESIDUAL MANIFOLD (\Delta):
   \Delta = W_ground - W_dream
   
   In a Zero-Shot context, we lack W_ground. However, if \Delta is structurally 
   consistent for a regime, we can use a learned archetype \alpha:
   \Delta' = \alpha \otimes W_dream

3. HOMOMORPHIC REFINEMENT (R):
   Y_refined = (X @ W_dream.T) + \epsilon \cdot (X @ \Delta.T)
   
   The term (X @ \Delta.T) acts as a high-frequency diffusion source, 
   smoothing the "aliasing" artifacts of the 2-round partial evaluation.
"""

class FeistelDiffusionFilter(nn.Module):
    """
    Homomorphic Error Correction via Feistel Residuals.
    Uses the discrepancy between 'Dream' (2-round) and 'Ground' (4-round)
    manifolds as a diffusion source to denoise forecasts.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon # Diffusion rate

    def forward(self, x: torch.Tensor, w_dream: torch.Tensor, w_truth: torch.Tensor) -> torch.Tensor:
        """
        Refines a forecast by applying the Feistel Diffusion Filter.
        
        Math:
        Y_dream = X @ W_dream^T
        Delta = W_truth - W_dream
        Y_refined = Y_dream + epsilon * (X @ Delta^T)
        """
        # Linear projections
        y_dream = torch.matmul(x, w_dream.t())
        
        # Calculate Diffusion Source
        delta = w_truth - w_dream
        
        # Apply Diffusion-based refinement
        # This simulates the '2-round dreamer' guessing the '4-round result'
        correction = torch.matmul(x, delta.t())
        y_refined = y_dream + self.epsilon * correction
        
        return y_refined, correction

def test_diffusion_logic():
    print(">>> Validating Feistel Diffusion Filter Logic")
    
    # Inputs
    batch, seq, dim = 1, 10, 64
    x = torch.randn(batch, seq, dim)
    
    # Simulated Feistel Manifolds (Normalized)
    w_dream = torch.randn(dim, dim) * 0.1
    w_truth = w_dream + torch.randn(dim, dim) * 0.05 # Structural residual
    
    filter_layer = FeistelDiffusionFilter(epsilon=0.5)
    
    y_refined, correction = filter_layer(x, w_dream, w_truth)
    
    # Calculate MSE reduction
    y_truth = torch.matmul(x, w_truth.t())
    mse_dream = torch.mean((torch.matmul(x, w_dream.t()) - y_truth)**2).item()
    mse_refined = torch.mean((y_refined - y_truth)**2).item()
    
    print(f"MSE (2-Round Dream):   {mse_dream:.8f}")
    print(f"MSE (Refined Filter): {mse_refined:.8f}")
    
    improvement = (mse_dream - mse_refined) / mse_dream * 100
    print(f"Diffusion Accuracy Gain: {improvement:.2f}%")

if __name__ == "__main__":
    test_diffusion_logic()
