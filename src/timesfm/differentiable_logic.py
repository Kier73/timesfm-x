import torch
import torch.nn as nn
import math

class HarmonicShift(nn.Module):
    """
    Gen 8: Continuous harmonic approximation of bitwise XOR.
    Enables differentiability on the Holographic Manifold.
    """
    def __init__(self, harmonics: int = 8):
        super().__init__()
        self.harmonics = harmonics
        # Harmonic amplitudes: decaying 1/k power to simulate bitwise significance
        self.register_buffer("k_vals", torch.pow(2.0, torch.arange(harmonics).float()))
        self.register_buffer("alpha", torch.pow(0.5, torch.arange(harmonics).float()))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the harmonic interference S(x, y).
        Approximates (x ^ y) as a smooth differentiable pulse field.
        """
        # x, y normalized to [0, 1]
        diff = (x - y).unsqueeze(-1) # [..., 1]
        
        # S(x, y) = Sum( alpha_k * cos(2^k * pi * diff) )
        interference = self.alpha * torch.cos(self.k_vals * math.pi * diff)
        return torch.sum(interference, dim=-1)

class STE_Quantizer(torch.autograd.Function):
    """Straight-Through Estimator for integer seed learning."""
    @staticmethod
    def forward(ctx, input):
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input).float()
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_quantize(x):
    return STE_Quantizer.apply(x)
