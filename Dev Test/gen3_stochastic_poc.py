import torch
import numpy as np
import math
from typing import Tuple

def fmix64(h: int) -> int:
    """MurmurHash3 64-bit finalizer mix for deterministic seeds."""
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

class StochasticSonarPOC:
    """
    POC for Langevin-style Spectral Sonar.
    Filters noise by probing the manifold density.
    """
    def __init__(self, n_probes: int = 5, jitter: float = 0.05):
        self.n_probes = n_probes
        self.jitter = jitter

    def scan_once(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard LS fit for spectral basis."""
        L = inputs.shape[0]
        pts = torch.linspace(-1, 1, L)
        
        # Simple FFT for dominant freq
        yf = torch.fft.fft(inputs)
        xf = torch.fft.fftfreq(L, d=(pts[1]-pts[0]).item())
        magnitudes = torch.abs(yf[1:])
        dominant_idx = torch.argmax(magnitudes) + 1
        freq = torch.abs(xf[dominant_idx]).item()
        
        X = torch.stack([
            pts**2, pts, torch.sin(freq * pts), torch.cos(freq * pts), torch.ones_like(pts)
        ], dim=1)
        
        try:
            beta = torch.linalg.lstsq(X, inputs.unsqueeze(1)).solution.flatten()
        except:
            beta = torch.zeros(5)
        return beta

    def stochastic_scan(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Langevin Probing: Jitter the inputs and measure coefficient stability.
        """
        betas = []
        for _ in range(self.n_probes):
            # Add stochastic jitter (Langevin-inspired)
            noise = torch.randn_like(inputs) * self.jitter
            beta = self.scan_once(inputs + noise)
            betas.append(beta)
        
        stacked = torch.stack(betas)
        mean_beta = torch.mean(stacked, dim=0)
        std_beta = torch.std(stacked, dim=0)
        
        # Coherence: Max periodic coefficient magnitude
        coherence = torch.abs(mean_beta[2:4]).max().item()
        
        # Coefficient of Variation (CV) = Mean of std/mean for non-zero coeffs
        cv = (std_beta / (torch.abs(mean_beta) + 1e-6)).mean().item()
        
        # Stability Score: Coherence dampened by instability
        score = coherence / (1.0 + cv)
        
        return mean_beta, score

def gen_lorenz(n=1000):
    x, y, z = 0.1, 0.0, 0.0
    s, r, b = 10, 28, 8/3
    data = []
    dt = 0.01
    for _ in range(n):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append(x)
    return torch.tensor(data, dtype=torch.float32)

def gen_noise(n=1000):
    return torch.randn(n)

def test_poc():
    sonar = StochasticSonarPOC(n_probes=10, jitter=0.1)
    
    # 1. Test Stable Chaotic Manifold (Lorenz)
    lorenz = gen_lorenz()
    beta_l, var_l = sonar.stochastic_scan(lorenz)
    
    # 2. Test High-Entropy Noise
    noise = gen_noise()
    beta_n, var_n = sonar.stochastic_scan(noise)
    
    print("=" * 60)
    print("GEN 3 STOCHASTIC SONAR POC RESULTS (Weighted Score)")
    print("=" * 60)
    print(f"Lorenz (Manifold)  | Stability Score: {var_l:.6f}")
    print(f"White Noise (Rand) | Stability Score: {var_n:.6f}")
    print("-" * 60)
    
    discrimination = var_l / (var_n + 1e-9)
    print(f"Discrimination Power (Separation): {discrimination:.1f}x")
    
    # Stability Gate Conjecture
    # Real manifolds should score > 0.1
    threshold = 0.05
    if var_l > threshold:
        print("Lorenz: STABLE (Latching OK)")
    if var_n < threshold:
        print("Noise: UNSTABLE (Latching REFUSED - Stability Gate Triggered)")

if __name__ == "__main__":
    test_poc()
