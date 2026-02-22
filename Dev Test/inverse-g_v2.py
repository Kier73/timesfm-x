import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Forward Gielis (Asymmetrical Support)
# ==========================================
def gielis_oscillator(t, freq, m, n1, n2, n3, a=1.0, b=1.0):
    """Generates the Supershape time-series, allowing n2 != n3 for asymmetry."""
    phi = 2 * np.pi * freq * t
    term1 = torch.abs(torch.cos(m * phi / 4.0) / a) ** n2
    term2 = torch.abs(torch.sin(m * phi / 4.0) / b) ** n3
    r = (term1 + term2) ** (-1.0 / n1)
    return r * torch.sin(phi)

# ==========================================
# 2. Iteration 2: Asymmetrical Morphological Sensor
# ==========================================
def sense_asymmetric_morphology(t, target_signal):
    """
    Variation 36: 5D Spectral-Holographic Ghost Matrix.
    Breaks symmetry by independently searching n2 and n3.
    """
    start_time = time.time()

    # --- PHASE 1: Spectral Sonar ---
    yf = torch.fft.rfft(target_signal)
    xf = torch.fft.rfftfreq(len(t), d=(t[1]-t[0]).item())
    sensed_freq = xf[torch.argmax(torch.abs(yf[1:])) + 1].item()

    # --- PHASE 2: 5D Holographic Projection (The Ana-Kata Volume) ---
    phi = 2 * np.pi * sensed_freq * t
    
    # Grid resolutions (Tuned to project ~80,000 shapes instantly without blowing out RAM)
    m_grid = torch.arange(1, 11, 1, dtype=torch.float32)      # 10 symmetries
    n1_grid = torch.logspace(-1.0, 1.0, 20)                   # 20 global convexities
    n2_grid = torch.logspace(-1.0, 1.0, 20)                   # 20 right-leaning topologies
    n3_grid = torch.logspace(-1.0, 1.0, 20)                   # 20 left-leaning topologies

    # Reshape for 5D PyTorch Broadcasting [M, N1, N2, N3, Time]
    M = m_grid.view(-1, 1, 1, 1, 1)
    N1 = n1_grid.view(1, -1, 1, 1, 1)
    N2 = n2_grid.view(1, 1, -1, 1, 1)
    N3 = n3_grid.view(1, 1, 1, -1, 1)
    PHI = phi.view(1, 1, 1, 1, -1)

    # Compute 80,000 universes simultaneously 
    term1 = torch.abs(torch.cos(M * PHI / 4.0)) ** N2
    term2 = torch.abs(torch.sin(M * PHI / 4.0)) ** N3
    R_grid = (term1 + term2) ** (-1.0 / N1)
    S_grid = R_grid * torch.sin(PHI) # Shape: [10, 20, 20, 20, T]

    # --- PHASE 3: Quantum Collapse (Find the exact topological match) ---
    target_view = target_signal.view(1, 1, 1, 1, -1)
    mse_landscape = torch.mean((S_grid - target_view)**2, dim=-1) # Shape: [10, 20, 20, 20]

    min_flat_idx = torch.argmin(mse_landscape).item()
    m_idx, n1_idx, n2_idx, n3_idx = np.unravel_index(min_flat_idx, mse_landscape.shape)

    sensed_m = m_grid[m_idx].item()
    sensed_n1 = n1_grid[n1_idx].item()
    sensed_n2 = n2_grid[n2_idx].item()
    sensed_n3 = n3_grid[n3_idx].item()
    
    compute_time = time.time() - start_time

    return {
        "freq": sensed_freq,
        "m": sensed_m,
        "n1": sensed_n1,
        "n2": sensed_n2,
        "n3": sensed_n3,
        "compute_time_ms": compute_time * 1000,
        "mse": mse_landscape[m_idx, n1_idx, n2_idx, n3_idx].item()
    }

# ==========================================
# 3. The Execution Test
# ==========================================
if __name__ == "__main__":
    print(">>> Initializing TimesFM-X Gen 6 (Asymmetrical Sensor)...")
    
    t = torch.linspace(0, 5, 500)
    
    # Ground Truth: A highly skewed, biological/chaotic waveform
    # Notice n2 (0.2) is drastically different from n3 (8.0)
    true_freq = 1.0
    true_m = 4.0
    true_n1 = 0.5
    true_n2 = 0.2  # Sharp/spiky on one side
    true_n3 = 8.0  # Flat/plateau on the other side
    
    target_signal = gielis_oscillator(t, freq=true_freq, m=true_m, n1=true_n1, n2=true_n2, n3=true_n3)
    
    print(f"\n[Ground Truth DNA] Freq: {true_freq} | M: {true_m} | N1: {true_n1} | N2: {true_n2} | N3: {true_n3}")
    print("-" * 60)
    
    print("[Sensor Active] Projecting 80,000 Ana-Kata Topologies...")
    sensed_dna = sense_asymmetric_morphology(t, target_signal)
    
    print(f"[Lock Achieved] Completed in {sensed_dna['compute_time_ms']:.2f} ms")
    print(f"  > Sensed Freq: {sensed_dna['freq']:.2f}")
    print(f"  > Sensed M   : {sensed_dna['m']:.2f}")
    print(f"  > Sensed N1  : {sensed_dna['n1']:.4f}")
    print(f"  > Sensed N2  : {sensed_dna['n2']:.4f}")
    print(f"  > Sensed N3  : {sensed_dna['n3']:.4f}")
    print(f"  > Manifold MSE Error: {sensed_dna['mse']:.8f}")
    
    reconstructed_signal = gielis_oscillator(
        t, 
        freq=sensed_dna['freq'], 
        m=sensed_dna['m'], 
        n1=sensed_dna['n1'], 
        n2=sensed_dna['n2'], 
        n3=sensed_dna['n3']
    )

    plt.figure(figsize=(12, 6))
    plt.plot(t.numpy(), target_signal.numpy(), label="Raw Asymmetrical Chaos (Target)", color="gray", linewidth=5, alpha=0.4)
    plt.plot(t.numpy(), reconstructed_signal.numpy(), label="Gen 6 Asymmetrical Hologram", color="blue", linestyle="--", linewidth=2)
    plt.title(f"Gen 6: Analytical Sensing of Asymmetrical Gielis Topology (MSE: {sensed_dna['mse']:.6f})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
