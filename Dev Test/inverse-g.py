import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. The Gielis Mathematical Foundation
# ==========================================
def gielis_oscillator(t, freq, m, n1, n2, n3, a=1.0, b=1.0):
    """Forward generation of the Supershape time-series."""
    phi = 2 * np.pi * freq * t
    term1 = torch.abs(torch.cos(m * phi / 4.0) / a) ** n2
    term2 = torch.abs(torch.sin(m * phi / 4.0) / b) ** n3
    r = (term1 + term2) ** (-1.0 / n1)
    return r * torch.sin(phi)

# ==========================================
# 2. The Analytical Sensor (TimesFM-X Gen 6)
# ==========================================
def sense_morphology(t, target_signal):
    """
    Variation 36: Spectral-Holographic Inverse Gielis
    Instead of guessing, we 'sense' the structural parameters by projecting 
    a Virtual Viewport over the Ana-Kata topology and finding the absolute resonance.
    """
    start_time = time.time()

    # --- PHASE 1: Spectral Sonar (Identify the Heartbeat) ---
    # We use Fast Fourier Transform to extract the exact fundamental frequency
    yf = torch.fft.rfft(target_signal)
    xf = torch.fft.rfftfreq(len(t), d=(t[1]-t[0]).item())
    
    # Ignore DC component (index 0)
    sensed_freq = xf[torch.argmax(torch.abs(yf[1:])) + 1].item()

    # --- PHASE 2: Morphological Hologram Projection ---
    # We construct a "Ghost Matrix" (Virtual Viewport). 
    # Instead of iterating, we broadcast the Gielis topology across a 4D tensor.
    # To keep memory O(1) conceptually, we assume structural symmetry (n2 = n3 = n23) for Iteration 1.
    
    phi = 2 * np.pi * sensed_freq * t
    
    # Discrete symmetry boundaries (m usually dictates the number of points/petals)
    m_grid = torch.arange(1, 10, 1, dtype=torch.float32) # [9]
    
    # Continuous curvature boundaries (Log-spaced to handle extreme spikes and plateaus)
    n1_grid = torch.logspace(-1.0, 1.0, 50) # [50]
    n23_grid = torch.logspace(-1.0, 1.0, 50) # [50]

    # Reshape for massive PyTorch broadcasting (The Virtual Layer)
    M = m_grid.view(-1, 1, 1, 1)
    N1 = n1_grid.view(1, -1, 1, 1)
    N23 = n23_grid.view(1, 1, -1, 1)
    PHI = phi.view(1, 1, 1, -1)

    # Compute the entire 4D manifold in one parallelized mathematical sweep
    term1 = torch.abs(torch.cos(M * PHI / 4.0)) ** N23
    term2 = torch.abs(torch.sin(M * PHI / 4.0)) ** N23
    R_grid = (term1 + term2) ** (-1.0 / N1)
    S_grid = R_grid * torch.sin(PHI) # Shape: [9, 50, 50, T]

    # --- PHASE 3: Holographic Jump & Lattice Lock ---
    # Calculate Mean Squared Error across the entire topology instantly
    target_view = target_signal.view(1, 1, 1, -1)
    mse_landscape = torch.mean((S_grid - target_view)**2, dim=-1)

    # Teleport to the absolute minimum resonance
    min_flat_idx = torch.argmin(mse_landscape)
    
    # Unravel the 1D index back into our 3D Morphological Coordinates
    m_idx, n1_idx, n23_idx = np.unravel_index(min_flat_idx.item(), mse_landscape.shape)

    sensed_m = m_grid[m_idx].item()
    sensed_n1 = n1_grid[n1_idx].item()
    sensed_n23 = n23_grid[n23_idx].item()
    
    compute_time = time.time() - start_time

    return {
        "freq": sensed_freq,
        "m": sensed_m,
        "n1": sensed_n1,
        "n2": sensed_n23,
        "n3": sensed_n23,
        "compute_time_ms": compute_time * 1000,
        "mse": mse_landscape[m_idx, n1_idx, n23_idx].item()
    }

# ==========================================
# 3. The Execution Test
# ==========================================
if __name__ == "__main__":
    print(">>> Initializing TimesFM-X Gen 6 Analytical Sensor...")
    
    # 1. Create the chaotic/spiky "Unknown" Signal (The Target)
    t = torch.linspace(0, 5, 500)
    # Ground Truth: Spiky Star
    true_freq = 1.0
    true_m = 5.0
    true_n1 = 0.5
    true_n23 = 1.0 
    
    target_signal = gielis_oscillator(t, freq=true_freq, m=true_m, n1=true_n1, n2=true_n23, n3=true_n23)
    
    print(f"\n[Ground Truth DNA] Freq: {true_freq} | M: {true_m} | N1: {true_n1} | N2/3: {true_n23}")
    print("-" * 50)
    
    # 2. Sense the parameters analytically
    print("[Sensor Active] Scanning Virtual Viewport for Morphological Resonance...")
    sensed_dna = sense_morphology(t, target_signal)
    
    print(f"[Lock Achieved] Completed in {sensed_dna['compute_time_ms']:.2f} ms")
    print(f"  > Sensed Freq: {sensed_dna['freq']:.2f}")
    print(f"  > Sensed M   : {sensed_dna['m']:.2f}")
    print(f"  > Sensed N1  : {sensed_dna['n1']:.4f}")
    print(f"  > Sensed N2/3: {sensed_dna['n2']:.4f}")
    print(f"  > Manifold MSE Error: {sensed_dna['mse']:.8f}")
    
    # 3. Reconstruct the sensed model
    reconstructed_signal = gielis_oscillator(
        t, 
        freq=sensed_dna['freq'], 
        m=sensed_dna['m'], 
        n1=sensed_dna['n1'], 
        n2=sensed_dna['n2'], 
        n3=sensed_dna['n3']
    )

    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(t.numpy(), target_signal.numpy(), label="Raw Chaotic Data (Target)", color="gray", linewidth=4, alpha=0.5)
    plt.plot(t.numpy(), reconstructed_signal.numpy(), label="Sensed Ana-Kata Tracker (Model)", color="red", linestyle="--", linewidth=2)
    plt.title(f"Gen 6: Analytical Sensing of Gielis Topology (MSE: {sensed_dna['mse']:.6f})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
