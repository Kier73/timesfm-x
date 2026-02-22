import torch
import numpy as np
import time

# ==========================================
# 1. Forward Gielis (Asymmetrical)
# ==========================================
def gielis_oscillator(t, freq, m, n1, n2, n3, a=1.0, b=1.0):
    phi = 2 * np.pi * freq * t
    term1 = torch.abs(torch.cos(m * phi / 4.0) / a) ** n2
    term2 = torch.abs(torch.sin(m * phi / 4.0) / b) ** n3
    r = (term1 + term2) ** (-1.0 / n1)
    return r * torch.sin(phi)

# ==========================================
# 2. Iteration 3: Coarse Jump + Lattice Lock
# ==========================================
def sense_with_lattice_lock(t, target_signal):
    start_time = time.time()

    # --- PHASE 1: Spectral Sonar ---
    yf = torch.fft.rfft(target_signal)
    xf = torch.fft.rfftfreq(len(t), d=(t[1]-t[0]).item())
    sensed_freq = xf[torch.argmax(torch.abs(yf[1:])) + 1].item()

    # --- PHASE 2: Coarse Holographic Jump (The "Macro-Net") ---
    phi = 2 * np.pi * sensed_freq * t
    m_grid = torch.arange(1, 11, 1, dtype=torch.float32)
    n1_grid = torch.logspace(-1.0, 1.0, 20)
    n2_grid = torch.logspace(-1.0, 1.0, 20)
    n3_grid = torch.logspace(-1.0, 1.0, 20)

    M = m_grid.view(-1, 1, 1, 1, 1)
    N1 = n1_grid.view(1, -1, 1, 1, 1)
    N2 = n2_grid.view(1, 1, -1, 1, 1)
    N3 = n3_grid.view(1, 1, 1, -1, 1)
    PHI = phi.view(1, 1, 1, 1, -1)

    term1 = torch.abs(torch.cos(M * PHI / 4.0)) ** N2
    term2 = torch.abs(torch.sin(M * PHI / 4.0)) ** N3
    R_grid = (term1 + term2) ** (-1.0 / N1)
    S_grid = R_grid * torch.sin(PHI)

    target_view = target_signal.view(1, 1, 1, 1, -1)
    mse_landscape = torch.mean((S_grid - target_view)**2, dim=-1)

    min_flat_idx = torch.argmin(mse_landscape).item()
    m_idx, n1_idx, n2_idx, n3_idx = np.unravel_index(min_flat_idx, mse_landscape.shape)

    coarse_m = m_grid[m_idx].item()
    coarse_n1 = n1_grid[n1_idx].item()
    coarse_n2 = n2_grid[n2_idx].item()
    coarse_n3 = n3_grid[n3_idx].item()
    
    coarse_time = time.time() - start_time

    # --- PHASE 3: Residue-Refined Lattice Lock (The "Micro-Net") ---
    # We construct a dense, high-resolution grid *only* around the coarse estimate.
    # M is structurally rigid (integers), so we don't refine it. We refine the continuous curvatures.
    
    # We span a +/- 10% bracket around the coarse coordinates.
    lock_time_start = time.time()
    
    n1_fine = torch.linspace(coarse_n1 * 0.8, coarse_n1 * 1.2, 50)
    n2_fine = torch.linspace(coarse_n2 * 0.8, coarse_n2 * 1.2, 50)
    n3_fine = torch.linspace(coarse_n3 * 0.8, coarse_n3 * 1.2, 50)

    # Re-Broadcast the new Micro-Net (125,000 localized topologies)
    N1_F = n1_fine.view(-1, 1, 1, 1)
    N2_F = n2_fine.view(1, -1, 1, 1)
    N3_F = n3_fine.view(1, 1, -1, 1)
    
    # M and Freq are locked from the Coarse Jump
    PHI_F = phi.view(1, 1, 1, -1)
    
    term1_f = torch.abs(torch.cos(coarse_m * PHI_F / 4.0)) ** N2_F
    term2_f = torch.abs(torch.sin(coarse_m * PHI_F / 4.0)) ** N3_F
    R_grid_f = (term1_f + term2_f) ** (-1.0 / N1_F)
    S_grid_f = R_grid_f * torch.sin(PHI_F)

    # Snapping to the absolute continuous minimum
    target_view_f = target_signal.view(1, 1, 1, -1)
    mse_landscape_f = torch.mean((S_grid_f - target_view_f)**2, dim=-1)
    
    min_flat_idx_f = torch.argmin(mse_landscape_f).item()
    n1_f_idx, n2_f_idx, n3_f_idx = np.unravel_index(min_flat_idx_f, mse_landscape_f.shape)
    
    fine_n1 = n1_fine[n1_f_idx].item()
    fine_n2 = n2_fine[n2_f_idx].item()
    fine_n3 = n3_fine[n3_f_idx].item()
    
    lock_time = time.time() - lock_time_start
    total_time = time.time() - start_time

    return {
        "freq": sensed_freq,
        "m": coarse_m,
        "n1": fine_n1,
        "n2": fine_n2,
        "n3": fine_n3,
        "coarse_mse": mse_landscape[m_idx, n1_idx, n2_idx, n3_idx].item(),
        "fine_mse": mse_landscape_f[n1_f_idx, n2_f_idx, n3_f_idx].item(),
        "coarse_time_ms": coarse_time * 1000,
        "lock_time_ms": lock_time * 1000,
        "total_time_ms": total_time * 1000
    }

# ==========================================
# 3. Execution
# ==========================================
t = torch.linspace(0, 5, 500)
true_freq, true_m, true_n1, true_n2, true_n3 = 1.0, 4.0, 0.5, 0.2, 8.0
target_signal = gielis_oscillator(t, freq=true_freq, m=true_m, n1=true_n1, n2=true_n2, n3=true_n3)

print(">>> Testing Gen 6: Lattice Lock Refinement")
print(f"Ground Truth -> N1: {true_n1} | N2: {true_n2} | N3: {true_n3}")
print("-" * 50)

res = sense_with_lattice_lock(t, target_signal)

print(f"Coarse Jump     -> N1: {res['n1']:.4f} | N2: {res['n2']:.4f} | N3: {res['n3']:.4f} | MSE: {res['coarse_mse']:.6f} | Time: {res['coarse_time_ms']:.1f}ms")
print(f"Lattice Lock    -> N1: {res['n1']:.4f} | N2: {res['n2']:.4f} | N3: {res['n3']:.4f} | MSE: {res['fine_mse']:.8f} | Time: {res['lock_time_ms']:.1f}ms")
print(f"Total Compute   : {res['total_time_ms']:.1f}ms")
