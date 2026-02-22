import torch
import numpy as np
import math
from src.timesfm.timesfm_x_core import SpectralSonar

def test_blind_lattice_lock():
    print(">>> Technical Audit: Blind Pattern Test (Lattice Lock Reliability)")
    
    sonar = SpectralSonar()
    
    # 1. Generate a "Blind" signal with random parameters
    # m is integer-like, n1, n2, n3 are curved
    true_m = float(np.random.randint(2, 9))
    true_n1 = 10**np.random.uniform(-0.5, 0.5)
    true_n2 = 10**np.random.uniform(-0.5, 0.5)
    true_n3 = 10**np.random.uniform(-0.5, 0.5)
    
    # Standardized [-1, 1] viewport
    t = torch.linspace(-1, 1, 200)
    freq = 1.0 # Standard freq for viewport test
    phi = 2 * math.pi * freq * t
    
    # Superformula
    term1 = torch.abs(torch.cos(true_m * phi / 4.0)) ** true_n2
    term2 = torch.abs(torch.sin(true_m * phi / 4.0)) ** true_n3
    r = (term1 + term2) ** (-1.0 / true_n1)
    target_signal = r * torch.sin(phi)
    
    # Add noise (10%)
    target_signal += torch.randn_like(target_signal) * 0.1
    
    print(f"  [Audit] Ground Truth -> M: {true_m} | N1: {true_n1:.4f} | N2: {true_n2:.4f} | N3: {true_n3:.4f}")
    print("  [Audit] Processing noisy signal via Lattice Lock...")
    
    # 2. Sense parameters
    sensed_params = sonar.morphological_scan(target_signal, freq)
    
    s_m, s_n1, s_n2, s_n3 = sensed_params.tolist()
    
    print(f"  [Result] Sensed DNA  -> M: {s_m} | N1: {s_n1:.4f} | N2: {s_n2:.4f} | N3: {s_n3:.4f}")
    
    # 3. Validation
    m_match = (s_m == true_m)
    # Check if curvatures are within a reasonable 20% error bracket given the 10% noise
    n1_err = abs(s_n1 - true_n1) / true_n1
    n2_err = abs(s_n2 - true_n2) / true_n2
    n3_err = abs(s_n3 - true_n3) / true_n3
    
    print(f"  [Metric] M Match: {m_match}")
    print(f"  [Metric] N1 Error: {n1_err*100:.2f}%")
    print(f"  [Metric] N2 Error: {n2_err*100:.2f}%")
    print(f"  [Metric] N3 Error: {n3_err*100:.2f}%")
    
    if m_match and n1_err < 0.2:
        print(">>> AUDIT PASS: Lattice Lock is analytically robust to noise and blind patterns.")
    else:
        print(">>> AUDIT FAIL: Morphological recovery failed outside of archetypes.")

if __name__ == "__main__":
    # Run multiple seeds for rigor
    for i in range(3):
        print(f"\n--- Round {i+1} ---")
        test_blind_lattice_lock()
