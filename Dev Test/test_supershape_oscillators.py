import torch
import numpy as np
import matplotlib.pyplot as plt

def gielis_radius(phi, m, a, b, n1, n2, n3):
    """
    Standard 2D Gielis Superformula.
    """
    term1 = torch.abs(torch.cos(m * phi / 4.0) / a) ** n2
    term2 = torch.abs(torch.sin(m * phi / 4.0) / b) ** n3
    return (term1 + term2) ** (-1.0 / n1)

def super_oscillator(t, freq, m, a, b, n1, n2, n3):
    """
    Generates a time-series signal based on the Superformula radius.
    """
    phi = 2 * np.pi * freq * t
    r = gielis_radius(phi, m, a, b, n1, n2, n3)
    return r * torch.sin(phi)

def test_supershape_oscillation():
    print(">>> Generating Supershape Oscillators...")
    t = torch.linspace(0, 5, 1000)
    
    # 1. Spiky Star Shape (Pulsed periodicity)
    s1 = super_oscillator(t, freq=1.0, m=5, a=1, b=1, n1=0.5, n2=1, n3=1)
    
    # 2. Rounded Square (Plateau periodicity)
    s2 = super_oscillator(t, freq=1.0, m=4, a=1, b=1, n1=100, n2=100, n3=100)
    
    # 3. Organic "Flower" (Harmonic complexity)
    s3 = super_oscillator(t, freq=1.0, m=6, a=1, b=1, n1=1, n2=7, n3=8)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(t.numpy(), s1.numpy(), label="Spiky Star (m=5)")
    plt.plot(t.numpy(), s2.numpy(), label="Rounded Square (m=4)")
    plt.plot(t.numpy(), s3.numpy(), label="Organic Flower (m=6)")
    plt.title("Gielis Super-Oscillators: Time-Series Basis Functions")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Save for user review
    out_path = "VL_Dev/supershape_oscillators.png"
    plt.savefig(out_path)
    print(f"Comparison plot saved to {out_path}")

if __name__ == "__main__":
    test_supershape_oscillation()
