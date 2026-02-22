import torch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from timesfm.gvm_zs_engine import GVM_ZS_Engine

def test_engine():
    print(">>> Testing GVM_ZS_Engine (Gen 8)...")
    engine = GVM_ZS_Engine(order=10, seed=0x1234)
    
    Q = torch.randn(1, 10, 64)
    K = torch.randn(1, 10, 64)
    V = torch.randn(1, 10, 64)
    
    resonance = engine(Q, K, V)
    print(f"Resonance shape: {resonance.shape}")
    print(f"Sample peak [0, 0]: {resonance[0, 0, 0]:.6f}")
    
    # Test Materialization
    print("Materializing delta at (0, 0)...")
    engine.materialize(0, 0, 1.0)
    
    resonance_new = engine(Q, K, V)
    print(f"Sample peak [0, 0] after materialization: {resonance_new[0, 0, 0]:.6f}")
    
    if abs(resonance_new[0, 0, 0] - 1.0) < 1e-5:
        print("PASS: Materialization verified.")
    else:
        print("FAIL: Materialization failed.")

if __name__ == "__main__":
    test_engine()
