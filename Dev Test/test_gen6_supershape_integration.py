import numpy as np
import sys
import os
import torch

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from timesfm.timesfm_x_model import TimesFMX

def test_gen6_integration():
    print(">>> Testing TimesFM-X Gen 6: Supershape & Sparse-HDC Integration")
    
    # Initialize model
    model = TimesFMX()
    
    # Mock foundation to avoid compilation error in research env
    print("  [Note] Mocking foundation path for isolated manifold test.")
    from unittest.mock import MagicMock
    import timesfm.timesfm_2p5.timesfm_2p5_torch as torch_mod
    torch_mod.TimesFM_2p5_200M_torch.forecast = MagicMock(return_value=(np.zeros((1, 16)), np.zeros((1, 16, 9))))
    
    # Generate spiky periodic signal (Star archetype)
    t = torch.linspace(0, 10, 100)
    # Using a high frequency to ensure resonance
    freq = 2.0
    phi = 2 * np.pi * freq * t
    # Star-like params: m=5, n1=0.5
    r = (torch.abs(torch.cos(5 * phi / 4.0))**1.0 + torch.abs(torch.sin(5 * phi / 4.0))**1.0)**(-1.0 / 0.5)
    signal = r * torch.sin(phi)
    
    inputs = [signal.numpy()]
    
    print("  [Step 1] Initial Forecast (Discovery & Induction)...")
    p1, _ = model.forecast(horizon=16, inputs=inputs, mode='base')
    
    # Verify morphological tracking
    print("  [Step 2] Verifying Morphological State...")
    # Check if a law was induced with morphological params
    laws = model.registry.laws
    print(f"    Active Laws in Registry: {len(laws)}")
    
    morph_index_size = model.registry.morph_index.ntotal
    print(f"    Morphological Index Size: {morph_index_size}")
    
    if morph_index_size > 0:
        print("    PASS: Morphological Manifold grounding successful.")
    else:
        print("    FAIL: Morphological Manifold was not indexed.")

    print("  [Step 3] Verification of Zero-Shot Shunting (Hotswap)...")
    # Call again - should be much faster via shunting
    p2, _ = model.forecast(horizon=16, inputs=inputs, mode='base')
    
    # Check hits on laws
    hits = sum(law.hits for law in model.registry.laws.values())
    print(f"    Total Registry Hits: {hits}")
    
    if hits > 1:
         print("    PASS: Zero-Shot Morphological Shunting active.")
    else:
         print("    FAIL: Shunting mechanism did not trigger.")

    print("  [Step 4] Checking 4D Ana-Kata Tracker...")
    if len(model.registry.ana_kata_tracker) > 0:
        print(f"    Tracking {len(model.registry.ana_kata_tracker)} manifold regime(s).")
        print("    PASS: Ana-Kata topology tracking online.")
    else:
        print("    FAIL: Ana-Kata tracker is empty.")

if __name__ == "__main__":
    test_gen6_integration()
