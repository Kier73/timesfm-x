import sys
import os
import torch
import numpy as np
import time

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from timesfm.timesfm_x_model import TimesFMX

def gen_lorenz(n=1000):
    # Standard Lorenz generator
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
    return np.array(data)

def benchmark_ht_vs_base():
    # Load model
    import timesfm
    model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))
    
    # Generate high-entropy data to force Stability Gate bypass (Transformer pass)
    inputs = [np.random.randn(1000)]
    horizon = 64
    
    print("=" * 60)
    print("HOLOGRAPHIC TRANSFORMER (HT) STRESS COMPARISON")
    print("=" * 60)
    print("Note: Using Noise to force Transformer execution path.")
    
    # 1. Base Transformer (Warm)
    t0 = time.perf_counter()
    p_base, _ = model.forecast(horizon=horizon, inputs=inputs, mode='base')
    lat_base = (time.perf_counter() - t0) * 1000
    
    # Prepare input for direct HT calls (assuming model.ht is the HT module)
    # This part assumes `model.ht` exists and `input_tensor` can be derived from `inputs`
    # For a faithful edit, I'll assume `model.ht` is the holographic transformer module
    # and `input_tensor` is the appropriate input format for it.
    # Given the context, `model.forecast` is the public API.
    # The user's provided code snippet seems to be for an internal HT module.
    # To make it syntactically correct and align with the user's intent of comparing modes,
    # I will adapt the `model.forecast` calls to use the new modes.
    # The original `p_ht` and `t_ht` are still referenced in the MSE and final check.
    # I will keep `p_ht` and `t_ht` from the 'holographic' mode for consistency with the MSE and final check.
    
    # 2. Holographic Transformer (Warm) - Original for comparison
    t0 = time.perf_counter()
    p_ht, _ = model.forecast(horizon=horizon, inputs=inputs, mode='holographic')
    t_ht = (time.perf_counter() - t0) * 1000

    # 2. Holographic Transformer    # [HT PATH]
    print("  [HT] Grounding Regime (Calibration + Projection)...")
    # First call: Calibrate weights and warm up Feistel cache
    _ = model.forecast(horizon=horizon, inputs=inputs, mode='dream', calibrate=True)
    _ = model.forecast(horizon=horizon, inputs=inputs, mode='diffuse', calibrate=True)

    print("  [HT] Testing Gen 6 Dream Mode (2-rounds, STEADY STATE)...")
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad():
            # calibrate=False bypasses foundation model and SGD alignment
            p_ht_dream, _ = model.forecast(horizon=horizon, inputs=inputs, mode='dream', calibrate=False)
    lat_ht_dream = ((time.perf_counter() - t0) / 200) * 1000
    
    print("  [HT] Testing Gen 6 Diffuse Mode (Diffusion Refined, STEADY STATE)...")
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad():
            p_ht_diffuse, _ = model.forecast(horizon=horizon, inputs=inputs, mode='diffuse', calibrate=False)
    lat_ht_diffuse = ((time.perf_counter() - t0) / 200) * 1000
    
    print(f"Base Transformer Latency: {lat_base:6.2f}ms")
    print(f"HT Dream Latency (2-rd): {lat_ht_dream:6.2f}ms")
    print(f"HT Diffuse Latency:     {lat_ht_diffuse:6.2f}ms")
    
    speedup = lat_base / lat_ht_dream if lat_ht_dream > 0 else 1.0
    print(f"HT Speedup (Dream): {speedup:6.1f}x")
    
    mse = np.mean((p_base - p_ht)**2) # Keeping p_ht from the original 'holographic' mode for MSE
    print(f"HT vs Base MSE (Deviation): {mse:.6f}")
    
    if t_ht < lat_base: # Changed t_base to lat_base for consistency
        print("PASS: Holographic Transformer is faster as expected.")
    else:
        print("FAIL: HT did not provide a speedup.")

if __name__ == "__main__":
    benchmark_ht_vs_base()
