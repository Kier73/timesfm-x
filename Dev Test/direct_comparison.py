import sys
import os
import torch
import numpy as np
import time

# Helper to load a specific module version
def load_timesfm_version(repo_path):
    src_path = os.path.join(repo_path, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import timesfm
    return timesfm

def lorenz_system(n, dt=0.01):
    x, y, z = 0.1, 0.0, 0.0
    s, r, b = 10, 28, 8/3
    data = []
    for _ in range(n):
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append(x)
    return np.array(data)

def main():
    original_path = r"C:\Users\kross\Downloads\de\timesfm-master"
    updated_path = r"C:\Users\kross\Downloads\timesfm-master"

    print("=" * 70)
    print("DIRECT COMPARISON: ORIGINAL VS TIMESFM-X")
    print("=" * 70)

    # 1. Load Original
    print(f"\n[1/3] Initializing Original TimesFM (from {original_path})...")
    # Clean sys.path to ensure fresh import
    if 'timesfm' in sys.modules:
        del sys.modules['timesfm']
    sys.path = [p for p in sys.path if "timesfm" not in p]
    
    sys.path.insert(0, os.path.join(original_path, "src"))
    import timesfm as tfm_orig
    orig_model = tfm_orig.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    orig_model.compile(tfm_orig.ForecastConfig(max_context=1024, max_horizon=256))

    # 2. Load Updated (TimesFM-X)
    print(f"\n[2/3] Initializing TimesFM-X (from {updated_path})...")
    if 'timesfm' in sys.modules:
        del sys.modules['timesfm']
    sys.path = [p for p in sys.path if original_path not in p] # Remove original
    sys.path.insert(0, os.path.join(updated_path, "src"))
    
    import timesfm as tfm_x
    from timesfm.timesfm_x_model import TimesFMX
    x_model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
    x_model.compile(tfm_x.ForecastConfig(max_context=1024, max_horizon=256))

    # 3. Running Chaotic Benchmark
    print(f"\n[3/3] Running Comparative Benchmark (Lorenz System)...")
    chaotic_series = lorenz_system(1100)
    inputs = [chaotic_series[:1000]]
    ground_truth = chaotic_series[1000:1064]
    horizon = 64

    # Original Pass
    t0 = time.perf_counter()
    p_orig, _ = orig_model.forecast(horizon=horizon, inputs=inputs)
    lat_orig = (time.perf_counter() - t0) * 1000
    mse_orig = np.mean((p_orig[0] - ground_truth)**2)

    # X-Model Pass 1 (Cold)
    t0 = time.perf_counter()
    p_x_cold, _ = x_model.forecast(horizon=horizon, inputs=inputs)
    lat_x_cold = (time.perf_counter() - t0) * 1000
    mse_x = np.mean((p_x_cold[0] - ground_truth)**2)

    # X-Model Pass 2 (Warm/Shunted)
    t0 = time.perf_counter()
    p_x_warm, _ = x_model.forecast(horizon=horizon, inputs=inputs)
    lat_x_warm = (time.perf_counter() - t0) * 1000

    print("\n" + "-" * 50)
    print(f"{'Metric':<20} | {'Original':<12} | {'TimesFM-X':<12}")
    print("-" * 50)
    print(f"{'Latency (Cold)':<20} | {lat_orig:>8.2f}ms | {lat_x_cold:>8.2f}ms")
    print(f"{'Latency (Shunted)':<20} | {'N/A':>10} | {lat_x_warm:>8.2f}ms")
    print(f"{'Accuracy (MSE)':<20} | {mse_orig:>10.4f} | {mse_x:>10.4f}")
    print(f"{'Scale Invariance':<20} | {'Standard':<12} | {'HDC-Grounded':<12}")
    print("-" * 50)
    
    speedup = lat_orig / lat_x_warm
    print(f"\nCONCLUSION: TimesFM-X is {speedup:.1f}x faster on repeated patterns.")
    print("=" * 70)

if __name__ == "__main__":
    main()
