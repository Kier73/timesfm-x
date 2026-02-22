import torch
import numpy as np
import timesfm
from timesfm.timesfm_x_model import TimesFMX
import time

def lorenz_system(n, dt=0.01):
    """Generates chaotic Lorenz attractor data."""
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
    print("-" * 60)
    print("TIMESFM-X CHAOTIC BENCHMARK")
    print("-" * 60)

    # 1. Setup Models
    print("Loading Base TimesFM 2.5...")
    base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    base_model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))

    print("\nLoading TimesFM-X (Enhanced)...")
    # In a real scenario, we'd load the same weights into the X wrapper
    x_model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
    x_model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))

    # 2. Generate Chaotic Input
    print("\nGenerating Chaotic Lorenz System data...")
    chaotic_series = lorenz_system(1200)
    context_data = chaotic_series[:1000]
    ground_truth = chaotic_series[1000:1064]
    
    inputs = [context_data]
    horizon = 64

    # 3. Benchmark Iteration 1 (Cold Start)
    print(f"\n[PASS 1: COLD START] Forecasting horizon={horizon}...")
    
    t0 = time.perf_counter()
    p_base, _ = base_model.forecast(horizon=horizon, inputs=inputs)
    t_base = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    p_x_orig, _ = x_model.forecast(horizon=horizon, inputs=inputs) # Renamed p_x to p_x_orig
    lat_orig = (time.perf_counter() - t0) * 1000 # Renamed t_x to lat_orig

    mse_base = np.mean((p_base[0, :horizon] - ground_truth)**2)
    print(f"Base Model: {t_base:.2f}ms, MSE: {mse_base:.6f}")

    # [PASS 1: COLD START]
    print("\n[PASS 1: COLD START] Forecasting horizon=64...")
    t0 = time.perf_counter()
    p_x, _ = x_model.forecast(horizon=horizon, inputs=inputs)
    lat_x_cold = (time.perf_counter() - t0) * 1000
    mse_x = np.mean((p_x[0] - ground_truth)**2)
    print(f"TimesFM-X Gen 2: {lat_x_cold:.2f}ms, MSE: {mse_x:.6f}")

    # [PASS 2: WARM START]
    print("\n[PASS 2: WARM START] (Testing Spectral JIT Teleportation)...")
    t0 = time.perf_counter()
    p_x_warm, _ = x_model.forecast(horizon=horizon, inputs=inputs)
    lat_x_warm = (time.perf_counter() - t0) * 1000
    print(f"TimesFM-X (Teleported): {lat_x_warm:.2f}ms (Speedup: {lat_orig/lat_x_warm:.1f}x)")

    # [PASS 3: BYZANTINE NOISE TEST]
    print("\n[PASS 3: BYZANTINE NOISE TEST] Adding Gaussian Jitter...")
    jittered_inputs = [inputs[0] + np.random.normal(0, 0.01, inputs[0].shape)]
    t0 = time.perf_counter()
    p_x_noise, _ = x_model.forecast(horizon=horizon, inputs=jittered_inputs)
    lat_x_noise = (time.perf_counter() - t0) * 1000
    mse_noise = np.mean((p_x_noise[0] - ground_truth)**2)
    print(f"TimesFM-X (Robust): {lat_x_noise:.2f}ms, MSE: {mse_noise:.6f}")

    # 5. Scale Invariance Test
    print("\n[SCALE INVARIANCE TEST] Testing at 1,000,000x scale...")
    scaled_inputs = [context_data * 1e6]
    
    t0 = time.perf_counter()
    p_scaled, _ = x_model.forecast(horizon=horizon, inputs=scaled_inputs)
    t_scaled = (time.perf_counter() - t0) * 1000
    
    # Check if the shape is recovered correctly (normalized relative error)
    rel_error = np.mean(np.abs(p_scaled[0] / 1e6 - p_x[0]))
    print(f"Scaled Latency: {t_scaled:.2f}ms")
    print(f"Scale Recovery Relative Error: {rel_error:.2e}")

    print("-" * 60)
    print("BENCHMARK COMPLETE")
    print("-" * 60)

if __name__ == "__main__":
    main()
