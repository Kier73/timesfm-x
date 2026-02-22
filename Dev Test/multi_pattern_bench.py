import sys
import os
import torch
import numpy as np
import time
from typing import List, Tuple

# --- Pattern Generators ---

def gen_lorenz(n=1100, dt=0.01):
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

def gen_henon(n=1100, a=1.4, b=0.3):
    x, y = 0.1, 0.0
    data = []
    for _ in range(n):
        x_next = 1 - a * x**2 + y
        y_next = b * x
        x, y = x_next, y_next
        data.append(x)
    return np.array(data)

def gen_rossler(n=1100, dt=0.01):
    x, y, z = 0.1, 0.0, 0.0
    a, b, c = 0.2, 0.2, 5.7
    data = []
    for _ in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append(x)
    return np.array(data)

def gen_sine(n=1100):
    t = np.linspace(0, 50, n)
    return np.sin(t)

def gen_sawtooth(n=1100):
    t = np.linspace(0, 50, n)
    return 2 * (t / (2 * np.pi) - np.floor(0.5 + t / (2 * np.pi)))

def gen_square(n=1100):
    t = np.linspace(0, 50, n)
    return np.sign(np.sin(t))

def gen_random_walk(n=1100):
    return np.cumsum(np.random.normal(0, 0.1, n))

def gen_trend_noise(n=1100):
    t = np.linspace(0, 10, n)
    return 0.5 * t + np.random.normal(0, 0.1, n)

def gen_seasonal(n=1100):
    t = np.linspace(0, 10, n)
    return 0.5 * t + np.sin(2 * np.pi * t) + np.random.normal(0, 0.05, n)

def gen_fm(n=1100):
    t = np.linspace(0, 10, n)
    return np.sin(2 * np.pi * (2 * t + 0.5 * t**2))

# --- Benchmarking Orchestrator ---

def main():
    original_path = r"C:\Users\kross\Downloads\de\timesfm-master"
    updated_path = r"C:\Users\kross\Downloads\timesfm-master"
    
    patterns = {
        "Lorenz (Chaos)": gen_lorenz(),
        "Henon (Chaos)": gen_henon(),
        "Rossler (Chaos)": gen_rossler(),
        "Sine (Periodic)": gen_sine(),
        "Sawtooth (Periodic)": gen_sawtooth(),
        "Square (Periodic)": gen_square(),
        "Random Walk (Stoch)": gen_random_walk(),
        "Trend + Noise": gen_trend_noise(),
        "Seasonal": gen_seasonal(),
        "FM Wave": gen_fm()
    }

    print("=" * 90)
    print(f"{'Pattern System':<25} | {'Orig MSE':<10} | {'X MSE':<10} | {'Orig Lat':<10} | {'X JIT Lat':<10}")
    print("=" * 90)

    # 1. Initialize result trackers
    results = []

    # 2. Setup Loop
    for name, series in patterns.items():
        # Setup data
        inputs = [series[:1000]]
        ground_truth = series[1000:1064]
        horizon = 64

        # Pass 1: Original
        if 'timesfm' in sys.modules: del sys.modules['timesfm']
        sys.path = [p for p in sys.path if "timesfm" not in p]
        sys.path.insert(0, os.path.join(original_path, "src"))
        import timesfm as tfm_orig
        orig_model = tfm_orig.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        orig_model.compile(tfm_orig.ForecastConfig(max_context=1024, max_horizon=256))
        
        t0 = time.perf_counter()
        p_orig, _ = orig_model.forecast(horizon=horizon, inputs=inputs)
        lat_orig = (time.perf_counter() - t0) * 1000
        mse_orig = np.mean((p_orig[0] - ground_truth)**2)

        # Pass 2: TimesFM-X Gen 2
        if 'timesfm' in sys.modules: del sys.modules['timesfm']
        sys.path = [p for p in sys.path if original_path not in p]
        sys.path.insert(0, os.path.join(updated_path, "src"))
        import timesfm as tfm_x
        from timesfm.timesfm_x_model import TimesFMX
        x_model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
        x_model.compile(tfm_x.ForecastConfig(max_context=1024, max_horizon=256))
        
        # Cold pass (to induce law)
        x_model.forecast(horizon=horizon, inputs=inputs)
        
        # Warm JIT pass
        t0 = time.perf_counter()
        p_x, _ = x_model.forecast(horizon=horizon, inputs=inputs)
        lat_x_jit = (time.perf_counter() - t0) * 1000
        mse_x = np.mean((p_x[0] - ground_truth)**2)

        print(f"{name:<25} | {mse_orig:>10.4f} | {mse_x:>10.4f} | {lat_orig:>8.2f}ms | {lat_x_jit:>8.2f}ms")
        results.append((name, mse_orig, mse_x, lat_orig, lat_x_jit))

    print("=" * 90)
    avg_speedup = np.mean([r[3]/r[4] for r in results])
    print(f"AVERAGE SPECTRAL JIT SPEEDUP: {avg_speedup:.1f}x")
    print("=" * 90)

if __name__ == "__main__":
    main()
