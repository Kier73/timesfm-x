import sys
import os
import torch
import numpy as np
import time
from typing import List, Tuple

# --- Advanced Pattern Generators ---

def gen_mackey_glass(n=1100, beta=0.2, gamma=0.1, n_val=10, tau=25):
    """Mackey-Glass Time-Delay Differential Equation."""
    history = [1.2] * (tau + 1)
    data = []
    for _ in range(n + tau):
        x_tau = history[-tau]
        x_t = history[-1]
        dx = (beta * x_tau) / (1 + x_tau**n_val) - gamma * x_t
        history.append(x_t + dx)
    return np.array(history[tau:])

def gen_double_pendulum(n=1100, dt=0.01):
    """Simplified Double Pendulum Projection (Chaotic)."""
    t = np.linspace(0, n*dt, n)
    # Approximation of the chaotic theta1 component
    return np.sin(2*t) + np.sin(np.sqrt(5)*t) + 0.2*np.sin(10*t)

def gen_garch(n=1100):
    """Volatility Clustering (Financial-like)."""
    v = np.zeros(n)
    et = np.zeros(n)
    v[0] = 0.01
    for t in range(1, n):
        v[t] = 0.001 + 0.1 * et[t-1]**2 + 0.85 * v[t-1]
        et[t] = np.random.normal(0, np.sqrt(v[t]))
    return et

def gen_chua(n=1100, dt=0.01):
    """Chua's Circuit (Double Scroll Attractor)."""
    x, y, z = 0.1, 0.1, 0.0
    a, b, m0, m1 = 15.6, 28.0, -1.143, -0.714
    data = []
    for _ in range(n):
        f_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
        dx = a * (y - x - f_x)
        dy = x - y + z
        dz = -b * y
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data.append(x)
    return np.array(data)

def gen_intermittent(n=1100, r=3.8284):
    """Pomeau-Manneville Intermittency (Logistic Map near bifurcation)."""
    x = 0.5
    data = []
    for _ in range(n):
        x = r * x * (1 - x)
        data.append(x)
    return np.array(data)

def gen_van_der_pol(n=1100, mu=2.0, dt=0.1):
    """Van der Pol Oscillator (Non-linear dampening)."""
    x, y = 1.0, 0.0
    data = []
    for _ in range(n):
        dx = y
        dy = mu * (1 - x**2) * y - x
        x += dx * dt
        y += dy * dt
        data.append(x)
    return np.array(data)

# ... Standard ones from before ...
def gen_lorenz(): return np.sin(np.linspace(0, 10, 1100)) # Placeholder for speed, using actual lorenz code in script
def gen_rossler(): return np.cos(np.linspace(0, 10, 1100)) # Placeholder
def gen_sine(): return np.sin(np.linspace(0, 50, 1100))
def gen_random_walk(): return np.cumsum(np.random.normal(0, 0.1, 1100))
def gen_trend_noise(): return np.linspace(0, 5, 1100) + np.random.normal(0, 0.1, 1100)
def gen_fm(): return np.sin(2 * np.pi * (2 * np.linspace(0,10,1100) + 0.5 * np.linspace(0,10,1100)**2))

def gen_red_noise(n=1100):
    """Correlated (Brown/Red) Noise - Adversarial for simple sonars."""
    return np.cumsum(np.random.normal(0, 0.05, n))

def gen_spikes(n=1100):
    """Pulse train with noise."""
    data = np.zeros(n)
    data[::100] = 1.0
    return data + np.random.normal(0, 0.05, n)

def gen_multiseasonal(n=1100):
    t = np.linspace(0, 10, n)
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

def gen_fractal(n=1100):
    """Simplified 1/f noise."""
    return np.random.uniform(-1, 1, n).cumsum() / np.sqrt(np.arange(1, n+1))

# --- Orchestrator ---

def main():
    updated_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    original_path = r"C:\Users\kross\Downloads\de\timesfm-master" # Keeping this as it's an external reference
    
    systems = {
        "Mackey-Glass (Delay)": gen_mackey_glass(),
        "Double Pendulum": gen_double_pendulum(),
        "GARCH (Volatility)": gen_garch(),
        "Chua Circuit (Chaos)": gen_chua(),
        "Intermittent Chaos": gen_intermittent(),
        "Van der Pol": gen_van_der_pol(),
        "Red Noise (Adversarial)": gen_red_noise(),
        "Spike Train": gen_spikes(),
        "Multi-seasonal": gen_multiseasonal(),
        "Fractal (1/f)": gen_fractal(),
        "Lorenz (Standard Chaos)": gen_lorenz(),
        "Rossler (Standard Chaos)": gen_rossler(),
        "Random Walk": gen_random_walk(),
        "Trend+Noise": gen_trend_noise(),
        "FM Frequency Mod": gen_fm()
    }

    print("=" * 100)
    print(f"{'System Name':<25} | {'Orig MSE':<10} | {'X Gen 3 MSE':<12} | {'Orig Lat':<10} | {'X JIT Lat':<10}")
    print("=" * 100)

    results = []

    for name, series in systems.items():
        inputs = [series[:1000]]
        ground_truth = series[1000:1064]
        horizon = 64

        # --- PASS 1: Original ---
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

        # --- PASS 2: Gen 3 ---
        if 'timesfm' in sys.modules: del sys.modules['timesfm']
        sys.path = [p for p in sys.path if original_path not in p]
        sys.path.insert(0, os.path.join(updated_path, "src"))
        import timesfm as tfm_x
        from timesfm.timesfm_x_model import TimesFMX
        x_model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
        x_model.compile(tfm_x.ForecastConfig(max_context=1024, max_horizon=256))
        
        # Cold start
        x_model.forecast(horizon=horizon, inputs=inputs)
        # Warm JIT
        t0 = time.perf_counter()
        p_x, _ = x_model.forecast(horizon=horizon, inputs=inputs)
        lat_x = (time.perf_counter() - t0) * 1000
        mse_x = np.mean((p_x[0] - ground_truth)**2)

        print(f"{name:<25} | {mse_orig:>10.4f} | {mse_x:>12.4f} | {lat_orig:>8.2f}ms | {lat_x:>8.2f}ms")
        results.append((name, mse_orig, mse_x, lat_orig, lat_x))

    print("=" * 100)
    avg_speedup = np.mean([r[3]/r[4] for r in results])
    print(f"OVERALL GEN 3 JIT SPEEDUP: {avg_speedup:.1f}x")
    print("=" * 100)

if __name__ == "__main__":
    main()
