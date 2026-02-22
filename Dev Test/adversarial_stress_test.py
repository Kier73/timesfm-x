import sys
import os
import torch
import numpy as np
import time

def load_x_model(repo_path):
    sys.path.insert(0, os.path.join(repo_path, "src"))
    import timesfm
    from timesfm.timesfm_x_model import TimesFMX
    model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))
    return model

def main():
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model = load_x_model(repo_path)
    horizon = 64

    print("=" * 70)
    print("TIMESFM-X ADVERSARIAL STRESS TEST")
    print("=" * 70)

    # --- TEST 1: PURE NOISE (GHOSTING) ---
    print("\n[TEST 1] Pure White Noise Interaction...")
    noise = [np.random.normal(0, 1.0, 1000)]
    p_noise, _ = model.forecast(horizon=horizon, inputs=noise)
    # Check if sonar snapped incorrectly
    print(f"  Resultant Mean Pred: {np.mean(np.abs(p_noise)):.6f}")
    if np.mean(np.abs(p_noise)) > 0.05:
        print("  WARNING: Possible 'Ghosting' detected. Model found a law in pure noise.")
    else:
        print("  PASS: Model remained neutral on noise.")

    # --- TEST 2: STRUCTURAL BREAK (FREQUENCY SHIFT) ---
    print("\n[TEST 2] Structural Break (Sine Freq Shift at Context Boundary)...")
    # Context is 10Hz, but the future is 20Hz
    t_context = np.linspace(0, 10, 1000)
    t_horizon = np.linspace(10, 10.64, 64)
    data_context = np.sin(2 * np.pi * 1.0 * t_context) # 1Hz
    data_ground = np.sin(2 * np.pi * 5.0 * t_horizon)  # 5Hz Shift
    
    p_break, _ = model.forecast(horizon=horizon, inputs=[data_context])
    mse_break = np.mean((p_break[0] - data_ground)**2)
    print(f"  Structural Break MSE: {mse_break:.4f}")
    print("  Note: This measures if the Spectral Jump resists the new (unseen) frequency.")

    # --- TEST 3: PRECISION LIMITS (10^15 SCALE) ---
    print("\n[TEST 3] Precision Limit Test (10^15 Scale)...")
    scale = 1e15
    original_sine = np.sin(np.linspace(0, 10, 1000))
    scaled_sine = [original_sine * scale]
    
    # Trigger Cold Start
    model.forecast(horizon=horizon, inputs=scaled_sine)
    # Trigger Shunt/Warm Start
    t0 = time.perf_counter()
    p_scaled, _ = model.forecast(horizon=horizon, inputs=scaled_sine)
    lat_scaled = (time.perf_counter() - t0) * 1000
    
    if lat_scaled < 1.0:
        print(f"  PASS: Shunted successfully at 1e15 scale ({lat_scaled:.2f}ms).")
    else:
        print(f"  FAIL: Shunt missed at extreme scale. Possible float absorption in signature.")

    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
