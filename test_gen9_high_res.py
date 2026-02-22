import torch
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import List, Tuple

# --- ENVIRONMENT SETUP ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from timesfm.timesfm_x_model import TimesFMX

def print_section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def test_gen9_ettm1_stress_test():
    print_section("GEN 9 HIGH-RESOLUTION STRESS TEST: ETTm1 (15-MINUTE DATA)")
    
    # 1. Load high-resolution data
    data_path = os.path.join(ROOT_DIR, 'Datasets', 'ETTm1.csv')
    if not os.path.exists(data_path):
        print(f"  [Error] Dataset ETTm1.csv not found.")
        return
        
    df = pd.read_csv(data_path)
    print(f"  [Dataset] Loaded {len(df)} points from ETTm1 (15-min intervals).")
    
    # Focus on 'OT' (Oil Temperature)
    target_series = df['OT'].values
    
    # 2. Sequential Sliding Window Test
    # We test multiple windows to see if holographic consensus holds across different temporal regimes
    context_len = 1024
    num_windows = 5
    print(f"  [Action] Running {num_windows} sequential holographic projections...")
    
    model = TimesFMX()
    
    # Mock foundation to isolate Gen 9 dynamics
    from unittest.mock import MagicMock
    import timesfm.timesfm_2p5.timesfm_2p5_torch as torch_mod
    torch_mod.TimesFM_2p5_200M_torch.forecast = MagicMock(return_value=(np.zeros((1, 16)), np.zeros((1, 16, 9))))
    
    latencies = []
    for i in range(num_windows):
        offset = i * 1000
        window = [target_series[offset : offset + context_len]]
        
        start = time.perf_counter()
        # Engage 'holographic' mode for pure GVM/HT shunting
        model.forecast(horizon=16, inputs=window, mode='holographic')
        end = time.perf_counter()
        
        ms = (end - start) * 1000
        latencies.append(ms)
        print(f"    - Window {i+1} [Offset {offset}]: {ms:.4f} ms")
        
    avg_latency = sum(latencies) / num_windows
    print(f"  [Result] Average Holographic Latency: {avg_latency:.4f} ms")
    
    # 3. High-Frequency Adaptation Test
    print_section("HIGH-FREQUENCY RECEPTION: GVM OVERLAY SATURATION")
    
    # Simulating a rapid noise-injection/burst on ETTm1 data
    burst_data = target_series[:context_len].copy()
    # Add a high-frequency jitter pulse at the end
    burst_data[-10:] += np.random.normal(0, 10.0, 10)
    
    print(f"  [Action] Injecting 10-point chaotic burst and triggering Deep Calibration...")
    
    # calibrate=True engages Full-Manifold Feedback materialization
    model.forecast(horizon=16, inputs=[burst_data], mode='diffuse', calibrate=True)
    
    # Verify materialization depth
    materialized_count = 0
    for layer in model.ht.layers:
        if hasattr(layer.attn, 'wavefront') and layer.attn.wavefront.h_cache is not None:
             # We check the last 10 points
             deltas = [layer.attn.wavefront.h_cache[context_len-1-k, context_len-1-k].item() for k in range(10)]
             materialized_count = sum(1 for d in deltas if abs(d) > 0)
             if materialized_count > 0:
                 print(f"  [Result] Layer {layer.__class__.__name__}: Materialized {materialized_count}/10 points in burst window.")
                 break
    
    if materialized_count >= 5: # Threshold for high-density capture
        print(f"  [Status] PASS: High-frequency regime captured by GVM Sparse Overlay.")
    else:
        print(f"  [Status] FAIL: Insufficient materialization for high-res burst.")

if __name__ == "__main__":
    try:
        test_gen9_ettm1_stress_test()
    except Exception as e:
        print(f"\n  [ERROR] High-resolution test failed: {e}")
        import traceback
        traceback.print_exc()
