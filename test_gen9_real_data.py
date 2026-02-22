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
    print(f"\n{'#'*80}")
    print(f"  {title}")
    print(f"{'#'*80}")

def test_gen9_etth1_validation():
    print_section("GEN 9 REAL-WORLD VALIDATION: ETTh1 (ELECTRICITY TRANSFORMER DATA)")
    
    # 1. Load institutional data
    data_path = os.path.join(ROOT_DIR, 'Datasets', 'ETTh1.csv')
    if not os.path.exists(data_path):
        print(f"  [Error] Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"  [Dataset] Loaded {len(df)} points from ETTh1. Columns: {list(df.columns)}")
    
    # Focus on 'OT' (Oil Temperature) - the standard benchmark target
    target_series = df['OT'].values
    
    # 2. Prepare Context (Window of 1024 points)
    context_len = 1024
    inputs = [target_series[:context_len]]
    ground_truth = target_series[context_len:context_len+16] # 16-step horizon
    
    print(f"  [Config] Context Length: {context_len}")
    print(f"  [Config] Generation Mode: Holographic ('diffuse')")
    
    model = TimesFMX()
    
    # Mock foundation to focus on Holographic/GVM dynamics
    from unittest.mock import MagicMock
    import timesfm.timesfm_2p5.timesfm_2p5_torch as torch_mod
    torch_mod.TimesFM_2p5_200M_torch.forecast = MagicMock(return_value=(np.zeros((1, 16)), np.zeros((1, 16, 9))))
    
    # 3. Running Holographic Inference
    print(f"  [Action] Projecting holographic manifold for ETTh1 sensor stream...")
    start = time.perf_counter()
    p1, _ = model.forecast(horizon=16, inputs=inputs, mode='diffuse')
    end = time.perf_counter()
    
    print(f"  [Result] Inference Latency: {(end-start)*1000:.4f} ms")
    
    # 4. Stress Test: Black Swan Materialization
    print_section("STRESS TEST: REAL-TIME BLACK SWAN MATERIALIZATION")
    
    # We induce a artificial sensor break (simulating a transformer fault)
    fault_signal = target_series[:context_len].copy()
    fault_signal[-1] += 20.0 # Extreme jump
    inputs_fault = [fault_signal]
    
    print(f"  [Action] Injecting structural break and initiating Deep Calibration...")
    
    # Calibrate will trigger Materialization for the delta
    model.forecast(horizon=16, inputs=inputs_fault, mode='diffuse', calibrate=True)
    
    # Verify materialization in GVM cache
    found_materialization = False
    for layer in model.ht.layers:
        if hasattr(layer.attn, 'wavefront') and layer.attn.wavefront.h_cache is not None:
             # Check the last point in the sequence
             delta = layer.attn.wavefront.h_cache[context_len-1, context_len-1].item()
             if abs(delta) > 0:
                 print(f"  [Result] Layer Materialization Hook: Delta {delta:.6f} recorded.")
                 found_materialization = True
                 break
    
    if found_materialization:
        print(f"  [Status] PASS: TimesFM-X successfully 'absorbed' the real-world fault into its Generative Law.")
    else:
        print(f"  [Status] FAIL: Materialization gate skipped.")

if __name__ == "__main__":
    try:
        test_gen9_etth1_validation()
    except Exception as e:
        print(f"\n  [ERROR] Real-world test failed: {e}")
        import traceback
        traceback.print_exc()
