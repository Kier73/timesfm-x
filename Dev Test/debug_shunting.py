import torch
import sys
import os
import numpy as np

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import timesfm
from timesfm.timesfm_x_model import TimesFMX

def debug_shunting():
    model = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))
    
    # Simple Sin signal
    data = [np.sin(np.linspace(0, 10, 1000))]
    horizon = 64
    
    print("--- PASS 1: Induction ---")
    sig1 = model.registry.get_signature(torch.tensor(data[0], dtype=torch.float32))
    print(f"Sig 1: {sig1}")
    model.forecast(horizon, data)
    
    print(f"Registry size: {len(model.registry.laws)}")
    print(f"Law sigs: {list(model.registry.laws.keys())}")
    
    print("--- PASS 2: Shunting ---")
    sig2 = model.registry.get_signature(torch.tensor(data[0], dtype=torch.float32))
    print(f"Sig 2: {sig2}")
    import time
    t0 = time.perf_counter()
    model.forecast(horizon, data)
    lat = (time.perf_counter() - t0) * 1000
    
    print(f"Pass 2 Latency: {lat:.2f}ms")
    if lat < 10.0:
        print("PASS: Shunting active.")
    else:
        print("FAIL: Shunting missed.")

if __name__ == "__main__":
    debug_shunting()
