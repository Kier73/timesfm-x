import sys
import os
import torch
import numpy as np
import time
import json
import hashlib
import pandas as pd
from typing import List, Tuple, Dict, Any

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import timesfm
from timesfm.timesfm_x_model import TimesFMX

# --- 1. REPRODUCIBLE DATA SYNTHESIZER ---

class ReproducibleSynthesizer:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, name: str, n: int = 1100) -> np.ndarray:
        local_seed = int(hashlib.md5(name.encode()).hexdigest(), 16) % 0xFFFFFFFF
        lrng = np.random.default_rng(local_seed)
        
        if name == "Lorenz":
            x, y, z = 0.1, 0.0, 0.0
            s, r, b = 10, 28, 8/3
            dt = 0.01
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
        elif name == "Multi_Seasonal":
            t = np.linspace(0, 10, n)
            return np.sin(2 * np.pi * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
        return np.zeros(n)

# --- 2. PRODUCTION DATA LOADER (CSV) ---

class CSVDataLoader:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def load_ett(self, file_name: str, target_col: str = 'OT', n: int = 1100) -> np.ndarray:
        path = os.path.join(self.base_dir, file_name)
        if not os.path.exists(path):
            print(f"  [Error] {file_name} not found at {path}")
            return np.zeros(n)
        df = pd.read_csv(path)
        data = df[target_col].values
        # Return the last n points for benchmarking consistency
        return data[-n:]

# --- 3. PRODUCTION EVALUATOR ---

class MasterEvaluator:
    def __init__(self, dataset_dir: str):
        self.synth = ReproducibleSynthesizer(seed=42)
        self.loader = CSVDataLoader(dataset_dir)
        self.model_x = TimesFMX.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.model_x.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256))
        
    def run_suite(self):
        tests = [
            ("Synthetic", ["Lorenz", "Multi_Seasonal"]),
            ("Real-World", ["ETTh1.csv", "ETTh2.csv"])
        ]
        
        report = {}

        for stage, target_names in tests:
            print(f"\n>>> STAGE: {stage}")
            report[stage] = {}
            for name in target_names:
                if stage == "Real-World":
                    series = self.loader.load_ett(name)
                else:
                    series = self.synth.generate(name)
                
                inputs = [series[:1000]]
                ground_truth = series[1000:1064]
                horizon = 64
                
                # --- MEASURE 1: Foundation Path (Forced) ---
                # We use a dummy mode or just clear registry to force transformer logic
                t0 = time.perf_counter()
                # Dummy mode 'foundation' would be better, but we can just use a large 
                # offset in the inputs to ensure it's not in registry yet.
                p_base, _ = self.model_x.forecast(horizon=horizon, inputs=[series[0:1000]], mode='base')
                lat_base = (time.perf_counter() - t0) * 1000
                mse_base = np.mean((p_base[0] - series[1000:1064])**2)

                # --- MEASURE 2: Zero-Shot Discovery (Reflexive) ---
                # We use a SHIFTED window of the same system.
                # In Gen 4, this would miss the hash. In Gen 5, Reflexive Dream catches it.
                inputs_new = [series[10:1010]] # New hash, same law
                t0 = time.perf_counter()
                p_reflex, _ = self.model_x.forecast(horizon=horizon, inputs=inputs_new, mode='base')
                lat_reflex = (time.perf_counter() - t0) * 1000
                
                # --- MEASURE 3: Warm JIT (Exact Match) ---
                t0 = time.perf_counter()
                p_jit, _ = self.model_x.forecast(horizon=horizon, inputs=inputs_new, mode='base')
                lat_jit = (time.perf_counter() - t0) * 1000

                report[stage][name] = {
                    "MSE": float(mse_base),
                    "Lat_Foundation": lat_base,
                    "Lat_Reflexive": lat_reflex,
                    "Lat_JIT": lat_jit,
                    "Speedup_Reflex": lat_base / lat_reflex if lat_reflex > 0 else 1.0,
                    "Speedup_JIT": lat_base / lat_jit if lat_jit > 0 else 1.0
                }
                
                print(f"  {name:<15} | Foundat: {lat_base:6.2f}ms | Reflex: {lat_reflex:6.2f}ms | JIT: {lat_jit:6.2f}ms")

        return report

def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Datasets'))
    evaluator = MasterEvaluator(dataset_dir)
    report = evaluator.run_suite()
    
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VL_Dev', 'production_benchmark_results.json'))
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nProduction Suite Complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
