import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from .timesfm_2p5 import timesfm_2p5_torch
from . import timesfm_x_core
from . import configs
from .holographic_transformer import HolographicTransformer

class TimesFMX(timesfm_2p5_torch.TimesFM_2p5_200M_torch):
    """
    TimesFM-X: Enhanced with Spectral Sonar and vGPU-style Shunting.
    Supports dual-path forecasting: Base Transformer and Holographic Transformer.
    """
    def __init__(self, **kwargs):
        # Base class init happens via from_pretrained usually, but we define components here
        self.hdc = timesfm_x_core.HypervectorEngine()
        self.sonar = timesfm_x_core.SpectralSonar()
        self.registry = timesfm_x_core.PatternRegistry(self.hdc)
        self.trinity = timesfm_x_core.TrinityConsensus(self.hdc)
        # Initialize Holographic Transformer (HT)
        self.ht = HolographicTransformer(embed_dim=1280, n_layers=4) # Matches 200M embed dim approx
        print("TimesFM-X Gen 3 initialized: Dual-Path (Base/Holographic) Engine online.")

    def forecast(self, horizon: int, inputs: List[np.ndarray], mode: str = 'base', calibrate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecasting with Gen 5/6: Zero-Shot Discovery and Semantic JIT.
        """
        output_points = []
        output_quantiles = []
        STABILITY_THRESHOLD = 0.05

        for each_input in inputs:
            # Prepare tensor for Sonar/HT
            input_t = torch.tensor(each_input, dtype=torch.float32, device=self.model.device)
            # Generate Hypervector for structural shunting
            input_hv = self.hdc.signal_to_hv(input_t)
            
            # --- PHASE 0: Mode Routing (Holographic Fast-Path) ---
            if mode in ('holographic', 'dream', 'diffuse'):
                # Gen 6: Separate Calibration (Slow) from Inference (Fast)
                if calibrate:
                    # Generate Base target for calibration (few-shot)
                    base_pf, base_qf = super().forecast(horizon, [each_input])
                    target_t = torch.tensor(base_pf, dtype=torch.float32, device=self.model.device)
                    self._calibrate_ht(input_t, target_t)
                    self.latest_qf = base_qf # Cache for headless mode
                else:
                    # Use cached or dummy quantiles in inference-only mode
                    base_qf = getattr(self, 'latest_qf', np.zeros((1, horizon, 9))) 
                
                # Gen 6: Mode Selection
                ht_mode = 'ground' if mode == 'holographic' else mode
                
                self.ht.eval()
                with torch.no_grad():
                    pf_final_t = self.ht(input_t.unsqueeze(0).unsqueeze(-1), mode=ht_mode)
                    pf_final = pf_final_t.detach().cpu().numpy()[:, -horizon:, 0]
                    output_points.append(pf_final)
                    output_quantiles.append(base_qf)
                continue

            # --- PHASE 1: JIT Teleportation (O(1) Law Recall + Reflexive Dream) ---
            event_sig = self.registry.get_signature(input_t)
            shunted_res = self.registry.shunt(event_sig, input_hv)
            if shunted_res is not None:
                output_points.append(shunted_res[0].cpu().numpy())
                output_quantiles.append(shunted_res[1].cpu().numpy())
                continue

            # --- PHASE 2: Spectral Analysis & Semantic JIT Gating ---
            # Gen 9 Adaptive Depth: Skip morphological search for holographic modes
            skip_m = mode in ('holographic', 'dream', 'diffuse')
            freq, beta, context_len, stability, gielis = self.sonar.stochastic_scan(input_t, skip_morphology=skip_m)
            
            # Gen 5/6: Semantic JIT Check
            periodic_resonance = torch.abs(beta[2:4]).max().item()
            is_periodic = periodic_resonance > 0.5 
            
            if stability < 0.05:
                intention_pf, qf_final = super().forecast(horizon, [each_input])
                pf_final = intention_pf
            else:
                # ...
                law_sig = self.registry.get_spectral_signature(beta)
                
                # Gen 5/6: Zero-Shot Shortcut
                shunted_res = self.registry.shunt(law_sig, beta=beta, gielis=gielis)
                if shunted_res is not None:
                    output_points.append(shunted_res[0].cpu().numpy())
                    output_quantiles.append(shunted_res[1].cpu().numpy())
                    continue
                
                # Gen 6: Determine HT Mode based on Resonance
                ht_mode = 'dream' if periodic_resonance > 0.8 else 'diffuse'
                
                input_tensor = torch.tensor(each_input).float().unsqueeze(0).unsqueeze(-1)
                law_pf_ht_t = self.ht(input_tensor, mode=ht_mode)
                law_pf_ht = law_pf_ht_t.detach().cpu().numpy()[:, -horizon:, 0]
                
                intention_pf, qf_final = super().forecast(horizon, [each_input])
                
                coherence = periodic_resonance
                blend_factor = min(0.6, coherence * 2.5) 
                
                pf_final = (1.0 - blend_factor) * intention_pf + blend_factor * law_pf_ht
                
                # Phase 5/6: Reflexive Induction (with Morphological Tracking)
                res_tensor = (torch.tensor(pf_final), torch.tensor(qf_final))
                self.registry.induce(event_sig, res_tensor, input_hv, law_sig=law_sig, beta=beta, gielis=gielis)
            output_points.append(pf_final)
            output_quantiles.append(qf_final)

        return np.concatenate(output_points, axis=0), np.concatenate(output_quantiles, axis=0)

    def _calibrate_ht(self, context: torch.Tensor, target: torch.Tensor, steps: int = 5):
        """Aligns HT output with Foundation Model targets using SGD."""
        optimizer = torch.optim.Adam(self.ht.calibration_head.parameters(), lr=0.1)
        self.ht.train()
        x = context.unsqueeze(0).unsqueeze(-1)
        
        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()
                pred = self.ht(x)[:, -target.shape[1]:, 0]
                if not pred.requires_grad:
                    # Fallback: force grad on leaf input to ensure graph construction
                    x.requires_grad_(True)
                    pred = self.ht(x)[:, -target.shape[1]:, 0]
                
                loss = torch.nn.functional.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                
        # Gen 9: Full-Manifold Feedback (Re-tuning the Generative Law)
        # Instead of just the first point, we harmonize the entire error residual.
        with torch.no_grad():
            final_pred_seq = self.ht(x)[:, -target.shape[1]:, 0] # [1, S]
            residuals = (target - final_pred_seq).squeeze(0) # [S]
            
            error_mean = torch.abs(residuals).mean().item()
            if error_mean > 0.005: # Global significance threshold
                # Materialize corrections along the temporal diagonal
                for s in range(residuals.shape[0]):
                    delta = residuals[s].item()
                    if abs(delta) > 0.001: # Individual point threshold
                        for layer in self.ht.layers:
                            if hasattr(layer.attn, 'wavefront'):
                                layer.attn.wavefront.materialize(s, s, delta)
                
                print(f"  [Gen 9] Full-Manifold Calibration: Materialized {residuals.shape[0]} points (Avg Error: {error_mean:.4f})")

