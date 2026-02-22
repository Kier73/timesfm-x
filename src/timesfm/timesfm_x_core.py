import torch
import torch.nn as nn
import numpy as np
import math
import faiss
from .rns_kernels import RNS_Core
from .supershape_engine import SupershapeEngine
from typing import List, Tuple, Dict, Any, Optional

def fmix64(h: int) -> int:
    """MurmurHash3 64-bit finalizer mix for deterministic seeds."""
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h

class HypervectorEngine(nn.Module):
    """
    4096-bit Hypervector Engine (HDC) for holographic logic.
    Supports bind (XOR), bundle (bitwise majority), and permutation.
    """
    DIMENSION = 4096
    
    def __init__(self):
        super().__init__()

    def from_seed(self, seed: int) -> torch.Tensor:
        """Vectorized deterministic bit-vector generation."""
        # Use a deterministic torch RNG for speed
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed & 0xFFFFFFFF)
        bits = torch.randint(0, 2, (self.DIMENSION,), generator=rng, device='cpu').float()
        return bits * 2.0 - 1.0

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binding via element-wise multiplication (equivalent to XOR on {-1, 1})."""
        return a * b

    def bundle(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Bundling via bitwise majority (sign of sum)."""
        if not vectors:
            return torch.zeros(self.DIMENSION)
        stacked = torch.stack(vectors)
        return torch.sign(torch.sum(stacked, dim=0))

    def signal_to_hv(self, signal: torch.Tensor) -> torch.Tensor:
        """Vectorized signal-to-HV conversion. Returns RAW float HV for Eco-Sampling."""
        L = signal.shape[0]
        indices = torch.linspace(0, L-1, 100, dtype=torch.long)
        sampled = signal[indices]
        val_seeds = torch.round(sampled * 1000).long()
        
        combined_hv = torch.zeros(self.DIMENSION, device='cpu')
        for i, v in enumerate(val_seeds):
            combined_hv += self.from_seed(i ^ v.item())
        
        return combined_hv # Return raw for Gen 6 Eco-Sampling

    def get_sparse_hv(self, hv: torch.Tensor, sparsity: float = 0.5) -> torch.Tensor:
        """
        Gen 6: Sparse-HDC (Eco-Sampling).
        Uses topological energy (magnitude of raw HV) for index priority.
        """
        L = hv.shape[0]
        n_key = int(L * sparsity)
        
        # Rigorous Sampling: Select dimensions with the highest cumulative energy
        indices = torch.argsort(torch.abs(hv), descending=True)[:n_key]
        sparse_hv = torch.zeros_like(hv)
        # Sign-project only the key dimensions
        sparse_hv[indices] = torch.sign(hv[indices])
        return sparse_hv

class TrinityConsensus(nn.Module):
    """
    Resolves computational truth via the convergence of Law, Intention, and Event.
    Notation: T = Law ^ Intention ^ Event
    """
    def __init__(self, engine: HypervectorEngine):
        super().__init__()
        self.engine = engine

    def resolve(self, law_sig: int, intention_sig: int, event_sig: int) -> torch.Tensor:
        v_law = self.engine.from_seed(law_sig)
        v_int = self.engine.from_seed(intention_sig)
        v_eve = self.engine.from_seed(event_sig)
        
        # Trinity Binding
        return self.engine.bind(self.engine.bind(v_law, v_int), v_eve)

    def verify(self, candidate: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """Returns similarity score (cosine similarity)."""
        return torch.cos(torch.nn.functional.cosine_similarity(candidate.unsqueeze(0), ground_truth.unsqueeze(0))).item()

class SpectralSonar(nn.Module):
    """
    Koopman Lifting module: extracts dominant frequencies and builds a holographic model.
    Inspired by Variation 36: Spectral-Holographic Manifold Solver.
    """
    def __init__(self, sample_density: int = 12):
        super().__init__()
        self.sample_density = sample_density

    def scan(self, inputs: torch.Tensor, time_axis: Optional[torch.Tensor] = None) -> Tuple[float, torch.Tensor, int]:
        """
        Rigorous Spectral Scan: Standardizes any input scale to [-1, 1].
        """
        L = inputs.shape[0]
        if time_axis is None:
            # Assume uniform sampling: map [0, L-1] to [-1, 1]
            t_orig = torch.linspace(0, L-1, L, device=inputs.device)
            duration = L - 1
        else:
            t_orig = time_axis
            duration = (t_orig[-1] - t_orig[0]).item()
            
        # Standardized Viewport: map inputs to normalized pts in [-1, 1]
        pts = torch.linspace(-1, 1, L, device=inputs.device)
        
        yf = torch.fft.rfft(inputs)
        # cycles per sample in original time-series
        xf = torch.fft.rfftfreq(L, d=1.0)
        dominant_idx = torch.argmax(torch.abs(yf[1:])) + 1
        
        # Map back to angular frequency in the standardized [-1, 1] viewport
        # Original f_hz = dominant_idx / L
        # Standardized pts has duration 2.0
        # So f_std = (f_hz * duration) / 2.0 ? 
        # Actually, let's just find where the peaks are relative to the viewport.
        # k = cycles per duration. In [-1, 1], duration is 2.
        # So sensed_freq = dominant_idx / 2.0 (cycles per unit T)
        sensed_freq = float(dominant_idx) / 1.0 # Scale to something that fits the viewport well
        sensed_freq = sensed_freq * math.pi
        
        X = torch.stack([
            pts**2, pts, torch.sin(sensed_freq * pts), torch.cos(sensed_freq * pts), torch.ones_like(pts)
        ], dim=1)
        
        try:
            beta = torch.linalg.lstsq(X, inputs.unsqueeze(1)).solution.flatten()
        except Exception:
            beta = torch.zeros(5, device=inputs.device)
            
        return sensed_freq, beta, L

    def morphological_scan(self, inputs: torch.Tensor, freq: float) -> torch.Tensor:
        """
        Gen 6: Lattice Lock Hierarchical Search.
        Analytical sensing of Gielis Morphological DNA.
        """
        L = inputs.shape[0]
        t = torch.linspace(-1, 1, L, device=inputs.device)
        phi = 2 * math.pi * freq * t
        
        # --- PHASE 1: Coarse Jump (The Macro-Net) ---
        m_grid = torch.arange(1, 11, 1, dtype=torch.float32, device=inputs.device)
        n1_grid = torch.logspace(-1.0, 1.0, 20, device=inputs.device)
        n2_grid = torch.logspace(-1.0, 1.0, 20, device=inputs.device)
        n3_grid = torch.logspace(-1.0, 1.0, 20, device=inputs.device)
        
        # Project 80,000 ghost topologies
        ghost_matrix = SupershapeEngine.ghost_projection(phi, m_grid, n1_grid, n2_grid, n3_grid)
        # Rigorous Scaling: Normalize each candidate shape to ensure we match TOPOLOGY, not amplitude.
        ghost_matrix = (ghost_matrix - ghost_matrix.mean(dim=-1, keepdim=True)) / (ghost_matrix.std(dim=-1, keepdim=True) + 1e-6)
        
        # Quantum Collapse: Find absolute minimum resonance
        target_view = (inputs - inputs.mean()) / (inputs.std() + 1e-6)
        target_view = target_view.view(1, 1, 1, 1, -1)
        
        mse_landscape = torch.mean((ghost_matrix - target_view)**2, dim=-1)
        min_idx = torch.argmin(mse_landscape)
        m_idx, n1_idx, n2_idx, n3_idx = np.unravel_index(min_idx.item(), mse_landscape.shape)
        
        c_m = m_grid[m_idx].item()
        c_n1 = n1_grid[n1_idx].item()
        c_n2 = n2_grid[n2_idx].item()
        c_n3 = n3_grid[n3_idx].item()
        
        # --- PHASE 2: Lattice Lock Refinement (The Micro-Net) ---
        # Refine curvature parameters (+/- 20% bracket)
        n1_f = torch.linspace(c_n1 * 0.8, c_n1 * 1.2, 30, device=inputs.device)
        n2_f = torch.linspace(c_n2 * 0.8, c_n2 * 1.2, 30, device=inputs.device)
        n3_f = torch.linspace(c_n3 * 0.8, c_n3 * 1.2, 30, device=inputs.device)
        
        # Re-broadcast localized topologies (27,000 universes)
        # We reuse ghost_projection logic but with fixed M
        N1_F = n1_f.view(-1, 1, 1, 1)
        N2_F = n2_f.view(1, -1, 1, 1)
        N3_F = n3_f.view(1, 1, -1, 1)
        PHI_F = phi.view(1, 1, 1, -1)
        
        # Compute micro-grid
        # (Directly inline for speed)
        term1_f = torch.abs(torch.cos(c_m * PHI_F / 4.0)) ** N2_F
        term2_f = torch.abs(torch.sin(c_m * PHI_F / 4.0)) ** N3_F
        R_grid_f = (term1_f + term2_f) ** (-1.0 / N1_F)
        S_grid_f = R_grid_f * torch.sin(PHI_F)
        
        # Rigorous Scaling (Micro)
        S_grid_f = (S_grid_f - S_grid_f.mean(dim=-1, keepdim=True)) / (S_grid_f.std(dim=-1, keepdim=True) + 1e-6)
        
        mse_f = torch.mean((S_grid_f - target_view.view(1, 1, 1, -1))**2, dim=-1)
        min_f_idx = torch.argmin(mse_f)
        n1_fi, n2_fi, n3_fi = np.unravel_index(min_f_idx.item(), mse_f.shape)
        
        return torch.tensor([
            c_m, n1_f[n1_fi].item(), n2_f[n2_fi].item(), n3_f[n3_fi].item()
        ], device=inputs.device)

    def stochastic_scan(self, inputs: torch.Tensor, n_probes: int = 2, jitter: float = 0.05, skip_morphology: bool = False) -> Tuple[float, torch.Tensor, int, float, torch.Tensor]:
        """
        Gen 6 Langevin Sonar: Adaptive Depth for Gen 9.
        skip_morphology: If True, bypasses the massive 80k-topology search.
        """
        betas = []
        freqs = []
        for _ in range(n_probes):
            noise = torch.randn_like(inputs) * jitter
            f, b, _ = self.scan(inputs + noise)
            betas.append(b)
            freqs.append(f)
            
        stacked_betas = torch.stack(betas)
        mean_beta = torch.mean(stacked_betas, dim=0)
        std_beta = torch.std(stacked_betas, dim=0)
        mean_freq = sum(freqs) / n_probes
        
        stability_score = torch.abs(mean_beta[2:4]).max().item() / (1.0 + (std_beta / (torch.abs(mean_beta) + 1e-6)).mean().item())
        
        # Gen 6 Morphological Fit (Adaptive Skip)
        if not skip_morphology:
            gielis_params = self.morphological_scan(inputs, mean_freq)
        else:
            gielis_params = torch.zeros(4, device=inputs.device)
            
        return mean_freq, mean_beta, inputs.shape[0], stability_score, gielis_params

    def jump(self, freq: float, beta: torch.Tensor, horizon: int, context_len: int) -> torch.Tensor:
        """
        Performs a 'Holographic Jump' to predict the future based on the spectral model.
        """
        # Calculate step size based on context length over [-1, 1] range
        step_size = 2.0 / context_len
        start_x = 1.0 + step_size
        end_x = 1.0 + (horizon * step_size)
        
        grid = torch.linspace(start_x, end_x, horizon, device=beta.device)
        forecast = (
            beta[0] * grid**2 + 
            beta[1] * grid + 
            beta[2] * torch.sin(freq * grid) + 
            beta[3] * torch.cos(freq * grid) + 
            beta[4]
        )
        return forecast

class Law:
    """
    Represents a promoted algorithmic 'Law' (verified forecast manifold).
    """
    def __init__(self, sig: int, result: Any):
        self.sig = sig
        self.result = result
        self.confidence = 1.0
        self.hits = 1

class PatternRegistry:
    """
    Gen 4 Vectorized Induction Registry.
    Uses FAISS for sub-linear similarity search over 4k-bit hypervectors.
    """
    def __init__(self, engine: HypervectorEngine):
        self.engine = engine
        self.laws: Dict[int, Law] = {}
        # FAISS Index for similarity-based structural shunting (HDC)
        self.index = faiss.IndexFlatL2(engine.DIMENSION)
        self.index_sigs: List[int] = [] 
        
        # Gen 5: Spectral Index for manifold-based discovery
        self.spectral_index = faiss.IndexFlatL2(5) # Beta is 5D
        self.spectral_sigs: List[int] = [] # Maps spectral index to result signature
        
        # RNS Core for hardware-invariant signatures
        self.rns = RNS_Core()
        
        # Gen 5: Reflexive Dreaming Map
        # Maps Spectral Signatures -> Manifold Laws
        self.dream_map: Dict[int, int] = {} 
        
        # Gen 6: Ana-Kata Tracker (Tracking 4D regime shifts)
        self.ana_kata_tracker: Dict[int, torch.Tensor] = {} 
        
        # Gen 6: Morphological Index (Gielis Params: [m, n1, n2, n3])
        self.morph_index = faiss.IndexFlatL2(4)
        self.morph_sigs: List[int] = []


    def get_signature(self, inputs: torch.Tensor) -> int:
        """Hardware-invariant signature via RNS math."""
        residues = self.rns.float_to_residues(inputs, scale=1000000)
        return self.rns.compute_signature(residues)

    def get_spectral_signature(self, beta: torch.Tensor) -> int:
        """Lower precision signature for spectral manifolds (noise robust)."""
        residues = self.rns.float_to_residues(beta, scale=10) # Even lower for Gen 5
        return self.rns.compute_signature(residues)

    def shunt(self, sig: int, hv: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None, gielis: Optional[torch.Tensor] = None) -> Optional[Any]:
        """
        Hybrid Shunting:
        1. Exact match (O(1) hash).
        2. Reflexive Dream Lookup (O(1) hash).
        3. Morphological Search (Sub-linear FAISS on Gielis params).
        4. Spectral Search (Sub-linear FAISS on beta).
        5. Structural match (Sub-linear FAISS on HDC).
        """
        # 1 & 2. Exact / Dream Match
        law = self.laws.get(sig)
        if not law:
            law_sig = self.dream_map.get(sig)
            if law_sig: law = self.laws.get(law_sig)
            
        if law:
            law.hits += 1
            return law.result
        
        # 3. Morphological Search (Gen 6 Lattice Lock result)
        if gielis is not None:
            g_np = gielis.detach().cpu().numpy().reshape(1, -1)
            dist, idx = self.morph_index.search(g_np, 1)
            if idx[0][0] != -1 and dist[0][0] < 0.01: # High precision required for Lattice Lock
                match_sig = self.morph_sigs[idx[0][0]]
                return self.laws[match_sig].result

        # 4. Spectral Search (Zero-Shot Discovery)
        if beta is not None:
            beta_np = beta.detach().cpu().numpy().reshape(1, -1)
            dist, idx = self.spectral_index.search(beta_np, 1)
            if idx[0][0] != -1 and dist[0][0] < 1.0: # Relaxed for real-world noise
                match_sig = self.spectral_sigs[idx[0][0]]
                # print(f"  [Spectral] Discovery Match! Dist: {dist[0][0]:.4f}")
                return self.laws[match_sig].result

        # 4. Structural Match (Fuzzy HDC)
        if hv is not None:
            hv_np = hv.detach().cpu().numpy().reshape(1, -1)
            dist, idx = self.index.search(hv_np, 1)
            if idx[0][0] != -1 and dist[0][0] < 50.0: # Threshold for structural identity
                match_sig = self.index_sigs[idx[0][0]]
                return self.laws[match_sig].result
                
        return None

    def induce(self, sig: int, result: Any, hv: torch.Tensor, law_sig: Optional[int] = None, beta: Optional[torch.Tensor] = None, gielis: Optional[torch.Tensor] = None):
        """
        Induces a candidate result into the registry.
        """
        if sig not in self.laws:
            self.laws[sig] = Law(sig, result)
            
            # Sparse-HDC Indication
            shv = self.engine.get_sparse_hv(hv)
            shv_np = shv.detach().cpu().numpy().reshape(1, -1)
            self.index.add(shv_np)
            self.index_sigs.append(sig)
            
            # Add to Spectral index
            if beta is not None:
                beta_np = beta.detach().cpu().numpy().reshape(1, -1)
                self.spectral_index.add(beta_np)
                self.spectral_sigs.append(sig)
            
            # Gen 6 Morphological tracking
            if gielis is not None:
                g_np = gielis.detach().cpu().numpy().reshape(1, -1)
                self.morph_index.add(g_np)
                self.morph_sigs.append(sig)
                
                # Tracking 4D Ana-Kata transforms
                # We assume a default zero-state for theta/psi for now
                self.ana_kata_tracker[sig] = gielis
            
            # Create Reflexive Link
            if law_sig is not None:
                self.dream_map[law_sig] = sig
                self.dream_map[sig] = law_sig
        else:
            self.laws[sig].hits += 1
