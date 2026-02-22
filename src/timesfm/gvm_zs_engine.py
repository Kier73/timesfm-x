import torch
import torch.nn as nn
import numpy as np
import ctypes
import os
import sys
from typing import Optional

class GVM_ZS_Engine(nn.Module):
    """
    Gen 8: GVM-Backed Zero-Shot Engine.
    Uses gvm_dynamics.dll for C-accelerated Hilbert and VRNS synthesis.
    Supports Sparse Overlay "Materialization" for adapting to regime shifts.
    """
    def __init__(self, order: int = 10, seed: int = 0xBEEF):
        super().__init__()
        self.order = order
        self.dim = 1 << order
        self.seed = seed
        
        # Load DLL
        dll_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Generative_Memory')
        dll_path = os.path.abspath(os.path.join(dll_dir, 'gvm_dynamics.dll'))
        
        try:
            self.lib = ctypes.CDLL(dll_path)
            self._setup_ctypes()
            self.ctx = self.lib.gmem_create(seed)
            print(f"  [GVM Engine] Gen 8 Backend Online (order={order}, seed={hex(seed)})")
            
            # Hybrid Cache: For small scales, pre-calculate the manifold into a tensor.
            # This avoids DLL overhead and for-loops during the hot forward pass.
            if order <= 10:
                print(f"  [GVM Engine] Hydrating 2D Cache ({self.dim}x{self.dim})...")
                self.register_buffer("h_cache", torch.zeros((self.dim, self.dim), dtype=torch.float32))
                
                # Vectorized Hydration
                num_pts = self.dim * self.dim
                y_coords, x_coords = torch.meshgrid(torch.arange(self.dim), torch.arange(self.dim), indexing='ij')
                
                flat_x = x_coords.flatten().numpy().astype(np.uint64)
                flat_y = y_coords.flatten().numpy().astype(np.uint64)
                
                h_indices = (ctypes.c_uint64 * num_pts)()
                self.lib.gmem_bulk_hilbert_encode(
                    self.dim, 
                    flat_x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    flat_y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    h_indices,
                    num_pts
                )
                
                res_buffer = (ctypes.c_float * num_pts)()
                self.lib.gmem_bulk_fetch_f32(self.ctx, h_indices, res_buffer, num_pts)
                
                self.h_cache.copy_(torch.from_numpy(np.frombuffer(res_buffer, dtype=np.float32)).view(self.dim, self.dim))
                print(f"  [GVM Engine] Cache Ready (Vectorized Path Active).")
            else:
                self.register_buffer("h_cache", None)
                
        except Exception as e:
            print(f"  [GVM Engine] ERROR loading backend: {e}")
            self.lib = None
            self.ctx = None

    def _setup_ctypes(self):
        self.lib.gmem_create.argtypes = [ctypes.c_uint64]
        self.lib.gmem_create.restype = ctypes.c_void_p
        
        self.lib.gmem_destroy.argtypes = [ctypes.c_void_p]
        self.lib.gmem_destroy.restype = None
        
        self.lib.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.gmem_fetch_f32.restype = ctypes.c_float
        
        self.lib.gmem_write_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_float]
        self.lib.gmem_write_f32.restype = None
        
        self.lib.gmem_hilbert_encode.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.gmem_hilbert_encode.restype = ctypes.c_uint64
        
        self.lib.gmem_bulk_hilbert_encode.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
        self.lib.gmem_bulk_hilbert_encode.restype = None
        
        self.lib.gmem_bulk_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64]
        self.lib.gmem_bulk_fetch_f32.restype = None

        self.lib.gmem_trinity_solve_rns.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, 
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
        ]
        self.lib.gmem_trinity_solve_rns.restype = ctypes.c_uint64

    def materialize(self, i: int, j: int, value: float):
        """Writes a delta to the Sparse Overlay and updates the cache if active."""
        if self.ctx and self.lib:
            h_idx = self.lib.gmem_hilbert_encode(self.dim, i, j)
            self.lib.gmem_write_f32(self.ctx, h_idx, value)
            
            # Sync cache if we are in small-scale mode
            if hasattr(self, 'h_cache') and self.h_cache is not None:
                if i < self.dim and j < self.dim:
                    self.h_cache[i, j] = value

    def resolve_manifold(self, i_indices: torch.Tensor, j_indices: torch.Tensor) -> torch.Tensor:
        """Resolves spatiotemporal wavefront via Hybrid Cache or C backend."""
        # Gen 8 Vectorized Path: Fast tensor lookup for small scales
        if hasattr(self, 'h_cache') and self.h_cache is not None:
            if i_indices.max() < self.dim and j_indices.max() < self.dim:
                return self.h_cache[i_indices.long(), j_indices.long()]

        if not self.ctx or not self.lib:
            # Fallback to zero if backend failure
            return torch.zeros_like(i_indices, dtype=torch.float32)

        # Procedural Path (Vectorized): Extreme scales (Million/Billion points)
        flat_i = i_indices.flatten().long()
        flat_j = j_indices.flatten().long()
        num_elements = flat_i.numel()
        
        # Scalar Optimization: If only one point, avoid buffer allocation overhead
        if num_elements == 1:
            h_idx = self.lib.gmem_hilbert_encode(self.dim, flat_i[0].item(), flat_j[0].item())
            val = self.lib.gmem_fetch_f32(self.ctx, h_idx)
            return torch.tensor([val], device=i_indices.device).view(i_indices.shape)

        # Vectorized Path (C-Level Bulk): Single DLL calls for all points
        # 1. Encode Hilbert indices (C-NATIVELY)
        h_indices = (ctypes.c_uint64 * num_elements)()
        # We need to pass the raw uint64 pointers
        self.lib.gmem_bulk_hilbert_encode(
            self.dim, 
            flat_i.numpy().astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            flat_j.numpy().astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            h_indices,
            num_elements
        )
        
        # 2. Bulk fetch result buffer
        res_buffer = (ctypes.c_float * num_elements)()
        
        # 3. DLL Single Call (Eliminates for-loop / ctypes context switching)
        self.lib.gmem_bulk_fetch_f32(self.ctx, h_indices, res_buffer, num_elements)
        
        return torch.tensor(res_buffer, device=i_indices.device).view(i_indices.shape)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mode: str = 'ground') -> torch.Tensor:
        """
        Gen 9: 2D Spatiotemporal Interference.
        Resolves the full S x S resonant mask.
        """
        B, S, D = Q.shape
        
        # 2D Grid: All-to-all interference
        s_indices = torch.arange(S, device=Q.device)
        i_idx, j_idx = torch.meshgrid(s_indices, s_indices, indexing='ij')
        
        # Resolve spatiotemporal interference for the entire attention landscape
        # This uses the high-speed Bulk Fetch path for massive S
        psi = self.resolve_manifold(i_idx, j_idx) # [S, S]
        
        # Return interference mask [1, S, S]
        return psi.unsqueeze(0)

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.gmem_destroy(self.ctx)
