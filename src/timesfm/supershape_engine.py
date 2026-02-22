import torch
import math
from typing import Tuple

class SupershapeEngine:
    """
    Gielis Superformula Engine for Morphological Time-Series Analysis.
    Provides kernels for 2D, 3D, and 4D hyperspace projections.
    """
    @staticmethod
    def r_gielis(phi: torch.Tensor, m: torch.Tensor, a: float, b: float, n1: torch.Tensor, n2: torch.Tensor, n3: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Gielis radius. Optimized for massive broadcasting.
        """
        term1 = torch.abs(torch.cos(m * phi / 4.0) / a) ** n2
        term2 = torch.abs(torch.sin(m * phi / 4.0) / b) ** n3
        return (term1 + term2) ** (-1.0 / n1)
    
    @staticmethod
    def ghost_projection(phi: torch.Tensor, m_grid: torch.Tensor, n1_grid: torch.Tensor, n2_grid: torch.Tensor, n3_grid: torch.Tensor) -> torch.Tensor:
        """
        Projects a 5D ghost matrix of Gielis oscillators.
        Shape: [M, N1, N2, N3, Time]
        """
        M = m_grid.view(-1, 1, 1, 1, 1)
        N1 = n1_grid.view(1, -1, 1, 1, 1)
        N2 = n2_grid.view(1, 1, -1, 1, 1)
        N3 = n3_grid.view(1, 1, 1, -1, 1)
        PHI = phi.view(1, 1, 1, 1, -1)
        
        term1 = torch.abs(torch.cos(M * PHI / 4.0)) ** N2
        term2 = torch.abs(torch.sin(M * PHI / 4.0)) ** N3
        R_grid = (term1 + term2) ** (-1.0 / N1)
        return R_grid * torch.sin(PHI)

    @staticmethod
    def super_oscillator(t: torch.Tensor, freq: float, params: torch.Tensor) -> torch.Tensor:
        """
        Generates a non-linear oscillator based on Gielis parameters.
        params: [m, a, b, n1, n2, n3]
        """
        phi = 2 * math.pi * freq * t
        m, a, b, n1, n2, n3 = params[0], params[1], params[2], params[3], params[4], params[5]
        r = SupershapeEngine.r_gielis(phi, m, a, b, n1, n2, n3)
        return r * torch.sin(phi)

    @staticmethod
    def ana_kata_project(r_phi: torch.Tensor, r_theta: torch.Tensor, r_psi: torch.Tensor, 
                        phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Projects a 4D hypershape into [x, y, z, w] Cartesian coordinates.
        Supports 'Ana-Kata' tracking.
        """
        x = r_psi * torch.cos(psi) * r_theta * torch.cos(theta) * r_phi * torch.cos(phi)
        y = r_psi * torch.cos(psi) * r_theta * torch.cos(theta) * r_phi * torch.sin(phi)
        z = r_psi * torch.cos(psi) * r_theta * torch.sin(theta)
        w = r_psi * torch.sin(psi)
        return torch.stack([x, y, z, w], dim=-1)
