import torch
import numpy as np

class RNS_Core:
    """
    Residue Number System (RNS) Implementation for PyTorch.
    Converts floating-point manifolds into bit-identical modular integer residues.
    """
    def __init__(self, primes: torch.Tensor = None):
        if primes is None:
            # First 5 primes that fit nicely in int32/int64 modular math
            self.primes = torch.tensor([1000003, 1000033, 1000037, 1000039, 1000081], dtype=torch.int64)
        else:
            self.primes = primes

    def float_to_residues(self, x: torch.Tensor, scale: int = 1000000) -> torch.Tensor:
        """
        Maps float -> Scaled Integer -> Prime Residues.
        Uses round() to prevent snap-errors from device-local float epsilon.
        """
        # 1. Scale, Round, and cast to long
        scaled_x = torch.round(x * scale).long()
        
        # 2. Compute residues
        # x mod p for each prime p
        residues = []
        for p in self.primes:
            residues.append(scaled_x % p)
        
        return torch.stack(residues, dim=-1)

    @staticmethod
    def modinv(a: int, m: int) -> int:
        if m == 1: return 0
        m0, y, x = m, 0, 1
        while a > 1:
            q = a // m
            t = m
            m = a % m
            a = t
            t = y
            y = x - q * y
            x = t
        if x < 0: x += m0
        return x

    def mrc_reconstruct(self, residues: torch.Tensor) -> int:
        """
        Mixed-Radix Reconstruction (MRC) for positional residue stability.
        X = a1 + a2*m1 + a3*(m1*m2) + ...
        """
        moduli = self.primes.tolist()
        res_list = residues.tolist()
        if not isinstance(res_list, list): res_list = [res_list]
        
        k = len(moduli)
        a = [0] * k
        a[0] = res_list[0]
        
        for i in range(1, k):
            temp = res_list[i]
            product = 1
            for j in range(i):
                term = (a[j] * product) % moduli[i]
                temp = (temp - term) % moduli[i]
                product = (product * moduli[j]) % moduli[i]
            
            inv_product = self.modinv(product, moduli[i])
            a[i] = (temp * inv_product) % moduli[i]
        
        result = 0
        current_m = 1
        for i in range(k):
            result += a[i] * current_m
            current_m *= moduli[i]
            
        return result

    def compute_signature(self, residues: torch.Tensor) -> int:
        """
        Bundles residues into a single 64-bit deterministic hash using MRC.
        """
        # Sum of residues is no longer enough for Gen 5.
        # We use MRC to get a positional reconstructed value.
        # This value is then hashed for the signature.
        
        # We take the mean residue per prime to consolidate the signal
        # before reconstruction if multiple points are passed (e.g. context window)
        if residues.dim() > 1:
            mean_res = torch.mean(residues.float(), dim=0).long()
        else:
            mean_res = residues
            
        combined_val = self.mrc_reconstruct(mean_res)
        
        # Murmur-style final mix (fmix64)
        h = int(combined_val) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        h = (h * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> 33
        return h & 0xFFFFFFFFFFFFFFFF
