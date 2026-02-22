import torch
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from timesfm.rns_kernels import RNS_Core

def modinv(a, m):
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

def mrc_reconstruct(residues, moduli):
    """Mixed-Radix Reconstruction algorithm."""
    k = len(moduli)
    a = [0] * k
    a[0] = residues[0]
    
    for i in range(1, k):
        temp = residues[i]
        product = 1
        for j in range(i):
            term = (a[j] * product) % moduli[i]
            temp = (temp - term) % moduli[i]
            product = (product * moduli[j]) % moduli[i]
        
        inv_product = modinv(product, moduli[i])
        a[i] = (temp * inv_product) % moduli[i]
    
    result = 0
    current_m = 1
    for i in range(k):
        result += a[i] * current_m
        current_m *= moduli[i]
        
    return result

def test_mrc():
    print(">>> Testing Mixed-Radix Reconstruction (MRC)")
    moduli = [1000003, 1000033, 1000037, 1000039, 1000081]
    
    # Test with a large value
    target_value = 123456789012345
    residues = [target_value % m for m in moduli]
    
    print(f"Target Value: {target_value}")
    print(f"Residues:     {residues}")
    
    reconstructed = mrc_reconstruct(residues, moduli)
    print(f"Reconstructed: {reconstructed}")
    
    if reconstructed == (target_value % (1000003 * 1000033 * 1000037 * 1000039 * 1000081)):
        print("PASS: MRC Reconstruction successful (within Dynamic Range).")
    else:
        print("FAIL: MRC Reconstruction mismatch.")

if __name__ == "__main__":
    test_mrc()
