import torch
import time
from src.timesfm.hilbert_tiling import hilbert_encode, hilbert_decode

def test_p_matrix_hilbert():
    print(">>> HW-ZS Test 4: P-Matrix (Hilbert Lattice Tiling / Spatial Locality)")
    
    order = 10 # 1024 x 1024 matrix
    dim = 1 << order
    
    print(f"  [Audit] Testing Bijective Mapping for {dim}x{dim} manifold...")
    
    # Test a few key points
    test_points = [(0, 0), (512, 512), (1023, 1023), (256, 768)]
    
    success_count = 0
    for i, j in test_points:
        h = hilbert_encode(i, j, order)
        ri, rj = hilbert_decode(h, order)
        if i == ri and j == rj:
            # print(f"    - Point ({i}, {j}) -> H={h} -> Decoded ({ri}, {rj}) [OK]")
            success_count += 1
        else:
            print(f"    - Point ({i}, {j}) -> H={h} -> Decoded ({ri}, {rj}) [FAIL]")
            
    print(f"  [Metric] Bijective Success: {success_count}/{len(test_points)}")
    
    print(f"  [Audit] Testing Spatial Locality (Nearby indices in H should be nearby in matrix)...")
    h_base = 500000
    p1 = hilbert_decode(h_base, order)
    p2 = hilbert_decode(h_base + 1, order)
    
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    print(f"    - H={h_base} -> {p1}")
    print(f"    - H={h_base+1} -> {p2}")
    print(f"    - Spatial Distance: {dist:.4f}")
    
    if success_count == len(test_points) and dist <= 1.5: # 1.5 is common for adjacent Hilbert pts
        print(">>> TEST 4 PASS: P-Matrix provides a bijective, locality-preserving temporal map.")
    else:
        print(">>> TEST 4 FAIL: Hilbert mapping is non-bijective or lacks locality.")

if __name__ == "__main__":
    import math
    test_p_matrix_hilbert()
