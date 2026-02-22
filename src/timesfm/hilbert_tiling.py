import torch

def hilbert_encode(i: int, j: int, order: int) -> int:
    """
    Encodes (i, j) coordinates into a 1D Hilbert index.
    Bijective mapping for 2D -> 1D.
    """
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (i & s) > 0 else 0
        ry = 1 if (j & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        
        # Rotate/Flip
        if ry == 0:
            if rx == 1:
                i = s - 1 - i
                j = s - 1 - j
            i, j = j, i
            
        s >>= 1
    return d

def hilbert_decode(d: int, order: int) -> tuple:
    """
    Decodes a 1D Hilbert index back into (i, j) coordinates.
    """
    i, j = 0, 0
    t = d
    s = 1
    while s < (1 << order):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        
        # Rotate/Flip
        if ry == 0:
            if rx == 1:
                i = s - 1 - i
                j = s - 1 - j
            i, j = j, i
            
        i += s * rx
        j += s * ry
        t //= 4
        s <<= 1
    return i, j

def hilbert_wavefront_tensor(order: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generates a full Hilbert Wavefront Map for an N x N matrix.
    N = 2^order.
    """
    n = 1 << order
    grid = torch.zeros(n, n, dtype=torch.long, device=device)
    for i in range(n):
        for j in range(n):
            grid[i, j] = hilbert_encode(i, j, order)
    return grid
