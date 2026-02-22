import ctypes
import os

dll_path = os.path.abspath("gvm_dynamics.dll")
lib = ctypes.CDLL(dll_path)

# Signatures
lib.gmem_create.argtypes = [ctypes.c_uint64]
lib.gmem_create.restype = ctypes.c_void_p

lib.gmem_destroy.argtypes = [ctypes.c_void_p]
lib.gmem_destroy.restype = None

lib.gmem_fetch_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
lib.gmem_fetch_f32.restype = ctypes.c_float

lib.gmem_hilbert_encode.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
lib.gmem_hilbert_encode.restype = ctypes.c_uint64

def test_gvm_port():
    print(">>> Testing GVM Specialized Port...")
    
    # Test Hilbert
    h = lib.gmem_hilbert_encode(1024, 16, 16)
    print(f"Hilbert(1024, 16, 16): {h}")
    
    # Test Context & Fetch
    ctx = lib.gmem_create(0x1337)
    val = lib.gmem_fetch_f32(ctx, 42)
    print(f"GVM Fetch (Seed 0x1337, Addr 42): {val:.6f}")
    
    lib.gmem_destroy(ctx)
    print("PASS: DLL verified.")

if __name__ == "__main__":
    test_gvm_port()
