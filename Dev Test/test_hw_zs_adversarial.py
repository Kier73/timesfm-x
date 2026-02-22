import torch
import math
from src.timesfm.hwm_zs_engine import HWM_ZS_Engine

def test_adversarial_aliasing():
    print(">>> ADVERSARIAL AUDIT: Testing for Sawtooth Aliasing (% 1000)")
    
    order = 10
    engine = HWM_ZS_Engine(order=order)
    
    # Check for periodic collapse at the modulo boundary
    # If the engineer is right, H=1000 and H=2000 should show high autocorrelation
    h1 = torch.arange(1000, 1010)
    h2 = torch.arange(2000, 2010)
    
    res1 = engine.resolve_manifold(h1 // 1024, h1 % 1024, 0xAAAA, 0xBBBB)
    res2 = engine.resolve_manifold(h2 // 1024, h2 % 1024, 0xAAAA, 0xBBBB)
    
    diff = torch.abs(res1 - res2).mean().item()
    print(f"  [Audit] Mean Divergence at Modulo Boundary (1000 vs 2000): {diff:.8f}")
    
    if diff < 1e-4:
        print("  [WARNING] ADVERSARIAL RISK: Aliasing detected. Modulo 1000 is causing periodic repetition.")
    else:
        print("  [OK] NOISE FLOOD: XOR variety is successfully masking the modulo periodicity.")

def test_adversarial_differentiability():
    print(">>> ADVERSARIAL AUDIT: Testing for Autograd Gradient Pass-through")
    
    # Create a parameter that should influence the wavefront
    seed_tensor = torch.tensor([0xBEEF], dtype=torch.float32, requires_grad=True)
    
    engine = HWM_ZS_Engine(order=10)
    
    # Attempt to resolve an element using a differentiable 'path'
    # NOTE: The current XOR logic is bitwise (integer), so it WILL break autograd.
    i, j = torch.tensor([512]), torch.tensor([512])
    
    try:
        # We manually call the logic to check for 'grad_fn'
        t = engine.h_map[i, j]
        # interaction_sig uses XOR...
        res = engine.resolve_manifold(i, j, seed_tensor.long().item(), 0xBBBB)
        
        # Check if 'res' has a grad_fn
        print(f"  [Audit] Result has grad_fn: {res.grad_fn}")
        if res.grad_fn is None:
            print("  [CRITICAL] BLOCKER: Wavefront is non-differentiable. Learning via backprop is impossible.")
        else:
            print("  [OK] Wavefront is differentiable.")
            
    except Exception as e:
        print(f"  [Error] Autograd Failure: {e}")

if __name__ == "__main__":
    test_adversarial_aliasing()
    test_adversarial_differentiability()
