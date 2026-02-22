import torch
import torch.optim as optim
from src.timesfm.hwm_zs_engine import HWM_ZS_Engine

def test_gen8_autograd_pass_through():
    print(">>> GEN 8 AUDIT: Autograd Gradient Pass-through Verification")
    
    # Initialize Engine with a learnable seed
    engine = HWM_ZS_Engine(order=10, seed=1234)
    optimizer = optim.SGD([engine.latent_seed], lr=0.1)
    
    # 1. FORWARD PASS (Differentiable Mode)
    # We sample a block of S points
    q = torch.randn(1, 100, 64)
    k = torch.randn(1, 100, 64)
    v = torch.randn(1, 100, 64)
    
    # Target: A specific resonance value we want to 'Learn'
    target = torch.ones(1, 100, 64) * 0.5 
    
    print(f"  [Audit] Initial Latent Seed: {engine.latent_seed.item():.4f}")
    
    # Run forward with mode != 'ground' to trigger differentiability
    out = engine(q, k, v, mode='dream')
    
    loss = torch.nn.functional.mse_loss(out, target)
    print(f"  [Audit] Initial Loss: {loss.item():.6f}")
    
    # 2. BACKWARD PASS
    loss.backward()
    
    grad = engine.latent_seed.grad
    print(f"  [Audit] Latent Seed Gradient: {grad.item() if grad is not None else 'NONE'}")
    
    if grad is not None and grad.item() != 0:
        print("  [OK] GRADIENT DETECTED: Holographic manifold is now differentiable.")
        
        # 3. OPTIMIZATION STEP
        optimizer.step()
        print(f"  [Audit] Updated Latent Seed: {engine.latent_seed.item():.4f}")
        
        # Verify optimization direction
        out_new = engine(q, k, v, mode='dream')
        loss_new = torch.nn.functional.mse_loss(out_new, target)
        print(f"  [Audit] New Loss: {loss_new.item():.6f}")
        
        if loss_new < loss:
            print(">>> TEST 8 PASS: Wavefront is learning via gradient descent.")
        else:
            print(">>> TEST 8 FAIL: Loss did not decrease.")
    else:
        print(">>> TEST 8 FAIL: No gradient Flow through the wavefront.")

if __name__ == "__main__":
    test_gen8_autograd_pass_through()
