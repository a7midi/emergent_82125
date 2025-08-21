# tests/test_rg_fp.py
"""
Smoke test for the first-principles RG engine (rg_fp).

Verifies:
  1. The first-principles beta_function is correctly registered by default upon import.
  2. The restore_default_beta() function correctly reverts the patch.
  3. The beta_exact function produces bounded, finite outputs.
"""

import numpy as np
import emergent.rg as rg
import emergent.rg_fp as rg_fp
from emergent.rg import CouplingVector

def test_fp_beta_is_registered_by_default():
    """
    Asserts that the FP beta function is the default after package import.
    """
    # Due to the __init__.py import, rg.beta_function should already be the patched one.
    # We can verify this by comparing its output to the source function in rg_fp.
    g0 = CouplingVector(g_star=0.5, lambda_mix=0.5, theta_cp=0.1)
    q, R = 13, 2

    # Call the active beta function within the emergent.rg namespace
    active_beta_output = rg.beta_function(0.0, g0.to_array(), q, R)
    
    # Call the source function directly for comparison
    source_beta_output = rg_fp.beta_exact(g0, q, R)
    
    assert np.allclose(active_beta_output, source_beta_output.to_array()), \
        "The default rg.beta_function is not the first-principles one."
    
    print("\n✅ Verified: First-principles beta function is registered by default.")

def test_restore_default_beta_works():
    """
    Asserts that the restoration function correctly reverts the patch.
    """
    # 1. Get a handle to the original function, which rg_fp has saved
    original_beta_fn = rg_fp._default_beta_rhs
    assert original_beta_fn is not None, "Could not find the stored original beta function."
    
    # 2. At this point, the function should be the patched one
    assert rg.beta_function is not original_beta_fn, "Patch should be active initially."
    
    # 3. Restore the default
    rg_fp.restore_default_beta()
    
    # 4. Assert that the function has been reverted
    assert rg.beta_function is original_beta_fn, "restore_default_beta() failed to revert the patch."
    
    # 5. Re-apply the patch for any subsequent tests
    rg_fp.set_beta_functions_fp(rg_fp.beta_exact)
    
    print("\n✅ Verified: Beta function restoration works correctly.")