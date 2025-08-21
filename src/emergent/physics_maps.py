# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib

class HooksProtocol:
    """Informal protocol (duck-typed)."""

def make_hooks_from_module(module_path: str):
    """
    Import `module_path` and call its make_hooks(). Perform a duck-typed check
    so downstream code can rely on the expected surface.
    """
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "make_hooks"):
        raise ImportError(f"{module_path} does not define make_hooks().")
    h = mod.make_hooks()

    # Required public surface
    required = ("gauge_couplings", "lambda_from_qR", "edm_from_rg")
    for name in required:
        if not hasattr(h, name):
            raise TypeError(
                f"{module_path}.make_hooks() returned an object without '{name}'."
            )

    # Optional helpers (used by the calibration / plotting helpers)
    # We accept their absence; callers will guard with hasattr.
    #   set_GeV0_by_anchors(mu_Z, k_Z, prefer="Z")
    #   get_GeV0()
    #   set_alpha_anchor(alpha)
    #   alpha_from_gauge(g1,g2,k)

    return h
