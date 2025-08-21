# -*- coding: utf-8 -*-
"""
Calibration helpers for the paper tests and the notebook.

Key fixes:
- Robust two-target calibration that *minimizes* the |sin² - target| error
  in g* even when there is no sign change (avoids false "roots").
- Optional coarse 2-D sweep over (g*, λ) before refining g* in 1-D.
- Exact α(M_Z) anchoring when the hooks provide set_alpha_anchor/get_alpha_anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import math
import numpy as np


# ---------------------------------------------------------------------
# Public coupling vector and bounds (imported by predict.py and tests)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CouplingVector:
    g_star: float
    lambda_mix: float
    theta_cp: float = 0.0


@dataclass
class _Bounds:
    g_lo: float = 0.05
    g_hi: float = 1.50
    l_lo: float = 0.05
    l_hi: float = 1.50


# ---------------------------------------------------------------------
# Simple, robust 1-D minimizer (golden-section) and 1-D bracket
# ---------------------------------------------------------------------

def _golden_minimize(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10, maxit: int = 200) -> float:
    """Golden-section search to *minimize* f on [a, b]."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618...
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = float(f(c))
    fd = float(f(d))
    it = 0
    while (b - a) > tol and it < maxit:
        it += 1
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(f(c))
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(f(d))
    return 0.5 * (a + b)


def _coarse_grid_argmin(f: Callable[[float], float], a: float, b: float, n: int = 41) -> float:
    xs = np.linspace(float(a), float(b), int(max(3, n)))
    vals = [float(f(x)) for x in xs]
    j = int(np.argmin(vals))
    return float(xs[j])


# ---------------------------------------------------------------------
# Main entry: two-anchor calibration
# ---------------------------------------------------------------------

def calibrate_two_anchors(
    g_template: CouplingVector,
    *,
    q: int,
    R: int,
    k_start: float,
    k_end: float,
    target_sin2_EW: float,
    mu_EW_GeV: float,
    hooks,
    mode: str = "g_only",
    target_alpha_EW: Optional[float] = None,
    bounds: _Bounds = _Bounds(),
    n_grid: int = 129,
) -> Dict:
    """
    Two-target calibration:
      (i) Fix physical clock GeV0 at the EW point.
      (ii) If available, enforce α_EM(M_Z) exactly by calling hooks.set_alpha_anchor.
      (iii) Choose (optionally λ, always g*) to minimize |sin²θ_W(EW) - target|.

    Returns:
      dict(g_star_cal, lambda_mix_cal, success, message, residual, residuals, bracket, GeV0, xi2_cal)

    NOTE: We avoid root-finding assumptions. Even if sin²(g*)-target has no sign change
          in the bracket, we get the *best* g* on the interval.
    """
    # 1) Fix the physical clock at the EW point (Z anchor)
    if hasattr(hooks, "set_GeV0_by_anchors"):
        # prefer Z so k_end maps to mu_Z
        try:
            hooks.set_GeV0_by_anchors(mu_Z=float(mu_EW_GeV), k_Z=float(k_end), prefer="Z")
        except TypeError:
            # older signature
            hooks.set_GeV0_by_anchors(float(mu_EW_GeV), float(k_end))

    # 2) Enforce α anchor exactly (if the hooks support it)
    if (target_alpha_EW is not None) and hasattr(hooks, "set_alpha_anchor"):
        hooks.set_alpha_anchor(float(target_alpha_EW))

    # Helper: evaluate sin^2 and α at EW for given (g*, λ)
    def _eval(g_star: float, lam: float) -> Tuple[float, float]:
        from .predict import predict_weak_mixing_curve  # local import to avoid cycles
        gv = type(g_template)(g_star=float(g_star), lambda_mix=float(lam),
                              theta_cp=g_template.theta_cp)
        _, summ = predict_weak_mixing_curve(
            gv, q=q, R=R, k_start=k_start, k_end=k_end,
            n_grid=81, bootstrap=0, seed=0, hooks=hooks
        )
        return float(summ["sin2_thetaW_EW"]), float(summ["alpha_EM_EW"])

    # 3) Decide λ (coarse 2-D search if requested), then refine g* in 1-D
    lam0 = float(g_template.lambda_mix)

    mode_lower = str(mode).strip().lower()
    if mode_lower.startswith("g_and_lambda"):
        Gs = np.linspace(bounds.g_lo, bounds.g_hi, 27)
        Ls = np.linspace(bounds.l_lo, bounds.l_hi, 27)
        best = (math.inf, lam0, float(g_template.g_star))
        for gv in Gs:
            for lv in Ls:
                s2, a = _eval(gv, lv)
                e1 = s2 - float(target_sin2_EW)
                e2 = (a - float(target_alpha_EW)) if (target_alpha_EW is not None) else 0.0
                f = e1*e1 + e2*e2
                if f < best[0]:
                    best = (f, float(lv), float(gv))
        lam0 = best[1]

    # Minimize the 1-D objective in g* (no sign-change assumption)
    def _obj_g(gv: float) -> float:
        s2, _ = _eval(float(gv), lam0)
        return (s2 - float(target_sin2_EW))**2

    # Coarse seed then golden-section refine
    g_seed = _coarse_grid_argmin(_obj_g, float(bounds.g_lo), float(bounds.g_hi), n=int(max(9, n_grid // 3)))
    g_lo, g_hi = float(bounds.g_lo), float(bounds.g_hi)
    # Shrink around the seed for robustness
    rad = max(0.02, 0.25 * (g_hi - g_lo))
    a = max(g_lo, g_seed - rad)
    b = min(g_hi, g_seed + rad)
    g_star_cal = _golden_minimize(_obj_g, a, b, tol=1e-12, maxit=300)

    s2_fin, a_fin = _eval(g_star_cal, lam0)

    # Safe GeV0 retrieval (function or property)
    GeV0_val = math.nan
    if hasattr(hooks, "get_GeV0"):
        g0_attr = getattr(hooks, "get_GeV0")
        GeV0_val = float(g0_attr() if callable(g0_attr) else g0_attr)

    return {
        "g_star_cal": float(g_star_cal),
        "lambda_mix_cal": float(lam0),
        "success": True,
        "message": "bounded minimization (root-free, α anchored)" if target_alpha_EW is not None
                   else "bounded minimization (root-free)",
        "residual": abs(s2_fin - float(target_sin2_EW)),
        "residuals": (
            s2_fin - float(target_sin2_EW),
            (a_fin - float(target_alpha_EW)) if (target_alpha_EW is not None) else float("nan"),
        ),
        "bracket": (float(a), float(b)),
        "GeV0": GeV0_val,
        "xi2_cal": None,
    }
