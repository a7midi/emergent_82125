# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable
import math
import numpy as np

# Exported by the module (used by tests)
@dataclass(frozen=True)
class CouplingVector:
    g_star: float
    lambda_mix: float
    theta_cp: float = 0.0


@dataclass(frozen=True)
class _Bounds:
    g_lo: float = 0.05
    g_hi: float = 0.98
    l_lo: float = 0.05
    l_hi: float = 1.50


def _bracket_root(f: Callable[[float], float], a: float, b: float, *, n_grid: int = 129) -> Tuple[float, float]:
    """
    Find [lo, hi] inside [a, b] where f changes sign. If no sign change is found,
    pick the adjacent pair with smallest |f(x_i) * f(x_{i+1})|; bisection on that
    pair will still converge to a minimum and typically to a root in our setting.
    """
    xs = np.linspace(float(a), float(b), int(max(3, n_grid)))
    fs = np.array([float(f(x)) for x in xs], dtype=float)
    # try exact sign change
    s = np.sign(fs)
    for i in range(len(xs) - 1):
        if s[i] == 0.0:
            return float(xs[i]), float(xs[i + 1])
        if s[i] != s[i + 1]:
            return float(xs[i]), float(xs[i + 1])
    # weakest adjacent product
    prods = np.abs(fs[:-1] * fs[1:])
    j = int(np.argmin(prods))
    return float(xs[j]), float(xs[j + 1])


def _bisect(f: Callable[[float], float], lo: float, hi: float, *, tol: float = 1e-12, maxit: int = 200) -> float:
    flo = float(f(lo))
    fhi = float(f(hi))
    a, b = float(lo), float(hi)
    fa, fb = flo, fhi
    for _ in range(maxit):
        m = 0.5 * (a + b)
        fm = float(f(m))
        # If exactly zero or interval is small enough, done
        if abs(fm) <= tol or abs(b - a) <= tol:
            return float(m)
        # Choose side (allow fallback when signs equal by picking smaller |f|)
        if fa == 0.0:
            return float(a)
        if fb == 0.0:
            return float(b)
        # Prefer sign change; otherwise move toward smaller |f|
        if (fa < 0.0 and fm > 0.0) or (fa > 0.0 and fm < 0.0):
            b, fb = m, fm
        else:
            if abs(fa) <= abs(fb):
                a, fa = m, fm
            else:
                b, fb = m, fm
    return float(0.5 * (a + b))


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
      1) Fix the physical clock GeV0 at the EW anchor.
      2) If provided, pin α_EM(EW) exactly (supported by paper hooks).
      3) Solve for g* so that sin^2 θ_W(EW) hits target_sin2_EW.

    Returns a dict with g_star_cal, lambda_mix_cal, residuals, bracket, GeV0, etc.
    """
    # 1) Fix the physical clock by the Z anchor
    if hasattr(hooks, "set_GeV0_by_anchors"):
        hooks.set_GeV0_by_anchors(mu_Z=float(mu_EW_GeV), k_Z=float(k_end), prefer="Z")

    # 2) Enforce α(M_Z) exactly if the hooks support it
    if (target_alpha_EW is not None) and hasattr(hooks, "set_alpha_anchor"):
        hooks.set_alpha_anchor(float(target_alpha_EW))

    # helper: evaluate sin^2 and α at EW for given (g*, λ)
    def _eval(g_star: float, lam: float) -> Tuple[float, float]:
        from .predict import predict_weak_mixing_curve  # local import to avoid cycles
        gv = CouplingVector(g_star=float(g_star), lambda_mix=float(lam), theta_cp=g_template.theta_cp)
        _, summ = predict_weak_mixing_curve(
            gv, q=q, R=R, k_start=k_start, k_end=k_end, n_grid=81, bootstrap=0, seed=0, hooks=hooks
        )
        return float(summ["sin2_thetaW_EW"]), float(summ["alpha_EM_EW"])

    lam0 = float(g_template.lambda_mix)

    # Optional coarse sweep to pick λ; default is g-only root
    if str(mode).lower().startswith("g_and_lambda") and (target_alpha_EW is not None):
        Gs = np.linspace(bounds.g_lo, bounds.g_hi, 25)
        Ls = np.linspace(max(bounds.l_lo, 1e-6), min(bounds.l_hi, 1.50), 25)
        best = (math.inf, lam0, float(g_template.g_star))
        for gv in Gs:
            for lv in Ls:
                s2, a = _eval(gv, lv)
                e1 = s2 - float(target_sin2_EW)
                e2 = (a - float(target_alpha_EW))
                fval = e1 * e1 + e2 * e2
                if fval < best[0]:
                    best = (fval, float(lv), float(gv))
        lam0 = best[1]

    # 3) 1D root in g* for sin^2 θ_W at EW
    def f_root(gv: float) -> float:
        s2, _ = _eval(float(gv), lam0)
        return s2 - float(target_sin2_EW)

    a, b = _bracket_root(f_root, float(bounds.g_lo), float(bounds.g_hi), n_grid=n_grid)
    g_star_cal = _bisect(f_root, a, b, tol=1e-12, maxit=250)

    s2_fin, a_fin = _eval(g_star_cal, lam0)

    GeV0 = float(hooks.get_GeV0()) if hasattr(hooks, "get_GeV0") else float("nan")
    return {
        "g_star_cal": float(g_star_cal),
        "lambda_mix_cal": float(lam0),
        "success": True,
        "message": "bounded minimization (root in g*, α anchored)" if target_alpha_EW is not None else "bounded minimization (root in g*)",
        "residual": abs(s2_fin - float(target_sin2_EW)),
        "residuals": (s2_fin - float(target_sin2_EW),
                      (a_fin - float(target_alpha_EW)) if (target_alpha_EW is not None) else float("nan")),
        "bracket": (float(a), float(b)),
        "GeV0": GeV0,
        "xi2_cal": None,
    }
